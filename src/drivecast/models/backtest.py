"""Rolling-origin backtest: train on the past, test on the next quarter, for every quarter.

For a test quarter Q the models are fitted on the training samples of the four quarters before
it. Training rows from the last 30 days before Q are dropped (an embargo): their labels look
into Q, which a model deployed on the first day of Q could not have known. The test quarter is
scored in full (every drive and day with a complete label), in batches, and evaluated per model.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drivecast.features.build import HORIZON_DAYS, feature_names, previous_quarter
from drivecast.models import evaluate
from drivecast.models.estimators import LightGBM, make
from drivecast.quality.published import quarter_bounds

MODELS = ("rule", "logreg", "lightgbm")
TRAIN_QUARTERS = 4
BATCH = 1_000_000


def train_quarters(test: str, n: int = TRAIN_QUARTERS) -> list[str]:
    quarters, q = [], test
    for _ in range(n):
        q = previous_quarter(q)
        quarters.append(q)
    return quarters[::-1]


def load_training(con: Any, files: list[str], test: str) -> pd.DataFrame:
    """The weighted training rows from ``files``, without the embargoed 30 days."""
    start, _ = quarter_bounds(test)
    listed = ", ".join(f"'{f}'" for f in files)
    cols = ", ".join(f"CAST({c} AS DOUBLE) AS {c}" for c in feature_names())
    frame: pd.DataFrame = con.execute(
        f"SELECT {cols}, label, weight FROM read_parquet([{listed}]) "
        f"WHERE label_complete AND date < DATE '{start}' - INTERVAL {HORIZON_DAYS} DAYS"
    ).df()
    return frame


def score_test(
    con: Any, features: str, test: str, models: dict[str, Any], scores_out: Path | None
) -> dict[str, Any]:
    """Score every complete-label row of the test quarter; arrays for the evaluation."""
    start, end = quarter_bounds(test)
    cols = ", ".join(f"CAST({c} AS DOUBLE) AS {c}" for c in feature_names())
    reader = con.execute(
        f"SELECT (hash(serial_number) >> 1)::BIGINT AS drive, "
        f"datediff('day', DATE '1970-01-01', date)::INTEGER AS day, label, "
        f"coalesce(days_to_failure, -1)::DOUBLE AS days_to_failure, "
        f"coalesce(days_to_failure BETWEEN 0 AND {HORIZON_DAYS - 1} AND "
        f"date + to_days(days_to_failure::INTEGER) < DATE '{end}', false) AS fails_in_quarter, "
        f"{cols} FROM read_parquet('{features}') WHERE label_complete "
        f"AND date >= DATE '{start}' AND date < DATE '{end}'"
    ).fetch_record_batch(BATCH)
    parts: dict[str, list[Any]] = {k: [] for k in ("drive", "day", "label", "dtf", "fiq")}
    scores: dict[str, list[Any]] = {name: [] for name in models}
    for batch in reader:
        frame = batch.to_pandas()
        parts["drive"].append(frame["drive"].to_numpy(np.int64))
        parts["day"].append(frame["day"].to_numpy(np.int32))
        parts["label"].append(frame["label"].to_numpy(bool))
        parts["dtf"].append(frame["days_to_failure"].to_numpy(np.float64))
        parts["fiq"].append(frame["fails_in_quarter"].to_numpy(bool))
        x = frame[feature_names()]
        for name, model in models.items():
            scores[name].append(model.score(x))
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    scored = {name: np.concatenate(v) for name, v in scores.items()}
    if scores_out is not None:
        table = pd.DataFrame(
            {"drive": arrays["drive"], "day": arrays["day"], "label": arrays["label"],
             "days_to_failure": arrays["dtf"], "fails_in_quarter": arrays["fiq"]}
            | {f"score_{n}": s for n, s in scored.items()}
        )  # fmt: skip
        scores_out.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(scores_out, index=False)
    return {"arrays": arrays, "scores": scored}


def run_quarter(
    con: Any,
    test: str,
    gold: str,
    *,
    models: tuple[str, ...] = MODELS,
    threads: int = 4,
    scores_out: Path | None = None,
) -> dict[str, Any]:
    """Fit on the four quarters before ``test`` and evaluate on ``test``; ``gold`` is a prefix."""
    begin = time.monotonic()
    quarters = train_quarters(test)
    files = [f"{gold}/train_{q}.parquet" for q in quarters]
    train = load_training(con, files, test)
    x, y, w = train[feature_names()], train["label"].to_numpy(bool), train["weight"].to_numpy()
    fitted: dict[str, Any] = {}
    fit_seconds: dict[str, float] = {}
    for name in models:
        t = time.monotonic()
        fitted[name] = make(name, threads=threads).fit(x, y, w)
        fit_seconds[name] = round(time.monotonic() - t, 1)
    scored = score_test(con, f"{gold}/features_{test}.parquet", test, fitted, scores_out)
    a = scored["arrays"]
    result: dict[str, Any] = {
        "quarter": test,
        "train_quarters": quarters,
        "train_rows": len(train),
        "train_positive_rows": int(y.sum()),
        "fit_seconds": fit_seconds,
        "models": {},
    }
    rows = evaluate.TestRows(a["drive"], a["day"], a["label"], a["dtf"], a["fiq"])
    for name, score in scored["scores"].items():
        result["models"][name] = evaluate.evaluate(rows, score)
    if isinstance(fitted.get("lightgbm"), LightGBM):
        result["importance"] = dict(list(fitted["lightgbm"].importance().items())[:20])
    result["seconds"] = round(time.monotonic() - begin, 1)
    return result


def backtest_quarters(first: str = "2015Q2", last: str | None = None) -> list[str]:
    """Test quarters: each needs four earlier quarters of training data (data starts 2013Q2)."""
    from drivecast.lake.silver import quarters

    return quarters(first, last)


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Per model: the mean and median of each metric over the quarters, and paired wins."""
    results = sorted(results, key=lambda r: r["quarter"])
    names = list(results[0]["models"])

    def metric(r: dict[str, Any], model: str) -> dict[str, float]:
        m = r["models"][model]
        return {
            "average_precision": m["average_precision"],
            "roc_auc": m["roc_auc"],
            "precision_at_25": m["daily"]["25"]["precision"],
            "recall_at_25": m["daily"]["25"]["recall"],
            "recall_at_1pct_false_alarms": m["false_alarm"]["0.01"]["recall"],
        }

    table = {n: pd.DataFrame([metric(r, n) for r in results]) for n in names}
    summary: dict[str, Any] = {
        "quarters": [r["quarter"] for r in results],
        "failed_drives": int(sum(r["models"][names[0]]["daily"]["25"]["failed"] for r in results)),
        "models": {
            n: {"mean": t.mean().round(4).to_dict(), "median": t.median().round(4).to_dict()}
            for n, t in table.items()
        },
        "wins": {},
    }
    for a in names:
        for b in names:
            if a < b:
                diff = table[a]["average_precision"] - table[b]["average_precision"]
                summary["wins"][f"{a}>{b}"] = int((diff > 0).sum())
                summary["wins"][f"{b}>{a}"] = int((diff < 0).sum())
    summary["by_quarter"] = {r["quarter"]: {n: metric(r, n) for n in names} for r in results}
    return summary
