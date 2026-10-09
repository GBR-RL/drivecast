"""How a deployed model decays: the same test quarter scored by models of different ages.

A model "deployed at" quarter D is fitted on the four quarters before D (with the 30-day embargo,
as in the backtest) and then kept. For a test quarter Q this module fits the models deployed
0, 1, 2, 3, 4 and 8 quarters earlier, and the one frozen at the first deployment (2015Q2), scores
all of Q with each and records their drift: how far the data each model was trained on is from
the quarter before Q (what is known when deciding to retrain) and from Q itself.

The results make every retraining policy computable without fitting anything again: a policy is
a sequence of model ages, and the performance of each age in each quarter is in the table.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drivecast.features.build import HORIZON_DAYS, feature_names
from drivecast.mlops import drift
from drivecast.models import evaluate
from drivecast.models.backtest import score_test, train_quarters
from drivecast.models.estimators import LightGBM, make
from drivecast.quality.published import quarter_bounds

AGES = (0, 1, 2, 3, 4, 8)
FIRST_DEPLOYMENT = "2015Q2"


def shift(quarter: str, n: int) -> str:
    """The quarter ``n`` quarters after ``quarter`` (before, if negative)."""
    index = int(quarter[:4]) * 4 + int(quarter[-1]) - 1 + n
    return f"{index // 4}Q{index % 4 + 1}"


def quarters_between(a: str, b: str) -> int:
    return (int(b[:4]) * 4 + int(b[-1])) - (int(a[:4]) * 4 + int(a[-1]))


def save(model: Any, stem: Path) -> Path:
    """LightGBM as its text format, the logistic regression pipeline with joblib."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, LightGBM):
        path = stem.with_suffix(".txt")
        model.model.booster_.save_model(str(path))
        return path
    import joblib

    path = stem.with_suffix(".joblib")
    joblib.dump(model.pipeline, path)
    return path


def _train_rows(con: Any, gold: str, deployed: str) -> pd.DataFrame:
    """The rows a model deployed at ``deployed`` is fitted on (with the embargo)."""
    start, _ = quarter_bounds(deployed)
    files = ", ".join(f"'{gold}/train_{q}.parquet'" for q in train_quarters(deployed))
    cols = ", ".join(f"CAST({c} AS FLOAT) AS {c}" for c in feature_names())
    frame: pd.DataFrame = con.execute(
        f"SELECT model, {cols}, label, weight FROM read_parquet([{files}]) "
        f"WHERE label_complete AND date < DATE '{start}' - INTERVAL {HORIZON_DAYS} DAYS"
    ).df()
    return frame


def _sample(con: Any, gold: str, quarter: str) -> pd.DataFrame:
    """A quarter's training sample, as a picture of its data (labels are not used)."""
    cols = ", ".join(f"CAST({c} AS FLOAT) AS {c}" for c in feature_names())
    frame: pd.DataFrame = con.execute(
        f"SELECT model, {cols}, weight FROM read_parquet('{gold}/train_{quarter}.parquet')"
    ).df()
    return frame


def _metrics(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "average_precision": m["average_precision"],
        "roc_auc": m["roc_auc"],
        "daily_25": m["daily"]["25"],
        "daily_100": m["daily"]["100"],
        "recall_at_1pct_false_alarms": m["false_alarm"]["0.01"]["recall"],
    }


def run_quarter(
    con: Any,
    test: str,
    gold: str,
    *,
    ages: tuple[int, ...] = AGES,
    threads: int = 4,
    models_out: Path | None = None,
) -> dict[str, Any]:
    begin = time.monotonic()
    plan: dict[str, tuple[str, str]] = {}  # name -> (family, deployed at)
    for age in ages:
        deployed = shift(test, -age)
        if quarters_between(FIRST_DEPLOYMENT, deployed) >= 0:
            plan[f"lightgbm_age{age}"] = ("lightgbm", deployed)
    plan["logreg_age0"] = ("logreg", test)
    if quarters_between(FIRST_DEPLOYMENT, test) >= 0:
        plan["lightgbm_frozen"] = ("lightgbm", FIRST_DEPLOYMENT)
        plan["logreg_frozen"] = ("logreg", FIRST_DEPLOYMENT)
    previous, current = _sample(con, gold, shift(test, -1)), _sample(con, gold, test)
    fitted: dict[str, Any] = {}
    drifts: dict[str, Any] = {}
    for name, (family, deployed) in plan.items():
        rows = _train_rows(con, gold, deployed)
        model = make(family, threads=threads).fit(
            rows[feature_names()], rows["label"].to_numpy(bool), rows["weight"].to_numpy()
        )
        fitted[name] = model
        top = list(model.importance())[:10] if isinstance(model, LightGBM) else None
        before = drift.summary(rows, previous, feature_names(), top)
        during = drift.summary(rows, current, feature_names(), top)
        drifts[name] = {
            "before": {k: v for k, v in before.items() if k != "psi"},
            "during": {k: v for k, v in during.items() if k != "psi"},
        }
        if models_out is not None and name in ("lightgbm_age0", "logreg_age0"):
            save(model, models_out / f"{family}_{test}")
        del rows
        gc.collect()
    scored = score_test(con, f"{gold}/features_{test}.parquet", test, fitted, None)
    a = scored["arrays"]
    rows_ = evaluate.TestRows(a["drive"], a["day"], a["label"], a["dtf"], a["fiq"])
    result: dict[str, Any] = {"quarter": test, "models": {}}
    for name, score in scored["scores"].items():
        family, deployed = plan[name]
        result["models"][name] = {
            "family": family,
            "deployed_at": deployed,
            "age": quarters_between(deployed, test),
            **_metrics(evaluate.evaluate(rows_, score)),
            "drift": drifts[name],
        }
    result["positive_rate"] = float(np.mean(a["label"]))
    result["seconds"] = round(time.monotonic() - begin, 1)
    return result
