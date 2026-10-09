"""How much a careless split inflates the result.

Three ways to hold out data, each evaluated with the same model (LightGBM) and the same metric
(average precision, with the sample weights so it estimates the full population):

- **time**: train on the four quarters before the test quarter (with the 30-day embargo) and test
  on the test quarter, as the model would be used;
- **random rows**: pool the same five quarters and hold out 20% of the rows at random, so a drive's
  neighbouring days, and its future, are in training;
- **random drives**: pool the same quarters and hold out 20% of the drives, so the future is in
  training but the held-out drives are not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from drivecast.features.build import HORIZON_DAYS, feature_names
from drivecast.models.backtest import train_quarters
from drivecast.models.estimators import LightGBM
from drivecast.quality.published import quarter_bounds

HOLDOUT = 0.2


def _load(con: Any, files: list[str]) -> pd.DataFrame:
    listed = ", ".join(f"'{f}'" for f in files)
    cols = ", ".join(f"CAST({c} AS DOUBLE) AS {c}" for c in feature_names())
    frame: pd.DataFrame = con.execute(
        f"SELECT serial_number, date, {cols}, label, weight FROM read_parquet([{listed}]) "
        "WHERE label_complete"
    ).df()
    return frame


def _fit_score(train: pd.DataFrame, test: pd.DataFrame, threads: int) -> float:
    from sklearn.metrics import average_precision_score

    model = LightGBM(threads=threads).fit(
        train[feature_names()], train["label"].to_numpy(bool), train["weight"].to_numpy()
    )
    score = model.score(test[feature_names()])
    return float(
        average_precision_score(test["label"], score, sample_weight=test["weight"].to_numpy())
    )


def compare_splits(
    con: Any, gold: str, test: str, threads: int = 4, seed: int = 13
) -> dict[str, Any]:
    quarters = train_quarters(test)
    past = _load(con, [f"{gold}/train_{q}.parquet" for q in quarters])
    present = _load(con, [f"{gold}/train_{test}.parquet"])
    start, _ = quarter_bounds(test)
    cutoff = pd.Timestamp(start) - pd.Timedelta(days=HORIZON_DAYS)
    result: dict[str, Any] = {"quarter": test, "train_quarters": quarters}
    result["time"] = _fit_score(past[pd.to_datetime(past["date"]) < cutoff], present, threads)
    pool = pd.concat([past, present], ignore_index=True)
    rng = np.random.default_rng(seed)
    held = rng.random(len(pool)) < HOLDOUT
    result["random_rows"] = _fit_score(pool[~held], pool[held], threads)
    drives = pool["serial_number"].unique()
    held_drives = set(drives[rng.random(len(drives)) < HOLDOUT])
    held = pool["serial_number"].isin(held_drives).to_numpy()
    result["random_drives"] = _fit_score(pool[~held], pool[held], threads)
    return result
