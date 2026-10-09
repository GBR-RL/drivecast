from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from drivecast.features.build import feature_names
from drivecast.models import backtest, evaluate
from drivecast.models.estimators import Rule, make


def test_daily_budget_flags_the_top_scores_of_each_day() -> None:
    day = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    score = np.array([0.9, 0.1, 0.5, 0.0, 0.2, 0.3], dtype=np.float32)
    assert evaluate.flagged_by_day(day, score, 2).tolist() == [True, False, True, False, True, True]
    # A zero score is never flagged, even when the budget is not used up.
    assert evaluate.flagged_by_day(day, score, 3)[3] == False  # noqa: E712


def test_daily_budget_precision_recall_and_lead_time() -> None:
    # Drive 1 fails on day 3 (label on days 0-3); drive 2 fails next quarter; drive 3 is healthy.
    drive = np.array([1, 1, 1, 1, 2, 2, 3, 3, 3, 3], dtype=np.int64)
    day = np.array([0, 1, 2, 3, 2, 3, 0, 1, 2, 3], dtype=np.int32)
    label = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    dtf = np.array([3, 2, 1, 0, 40, 39, -1, -1, -1, -1], dtype=np.float64)
    fiq = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    score = np.array([0.1, 0.2, 0.9, 0.9, 0.8, 0.8, 0.5, 0.5, 0.1, 0.1], dtype=np.float32)
    rows = evaluate.TestRows(drive, day, label, dtf, fiq)
    r = evaluate.daily_budget(rows, score, k=1)
    # Flagged: day 0 drive 3, day 1 drive 3, day 2 drive 1, day 3 drive 1.
    assert r["precision"] == 0.5
    assert (r["recall"], r["detected"], r["failed"]) == (1.0, 1, 1)
    assert r["median_lead_days"] == 1.0  # first flagged one day before failing


def test_false_alarm_threshold_comes_from_healthy_drives() -> None:
    rng = np.random.default_rng(0)
    healthy = rng.random(1000).astype(np.float32)
    drive = np.r_[np.arange(1000), [5000, 5001]].astype(np.int64)
    score = np.r_[healthy, [0.999999, 0.2]].astype(np.float32)
    label = np.r_[np.zeros(1000, bool), [True, True]]
    rows = evaluate.TestRows(drive, np.zeros(len(drive), np.int32), label,
                             np.zeros(len(drive)), label)  # fmt: skip
    r = evaluate.at_false_alarm_rate(rows, score, 0.01)
    assert r["recall"] == 0.5
    assert r["threshold"] == pytest.approx(np.quantile(healthy, 0.99))


def test_rule_counts_warning_attributes() -> None:
    x = pd.DataFrame({c: [0, 1] for c in ("smart_5", "smart_187", "smart_188")} |
                     {"smart_197": [None, 3], "smart_198": [0, 0]})  # fmt: skip
    assert Rule().score(x).tolist() == [0.0, 3.0 + 1.0]


def test_training_drops_the_embargo_before_the_test_quarter(tmp_path: Path) -> None:
    start = date(2020, 3, 1)
    rows = []
    for i in range(31 + 31):  # 2020-03-01 .. 2020-05-01
        d = start + timedelta(days=i)
        rows.append(dict.fromkeys(feature_names(), 0.0) | {"date": d, "label": i % 2 == 0,
                     "label_complete": True, "weight": 1.0})  # fmt: skip
    path = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(path)
    train = backtest.load_training(duckdb.connect(), [path.as_posix()], "2020Q2")
    # Test starts 2020-04-01; rows on or after 2020-03-02 are within 30 days and dropped.
    assert len(train) == 1


def test_train_quarters_are_the_four_before() -> None:
    assert backtest.train_quarters("2016Q1") == ["2015Q1", "2015Q2", "2015Q3", "2015Q4"]


def test_models_rank_an_obvious_signal_above_noise() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    x = pd.DataFrame(rng.normal(size=(n, len(feature_names()))), columns=feature_names())
    y = rng.random(n) < 0.1
    x.loc[y, "smart_5"] += 3
    w = np.ones(n)
    for name in ("logreg", "lightgbm"):
        model = make(name, threads=1).fit(x, y, w)
        s = model.score(x)
        assert s[y].mean() > s[~y].mean()
