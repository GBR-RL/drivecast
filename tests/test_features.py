from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from drivecast.features import build
from drivecast.lake.silver import SMART_NORMALIZED, SMART_RAW


def silver_file(path: Path, rows: list[dict[str, Any]]) -> str:
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    for c in ("model", "manufacturer", "datacenter"):
        frame[c] = frame.get(c, "x")
    smart = [f"smart_{i}_raw" for i in SMART_RAW] + [
        f"smart_{i}_normalized" for i in SMART_NORMALIZED
    ]
    for c in smart:
        if c not in frame:
            frame[c] = pd.Series([None] * len(frame), dtype="Int64")
    frame.to_parquet(path)
    return path.as_posix()


def days(start: str, n: int) -> list[str]:
    first = date.fromisoformat(start)
    return [(first + timedelta(days=i)).isoformat() for i in range(n)]


def test_features_look_back_and_the_label_looks_ahead(tmp_path: Path) -> None:
    rows = []
    # Drive A grows reallocated sectors from 2020-03-20 and fails on 2020-04-20.
    for i, d in enumerate(days("2020-03-01", 51)):
        rows.append({"date": d, "serial_number": "A", "manufacturer": "Seagate",
                     "capacity_bytes": 4e12, "failure": int(d == "2020-04-20"),
                     "smart_5_raw": max(0, i - 19), "smart_9_raw": 24 * (100 + i),
                     "smart_194_raw": 30 + (i == 40)})  # fmt: skip
    # Drive B is healthy throughout.
    for d in days("2020-03-01", 61):
        rows.append({"date": d, "serial_number": "B", "manufacturer": "HGST",
                     "capacity_bytes": 8e12, "failure": 0, "smart_5_raw": 0,
                     "smart_9_raw": 24 * 1000, "smart_194_raw": 25})  # fmt: skip
    silver = silver_file(tmp_path / "silver.parquet", rows)
    drives = tmp_path / "drives.parquet"
    pd.DataFrame(
        {
            "serial_number": ["A", "B"],
            "first_date": pd.to_datetime(["2020-03-01", "2020-03-01"]).date,
            "last_date": pd.to_datetime(["2020-04-20", "2020-04-30"]).date,
            "failure_date": [date(2020, 4, 20), None],
        }
    ).to_parquet(drives)
    con = duckdb.connect()
    out = tmp_path / "features_2020Q2.parquet"
    stats = build.build_quarter(con, "2020Q2", [silver], drives.as_posix(), out)
    assert stats["lake_end"] == "2020-04-30"
    f = (
        duckdb.sql(f"SELECT * FROM read_parquet('{out.as_posix()}')")
        .df()
        .set_index(["serial_number", "date"])
    )
    assert set(f.index.get_level_values("date").astype(str)) >= {"2020-04-01", "2020-04-20"}
    assert f.index.get_level_values("date").min() == pd.Timestamp("2020-04-01")  # the quarter
    a = f.loc["A"]
    first = a.iloc[0]  # 2020-04-01: index 31, smart_5 = 12
    assert first["smart_5"] == 12
    assert first["smart_5_d7"] == 7  # 12 - 5 (2020-03-25)
    assert first["smart_5_d30"] == 12  # 12 - 0 (2020-03-02 and earlier)
    assert first["temp_max_30d"] == 30
    assert first["maker_seagate"] == 1
    assert first["counters_growing_30d"] == 1
    assert first["age_days"] == 131
    # Label: fails within 30 days, counted from the failure day back.
    assert a["label"].all()  # 2020-04-01 .. 2020-04-20 are all within 29 days of failing
    assert int(a["days_to_failure"].iloc[0]) == 19
    b = f.loc["B"]
    assert not b["label"].any()
    # The last 30 days of the lake cannot have a complete negative label.
    assert not b["label_complete"].any()
    assert a["label_complete"].all()  # a known failure is a complete label


def test_training_sample_keeps_positives_and_weights_negatives(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "serial_number": [f"S{i}" for i in range(4000)],
            "date": [date(2020, 1, 1)] * 4000,
            "label": [i < 10 for i in range(4000)],
            "label_complete": [True] * 3990 + [False] * 10,
        }
    )
    features = tmp_path / "features.parquet"
    frame.to_parquet(features)
    stats = build.sample_training_rows(
        duckdb.connect(), features, tmp_path / "train.parquet", negative_rate=0.05
    )
    sample = pd.read_parquet(tmp_path / "train.parquet")
    assert stats["positive_rows"] == 10
    assert 120 < len(sample) - 10 < 280  # about 5% of 3,980 negatives
    assert set(sample.loc[~sample["label"], "weight"]) == {20.0}
    assert not sample["serial_number"].isin([f"S{i}" for i in range(3990, 4000)]).any()


def test_feature_names_match_the_columns() -> None:
    names = build.feature_names()
    assert len(names) == len(set(names))
    assert "smart_5_d30" in names
    assert "label" not in names
