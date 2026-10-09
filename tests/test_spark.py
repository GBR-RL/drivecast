import shutil
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from drivecast.features import build, compare

pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Spark needs Java", allow_module_level=True)

from test_features import days, silver_file


def test_spark_and_duckdb_compute_the_same_features(tmp_path: Path) -> None:
    from drivecast.features import spark

    rows = []
    for i, d in enumerate(days("2020-03-01", 51)):
        rows.append({"date": d, "serial_number": "A", "manufacturer": "Seagate",
                     "capacity_bytes": 4e12, "failure": int(d == "2020-04-20"),
                     "smart_5_raw": max(0, i - 19), "smart_9_raw": 24 * (100 + i),
                     "smart_194_raw": 30 + (i == 40),
                     "smart_197_raw": None if i % 3 else i})  # fmt: skip
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
    left = tmp_path / "duck.parquet"
    build.build_quarter(con, "2020Q2", [silver], drives.as_posix(), left)
    session = spark.session(threads=2, memory="1g")
    try:
        spark.build_quarter(session, "2020Q2", [silver], drives.as_posix(), tmp_path / "spark")
    finally:
        session.stop()
    result = compare.compare(con, left.as_posix(), (tmp_path / "spark" / "*.parquet").as_posix())
    assert result["only_left"] == result["only_right"] == 0
    assert result["joined"] == result["left_rows"] > 0
    assert result["differing_values"] == {}
