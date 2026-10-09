from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb

from drivecast.lake import silver
from drivecast.quality.published import quarter_months

HDD = 4_000_787_030_016


def row(day: str, serial: str, **values: Any) -> dict[str, Any]:
    base = {"date": day, "serial_number": serial, "model": "ST4000DM000", "capacity_bytes": HDD,
            "failure": 0, "smart_9_raw": 1000, "smart_5_raw": 0}  # fmt: skip
    return base | values


def test_silver_applies_the_quality_rules(bronze: Callable[..., Any], tmp_path: Path) -> None:
    con = bronze(
        [
            row("2020-01-01", "A", smart_9_raw=1000),
            row("2020-01-02", "A", capacity_bytes=-1, smart_9_raw=1024),
            row("2020-01-01", "B"),
            row("2020-01-01", "B", failure=1),  # duplicate; the failure wins
            row("2020-01-02", "B"),  # after the failure
            row("2020-01-01", "C", model="DELLBOSS VD", capacity_bytes=480_036_847_616),
            row("2020-01-01", "D", model="HGST HMS5C4040BLE640"),
            row("2019-12-31", "D", model="HGST HMS5C4040BLE640"),  # another quarter
            row("2020-04-01", "A"),  # another quarter
        ]
    )
    drives = tmp_path / "drives.parquet"
    assert silver.build_drives(con, drives) == 4
    table = {
        r[0]: r[1:]
        for r in duckdb.sql(
            f"SELECT serial_number, manufacturer, capacity_bytes, failure_date, hours_first, "
            f"hours_last, not_data_drive FROM read_parquet('{drives.as_posix()}')"
        ).fetchall()
    }
    assert table["A"][:2] == ("Seagate", HDD)
    assert (table["A"][3], table["A"][4]) == (1000, 1000)  # hours at first and last report
    assert str(table["B"][2]) == "2020-01-01"
    assert table["C"][5] is True
    assert table["D"][0] == "HGST"

    stats = silver.build_quarter(con, "2020Q1", drives.as_posix(), tmp_path / "s.parquet")
    assert (stats["rows"], stats["drives"], stats["failures"]) == (4, 3, 1)
    rows = duckdb.sql(
        f"SELECT serial_number, date::VARCHAR, capacity_bytes, failure "
        f"FROM read_parquet('{(tmp_path / 's.parquet').as_posix()}')"
    ).fetchall()
    assert rows == [
        ("A", "2020-01-01", HDD, 0),
        ("A", "2020-01-02", HDD, 0),  # capacity -1 replaced by the drive's own
        ("B", "2020-01-01", HDD, 1),
        ("D", "2020-01-01", HDD, 0),
    ]


def test_quarters_and_their_months() -> None:
    assert silver.quarters("2013Q2", "2014Q1") == ["2013Q2", "2013Q3", "2013Q4", "2014Q1"]
    assert len(silver.quarters()) == 53
    assert quarter_months("2026Q2") == ["2026-04", "2026-05", "2026-06"]
