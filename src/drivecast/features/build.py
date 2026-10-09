"""Gold layer: one feature row per drive and day, and the label "fails within the next 30 days".

Features only look backwards from the day they describe: the value of each SMART attribute on
that day, how much each error counter grew over the last 7 and 30 days, and a few drive facts.
A quarter is built from its own silver file plus the last 30 days of the quarter before (for the
windows); the label comes from the drive table, which knows failures in later quarters too.

Rows in the last 30 days of the lake do not have a complete label (a drive could fail after the
data ends); they are marked with ``label_complete = false`` and must not be used to train or
score.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from drivecast.quality.published import quarter_bounds

HORIZON_DAYS = 30
LOOKBACK_DAYS = 30
# Error counters: their growth matters more than their level.
COUNTERS = (5, 10, 183, 184, 187, 188, 189, 197, 198, 199)
# Other raw values used as they are.
LEVELS = (1, 7, 9, 12, 192, 193, 194, 240, 241, 242)
NORMALIZED = (1, 3, 5, 7, 187, 188, 194, 197, 198)
MANUFACTURERS = ("Seagate", "HGST", "WDC", "Toshiba")


def feature_names() -> list[str]:
    names = ["age_days", "days_seen", "capacity_tb", "temp_max_30d"]
    names += [f"maker_{m.lower()}" for m in MANUFACTURERS]
    for i in COUNTERS:
        names += [f"smart_{i}", f"smart_{i}_d7", f"smart_{i}_d30"]
    names += [f"smart_{i}" for i in LEVELS if i not in COUNTERS]
    names += [f"smart_{i}_n" for i in NORMALIZED]
    names += ["counters_growing_30d"]
    return names


def window_sql(start: str, end: str, lake_end: str) -> str:
    """Feature rows for the days in [start, end) from the views "silver" and "drives"."""
    w7 = (
        "(PARTITION BY serial_number ORDER BY date "
        "RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW)"
    )
    w30 = (
        "(PARTITION BY serial_number ORDER BY date "
        f"RANGE BETWEEN INTERVAL {LOOKBACK_DAYS} DAYS PRECEDING AND CURRENT ROW)"
    )
    cols = []
    for i in COUNTERS:
        v = f"s.smart_{i}_raw"
        cols += [
            f"{v} AS smart_{i}",
            f"{v} - first_value({v} IGNORE NULLS) OVER {w7} AS smart_{i}_d7",
            f"{v} - first_value({v} IGNORE NULLS) OVER {w30} AS smart_{i}_d30",
        ]
    cols += [f"s.smart_{i}_raw AS smart_{i}" for i in LEVELS if i not in COUNTERS]
    cols += [f"s.smart_{i}_normalized AS smart_{i}_n" for i in NORMALIZED]
    growing = " + ".join(f"(coalesce(smart_{i}_d30, 0) > 0)::INT" for i in COUNTERS)
    makers = ", ".join(f"(s.manufacturer = '{m}')::INT AS maker_{m.lower()}" for m in MANUFACTURERS)
    return f"""
    WITH windowed AS (
        SELECT s.date, s.serial_number, s.model, s.manufacturer,
            s.smart_9_raw / 24.0 AS age_days,
            datediff('day', d.first_date, s.date) AS days_seen,
            round(s.capacity_bytes / 1e12, 1) AS capacity_tb,
            max(s.smart_194_raw) OVER {w30} AS temp_max_30d,
            {makers},
            {", ".join(cols)},
            datediff('day', s.date, d.failure_date) AS days_to_failure
        FROM silver s JOIN drives d USING (serial_number)
    )
    SELECT *, {growing} AS counters_growing_30d,
        coalesce(days_to_failure BETWEEN 0 AND {HORIZON_DAYS - 1}, false) AS label,
        (date <= DATE '{lake_end}' - INTERVAL {HORIZON_DAYS} DAYS OR days_to_failure IS NOT NULL)
            AS label_complete
    FROM windowed
    WHERE date >= DATE '{start}' AND date < DATE '{end}'
    """


def previous_quarter(quarter: str) -> str:
    year, q = int(quarter[:4]), int(quarter[-1])
    return f"{year - 1}Q4" if q == 1 else f"{year}Q{q - 1}"


def build_quarter(
    con: Any,
    quarter: str,
    silver_files: list[str],
    drives: str,
    out: Path,
    *,
    lake_end: str | None = None,
) -> dict[str, Any]:
    """Write the feature rows of one quarter; ``silver_files`` must include the quarter before.

    ``lake_end`` is the last day of data; by default the last report in the drive table.
    """
    start, end = quarter_bounds(quarter)
    listed = ", ".join(f"'{f}'" for f in silver_files)
    con.execute(
        f"CREATE OR REPLACE VIEW silver AS SELECT * FROM read_parquet([{listed}]) "
        f"WHERE date >= DATE '{start}' - INTERVAL {LOOKBACK_DAYS} DAYS AND date < DATE '{end}'"
    )
    con.execute(f"CREATE OR REPLACE VIEW drives AS SELECT * FROM read_parquet('{drives}')")
    if lake_end is None:
        (last,) = con.execute("SELECT max(last_date) FROM drives").fetchone()
        lake_end = str(last)
    out.parent.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    con.execute(
        f"COPY ({window_sql(start, end, lake_end)} ORDER BY date, serial_number) "
        f"TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 500000)"
    )
    seconds = time.monotonic() - begin
    rows, positives, complete, drives_n = con.execute(
        f"SELECT count(*), sum(label::INT), sum(label_complete::INT), "
        f"count(DISTINCT serial_number) FROM read_parquet('{out.as_posix()}')"
    ).fetchone()
    return {
        "quarter": quarter,
        "file": out.name,
        "lake_end": lake_end,
        "rows": int(rows),
        "drives": int(drives_n),
        "positive_rows": int(positives or 0),
        "label_complete_rows": int(complete or 0),
        "bytes": out.stat().st_size,
        "seconds": round(seconds, 1),
    }


def sample_training_rows(
    con: Any, features: Path, out: Path, negative_rate: float = 0.01
) -> dict[str, Any]:
    """Every positive row and a fixed hash sample of negatives, weighted back to the full set."""
    per_mille = round(negative_rate * 1000)
    con.execute(
        f"""COPY (
        SELECT *, CASE WHEN label THEN 1.0 ELSE {1 / negative_rate} END AS weight
        FROM read_parquet('{features.as_posix()}')
        WHERE label_complete
          AND (label OR hash(serial_number || date::VARCHAR) % 1000 < {per_mille})
    ) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"""
    )
    rows, positives = con.execute(
        f"SELECT count(*), sum(label::INT) FROM read_parquet('{out.as_posix()}')"
    ).fetchone()
    return {"file": out.name, "rows": int(rows), "positive_rows": int(positives or 0)}
