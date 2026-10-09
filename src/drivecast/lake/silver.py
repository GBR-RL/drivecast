"""Silver layer: data drives only, one row per drive and day, with the data-quality rules applied.

Some rules need a drive's whole history (when it first failed, its usual capacity, whether it is
ever a boot drive), so a drive table is built first from all of bronze. Each quarter is then
written on its own:

- SSDs and boot drives are dropped (Backblaze excludes them from its hard-drive statistics);
- duplicate (serial number, date) rows are collapsed, keeping a failure if one of them has it;
- rows after a drive's first failure are dropped (a failed drive is replaced, not repaired);
- model and capacity are the drive's most common values (a few drives report several, and
  -1 for an unknown capacity);
- only the SMART attributes that some manufacturer reports for most of its drives are kept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from drivecast.quality.checks import not_data_drive_sql
from drivecast.quality.published import quarter_bounds

SMART_RAW = (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 22, 183, 184, 187, 188, 189, 190, 191, 192, 193,
             194, 195, 196, 197, 198, 199, 200, 240, 241, 242)  # fmt: skip
SMART_NORMALIZED = (1, 3, 5, 7, 9, 10, 187, 188, 192, 193, 194, 197, 198, 199)

MANUFACTURER = """CASE
    WHEN model ILIKE 'ST%' OR model ILIKE 'Seagate%' THEN 'Seagate'
    WHEN model ILIKE 'HGST%' OR model ILIKE 'Hitachi%' THEN 'HGST'
    WHEN model ILIKE 'WDC%' OR model ILIKE 'WUH%' OR model ILIKE 'WD%' THEN 'WDC'
    WHEN model ILIKE 'TOSHIBA%' OR model ILIKE 'MG%' THEN 'Toshiba'
    WHEN model ILIKE 'SAMSUNG%' THEN 'Samsung'
    ELSE 'other' END"""


def drives_sql() -> str:
    """One row per serial number over all of bronze."""
    return f"""
    SELECT serial_number,
        mode(model) AS model,
        count(DISTINCT model) AS model_names,
        mode(capacity_bytes) FILTER (WHERE capacity_bytes > 0) AS capacity_bytes,
        min(date) AS first_date,
        max(date) AS last_date,
        count(DISTINCT date) AS days,
        min(date) FILTER (WHERE failure = 1) AS failure_date,
        arg_min(smart_9_raw, date) FILTER (WHERE smart_9_raw IS NOT NULL) AS hours_first,
        min(date) FILTER (WHERE smart_9_raw IS NOT NULL) AS hours_first_date,
        arg_max(smart_9_raw, date) FILTER (WHERE smart_9_raw IS NOT NULL) AS hours_last,
        bool_or({not_data_drive_sql()}) AS not_data_drive
    FROM bronze GROUP BY serial_number
    """


def build_drives(con: Any, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY (SELECT *, {MANUFACTURER} AS manufacturer FROM ({drives_sql()}) "
        f"ORDER BY serial_number) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"
    )
    (n,) = con.execute(f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()
    return int(n)


def silver_columns() -> str:
    raw = [f"b.smart_{i}_raw" for i in SMART_RAW]
    norm = [f"b.smart_{i}_normalized" for i in SMART_NORMALIZED]
    return ", ".join(raw + norm)


def build_quarter(con: Any, quarter: str, drives: str, out: Path) -> dict[str, Any]:
    """Write one quarter of silver from the view ``bronze`` and the drive table."""
    start, end = quarter_bounds(quarter)
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""COPY (
        WITH b AS (
            SELECT * FROM bronze WHERE date >= DATE '{start}' AND date < DATE '{end}'
            QUALIFY row_number() OVER (PARTITION BY serial_number, date
                                       ORDER BY failure DESC NULLS LAST) = 1
        )
        SELECT b.date, b.serial_number, d.model, d.manufacturer, d.capacity_bytes,
            coalesce(b.failure, 0)::TINYINT AS failure, b.datacenter, {silver_columns()}
        FROM b JOIN read_parquet('{drives}') d USING (serial_number)
        WHERE NOT d.not_data_drive AND (d.failure_date IS NULL OR b.date <= d.failure_date)
        ORDER BY b.serial_number, b.date
    ) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 500000)"""
    )
    rows, drives_n, failures = con.execute(
        f"SELECT count(*), count(DISTINCT serial_number), sum(failure) "
        f"FROM read_parquet('{out.as_posix()}')"
    ).fetchone()
    return {
        "quarter": quarter,
        "file": out.name,
        "rows": int(rows),
        "drives": int(drives_n),
        "failures": int(failures or 0),
        "bytes": out.stat().st_size,
    }


def quarters(first: str = "2013Q2", last: str = "2026Q2") -> list[str]:
    year, q = int(first[:4]), int(first[-1])
    result = []
    while (year, q) <= (int(last[:4]), int(last[-1])):
        result.append(f"{year}Q{q}")
        year, q = (year + 1, 1) if q == 4 else (year, q + 1)
    return result
