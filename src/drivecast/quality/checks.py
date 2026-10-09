"""Data-quality checks over the bronze lake, counted per year.

Nothing is fixed here. Each check counts the rows or drives it affects, so the silver layer can
apply a documented rule for each problem and the report shows how much data every rule touches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from drivecast.lake.schema import SMART_IDS

# Models that are SSDs or boot drives rather than data drives (Backblaze excludes both from its
# hard-drive statistics). Matched case-insensitively against the model name.
NOT_DATA_DRIVE = (
    "SSD",
    "DELLBOSS",
    "Micron",
    "CT250",
    "CT500",
    "WDS",
    "SSDSC",
    "MTFDDAV",
    "SATADOM",
)

# Data drives in the fleet are 1 TB or larger (early pods used 1 TB WD Greens); smaller ones are
# 250-500 GB laptop drives used for booting.
MIN_DATA_DRIVE_BYTES = 900_000_000_000


def not_data_drive_sql(model: str = "model", capacity: str = "capacity_bytes") -> str:
    names = " OR ".join(f"{model} ILIKE '%{n}%'" for n in NOT_DATA_DRIVE)
    return f"({names} OR ({capacity} > 0 AND {capacity} < {MIN_DATA_DRIVE_BYTES}))"


@dataclass(frozen=True)
class Check:
    name: str
    unit: str  # "rows", "drives" or "days"
    description: str
    sql: str  # returns (year, count) rows; reads the view "bronze"


def _normalized_out_of_range() -> str:
    terms = " + ".join(
        f"coalesce((smart_{i}_normalized NOT BETWEEN 0 AND 255)::INT, 0)" for i in SMART_IDS
    )
    return f"SELECT year(date), sum({terms}) FROM bronze GROUP BY 1"


def _raw_negative() -> str:
    terms = " + ".join(f"coalesce((smart_{i}_raw < 0)::INT, 0)" for i in SMART_IDS)
    return f"SELECT year(date), sum({terms}) FROM bronze GROUP BY 1"


# Reports per day against the drives alive that day: first report <= day <= last report.
DAILY = (
    "span AS (SELECT serial_number, min(date) AS first, max(date) AS last FROM bronze GROUP BY 1), "
    "events AS (SELECT first AS date, 1 AS step FROM span UNION ALL "
    "SELECT last + 1, -1 FROM span), "
    "alive AS (SELECT date, sum(sum(step)) OVER (ORDER BY date) AS alive FROM events GROUP BY 1), "
    "reports AS (SELECT date, count(*) AS n FROM bronze GROUP BY 1), "
    "daily AS (SELECT r.date, r.n, (SELECT a.alive FROM alive a WHERE a.date <= r.date "
    "ORDER BY a.date DESC LIMIT 1) AS alive FROM reports r)"
)

FIRST_FAILURE = (
    "first_failure AS (SELECT serial_number, min(date) AS failed_on FROM bronze "
    "WHERE failure = 1 GROUP BY 1)"
)

CHECKS: tuple[Check, ...] = (
    Check(
        "rows",
        "rows",
        "Rows in the bronze lake",
        "SELECT year(date), count(*) FROM bronze GROUP BY 1",
    ),
    Check(
        "drives",
        "drives",
        "Distinct serial numbers reporting in the year",
        "SELECT year(date), count(DISTINCT serial_number) FROM bronze GROUP BY 1",
    ),
    Check(
        "failures",
        "rows",
        "Rows with failure = 1",
        "SELECT year(date), count(*) FROM bronze WHERE failure = 1 GROUP BY 1",
    ),
    Check(
        "missing_key",
        "rows",
        "Rows without a date, serial number or model",
        "SELECT year(date), count(*) FROM bronze WHERE date IS NULL OR serial_number IS NULL "
        "OR model IS NULL GROUP BY 1",
    ),
    Check(
        "duplicate_rows",
        "rows",
        "Extra rows for a (serial number, date) that is already present",
        "SELECT year(date), sum(n - 1) FROM (SELECT serial_number, date, count(*) AS n "
        "FROM bronze GROUP BY 1, 2 HAVING n > 1) GROUP BY 1",
    ),
    Check(
        "failure_not_binary",
        "rows",
        "Rows whose failure flag is not 0 or 1",
        "SELECT year(date), count(*) FROM bronze WHERE failure NOT IN (0, 1) "
        "OR failure IS NULL GROUP BY 1",
    ),
    Check(
        "several_failures",
        "drives",
        "Drives with more than one failure row (year of the first)",
        "SELECT year(first), count(*) FROM (SELECT serial_number, min(date) AS first FROM bronze "
        "WHERE failure = 1 GROUP BY 1 HAVING count(*) > 1) GROUP BY 1",
    ),
    Check(
        "reported_after_failure",
        "drives",
        "Drives that keep reporting after their first failure",
        f"WITH {FIRST_FAILURE} SELECT year(failed_on), count(DISTINCT serial_number) FROM bronze "
        "JOIN first_failure USING (serial_number) WHERE date > failed_on GROUP BY 1",
    ),
    Check(
        "rows_after_failure",
        "rows",
        "Rows reported after the drive's first failure",
        f"WITH {FIRST_FAILURE} SELECT year(date), count(*) FROM bronze "
        "JOIN first_failure USING (serial_number) WHERE date > failed_on GROUP BY 1",
    ),
    Check(
        "capacity_negative",
        "rows",
        "Rows with capacity_bytes below zero (Backblaze writes -1 when unknown)",
        "SELECT year(date), count(*) FROM bronze WHERE capacity_bytes < 0 GROUP BY 1",
    ),
    Check(
        "capacity_changes",
        "drives",
        "Drives reporting more than one positive capacity",
        "SELECT year(first), count(*) FROM (SELECT serial_number, min(date) AS first FROM bronze "
        "WHERE capacity_bytes > 0 GROUP BY 1 HAVING count(DISTINCT capacity_bytes) > 1) "
        "GROUP BY 1",
    ),
    Check(
        "model_changes",
        "drives",
        "Drives reporting more than one model name",
        "SELECT year(first), count(*) FROM (SELECT serial_number, min(date) AS first FROM bronze "
        "GROUP BY 1 HAVING count(DISTINCT model) > 1) GROUP BY 1",
    ),
    Check(
        "power_on_hours_decrease",
        "drives",
        "Drives whose power-on hours (SMART 9) go down between reports",
        "SELECT year(first), count(*) FROM (SELECT serial_number, min(date) AS first FROM "
        "(SELECT serial_number, date, smart_9_raw - lag(smart_9_raw) OVER (PARTITION BY "
        "serial_number ORDER BY date) AS step FROM bronze) WHERE step < 0 GROUP BY 1) GROUP BY 1",
    ),
    Check(
        "not_data_drive",
        "rows",
        "Rows from SSDs and boot drives",
        f"SELECT year(date), count(*) FROM bronze WHERE {not_data_drive_sql()} GROUP BY 1",
    ),
    Check(
        "normalized_out_of_range",
        "values",
        "SMART normalized values outside 0-255",
        _normalized_out_of_range(),
    ),
    Check("raw_negative", "values", "SMART raw values below zero", _raw_negative()),
    Check(
        "partial_days",
        "days",
        "Days on which fewer than 80% of the drives alive that day (seen before and after) report",
        f"WITH {DAILY} SELECT year(date), count(*) FROM daily WHERE n < 0.8 * alive GROUP BY 1",
    ),
    Check(
        "missing_days",
        "days",
        "Calendar days without any report between the first and the last day",
        "SELECT year(d), count(*) FROM (SELECT unnest(generate_series(min(date), max(date), "
        "INTERVAL 1 DAY))::DATE AS d FROM bronze) WHERE d NOT IN (SELECT DISTINCT date "
        "FROM bronze) GROUP BY 1",
    ),
)


def run(con: Any, checks: tuple[Check, ...] = CHECKS) -> dict[str, Any]:
    """Run every check against the view ``bronze``; counts per year and in total."""
    results: dict[str, Any] = {}
    for check in checks:
        rows = con.execute(check.sql).fetchall()
        years = [year for year, _ in rows]
        if len(years) != len(set(years)):
            raise ValueError(f"check {check.name} returned a year more than once")
        by_year = {str(year): int(n) for year, n in sorted(rows) if year is not None and n}
        results[check.name] = {
            "unit": check.unit,
            "description": check.description,
            "total": sum(by_year.values()),
            "by_year": by_year,
        }
    return results


def partial_day_ranges(con: Any) -> list[dict[str, Any]]:
    """Consecutive runs of partial days, with the drives reported on them."""
    rows = con.execute(
        f"WITH {DAILY}, p AS (SELECT date, n, alive, date - (row_number() OVER "
        "(ORDER BY date))::INT AS grp FROM daily WHERE n < 0.8 * alive) "
        "SELECT min(date), max(date), count(*), min(n), max(alive) FROM p GROUP BY grp ORDER BY 1"
    ).fetchall()
    return [
        {"from": str(a), "to": str(b), "days": int(n), "min_reports": int(lo), "alive": int(t)}
        for a, b, n, lo, t in rows
    ]
