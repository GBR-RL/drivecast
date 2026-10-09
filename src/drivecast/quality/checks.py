"""Data-quality checks over the bronze lake, counted per year.

Nothing is fixed here. Each check counts the rows, drives or days it affects, so the silver layer
can apply a documented rule for each problem and the report shows how much data every rule
touches.

The lake has 744M rows, too many for one query to group or window over on a CI runner, so the
checks are built from three smaller things, one bronze month at a time:

- row checks, counted inside each month (a duplicate is the same drive and day, so always in
  one month);
- the drive table, combined from monthly per-drive partials, for checks about a drive's whole
  history (failing twice, reporting after failing, changing model);
- the number of reports per day, compared with the drives alive that day.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
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
PARTIAL_SHARE = 0.8


def not_data_drive_sql(model: str = "model", capacity: str = "capacity_bytes") -> str:
    names = " OR ".join(f"{model} ILIKE '%{n}%'" for n in NOT_DATA_DRIVE)
    return f"({names} OR ({capacity} > 0 AND {capacity} < {MIN_DATA_DRIVE_BYTES}))"


@dataclass(frozen=True)
class Check:
    name: str
    unit: str  # "rows", "drives", "days" or "values"
    description: str


CHECKS: tuple[Check, ...] = (
    Check("rows", "rows", "Rows in the bronze lake"),
    Check("drives", "drives", "Distinct serial numbers reporting in the year"),
    Check("failures", "rows", "Rows with failure = 1"),
    Check("missing_key", "rows", "Rows without a date, serial number or model"),
    Check("duplicate_rows", "rows", "Extra rows for a (serial number, date) already present"),
    Check("failure_not_binary", "rows", "Rows whose failure flag is not 0 or 1"),
    Check("several_failures", "drives", "Drives with more than one failure row"),
    Check("reported_after_failure", "drives", "Drives that report after their first failure"),
    Check("rows_after_failure", "rows", "Rows reported after the drive's first failure"),
    Check("capacity_negative", "rows", "Rows with capacity_bytes below zero (-1 means unknown)"),
    Check("capacity_changes", "drives", "Drives reporting more than one positive capacity"),
    Check("model_changes", "drives", "Drives reporting more than one model name"),
    Check(
        "power_on_hours_decrease",
        "drives",
        "Drives whose power-on hours (SMART 9) go down between reports",
    ),
    Check("not_data_drive", "rows", "Rows from SSDs and boot drives"),
    Check("normalized_out_of_range", "values", "SMART normalized values outside 0-255"),
    Check("raw_negative", "values", "SMART raw values below zero"),
    Check(
        "partial_days",
        "days",
        "Days on which fewer than 80% of the drives alive that day (seen before and after) report",
    ),
    Check("missing_days", "days", "Calendar days without any report between the first and last"),
)  # fmt: skip


def _row_checks_sql(file: str) -> str:
    out_of_range = " + ".join(
        f"coalesce((smart_{i}_normalized NOT BETWEEN 0 AND 255)::INT, 0)" for i in SMART_IDS
    )
    negative = " + ".join(f"coalesce((smart_{i}_raw < 0)::INT, 0)" for i in SMART_IDS)
    return f"""
    SELECT year(date) AS year,
        count(*) AS rows,
        count(*) FILTER (WHERE failure = 1) AS failures,
        count(*) FILTER (WHERE date IS NULL OR serial_number IS NULL OR model IS NULL)
            AS missing_key,
        count(*) FILTER (WHERE failure IS NULL OR failure NOT IN (0, 1)) AS failure_not_binary,
        count(*) FILTER (WHERE capacity_bytes < 0) AS capacity_negative,
        count(*) FILTER (WHERE {not_data_drive_sql()}) AS not_data_drive,
        sum({out_of_range}) AS normalized_out_of_range,
        sum({negative}) AS raw_negative
    FROM read_parquet('{file}') GROUP BY 1
    """


def _month_partial_sql(file: str) -> str:
    """One row per drive and month: what the drive-level checks need."""
    return f"""
    SELECT serial_number, min(date) AS first_date, max(date) AS last_date,
        count(*) FILTER (WHERE failure = 1) AS failure_rows,
        min(date) FILTER (WHERE failure = 1) AS failure_date,
        count(DISTINCT model) AS models, min(model) AS model,
        count(DISTINCT capacity_bytes) FILTER (WHERE capacity_bytes > 0) AS capacities,
        min(capacity_bytes) FILTER (WHERE capacity_bytes > 0) AS capacity,
        arg_min(smart_9_raw, date) FILTER (WHERE smart_9_raw IS NOT NULL) AS hours_first,
        arg_max(smart_9_raw, date) FILTER (WHERE smart_9_raw IS NOT NULL) AS hours_last
    FROM read_parquet('{file}') WHERE serial_number IS NOT NULL GROUP BY 1
    """


def _hours_down_in_month_sql(file: str) -> str:
    return f"""
    SELECT DISTINCT serial_number FROM (
        SELECT serial_number, smart_9_raw - lag(smart_9_raw) OVER (
            PARTITION BY serial_number ORDER BY date) AS step
        FROM read_parquet('{file}') WHERE smart_9_raw IS NOT NULL
    ) WHERE step < 0
    """


def _by_year(rows: list[tuple[Any, Any]]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for year, n in rows:
        if year is not None and n:
            out[str(year)] += int(n)
    return dict(sorted(out.items()))


def run(con: Any, files: list[str], work: Path) -> dict[str, Any]:
    """Run every check over the bronze files; counts per year and in total."""
    work.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    row_fields = ("rows", "failures", "missing_key", "failure_not_binary", "capacity_negative",
                  "not_data_drive", "normalized_out_of_range", "raw_negative")  # fmt: skip
    con.execute("CREATE OR REPLACE TABLE daily (date DATE, n BIGINT)")
    con.execute("CREATE OR REPLACE TABLE hours_down (serial_number VARCHAR)")
    con.execute("CREATE OR REPLACE TABLE yearly_drives (year INTEGER, serial_number VARCHAR)")
    partials = []
    for i, file in enumerate(files):
        for row in con.execute(_row_checks_sql(file)).fetchall():
            year = str(row[0])
            for name, value in zip(row_fields, row[1:], strict=True):
                counts[name][year] += int(value or 0)
        for year, n in con.execute(
            f"SELECT year(date), sum(n - 1) FROM (SELECT serial_number, date, count(*) AS n "
            f"FROM read_parquet('{file}') GROUP BY 1, 2 HAVING n > 1) GROUP BY 1"
        ).fetchall():
            counts["duplicate_rows"][str(year)] += int(n)
        con.execute(
            f"INSERT INTO daily SELECT date, count(*) FROM read_parquet('{file}') "
            f"WHERE date IS NOT NULL GROUP BY 1"
        )
        con.execute(f"INSERT INTO hours_down {_hours_down_in_month_sql(file)}")
        con.execute(
            f"INSERT INTO yearly_drives SELECT DISTINCT year(date), serial_number "
            f"FROM read_parquet('{file}') WHERE serial_number IS NOT NULL"
        )
        partial = work / f"check_partial_{i:04d}.parquet"
        con.execute(f"COPY ({_month_partial_sql(file)}) TO '{partial.as_posix()}' (FORMAT parquet)")
        partials.append(f"'{partial.as_posix()}'")
    con.execute(
        f"CREATE OR REPLACE VIEW months AS SELECT * FROM read_parquet([{', '.join(partials)}])"
    )
    con.execute(
        "CREATE OR REPLACE TABLE spans AS SELECT serial_number, min(first_date) AS first_date, "
        "max(last_date) AS last_date, sum(failure_rows) AS failure_rows, "
        "min(failure_date) AS failure_date FROM months GROUP BY 1"
    )
    for file in files:
        for year, n in con.execute(
            f"SELECT year(b.date), count(*) FROM read_parquet('{file}') b JOIN spans s "
            f"USING (serial_number) WHERE b.date > s.failure_date GROUP BY 1"
        ).fetchall():
            counts["rows_after_failure"][str(year)] += int(n)
    counts["drives"] = Counter(
        _by_year(
            con.execute(
                "SELECT year, count(DISTINCT serial_number) FROM yearly_drives GROUP BY 1"
            ).fetchall()
        )
    )
    drive_checks = {
        "several_failures": "SELECT year(failure_date), count(*) FROM spans "
        "WHERE failure_rows > 1 GROUP BY 1",
        "reported_after_failure": "SELECT year(failure_date), count(*) FROM spans "
        "WHERE last_date > failure_date GROUP BY 1",
        "capacity_changes": "SELECT year(first), count(*) FROM (SELECT serial_number, "
        "min(first_date) AS first FROM months GROUP BY 1 HAVING max(capacities) > 1 OR "
        "count(DISTINCT capacity) > 1) GROUP BY 1",
        "model_changes": "SELECT year(first), count(*) FROM (SELECT serial_number, "
        "min(first_date) AS first FROM months GROUP BY 1 HAVING max(models) > 1 OR "
        "count(DISTINCT model) > 1) GROUP BY 1",
        # Down within a month, or from one month's last report to the next month's first.
        "power_on_hours_decrease": "WITH steps AS (SELECT serial_number, first_date, "
        "hours_first < lag(hours_last) OVER (PARTITION BY serial_number ORDER BY first_date) "
        "AS down FROM months WHERE hours_first IS NOT NULL), "
        "drives AS (SELECT serial_number FROM steps WHERE down UNION "
        "SELECT serial_number FROM hours_down) "
        "SELECT year(s.first_date), count(*) FROM drives JOIN spans s USING (serial_number) "
        "GROUP BY 1",
    }
    for name, sql in drive_checks.items():
        counts[name] = Counter(_by_year(con.execute(sql).fetchall()))
    for name, sql in _day_checks().items():
        counts[name] = Counter(_by_year(con.execute(sql).fetchall()))
    for partial in work.glob("check_partial_*.parquet"):
        partial.unlink()
    results: dict[str, Any] = {}
    for check in CHECKS:
        by_year = dict(sorted((y, n) for y, n in counts[check.name].items() if n))
        results[check.name] = {
            "unit": check.unit,
            "description": check.description,
            "total": sum(by_year.values()),
            "by_year": by_year,
        }
    return results


# Reports per day against the drives alive that day: first report <= day <= last report.
DAILY = (
    "events AS (SELECT first_date AS date, 1 AS step FROM spans UNION ALL "
    "SELECT last_date + 1, -1 FROM spans), "
    "alive AS (SELECT date, sum(sum(step)) OVER (ORDER BY date) AS alive FROM events GROUP BY 1), "
    "reports AS (SELECT date, sum(n) AS n FROM daily GROUP BY 1), "
    "per_day AS (SELECT r.date, r.n, (SELECT a.alive FROM alive a WHERE a.date <= r.date "
    "ORDER BY a.date DESC LIMIT 1) AS alive FROM reports r)"
)


def _day_checks() -> dict[str, str]:
    return {
        "partial_days": f"WITH {DAILY} SELECT year(date), count(*) FROM per_day "
        f"WHERE n < {PARTIAL_SHARE} * alive GROUP BY 1",
        "missing_days": "SELECT year(d), count(*) FROM (SELECT unnest(generate_series("
        "min(date), max(date), INTERVAL 1 DAY))::DATE AS d FROM daily) "
        "WHERE d NOT IN (SELECT date FROM daily) GROUP BY 1",
    }


def partial_day_ranges(con: Any) -> list[dict[str, Any]]:
    """Consecutive runs of partial days, with the reports and drives alive on them.

    Uses the tables left by ``run``.
    """
    rows = con.execute(
        f"WITH {DAILY}, p AS (SELECT date, n, alive, date - (row_number() OVER "
        f"(ORDER BY date))::INT AS grp FROM per_day WHERE n < {PARTIAL_SHARE} * alive) "
        "SELECT min(date), max(date), count(*), min(n), max(alive) FROM p GROUP BY grp ORDER BY 1"
    ).fetchall()
    return [
        {"from": str(a), "to": str(b), "days": int(n), "min_reports": int(lo), "alive": int(t)}
        for a, b, n, lo, t in rows
    ]
