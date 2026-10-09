"""Numbers Backblaze published for some quarters, and the same numbers computed from the lake.

If the lake was ingested correctly, applying a quarter's published inclusion rule to it should
reproduce the published drive counts, drive days and failures. Backblaze also removes drives in
certification testing, which the data does not mark, so small differences are expected there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from drivecast.quality.checks import not_data_drive_sql

AFR_DAYS = 365


@dataclass(frozen=True)
class Published:
    quarter: str
    url: str
    afr: float
    min_drives: int  # a model is included if its drive count at quarter end exceeds this
    min_drive_days: int  # ... and its drive days in the quarter exceed this
    drives: int | None = None  # all drives at quarter end, boot drives included
    boot_drives: int | None = None
    analyzed: int | None = None  # data drives of the included models at quarter end
    drive_days: int | None = None
    failures: int | None = None


BLOG = "https://www.backblaze.com/blog/backblaze-drive-stats-for-"
PUBLISHED: tuple[Published, ...] = (
    Published(
        "2022Q3", f"{BLOG}q3-2022/", afr=1.64, min_drives=59, min_drive_days=0,
        drives=230_897, boot_drives=4_200, analyzed=226_309,
    ),
    Published(
        "2025Q2", f"{BLOG}q2-2025/", afr=1.36, min_drives=100, min_drive_days=10_000,
        drives=321_201, boot_drives=3_971,
    ),
    Published(
        "2026Q1", f"{BLOG}q1-2026/", afr=1.24, min_drives=100, min_drive_days=10_000,
        drives=345_662, boot_drives=3_907, analyzed=341_263, drive_days=30_203_180,
        failures=1_030,
    ),
    Published(
        "2026Q2", f"{BLOG}q2-2026/", afr=1.73, min_drives=100, min_drive_days=10_000,
        drives=359_101, boot_drives=3_881, analyzed=354_415, drive_days=31_553_350,
        failures=1_498,
    ),
)  # fmt: skip


def quarter_bounds(quarter: str) -> tuple[str, str]:
    year, q = int(quarter[:4]), int(quarter[-1])
    start_month = 3 * (q - 1) + 1
    end = f"{year + 1}-01-01" if q == 4 else f"{year}-{start_month + 3:02d}-01"
    return f"{year}-{start_month:02d}-01", end


def quarter_months(quarter: str) -> list[str]:
    start, _ = quarter_bounds(quarter)
    year, month = int(start[:4]), int(start[5:7])
    return [f"{year}-{m:02d}" for m in range(month, month + 3)]


def afr(failures: int, drive_days: int) -> float:
    """Annualized failure rate in percent, as Backblaze defines it."""
    return 100.0 * failures / (drive_days / AFR_DAYS) if drive_days else 0.0


def compute(con: Any, pub: Published) -> dict[str, Any]:
    """The published quantities computed from the view ``bronze`` for one quarter."""
    start, end = quarter_bounds(pub.quarter)
    boot = not_data_drive_sql()
    row = con.execute(
        f"WITH q AS (SELECT * FROM bronze WHERE date >= DATE '{start}' AND date < DATE '{end}'), "
        f"last AS (SELECT * FROM q WHERE date = (SELECT max(date) FROM q)), "
        f"ends AS (SELECT model, count(*) AS n FROM last WHERE NOT {boot} GROUP BY 1), "
        f"days AS (SELECT model, count(*) AS dd, sum(failure) AS f FROM q WHERE NOT {boot} "
        f"GROUP BY 1), "
        f"models AS (SELECT model, coalesce(n, 0) AS n, dd, f FROM days LEFT JOIN ends "
        f"USING (model)) "
        f"SELECT (SELECT count(*) FROM last), (SELECT count(*) FROM last WHERE {boot}), "
        f"(SELECT max(date) FROM q), "
        f"sum(n) FILTER (WHERE n > {pub.min_drives} AND dd > {pub.min_drive_days}), "
        f"sum(dd) FILTER (WHERE n > {pub.min_drives} AND dd > {pub.min_drive_days}), "
        f"sum(f) FILTER (WHERE n > {pub.min_drives} AND dd > {pub.min_drive_days}), "
        f"count(*) FILTER (WHERE n > {pub.min_drives} AND dd > {pub.min_drive_days}) "
        f"FROM models"
    ).fetchone()
    assert row is not None
    drives, boot_drives, last_day, analyzed, drive_days, failures, models = row
    return {
        "last_day": str(last_day),
        "drives": int(drives or 0),
        "boot_drives": int(boot_drives or 0),
        "analyzed": int(analyzed or 0),
        "drive_days": int(drive_days or 0),
        "failures": int(failures or 0),
        "models": int(models or 0),
        "afr": round(afr(int(failures or 0), int(drive_days or 0)), 2),
    }


def reconcile(con: Any, published: tuple[Published, ...] = PUBLISHED) -> list[dict[str, Any]]:
    """Published against computed, for every published quarter present in the lake."""
    rows = []
    for pub in published:
        start, end = quarter_bounds(pub.quarter)
        (n,) = con.execute(
            f"SELECT count(*) FROM bronze WHERE date >= DATE '{start}' AND date < DATE '{end}'"
        ).fetchone()
        if not n:
            continue
        ours = compute(con, pub)
        fields = ("drives", "boot_drives", "analyzed", "drive_days", "failures", "afr")
        rows.append(
            {
                "quarter": pub.quarter,
                "source": pub.url,
                "rule": {"min_drives": pub.min_drives, "min_drive_days": pub.min_drive_days},
                "published": {f: getattr(pub, f) for f in fields},
                "lake": ours,
            }
        )
    return rows
