from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from drivecast.quality import checks, published, report

HDD = 4_000_787_030_016


def row(day: str, serial: str, **values: Any) -> dict[str, Any]:
    base = {"date": day, "serial_number": serial, "model": "M1", "capacity_bytes": HDD,
            "failure": 0, "smart_9_raw": 100}  # fmt: skip
    return base | values


def totals(con: Any) -> dict[str, int]:
    return {name: r["total"] for name, r in checks.run(con).items()}


def test_checks_count_each_problem(bronze: Callable[..., Any]) -> None:
    con = bronze(
        [
            row("2026-03-29", "A", smart_9_raw=100),
            row("2026-03-30", "A", smart_9_raw=124),
            row("2026-03-31", "A", smart_9_raw=110),  # power-on hours go down
            row("2026-03-30", "B", failure=1),
            row("2026-03-31", "B"),  # reported after its failure
            row("2026-03-29", "C"),
            row("2026-03-29", "C"),  # duplicate
            row("2026-03-29", "D", model="DELLBOSS VD", capacity_bytes=480_036_847_616),
            row("2026-03-29", "E", capacity_bytes=-1),
            row("2026-03-30", "E", model="M2"),
            row("2026-03-30", "F", smart_5_normalized=300, smart_5_raw=-5),
        ]
    )
    t = totals(con)
    assert (t["rows"], t["failures"]) == (11, 1)
    assert t["duplicate_rows"] == 1
    assert t["reported_after_failure"] == 1
    assert t["rows_after_failure"] == 1
    assert t["power_on_hours_decrease"] == 1
    assert t["not_data_drive"] == 1
    assert t["capacity_negative"] == 1
    assert t["model_changes"] == 1
    assert t["normalized_out_of_range"] == 1
    assert t["raw_negative"] == 1
    assert t["several_failures"] == t["missing_key"] == t["failure_not_binary"] == 0


def test_partial_and_missing_days(bronze: Callable[..., Any]) -> None:
    start = date(2020, 1, 1)
    rows = []
    for d in range(12):
        day = (start + timedelta(days=d)).isoformat()
        if d == 7:
            continue  # no file for this day
        reporting = range(2) if d in (4, 5) else range(10)  # two days with an outage
        rows += [row(day, f"S{i}") for i in reporting]
    con = bronze(rows)
    t = totals(con)
    assert t["partial_days"] == 2
    assert t["missing_days"] == 1
    assert checks.partial_day_ranges(con) == [
        {"from": "2020-01-05", "to": "2020-01-06", "days": 2, "min_reports": 2, "alive": 10}
    ]


def test_reconciliation_applies_the_inclusion_rule(bronze: Callable[..., Any]) -> None:
    rows = []
    for day in ("2026-03-30", "2026-03-31"):
        rows += [row(day, s) for s in ("A", "B", "C")]
        rows.append(row(day, "D", model="M2"))  # one drive of its model: excluded
        rows.append(row(day, "X", model="DELLBOSS VD", capacity_bytes=480_036_847_616))
    rows.append(row("2026-03-30", "Z", failure=1))  # failed before the last day
    con = bronze(rows)
    pub = published.Published("2026Q1", "u", afr=1.0, min_drives=1, min_drive_days=0)
    ours = published.compute(con, pub)
    assert ours["last_day"] == "2026-03-31"
    assert (ours["drives"], ours["boot_drives"], ours["analyzed"]) == (5, 1, 3)
    assert (ours["drive_days"], ours["failures"], ours["models"]) == (7, 1, 1)
    assert ours["afr"] == round(100 * 1 / (7 / 365), 2)


def test_quarter_bounds() -> None:
    assert published.quarter_bounds("2026Q2") == ("2026-04-01", "2026-07-01")
    assert published.quarter_bounds("2025Q4") == ("2025-10-01", "2026-01-01")


def test_markdown_report_lists_checks_and_reconciliation(bronze: Callable[..., Any]) -> None:
    con = bronze([row("2026-03-31", "A"), row("2026-03-31", "B", failure=1)])
    md = report.markdown(report.build(con))
    assert "| 2026 | 2 | 2 | 1 |" in md
    assert "Extra rows for a (serial number, date)" in md
    assert "[2026Q1](https://www.backblaze.com/blog/backblaze-drive-stats-for-q1-2026/)" in md
