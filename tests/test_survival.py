import numpy as np
import pandas as pd
import pytest

from drivecast.analysis import survival


def drives(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {"model": "ST4000DM000", "manufacturer": "Seagate", "capacity_bytes": 4e12,
            "not_data_drive": False, "failure_date": None}  # fmt: skip
    frame = pd.DataFrame([base | r for r in rows])
    for column in ("first_date", "last_date", "hours_first_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def test_age_comes_from_power_on_hours() -> None:
    life, dropped = survival.lifetimes(
        drives(
            [
                # first seen 2020-01-01, hours read on 2020-01-11 say 1,000 days of age
                {"serial_number": "A", "first_date": "2020-01-01", "last_date": "2020-12-31",
                 "hours_first_date": "2020-01-11", "hours_first": 24_000},
                {"serial_number": "B", "first_date": "2020-01-01", "last_date": "2020-01-02",
                 "hours_first_date": "2020-01-01", "hours_first": None},
                {"serial_number": "C", "first_date": "2020-01-01", "last_date": "2020-01-02",
                 "hours_first_date": "2020-01-01", "hours_first": 24 * 365 * 20},
            ]
        )
    )  # fmt: skip
    assert dropped == {"no_power_on_hours": 1, "implausible_age": 1}
    a = life.iloc[0]
    assert a["entry_age"] == 990
    assert a["exit_age"] == 990 + 365 + 1
    assert not a["failed"]


def test_poisson_interval_is_exact() -> None:
    low, high = survival.poisson_interval(0, 100.0)
    assert low == 0
    assert high == pytest.approx(3.689, abs=1e-3)  # chi2(0.975, 2) / 2
    low, high = survival.poisson_interval(10, 100.0)
    assert low < 10 < high


def test_hazard_exposure_is_the_overlap_with_each_age_bucket() -> None:
    life = pd.DataFrame(
        {"entry_age": [0.0, 100.0], "exit_age": [150.0, 182.5], "failed": [True, False]}
    )
    h = survival.hazard_by_age(life, width_days=91.25, max_years=1)
    years = h["drive_years"].to_numpy() * survival.DAYS_PER_YEAR
    assert np.allclose(years[:3], [91.25, 58.75 + 82.5, 0.0])
    assert h["failures"].tolist()[:2] == [0, 1]


def test_late_entry_changes_survival() -> None:
    # Old drives that were already 5 years old when first seen and then survived, plus new
    # drives that fail in their first year. Treating the old ones as watched from new hides the
    # early failures behind thousands of survivor years they were never observed for.
    rows = [{"entry_age": 5 * 365.0, "exit_age": 6 * 365.0, "failed": False}] * 50
    rows += [{"entry_age": 0.0, "exit_age": 200.0, "failed": True}] * 5
    rows += [{"entry_age": 0.0, "exit_age": 400.0, "failed": False}] * 5
    life = pd.DataFrame(rows).assign(manufacturer="Seagate", serial_number="x")
    truncated = survival.kaplan_meier(life, "manufacturer", (1,), min_drives=1)
    naive = survival.kaplan_meier(life, "manufacturer", (1,), min_drives=1, late_entry=False)
    assert truncated["Seagate"]["years"]["1"]["survival"] == pytest.approx(0.5)
    assert naive["Seagate"]["years"]["1"]["survival"] == pytest.approx(55 / 60)
