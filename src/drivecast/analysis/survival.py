"""How long drives live: AFR with exact intervals, the hazard by age, Kaplan-Meier and Cox models.

Age comes from power-on hours (SMART 9), not from the first report: many drives were already in
service when the data starts in 2013, or when they were moved into the fleet. A drive therefore
enters observation at the age it had when first seen (left truncation) and leaves it at its last
report, failed or not (right censoring). Ignoring the late entry would count old survivors as if
they had been watched since new and make every curve look better than it is.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2

DAYS_PER_YEAR = 365.0
MAX_AGE_YEARS = 15.0


def lifetimes(drives: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Entry and exit age (days) and the failure flag per data drive, with the drives dropped."""
    d = drives[~drives["not_data_drive"]].copy()
    dropped = {"no_power_on_hours": int(d["hours_first"].isna().sum())}
    d = d.dropna(subset=["hours_first"])
    first = pd.to_datetime(d["first_date"])
    last = pd.to_datetime(d["last_date"])
    hours_date = pd.to_datetime(d["hours_first_date"])
    # Age on the first report: the hours read later, minus the days in between.
    d["entry_age"] = d["hours_first"] / 24.0 - (hours_date - first).dt.days
    d["exit_age"] = d["entry_age"] + (last - first).dt.days + 1
    d["failed"] = d["failure_date"].notna()
    bad = (d["entry_age"] < 0) | (d["exit_age"] > MAX_AGE_YEARS * DAYS_PER_YEAR)
    dropped["implausible_age"] = int(bad.sum())
    d = d[~bad]
    d["cohort"] = first[~bad].dt.year
    d["capacity_tb"] = (d["capacity_bytes"] / 1e12).round(0)
    columns = ["serial_number", "model", "manufacturer", "capacity_tb", "cohort", "entry_age",
               "exit_age", "failed"]  # fmt: skip
    return d[columns].reset_index(drop=True), dropped


def poisson_interval(failures: int, years: float, level: float = 0.95) -> tuple[float, float]:
    """Exact (Garwood) interval for an annualized failure rate, in percent."""
    alpha = 1 - level
    low = chi2.ppf(alpha / 2, 2 * failures) / 2 if failures else 0.0
    high = chi2.ppf(1 - alpha / 2, 2 * failures + 2) / 2
    return 100 * low / years, 100 * high / years


def afr_table(life: pd.DataFrame, by: str, min_drive_years: float = 1000) -> pd.DataFrame:
    """AFR per group with exact 95% intervals (drive years from the observed spans)."""
    years = (life["exit_age"] - life["entry_age"]) / DAYS_PER_YEAR
    g = (
        life.assign(years=years)
        .groupby(by)
        .agg(
            drives=("serial_number", "size"),
            drive_years=("years", "sum"),
            failures=("failed", "sum"),
        )
    )
    g = g[g["drive_years"] >= min_drive_years].copy()
    g["afr"] = 100 * g["failures"] / g["drive_years"]
    bounds = [
        poisson_interval(int(f), y) for f, y in zip(g["failures"], g["drive_years"], strict=True)
    ]
    g["afr_low"] = [b[0] for b in bounds]
    g["afr_high"] = [b[1] for b in bounds]
    return g.sort_values("drive_years", ascending=False)


def hazard_by_age(
    life: pd.DataFrame, width_days: float = 91.25, max_years: float = 12
) -> pd.DataFrame:
    """Failures per drive year in age buckets: exposure is each drive's overlap with a bucket."""
    edges = np.arange(0, max_years * DAYS_PER_YEAR + width_days, width_days)
    entry = life["entry_age"].to_numpy()
    exit_ = life["exit_age"].to_numpy()
    exposure = np.clip(
        np.minimum(exit_[:, None], edges[None, 1:]) - np.maximum(entry[:, None], edges[None, :-1]),
        0,
        None,
    ).sum(axis=0)
    failed_age = exit_[life["failed"].to_numpy()]
    failures, _ = np.histogram(failed_age, bins=edges)
    years = exposure / DAYS_PER_YEAR
    out = pd.DataFrame(
        {"age_from_years": edges[:-1] / DAYS_PER_YEAR, "drive_years": years, "failures": failures}
    )
    out["afr"] = np.where(years > 0, 100 * failures / np.maximum(years, 1e-9), np.nan)
    return out


def kaplan_meier(life: pd.DataFrame, by: str, at_years: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
                 min_drives: int = 2000, *, late_entry: bool = True) -> dict[str, Any]:  # fmt: skip
    """Kaplan-Meier survival at whole years, per group.

    With ``late_entry=False`` every drive is treated as watched from age zero, the common mistake
    this module exists to avoid; it is computed only to show the size of the bias.
    """
    from lifelines import KaplanMeierFitter

    out: dict[str, Any] = {}
    for name, g in life.groupby(by):
        if len(g) < min_drives:
            continue
        km = KaplanMeierFitter()
        entry = g["entry_age"] / DAYS_PER_YEAR if late_entry else None
        km.fit(g["exit_age"] / DAYS_PER_YEAR, g["failed"], entry=entry)
        ci = km.confidence_interval_survival_function_
        rows = {}
        for year in at_years:
            if year > g["exit_age"].max() / DAYS_PER_YEAR:
                break
            at_risk = int(((g["entry_age"] <= year * DAYS_PER_YEAR)
                           & (g["exit_age"] > year * DAYS_PER_YEAR)).sum())  # fmt: skip
            idx = km.survival_function_.index.searchsorted(year, side="right") - 1
            rows[str(year)] = {
                "survival": float(km.survival_function_.iloc[max(idx, 0), 0]),
                "low": float(ci.iloc[max(idx, 0), 0]),
                "high": float(ci.iloc[max(idx, 0), 1]),
                "at_risk": at_risk,
            }
        out[str(name)] = {"drives": len(g), "failures": int(g["failed"].sum()), "years": rows}
    return out


def curves(life: pd.DataFrame, by: str, groups: list[str], step: float = 0.1,
           max_years: float = 8) -> dict[str, list[tuple[float, float]]]:  # fmt: skip
    """Survival curves on a grid of ages, for charts."""
    from lifelines import KaplanMeierFitter

    grid: NDArray[np.float64] = np.arange(0, max_years + step, step)
    result = {}
    for name in groups:
        g = life[life[by] == name]
        km = KaplanMeierFitter()
        km.fit(g["exit_age"] / DAYS_PER_YEAR, g["failed"], entry=g["entry_age"] / DAYS_PER_YEAR)
        limit = float(g["exit_age"].max() / DAYS_PER_YEAR)
        values = km.survival_function_at_times(grid[grid <= limit]).to_numpy()
        result[name] = [(round(float(t), 2), round(float(v), 5)) for t, v in
                        zip(grid[grid <= limit], values, strict=True)]  # fmt: skip
    return result


def cox(life: pd.DataFrame, min_drives: int = 2000) -> dict[str, Any]:
    """Cox model of the hazard by manufacturer, capacity and cohort, with late entry."""
    from lifelines import CoxPHFitter

    keep = life.groupby("manufacturer")["serial_number"].transform("size") >= min_drives
    d = life[keep].copy()
    d["start"] = d["entry_age"] / DAYS_PER_YEAR
    d["stop"] = d["exit_age"] / DAYS_PER_YEAR
    d["failed"] = d["failed"].astype(int)
    reference = d["manufacturer"].value_counts().index[0]
    dummies = pd.get_dummies(d["manufacturer"], prefix="maker", dtype=float).drop(
        columns=f"maker_{reference}"
    )
    x = pd.concat([d[["start", "stop", "failed", "capacity_tb", "cohort"]], dummies], axis=1)
    x["cohort"] = x["cohort"] - x["cohort"].min()
    constant = [c for c in ("capacity_tb", "cohort", *dummies.columns) if x[c].std() == 0]
    x = x.drop(columns=constant)
    model = CoxPHFitter(penalizer=0.0)
    model.fit(x, duration_col="stop", event_col="failed", entry_col="start")
    summary = model.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    return {
        "reference_manufacturer": reference,
        "drives": len(x),
        "failures": int(x["failed"].sum()),
        "concordance": float(model.concordance_index_),
        "hazard_ratios": {
            name: {
                "hr": float(r.iloc[0]),
                "low": float(r.iloc[1]),
                "high": float(r.iloc[2]),
                "p": float(r.iloc[3]),
            }
            for name, r in summary.iterrows()
        },
    }


def analyse(drives: pd.DataFrame) -> dict[str, Any]:
    life, dropped = lifetimes(drives)
    makers = life["manufacturer"].value_counts()
    big = [m for m, n in makers.items() if n >= 2000]
    return {
        "drives": len(life),
        "failures": int(life["failed"].sum()),
        "dropped": dropped,
        "afr_by_manufacturer": afr_table(life, "manufacturer")
        .round(4)
        .reset_index()
        .to_dict("records"),
        "afr_by_model": afr_table(life, "model", min_drive_years=5000)
        .round(4)
        .reset_index()
        .to_dict("records"),
        "hazard_by_age": hazard_by_age(life).round(4).to_dict("records"),
        "kaplan_meier": kaplan_meier(life, "manufacturer"),
        "kaplan_meier_ignoring_late_entry": kaplan_meier(life, "manufacturer", late_entry=False),
        "curves": curves(life, "manufacturer", [str(m) for m in big]),
        "cox": cox(life),
    }
