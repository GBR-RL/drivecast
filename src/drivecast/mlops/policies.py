"""Retraining policies, computed from the staleness table, and whether drift predicts decay.

A policy decides at the start of every quarter whether to keep the deployed model or deploy a
freshly fitted one. Its performance in a quarter is the measured performance of a model of that
age in that quarter, so no model is fitted here.

- **never**: the model deployed in 2015Q2, kept forever;
- **yearly**: a new model every fourth quarter;
- **quarterly**: a new model every quarter (the backtest);
- **on drift**: a new model when the mean PSI of the deployed model's ten most important
  features, between its training data and the quarter that just ended, exceeds a threshold
  (and at the latest when it is four quarters old).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drivecast.mlops.staleness import FIRST_DEPLOYMENT, quarters_between

MAX_AGE = 4
METRICS = ("average_precision", "recall_at_25", "recall_at_1pct_false_alarms")


def _row(model: dict[str, Any]) -> dict[str, float]:
    return {
        "average_precision": model["average_precision"],
        "recall_at_25": model["daily_25"]["recall"],
        "recall_at_1pct_false_alarms": model["recall_at_1pct_false_alarms"],
        "drift": model["drift"]["before"]["psi_mean"],
    }


def table(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """quarter -> model age ("0".."8", or "frozen") -> metrics and drift, for LightGBM."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for r in results:
        q = r["quarter"]
        out[q] = {}
        for name, m in r["models"].items():
            if m["family"] != "lightgbm":
                continue
            key = "frozen" if name.endswith("frozen") else str(m["age"])
            out[q][key] = _row(m)
    return out


def simulate(t: dict[str, dict[str, dict[str, float]]], policy: str,
             threshold: float = 0.0) -> dict[str, Any]:  # fmt: skip
    quarters = sorted(t)
    ages: list[str] = []
    deployed = FIRST_DEPLOYMENT
    retrains = 0
    for q in quarters:
        if policy == "never":
            key = "frozen"
        elif policy == "quarterly":
            key = "0"
            retrains += 1
        elif policy == "yearly":
            age = quarters_between(FIRST_DEPLOYMENT, q) % 4
            retrains += age == 0
            key = str(age)
        elif policy == "drift":
            age = quarters_between(deployed, q)
            current = t[q].get(str(age))
            if age > MAX_AGE or current is None or (age > 0 and current["drift"] > threshold):
                deployed, age = q, 0
                retrains += 1
            key = str(age)
        else:
            raise KeyError(policy)
        ages.append(key)
    rows = [t[q][k] for q, k in zip(quarters, ages, strict=True)]
    return {
        "policy": policy if policy != "drift" else f"drift>{threshold}",
        "retrains": int(retrains),
        "mean": {m: float(np.mean([r[m] for r in rows])) for m in METRICS},
        "ages": dict(zip(quarters, ages, strict=True)),
    }


def compare(results: list[dict[str, Any]],
            thresholds: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2)) -> dict[str, Any]:  # fmt: skip
    t = table(results)
    policies = [simulate(t, p) for p in ("never", "yearly", "quarterly")]
    policies += [simulate(t, "drift", th) for th in thresholds]
    return {"quarters": sorted(t), "policies": policies, "drift_vs_decay": drift_vs_decay(t)}


def drift_vs_decay(t: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    """Spearman correlation of a model's drift with its loss of AP against a fresh model."""
    from scipy.stats import spearmanr

    drifts, decays, ages = [], [], []
    for by_age in t.values():
        fresh = by_age.get("0")
        if fresh is None or fresh["average_precision"] <= 0:
            continue
        for key, row in by_age.items():
            if key in ("0", "frozen"):
                continue
            drifts.append(row["drift"])
            decays.append(row["average_precision"] / fresh["average_precision"] - 1)
            ages.append(int(key))
    if len(drifts) < 3:
        return {"pairs": len(drifts)}
    rho_drift = spearmanr(drifts, decays)
    rho_age = spearmanr(ages, decays)
    return {
        "pairs": len(drifts),
        "spearman_drift_decay": float(rho_drift.statistic),
        "p_drift": float(rho_drift.pvalue),
        "spearman_age_decay": float(rho_age.statistic),
        "p_age": float(rho_age.pvalue),
        "mean_relative_ap_by_age": {
            str(a): float(np.mean([d for d, g in zip(decays, ages, strict=True) if g == a]))
            for a in sorted(set(ages))
        },
    }
