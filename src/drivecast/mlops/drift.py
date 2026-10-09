"""Data drift between the data a model was trained on and the data it now sees.

- **PSI** (population stability index) per feature, on ten quantile bins of the reference data:
  ``sum((current - reference) * ln(current / reference))`` over the bins. Below 0.1 is usually
  read as stable, above 0.25 as a large shift. Weighted, so the sampled negatives count for the
  population they stand for.
- **Unseen models**: the share of drive-days whose drive model never appears in the training
  data. New drive models enter the fleet every year.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

BINS = 10
EPS = 1e-4


def psi(
    reference: NDArray[np.float64],
    current: NDArray[np.float64],
    w_ref: NDArray[np.float64] | None = None,
    w_cur: NDArray[np.float64] | None = None,
) -> float:
    """PSI of one feature; missing values are a bin of their own."""
    w_ref = np.ones(len(reference)) if w_ref is None else w_ref
    w_cur = np.ones(len(current)) if w_cur is None else w_cur
    ref_ok, cur_ok = ~np.isnan(reference), ~np.isnan(current)
    edges = np.unique(np.quantile(reference[ref_ok], np.linspace(0, 1, BINS + 1)[1:-1])) if (
        ref_ok.any()
    ) else np.array([])  # fmt: skip

    def shares(values: NDArray[np.float64], ok: NDArray[np.bool_], w: NDArray[np.float64]) -> Any:
        counts = np.bincount(np.searchsorted(edges, values[ok], side="right"), weights=w[ok],
                             minlength=len(edges) + 1)  # fmt: skip
        out = np.append(counts, w[~ok].sum())
        return np.maximum(out / max(w.sum(), EPS), EPS)

    p, q = shares(reference, ref_ok, w_ref), shares(current, cur_ok, w_cur)
    return float(np.sum((q - p) * np.log(q / p)))


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str]
) -> dict[str, float]:
    """PSI of every feature, using the "weight" column of both frames when present."""
    w_ref = reference["weight"].to_numpy(np.float64) if "weight" in reference else None
    w_cur = current["weight"].to_numpy(np.float64) if "weight" in current else None
    return {
        f: psi(reference[f].to_numpy(np.float64), current[f].to_numpy(np.float64), w_ref, w_cur)
        for f in features
    }


def unseen_models(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    """Weighted share of the current rows whose drive model is not in the reference."""
    known = set(reference["model"].unique())
    w = current["weight"].to_numpy(np.float64) if "weight" in current else np.ones(len(current))
    unseen = ~current["model"].isin(known).to_numpy()
    return float(w[unseen].sum() / w.sum()) if len(w) else 0.0


def summary(reference: pd.DataFrame, current: pd.DataFrame, features: list[str],
            top: list[str] | None = None) -> dict[str, Any]:  # fmt: skip
    """Drift of every feature, its mean and maximum, and the share of unseen drive models."""
    per_feature = feature_drift(reference, current, features)
    watched = top or features
    return {
        "psi_mean": float(np.mean([per_feature[f] for f in watched])),
        "psi_max": float(max(per_feature[f] for f in watched)),
        "features_over_0_25": int(sum(per_feature[f] > 0.25 for f in watched)),
        "unseen_models": unseen_models(reference, current),
        "psi": {f: round(v, 4) for f, v in per_feature.items()},
    }
