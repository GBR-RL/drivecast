"""How well a score finds failing drives in a test quarter.

Accuracy is meaningless here: on a given day about 0.15% of drives fail within 30 days. The
metrics are the ones an operations team would use.

- **Daily budget.** Every day the k highest-scoring drives are flagged (a replacement crew can
  check k drives a day). Precision: the share of flagged drive-days whose drive fails within 30
  days. Recall: the share of drives failing in the quarter that were flagged at least once in
  their last 30 days. Lead time: days from the first flag to the failure.
- **Fixed false-alarm rate.** The share of failing drives detected when the threshold flags a
  given share of the healthy drives at least once in the quarter. The threshold is set on the
  test quarter itself, so this compares rankings; it is not a deployable threshold.
- **Average precision** over drive-days, as a threshold-free summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

DAILY_BUDGETS = (10, 25, 50, 100)
FALSE_ALARM_RATES = (0.005, 0.01, 0.02)


@dataclass(frozen=True)
class TestRows:
    """The scored rows of a test quarter (one per drive and day)."""

    __test__ = False  # not a pytest test class

    drive: NDArray[np.int64]
    day: NDArray[np.int32]
    label: NDArray[np.bool_]  # fails within 30 days
    days_to_failure: NDArray[np.float64]  # -1 when the drive does not fail
    fails_in_quarter: NDArray[np.bool_]  # label, and the failure is inside the quarter


def flagged_by_day(day: NDArray[np.int32], score: NDArray[np.float32], k: int) -> NDArray[np.bool_]:
    """True for the k highest scores of each day, among scores above zero.

    A score of zero means "no reason to suspect" (the rule's score for a drive with no warning
    counter), so such drives are never flagged just to fill the budget.
    """
    order = np.lexsort((-score, day))
    sorted_days = day[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_days)) + 1]
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    flagged = np.zeros(len(order), dtype=bool)
    flagged[order[rank < k]] = True
    return flagged & (score > 0)


def daily_budget(rows: TestRows, score: NDArray[np.float32], k: int) -> dict[str, Any]:
    drive, label = rows.drive, rows.label
    flagged = flagged_by_day(rows.day, score, k)
    failed = np.unique(drive[label & rows.fails_in_quarter])
    hits = flagged & label & rows.fails_in_quarter
    detected = np.unique(drive[hits])
    lead: NDArray[np.float64] = np.array([])
    if len(detected):
        # Earliest flagged day before failure, per detected drive: the largest days_to_failure.
        d, t = drive[hits], rows.days_to_failure[hits]
        order = np.lexsort((-t, d))
        first = np.r_[True, np.diff(d[order]) != 0]
        lead = t[order][first]
    low, high = bootstrap_recall(np.isin(failed, detected))
    return {
        "precision": float(label[flagged].mean()) if flagged.any() else 0.0,
        "flagged_per_day": float(flagged.sum() / max(1, len(np.unique(rows.day)))),
        "recall": len(detected) / len(failed) if len(failed) else float("nan"),
        "recall_ci": [low, high],
        "detected": len(detected),
        "failed": len(failed),
        "median_lead_days": float(np.median(lead)) if len(lead) else float("nan"),
    }


def at_false_alarm_rate(
    rows: TestRows, score: NDArray[np.float32], rate: float
) -> dict[str, float]:
    """Recall of failing drives when ``rate`` of the healthy drives raise an alarm."""
    drive, label = rows.drive, rows.label
    positive_drives = np.unique(drive[label])
    healthy = ~np.isin(drive, positive_drives)
    hd, hs = drive[healthy], score[healthy]
    order = np.lexsort((-hs, hd))
    first = np.r_[True, np.diff(hd[order]) != 0]
    healthy_max = hs[order][first]
    threshold = float(np.quantile(healthy_max, 1 - rate))
    window = label & rows.fails_in_quarter
    fd, fs = drive[window], score[window]
    order = np.lexsort((-fs, fd))
    first = np.r_[True, np.diff(fd[order]) != 0]
    failed_max = fs[order][first]
    return {
        "recall": float((failed_max > threshold).mean()) if len(failed_max) else float("nan"),
        "threshold": threshold,
        "healthy_drives": len(healthy_max),
        "failed_drives": len(failed_max),
    }


def bootstrap_recall(
    detected: NDArray[np.bool_], n: int = 1000, seed: int = 13
) -> tuple[float, float]:
    """95% interval of a recall over failing drives, resampling the drives."""
    if len(detected) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(detected.astype(np.float64), size=(n, len(detected)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate(rows: TestRows, score: NDArray[np.float32]) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    label = rows.label
    result: dict[str, Any] = {
        "rows": len(score),
        "positive_rows": int(label.sum()),
        "average_precision": float(average_precision_score(label, score)),
        "roc_auc": float(roc_auc_score(label, score)) if 0 < label.sum() < len(label) else None,
        "daily": {},
        "false_alarm": {},
    }
    for k in DAILY_BUDGETS:
        result["daily"][str(k)] = daily_budget(rows, score, k)
    for rate in FALSE_ALARM_RATES:
        result["false_alarm"][str(rate)] = at_false_alarm_rate(rows, score, rate)
    return result
