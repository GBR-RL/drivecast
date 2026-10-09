"""Champion and challenger: which model family to deploy each quarter.

At the start of every quarter each family (the rule, logistic regression, LightGBM) has a fresh
model. The deployed family, the champion, is replaced only if a challenger did clearly better on
the most recent data whose labels are already known: the quarter that just ended, without its
last 30 days. Those scores come from the backtest models of that quarter, fitted on even earlier
data, so the decision never uses data that its models were fitted on, nor any label from the
future.

The gate is compared with always deploying one family, and with an oracle that knows the best
family of every quarter in advance (the upper bound of any selection rule).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drivecast.features.build import HORIZON_DAYS
from drivecast.quality.published import quarter_bounds

FAMILIES = ("rule", "logreg", "lightgbm")
MARGIN = 0.05


def validation_ap(con: Any, scores: str, decided_for: str) -> dict[str, float]:
    """AP of every family on the scores of the quarter before ``decided_for``, using only the
    rows whose label is known on the first day of ``decided_for``."""
    from sklearn.metrics import average_precision_score

    start, _ = quarter_bounds(decided_for)
    frame = con.execute(
        f"SELECT label, {', '.join(f'score_{f}' for f in FAMILIES)} FROM read_parquet('{scores}') "
        f"WHERE day < datediff('day', DATE '1970-01-01', DATE '{start}') - {HORIZON_DAYS}"
    ).df()
    return {
        f: float(average_precision_score(frame["label"], frame[f"score_{f}"])) for f in FAMILIES
    }


def simulate(
    test_ap: dict[str, dict[str, float]],
    validation: dict[str, dict[str, float]],
    *,
    start: str = "lightgbm",
    margin: float = MARGIN,
) -> dict[str, Any]:
    """Decide quarter by quarter; ``validation[q]`` is known at the start of q."""
    champion = start
    decisions = []
    for q in sorted(test_ap):
        promoted = False
        if q in validation:
            val = validation[q]
            best = max(FAMILIES, key=lambda f: val[f])
            if best != champion and val[best] > val[champion] * (1 + margin):
                champion, promoted = best, True
        decisions.append({"quarter": q, "champion": champion, "promoted": promoted,
                          "test_ap": test_ap[q][champion]})  # fmt: skip
    quarters = sorted(test_ap)
    mean = {f"always_{f}": float(np.mean([test_ap[q][f] for q in quarters])) for f in FAMILIES}
    mean["gate"] = float(np.mean([d["test_ap"] for d in decisions]))
    mean["oracle"] = float(np.mean([max(test_ap[q].values()) for q in quarters]))
    return {
        "margin": margin,
        "promotions": sum(d["promoted"] for d in decisions),
        "mean_test_ap": mean,
        "decisions": decisions,
    }
