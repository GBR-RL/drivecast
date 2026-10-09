from pathlib import Path
from typing import Any

import pytest

from drivecast.mlops import gate, policies


def staleness_result(quarter: str, ap_by_age: dict[str, float], drift: float) -> dict[str, Any]:
    models = {}
    for key, ap in ap_by_age.items():
        age = 99 if key == "frozen" else int(key)
        name = "lightgbm_frozen" if key == "frozen" else f"lightgbm_age{age}"
        models[name] = {
            "family": "lightgbm", "age": age, "average_precision": ap,
            "daily_25": {"recall": ap}, "recall_at_1pct_false_alarms": ap,
            "drift": {"before": {"psi_mean": drift * max(age, 0) if key != "frozen" else 1.0}},
        }  # fmt: skip
    return {"quarter": quarter, "models": models}


def test_policies_look_up_the_age_they_would_deploy() -> None:
    results = [
        staleness_result(
            q, {"0": 0.10, "1": 0.09, "2": 0.08, "3": 0.07, "4": 0.06, "frozen": 0.05}, drift=0.06
        )
        for q in ("2015Q2", "2015Q3", "2015Q4", "2016Q1", "2016Q2", "2016Q3")
    ]
    t = policies.table(results)
    never = policies.simulate(t, "never")
    assert never["mean"]["average_precision"] == pytest.approx(0.05)
    quarterly = policies.simulate(t, "quarterly")
    assert (quarterly["retrains"], quarterly["mean"]["average_precision"]) == (
        6,
        pytest.approx(0.10),
    )
    yearly = policies.simulate(t, "yearly")
    assert list(yearly["ages"].values()) == ["0", "1", "2", "3", "0", "1"]
    # Drift 0.06 per quarter of age: a 0.1 threshold retrains when the model is two quarters old.
    on_drift = policies.simulate(t, "drift", threshold=0.1)
    assert list(on_drift["ages"].values()) == ["0", "1", "0", "1", "0", "1"]


def test_drift_vs_decay_reports_correlations() -> None:
    results = [
        staleness_result(q, {"0": 0.10, "1": 0.09, "2": 0.07, "4": 0.04}, drift=0.05)
        for q in ("2016Q1", "2016Q2", "2016Q3")
    ]
    d = policies.drift_vs_decay(policies.table(results))
    assert d["pairs"] == 9
    assert d["spearman_age_decay"] < -0.9
    assert d["mean_relative_ap_by_age"]["4"] == pytest.approx(-0.6)


def test_gate_promotes_only_a_clearly_better_challenger() -> None:
    test_ap = {
        "2016Q1": {"rule": 0.02, "logreg": 0.10, "lightgbm": 0.09},
        "2016Q2": {"rule": 0.02, "logreg": 0.11, "lightgbm": 0.08},
        "2016Q3": {"rule": 0.02, "logreg": 0.07, "lightgbm": 0.12},
    }
    validation = {
        "2016Q2": {"rule": 0.02, "logreg": 0.103, "lightgbm": 0.100},  # within the margin
        "2016Q3": {"rule": 0.02, "logreg": 0.120, "lightgbm": 0.100},  # clearly better
    }
    r = gate.simulate(test_ap, validation, start="lightgbm", margin=0.05)
    assert [d["champion"] for d in r["decisions"]] == ["lightgbm", "lightgbm", "logreg"]
    assert r["promotions"] == 1
    assert r["mean_test_ap"]["gate"] == pytest.approx((0.09 + 0.08 + 0.07) / 3)
    assert r["mean_test_ap"]["oracle"] == pytest.approx((0.10 + 0.11 + 0.12) / 3)


def test_registry_logs_runs_and_moves_the_champion_alias(tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from drivecast.mlops import registry

    models = tmp_path / "models"
    models.mkdir()
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(50, 3)), rng.random(50) < 0.5
    for q in ("2016Q1", "2016Q2"):
        joblib.dump(LogisticRegression().fit(x, y), models / f"logreg_{q}.joblib")
    model = {"average_precision": 0.1, "roc_auc": 0.8,
             "daily": {"25": {"precision": 0.3, "recall": 0.2}, "100": {"recall": 0.4}},
             "false_alarm": {"0.01": {"recall": 0.5}}}  # fmt: skip
    backtests = {
        q: {
            "train_quarters": ["2015Q1"],
            "train_rows": 10,
            "train_positive_rows": 1,
            "models": {"rule": model, "logreg": model},
        }
        for q in ("2016Q1", "2016Q2")
    }
    decisions = {"decisions": [{"quarter": "2016Q1", "champion": "rule"},
                               {"quarter": "2016Q2", "champion": "logreg"}]}  # fmt: skip
    summary = registry.build(tmp_path / "store", backtests, models, decisions)
    assert (summary["runs"], summary["versions"]) == (4, 2)
    assert summary["champion"] == {"version": "2", "family": "logreg", "quarter": "2016Q2"}
