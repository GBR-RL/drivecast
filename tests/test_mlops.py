import numpy as np
import pandas as pd
import pytest

from drivecast.mlops import drift, staleness


def test_psi_is_zero_for_the_same_data_and_grows_with_a_shift() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=20_000)
    assert drift.psi(a, a) == pytest.approx(0.0, abs=1e-9)
    small = drift.psi(a, rng.normal(0.1, 1, 20_000))
    large = drift.psi(a, rng.normal(1.0, 1, 20_000))
    assert 0 < small < 0.1
    assert large > 0.25


def test_psi_counts_missing_values_as_their_own_bin() -> None:
    a = np.r_[np.arange(1000.0), np.full(1000, np.nan)]
    b = np.arange(2000.0) % 1000
    assert drift.psi(a, b) > 1  # half of the reference was missing, none of the current is


def test_psi_uses_weights() -> None:
    a = np.r_[np.zeros(100), np.ones(100)]
    b = np.r_[np.zeros(100), np.ones(10)]
    w_b = np.r_[np.ones(100), np.full(10, 10.0)]  # the ten ones stand for a hundred
    assert drift.psi(a, b, w_cur=w_b) == pytest.approx(0.0, abs=1e-9)


def test_unseen_models_share() -> None:
    ref = pd.DataFrame({"model": ["A", "B"]})
    cur = pd.DataFrame({"model": ["A", "C", "C", "B"], "weight": [1.0, 1.0, 2.0, 0.0]})
    assert drift.unseen_models(ref, cur) == 0.75


def test_quarter_arithmetic() -> None:
    assert staleness.shift("2017Q2", -8) == "2015Q2"
    assert staleness.shift("2016Q4", 1) == "2017Q1"
    assert staleness.quarters_between("2015Q2", "2017Q1") == 7
