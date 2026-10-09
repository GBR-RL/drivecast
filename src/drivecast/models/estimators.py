"""The models compared in the backtest, behind one small interface.

- ``rule``: Backblaze's own heuristic, a drive is suspect when any of SMART 5, 187, 188, 197 or
  198 is above zero; the score is how many are.
- ``logreg``: logistic regression on signed-log features, imputed and standardised.
- ``lightgbm``: gradient-boosted trees on the raw features (missing values are native).

Every model is fitted on the weighted training sample (all positives, 1% of negatives at weight
100) and returns a score where higher means more likely to fail within 30 days.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

RULE_ATTRIBUTES = ("smart_5", "smart_187", "smart_188", "smart_197", "smart_198")


class Model(Protocol):
    name: str

    def fit(self, x: pd.DataFrame, y: NDArray[np.bool_], w: NDArray[np.float64]) -> Model: ...

    def score(self, x: pd.DataFrame) -> NDArray[np.float32]: ...


class Rule:
    name = "rule"

    def fit(self, x: pd.DataFrame, y: NDArray[np.bool_], w: NDArray[np.float64]) -> Rule:
        return self

    def score(self, x: pd.DataFrame) -> NDArray[np.float32]:
        flags = np.zeros(len(x), dtype=np.float32)
        for c in RULE_ATTRIBUTES:
            flags += (x[c].fillna(0).to_numpy() > 0).astype(np.float32)
        return np.asarray(flags, dtype=np.float32)


def _signed_log(a: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.sign(a) * np.log1p(np.abs(a)), dtype=np.float64)


class LogReg:
    name = "logreg"

    def __init__(self, c: float = 0.1, seed: int = 13) -> None:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler

        self.pipeline = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            FunctionTransformer(_signed_log),
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2000, random_state=seed),
        )

    def fit(self, x: pd.DataFrame, y: NDArray[np.bool_], w: NDArray[np.float64]) -> LogReg:
        self.pipeline.fit(x.to_numpy(np.float64), y, logisticregression__sample_weight=w)
        return self

    def score(self, x: pd.DataFrame) -> NDArray[np.float32]:
        p = np.asarray(self.pipeline.predict_proba(x.to_numpy(np.float64)))[:, 1]
        return np.asarray(p, dtype=np.float32)


class LightGBM:
    name = "lightgbm"

    def __init__(self, seed: int = 13, threads: int = 4, **params: Any) -> None:
        self.params: dict[str, Any] = {
            "objective": "binary",
            "learning_rate": 0.05,
            "n_estimators": 400,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": seed,
            "n_jobs": threads,
            "verbose": -1,
        } | params

    def fit(self, x: pd.DataFrame, y: NDArray[np.bool_], w: NDArray[np.float64]) -> LightGBM:
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(x, y, sample_weight=w)
        return self

    def score(self, x: pd.DataFrame) -> NDArray[np.float32]:
        p = np.asarray(self.model.predict_proba(x))[:, 1]
        return np.asarray(p, dtype=np.float32)

    def importance(self) -> dict[str, float]:
        gain = self.model.booster_.feature_importance(importance_type="gain")
        names = self.model.booster_.feature_name()
        total = float(gain.sum()) or 1.0
        return {n: float(g) / total for n, g in sorted(zip(names, gain, strict=True),
                                                       key=lambda t: -t[1])}  # fmt: skip


def make(name: str, threads: int = 4) -> Model:
    if name == "rule":
        return Rule()
    if name == "logreg":
        return LogReg()
    if name == "lightgbm":
        return LightGBM(threads=threads)
    raise KeyError(name)
