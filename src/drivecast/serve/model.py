"""The deployed model: a model file and its metadata, chosen by the registry's champion alias.

``export`` resolves the alias in the MLflow store at build time and copies the champion's model
file next to a ``metadata.json``; the service only needs that directory, not MLflow.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from drivecast.features.build import feature_names
from drivecast.models.estimators import Rule

SUFFIX = {"lightgbm": ".txt", "logreg": ".joblib"}


@dataclass
class ServedModel:
    family: str
    quarter: str
    version: str | None
    model: Any

    def score(self, x: pd.DataFrame) -> NDArray[np.float32]:
        x = x[feature_names()].astype("float64")
        if self.family == "lightgbm":
            p = self.model.predict(x.to_numpy())
        elif self.family == "logreg":
            p = self.model.predict_proba(x.to_numpy())[:, 1]
        else:
            p = Rule().score(x)
        return np.asarray(p, dtype=np.float32)

    def describe(self) -> dict[str, Any]:
        return {"family": self.family, "quarter": self.quarter, "version": self.version}


def load(model_dir: Path) -> ServedModel:
    meta = json.loads((model_dir / "metadata.json").read_text())
    family = meta["family"]
    path = model_dir / f"model{SUFFIX.get(family, '')}"
    if family == "lightgbm":
        import lightgbm as lgb

        model: Any = lgb.Booster(model_file=str(path))
    elif family == "logreg":
        import joblib

        model = joblib.load(path)
    else:
        model = None
    return ServedModel(family, meta["quarter"], meta.get("version"), model)


def champion_from_store(store: Path) -> dict[str, str]:
    """Family, quarter and version that the store's champion alias points to."""
    import os

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from mlflow import MlflowClient

    from drivecast.mlops.registry import REGISTERED

    client = MlflowClient(tracking_uri=f"sqlite:///{(store / 'mlflow.db').as_posix()}")
    version = client.get_model_version_by_alias(REGISTERED, "champion")
    return {"family": version.tags["family"], "quarter": version.tags["quarter"],
            "version": str(version.version)}  # fmt: skip


def latest(models_dir: Path, family: str = "lightgbm") -> dict[str, str]:
    """Without a store: the most recent model of a family in ``models_dir``."""
    files = sorted(models_dir.glob(f"{family}_*{SUFFIX[family]}"))
    if not files:
        raise FileNotFoundError(f"no {family} model in {models_dir}")
    return {"family": family, "quarter": files[-1].stem.split("_")[-1], "version": ""}


def export(models_dir: Path, out: Path, store: Path | None = None) -> dict[str, str]:
    """Copy the champion (or the latest LightGBM without a store) to ``out`` for serving."""
    chosen = champion_from_store(store) if store is not None else latest(models_dir)
    family, quarter = chosen["family"], chosen["quarter"]
    out.mkdir(parents=True, exist_ok=True)
    if family in SUFFIX:
        shutil.copy(
            models_dir / f"{family}_{quarter}{SUFFIX[family]}", out / f"model{SUFFIX[family]}"
        )
    meta = chosen | {"features": ",".join(feature_names())}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return chosen
