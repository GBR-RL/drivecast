"""An MLflow store with every backtest run and a model registry driven by the gate.

Each test quarter gets one run per model family, with its parameters and its backtest metrics.
The models fitted for that quarter (LightGBM and logistic regression) are logged in MLflow's
model format and registered as versions of one registered model. The alias ``champion`` follows
the gate's decisions quarter by quarter, so at the end it points to the model the gate would
deploy now, and the version tags record when each version was champion.

The store is a SQLite file plus an artifact directory. In CI it is written through a local
tracking server, so it opens anywhere with
``mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --artifacts-destination
mlflow/artifacts``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from drivecast.models.estimators import LightGBM

REGISTERED = "drivecast-failure-30d"
# Types in the logistic-regression pipeline that MLflow's skops format does not trust by
# default: the signed-log transform and numpy dtypes.
TRUSTED = ["drivecast.models.estimators._signed_log", "numpy.dtype"]
EXPERIMENT = "drivecast-backtest"


def _metrics(m: dict[str, Any]) -> dict[str, float]:
    return {
        "average_precision": m["average_precision"],
        "roc_auc": m["roc_auc"] or 0.0,
        "precision_at_25": m["daily"]["25"]["precision"],
        "recall_at_25": m["daily"]["25"]["recall"],
        "recall_at_100": m["daily"]["100"]["recall"],
        "recall_at_1pct_false_alarms": m["false_alarm"]["0.01"]["recall"],
    }


def _connect(store: Path, tracking_uri: str | None) -> None:
    import mlflow

    if tracking_uri:
        # A tracking server that proxies artifacts stores them as mlflow-artifacts:/ URIs,
        # relative to its artifact root, so the published store opens anywhere.
        mlflow.set_tracking_uri(tracking_uri)
        location = None
    else:
        mlflow.set_tracking_uri(f"sqlite:///{(store / 'mlflow.db').as_posix()}")
        location = (store / "artifacts").as_uri()
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=location)
    mlflow.set_experiment(EXPERIMENT)


def build(
    store: Path,
    backtests: dict[str, dict[str, Any]],
    models_dir: Path,
    gate: dict[str, Any],
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Log every quarter in order, register its models and move the champion alias.

    Without ``tracking_uri`` the store is a SQLite file and an artifact folder under ``store``.
    """
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import joblib
    import lightgbm as lgb
    import mlflow
    from mlflow import MlflowClient

    store = store.resolve()
    store.mkdir(parents=True, exist_ok=True)
    _connect(store, tracking_uri)
    client = MlflowClient()
    champions = {d["quarter"]: d["champion"] for d in gate["decisions"]}
    versions: dict[tuple[str, str], str] = {}
    for quarter in sorted(backtests):
        result = backtests[quarter]
        for family, m in result["models"].items():
            with mlflow.start_run(run_name=f"{quarter} {family}") as run:
                is_champion = str(champions.get(quarter) == family).lower()
                mlflow.set_tags({"quarter": quarter, "family": family, "champion": is_champion})
                mlflow.log_params({
                    "train_quarters": ",".join(result["train_quarters"]),
                    "embargo_days": 30,
                    "train_rows": result["train_rows"],
                    "train_positive_rows": result["train_positive_rows"],
                })  # fmt: skip
                if family == "lightgbm":
                    mlflow.log_params({f"lgbm_{k}": v for k, v in LightGBM().params.items()
                                       if k not in ("n_jobs", "verbose")})  # fmt: skip
                mlflow.log_metrics(_metrics(m))
                if "importance" in result and family == "lightgbm":
                    mlflow.log_dict(result["importance"], "importance.json")
                suffix = "txt" if family == "lightgbm" else "joblib"
                model_path = models_dir / f"{family}_{quarter}.{suffix}"
                if family != "rule" and model_path.exists():
                    if family == "lightgbm":
                        info = mlflow.lightgbm.log_model(
                            lgb.Booster(model_file=str(model_path)), name="model"
                        )
                    else:
                        info = mlflow.sklearn.log_model(
                            joblib.load(model_path), name="model", skops_trusted_types=TRUSTED
                        )
                    number = str(mlflow.register_model(info.model_uri, REGISTERED).version)
                    tags = {"quarter": quarter, "family": family, "run": run.info.run_id}
                    for key, value in tags.items():
                        client.set_model_version_tag(REGISTERED, number, key, value)
                    versions[(quarter, family)] = number
        champion = champions.get(quarter)
        if champion and (quarter, champion) in versions:
            number = versions[(quarter, champion)]
            client.set_registered_model_alias(REGISTERED, "champion", number)
            client.set_model_version_tag(REGISTERED, number, "champion_for", quarter)
    final = client.get_model_version_by_alias(REGISTERED, "champion") if versions else None
    summary = {
        "store": store.as_posix(),
        "registered_model": REGISTERED,
        "runs": sum(len(r["models"]) for r in backtests.values()),
        "versions": len(versions),
        "champion": None if final is None else {
            "version": str(final.version), "family": final.tags.get("family"),
            "quarter": final.tags.get("quarter"),
        },
    }  # fmt: skip
    (store / "registry.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
