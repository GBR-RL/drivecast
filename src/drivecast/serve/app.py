"""HTTP service: the risk that a drive fails within 30 days, from its recent SMART readings.

- ``GET /health``: liveness and readiness, with the model being served;
- ``POST /score``: drives with their recent daily readings; returns each drive's score for its
  last day, the features computed by the batch pipeline's SQL, and which of Backblaze's five
  warning attributes are above zero;
- ``POST /score/features``: rows of features already computed by the pipeline.

The model directory comes from ``DRIVECAST_MODEL_DIR`` (default ``/app/model``).
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from drivecast.features.build import feature_names
from drivecast.models.estimators import RULE_ATTRIBUTES
from drivecast.serve.features import READING_COLUMNS, last_day_features
from drivecast.serve.model import ServedModel, load

MAX_DRIVES = 1000
MAX_READINGS = 400


class Drive(BaseModel):
    serial_number: str
    model: str
    capacity_bytes: int | None = None
    first_seen: date | None = None
    readings: Annotated[list[dict[str, Any]], Field(min_length=1, max_length=MAX_READINGS)]

    @field_validator("readings")
    @classmethod
    def known_columns(cls, readings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in readings:
            if "date" not in r:
                raise ValueError("every reading needs a date")
            unknown = set(r) - READING_COLUMNS - {"date"}
            if unknown:
                raise ValueError(f"unknown reading fields: {sorted(unknown)}")
            r["date"] = date.fromisoformat(str(r["date"]))
        return readings


class ScoreRequest(BaseModel):
    drives: Annotated[list[Drive], Field(min_length=1, max_length=MAX_DRIVES)]


class FeatureRequest(BaseModel):
    rows: Annotated[list[dict[str, float | None]], Field(min_length=1, max_length=10_000)]


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state["model"] = load(Path(os.environ.get("DRIVECAST_MODEL_DIR", "/app/model")))
    yield
    state.clear()


app = FastAPI(title="drivecast", lifespan=lifespan)


def _model() -> ServedModel:
    model = state.get("model")
    if not isinstance(model, ServedModel):
        raise HTTPException(503, "model not loaded")
    return model


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": _model().describe()}


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    model = _model()
    start = time.perf_counter()
    drives = [d.model_dump() for d in request.drives]
    frame = last_day_features(duckdb.connect(), drives)
    scores = model.score(frame)
    results = []
    for (_, row), s in zip(frame.iterrows(), scores, strict=True):
        results.append({
            "serial_number": row["serial_number"],
            "date": str(row["date"])[:10],
            "score": float(s),
            "warnings": [a for a in RULE_ATTRIBUTES if pd.notna(row[a]) and row[a] > 0],
            "features": {f: (None if pd.isna(row[f]) else float(row[f])) for f in feature_names()},
        })  # fmt: skip
    return {"model": model.describe(), "results": results,
            "milliseconds": round(1000 * (time.perf_counter() - start), 1)}  # fmt: skip


@app.post("/score/features")
def score_features(request: FeatureRequest) -> dict[str, Any]:
    model = _model()
    frame = pd.DataFrame(request.rows)
    missing = set(feature_names()) - set(frame.columns)
    if missing:
        raise HTTPException(422, f"missing features: {sorted(missing)}")
    return {"model": model.describe(), "scores": [float(s) for s in model.score(frame)]}
