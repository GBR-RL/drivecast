"""Paths and constants shared by the pipeline stages."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("DRIVECAST_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"
LAKE = DATA / "lake"
RUNS = ROOT / "runs"
SEED = 13
