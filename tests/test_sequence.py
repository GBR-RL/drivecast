from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from drivecast.models import sequence


def test_windows_place_each_day_and_mark_missing_days(tmp_path: Path) -> None:
    start = date(2020, 1, 1)
    days = [start + timedelta(days=i) for i in range(40) if i != 35]  # 2020-02-05 missing
    silver = pd.DataFrame(
        {
            "serial_number": "A",
            "date": days,
            "smart_5_raw": [float(i) for i in range(len(days))],
            "smart_194_raw": 25.0,
            "smart_9_raw": 1000.0,
        }
    )
    for i in sequence.COUNTERS:
        if f"smart_{i}_raw" not in silver:
            silver[f"smart_{i}_raw"] = None
    path = tmp_path / "silver.parquet"
    silver.to_parquet(path)
    keys = pd.DataFrame({"serial_number": ["A", "B"], "date": [date(2020, 2, 9), date(2020, 2, 9)]})
    x = sequence.windows(duckdb.connect(), keys, [path.as_posix()])
    assert x.shape == (2, sequence.WINDOW, sequence.CHANNELS)
    mask = x[0, :, -1]
    assert mask[-1] == 1  # the key's own day
    assert mask[-5] == 0  # 2020-02-05 did not report
    assert mask.sum() == sequence.WINDOW - 1
    reallocated = x[0, :, sequence.COUNTERS.index(5)]
    assert reallocated[-1] == pytest.approx(np.log1p(38) / 10)  # 2020-02-09 is the 39th day
    assert not x[1].any()  # an unknown drive has an empty window


def test_gru_learns_a_rising_counter() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n = 600
    seq = rng.normal(scale=0.1, size=(n, sequence.WINDOW, sequence.CHANNELS)).astype(np.float32)
    y = rng.random(n) < 0.3
    seq[y, -10:, 1] += np.linspace(0, 1, 10, dtype=np.float32)  # counter grows before failing
    static = np.zeros((n, len(sequence.STATIC) + 1), dtype=np.float32)
    model = sequence.GRUModel(hidden=8, epochs=20, threads=1, batch=64)
    model.fit(seq, static, y, np.ones(n))
    s = model.score(seq, static)
    assert s[y].mean() > s[~y].mean() + 0.2
