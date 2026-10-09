"""A recurrent model over each drive's last 30 days, compared with the per-day models.

The per-day models see a drive's history only through hand-made summaries (the growth of a
counter over 7 and 30 days). The GRU here reads the raw daily sequence of the main SMART
attributes instead, with a mask for days the drive did not report, plus a few static facts.

Scoring every drive on every day with a sequence model is 30 times the input of a per-day model,
so the comparison uses weekly checks: every drive is scored on every 7th day of the test quarter,
and all models are evaluated on exactly those drive-days.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from drivecast.features.build import HORIZON_DAYS, previous_quarter
from drivecast.models import evaluate
from drivecast.models.backtest import train_quarters
from drivecast.quality.published import quarter_bounds

WINDOW = 30
COUNTERS = (1, 5, 7, 10, 183, 184, 187, 188, 189, 197, 198, 199)
STATIC = ("Seagate", "HGST", "WDC", "Toshiba")
CHANNELS = len(COUNTERS) + 3  # counters, temperature, age, reported-that-day mask
CHECK_EVERY = 7
NEGATIVE_SHARE = 1.0  # of the training sample's negatives (already 1% of all)


def _values_sql() -> str:
    counters = ", ".join(f"s.smart_{i}_raw::DOUBLE AS c{i}" for i in COUNTERS)
    return f"{counters}, s.smart_194_raw::DOUBLE AS temp, s.smart_9_raw::DOUBLE AS hours"


def windows(con: Any, keys: pd.DataFrame, silver: list[str]) -> NDArray[np.float32]:
    """Daily inputs of the 30 days up to each key's date: array (keys, 30, channels)."""
    con.register("keys_frame", keys[["serial_number", "date"]].reset_index(drop=True))
    con.execute("CREATE OR REPLACE TEMP TABLE k AS SELECT row_number() OVER () - 1 AS idx, * "
                "FROM keys_frame")  # fmt: skip
    listed = ", ".join(f"'{f}'" for f in silver)
    long = con.execute(
        f"SELECT k.idx, datediff('day', s.date, k.date) AS lag, {_values_sql()} "
        f"FROM k JOIN read_parquet([{listed}]) s ON s.serial_number = k.serial_number "
        f"AND s.date BETWEEN k.date - INTERVAL {WINDOW - 1} DAYS AND k.date"
    ).df()
    con.unregister("keys_frame")
    x = np.zeros((len(keys), WINDOW, CHANNELS), dtype=np.float32)
    idx = long["idx"].to_numpy(np.int64)
    pos = WINDOW - 1 - long["lag"].to_numpy(np.int64)
    for j, i in enumerate(COUNTERS):
        v = long[f"c{i}"].to_numpy(np.float64)
        x[idx, pos, j] = np.nan_to_num(np.sign(v) * np.log1p(np.abs(v)) / 10.0)
    x[idx, pos, len(COUNTERS)] = np.nan_to_num(long["temp"].to_numpy(np.float64) / 50.0)
    x[idx, pos, len(COUNTERS) + 1] = np.nan_to_num(long["hours"].to_numpy(np.float64) / 50_000.0)
    x[idx, pos, len(COUNTERS) + 2] = 1.0
    return x


def statics(frame: pd.DataFrame) -> NDArray[np.float32]:
    cols = [(frame["manufacturer"] == m).to_numpy(np.float32) for m in STATIC]
    cols.append((frame["capacity_bytes"].fillna(0).to_numpy(np.float64) / 2e13).astype(np.float32))
    return np.stack(cols, axis=1)


class GRUModel:
    name = "gru"

    def __init__(self, hidden: int = 32, epochs: int = 8, seed: int = 13, threads: int = 4,
                 batch: int = 1024) -> None:  # fmt: skip
        self.hidden, self.epochs, self.seed, self.threads = hidden, epochs, seed, threads
        self.batch = batch

    def _net(self) -> Any:
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self, channels: int, statics: int, hidden: int) -> None:
                super().__init__()
                self.gru = nn.GRU(channels, hidden, batch_first=True)
                self.head = nn.Sequential(nn.Linear(hidden + statics, 32), nn.ReLU(),
                                          nn.Linear(32, 1))  # fmt: skip

            def forward(self, seq: Any, static: Any) -> Any:
                _, h = self.gru(seq)
                return self.head(torch.cat([h[-1], static], dim=1)).squeeze(1)

        return Net(CHANNELS, len(STATIC) + 1, self.hidden)

    def fit(self, seq: NDArray[np.float32], static: NDArray[np.float32], y: NDArray[np.bool_],
            w: NDArray[np.float64]) -> GRUModel:  # fmt: skip
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        self.net = self._net()
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        weights = torch.tensor(w / w.mean(), dtype=torch.float32)
        target = torch.tensor(y, dtype=torch.float32)
        data_seq, data_static = torch.from_numpy(seq), torch.from_numpy(static)
        rng = np.random.default_rng(self.seed)
        self.losses = []
        for _ in range(self.epochs):
            order = torch.from_numpy(rng.permutation(len(y)))
            total = 0.0
            for b in range(0, len(y), self.batch):
                i = order[b : b + self.batch]
                logits = self.net(data_seq[i], data_static[i])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target[i], weight=weights[i]
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(i)
            self.losses.append(total / len(y))
        return self

    def score(self, seq: NDArray[np.float32], static: NDArray[np.float32]) -> NDArray[np.float32]:
        import torch

        self.net.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(seq), 8192):
                logits = self.net(torch.from_numpy(seq[b : b + 8192]),
                                  torch.from_numpy(static[b : b + 8192]))  # fmt: skip
                out.append(torch.sigmoid(logits).numpy())
        return np.concatenate(out).astype(np.float32)


def run_quarter(con: Any, test: str, gold: str, silver: str, scores: str, *,
                threads: int = 4, epochs: int = 8) -> dict[str, Any]:  # fmt: skip
    """Fit the GRU like the backtest models and compare all models on weekly checks."""
    begin = time.monotonic()
    start, end = quarter_bounds(test)
    quarters = train_quarters(test)
    train_files = ", ".join(f"'{gold}/train_{q}.parquet'" for q in quarters)
    share = round(NEGATIVE_SHARE * 1000)
    train = con.execute(
        f"SELECT serial_number, date, manufacturer, label, weight / {NEGATIVE_SHARE} AS weight "
        f"FROM read_parquet([{train_files}]) WHERE label_complete "
        f"AND date < DATE '{start}' - INTERVAL {HORIZON_DAYS} DAYS "
        f"AND (label OR hash(serial_number || 'gru') % 1000 < {share})"
    ).df()
    # Balanced classes: only the ranking matters for the metrics, and the sampled negatives
    # population weights would leave the positives almost no say in the loss.
    positive = train["label"].to_numpy(bool)
    train["weight"] = np.where(positive, 0.5 / positive.mean(), 0.5 / (1 - positive.mean()))
    drives = f"{silver}/drives.parquet"
    capacity = con.execute(
        f"SELECT serial_number, capacity_bytes FROM read_parquet('{drives}')"
    ).df()
    train = train.merge(capacity, on="serial_number", how="left")
    silver_train = [
        f"{silver}/silver_{q}.parquet" for q in [previous_quarter(quarters[0]), *quarters]
    ]
    silver_train = [f for f in silver_train if Path(f).exists()]
    t = time.monotonic()
    seq = windows(con, train, silver_train)
    build_seconds = time.monotonic() - t
    model = GRUModel(epochs=epochs, threads=threads)
    t = time.monotonic()
    model.fit(seq, statics(train), train["label"].to_numpy(bool), train["weight"].to_numpy())
    fit_seconds = time.monotonic() - t
    del seq
    # Weekly checks of every drive in the test quarter, with the backtest's scores joined.
    silver_test = [f"{silver}/silver_{q}.parquet" for q in (previous_quarter(test), test)]
    silver_test = [f for f in silver_test if Path(f).exists()]
    listed = ", ".join(f"'{f}'" for f in silver_test)
    rows = con.execute(
        f"WITH s AS (SELECT * FROM read_parquet([{listed}]) WHERE date >= DATE '{start}' "
        f"AND date < DATE '{end}' AND datediff('day', DATE '{start}', date) % {CHECK_EVERY} = 0) "
        f"SELECT s.serial_number, s.date, s.manufacturer, s.capacity_bytes, b.* "
        f"FROM s JOIN read_parquet('{scores}') b "
        f"ON b.drive = (hash(s.serial_number) >> 1)::BIGINT "
        f"AND b.day = datediff('day', DATE '1970-01-01', s.date)"
    ).df()
    scored: dict[str, NDArray[np.float32]] = {
        c.removeprefix("score_"): rows[c].to_numpy(np.float32)
        for c in rows.columns if c.startswith("score_")
    }  # fmt: skip
    gru_scores = []
    t = time.monotonic()
    for _, part in rows.groupby("date", sort=True):
        x = windows(con, part, silver_test)
        gru_scores.append(pd.Series(model.score(x, statics(part)), index=part.index))
    scored["gru"] = pd.concat(gru_scores).reindex(rows.index).to_numpy(np.float32)
    score_seconds = time.monotonic() - t
    test_rows = evaluate.TestRows(
        rows["drive"].to_numpy(np.int64), rows["day"].to_numpy(np.int32),
        rows["label"].to_numpy(bool), rows["days_to_failure"].to_numpy(np.float64),
        rows["fails_in_quarter"].to_numpy(bool),
    )  # fmt: skip
    return {
        "quarter": test,
        "train_quarters": quarters,
        "train_sequences": len(train),
        "train_positive": int(train["label"].sum()),
        "check_days": int(rows["date"].nunique()),
        "test_rows": len(rows),
        "epochs": epochs,
        "losses": [round(v, 5) for v in model.losses],
        "seconds": {
            "windows": round(build_seconds, 1),
            "fit": round(fit_seconds, 1),
            "score": round(score_seconds, 1),
            "total": round(time.monotonic() - begin, 1),
        },
        "models": {name: evaluate.evaluate(test_rows, s) for name, s in scored.items()},
    }
