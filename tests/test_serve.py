from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

from drivecast.features import build
from drivecast.lake.silver import SMART_NORMALIZED, SMART_RAW
from drivecast.serve import model as served

pytest.importorskip("fastapi")


def history(serial: str, model: str, n: int, grow_from: int) -> list[dict[str, Any]]:
    start = date(2020, 3, 1)
    return [
        {"date": start + timedelta(days=i), "smart_5_raw": max(0, i - grow_from),
         "smart_9_raw": 24 * (500 + i), "smart_194_raw": 30 + i % 3,
         "smart_197_raw": None if i % 4 else i, "smart_187_normalized": 100 - (i > 30)}
        for i in range(n) if i != 33  # one day without a report
    ]  # fmt: skip


DRIVES = [
    {"serial_number": "A", "model": "ST4000DM000", "capacity_bytes": 4_000_787_030_016,
     "first_seen": date(2020, 3, 1), "readings": history("A", "ST4000DM000", 45, 20)},
    {"serial_number": "B", "model": "HGST HMS5C4040BLE640", "capacity_bytes": 4_000_787_030_016,
     "first_seen": date(2020, 3, 1), "readings": history("B", "HGST", 45, 99)},
]  # fmt: skip


def test_service_features_equal_the_batch_features(tmp_path: Path) -> None:
    rows = []
    for d in DRIVES:
        maker = "Seagate" if d["model"].startswith("ST") else "HGST"
        for r in d["readings"]:
            rows.append({"serial_number": d["serial_number"], "model": d["model"],
                         "manufacturer": maker, "capacity_bytes": d["capacity_bytes"],
                         "failure": 0, "datacenter": None, **r})  # fmt: skip
    silver = pd.DataFrame(rows)
    for c in [f"smart_{i}_raw" for i in SMART_RAW] + [
        f"smart_{i}_normalized" for i in SMART_NORMALIZED
    ]:
        if c not in silver:
            silver[c] = pd.Series([None] * len(silver), dtype="Int64")
    silver.to_parquet(tmp_path / "silver.parquet")
    pd.DataFrame({"serial_number": ["A", "B"], "first_date": [date(2020, 3, 1)] * 2,
                  "last_date": [date(2020, 4, 14)] * 2,
                  "failure_date": pd.Series([None, None], dtype="object")}).to_parquet(
        tmp_path / "drives.parquet")  # fmt: skip
    con = duckdb.connect()
    out = tmp_path / "features.parquet"
    build.build_quarter(con, "2020Q2", [(tmp_path / "silver.parquet").as_posix()],
                        (tmp_path / "drives.parquet").as_posix(), out)  # fmt: skip
    batch = pd.read_parquet(out)
    batch = batch[batch["date"] == batch["date"].max()].set_index("serial_number")
    from drivecast.serve.features import last_day_features

    service = last_day_features(duckdb.connect(), DRIVES).set_index("serial_number")
    for f in build.feature_names():
        left = batch.loc[["A", "B"], f].astype("float64").to_numpy()
        right = service.loc[["A", "B"], f].astype("float64").to_numpy()
        assert np.allclose(left, right, equal_nan=True), f


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(400, len(build.feature_names()))),
                     columns=build.feature_names())  # fmt: skip
    y = (x["smart_5"] > 1).to_numpy()
    models = tmp_path / "models"
    models.mkdir()
    lgb.LGBMClassifier(n_estimators=20, verbose=-1).fit(x, y).booster_.save_model(
        str(models / "lightgbm_2026Q2.txt")
    )
    out = tmp_path / "model"
    served.export(models, out)
    return out


def test_api_scores_drives_and_validates_input(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from drivecast.serve.app import app

    monkeypatch.setenv("DRIVECAST_MODEL_DIR", str(model_dir))
    payload = {"drives": [d | {"readings": [r | {"date": r["date"].isoformat()}
                                            for r in d["readings"]],
                               "first_seen": d["first_seen"].isoformat()}
                          for d in DRIVES]}  # fmt: skip
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["model"] == {"family": "lightgbm", "quarter": "2026Q2", "version": ""}
        r = client.post("/score", json=payload)
        assert r.status_code == 200
        results = {x["serial_number"]: x for x in r.json()["results"]}
        assert results["A"]["date"] == "2020-04-14"
        assert results["A"]["warnings"] == ["smart_5", "smart_197"]
        assert results["B"]["warnings"] == ["smart_197"]
        assert 0 <= results["A"]["score"] <= 1
        bad = {"drives": [payload["drives"][0] | {"readings": [{"date": "2020-03-01", "x": 1}]}]}
        assert client.post("/score", json=bad).status_code == 422
        assert client.post("/score/features", json={"rows": [{"smart_5": 1.0}]}).status_code == 422


def test_watchlist_ranks_every_drive_on_the_last_day(tmp_path: Path, model_dir: Path) -> None:
    from drivecast.serve import watchlist

    rows = []
    for d in DRIVES:
        maker = "Seagate" if d["model"].startswith("ST") else "HGST"
        for r in d["readings"]:
            rows.append({"serial_number": d["serial_number"], "model": d["model"],
                         "manufacturer": maker, "capacity_bytes": d["capacity_bytes"],
                         "failure": 0, "datacenter": None, **r})  # fmt: skip
    silver = pd.DataFrame(rows)
    columns = [f"smart_{i}_raw" for i in SMART_RAW]
    columns += [f"smart_{i}_normalized" for i in SMART_NORMALIZED]
    for c in columns:
        if c not in silver:
            silver[c] = pd.Series([None] * len(silver), dtype="Int64")
    silver.to_parquet(tmp_path / "silver.parquet")
    pd.DataFrame({"serial_number": ["A", "B"], "first_date": [date(2020, 3, 1)] * 2,
                  "last_date": [date(2020, 4, 14)] * 2,
                  "failure_date": pd.Series([None, None], dtype="object")}).to_parquet(
        tmp_path / "drives.parquet")  # fmt: skip
    out = tmp_path / "watch"
    summary = watchlist.build(duckdb.connect(), [(tmp_path / "silver.parquet").as_posix()],
                              (tmp_path / "drives.parquet").as_posix(), model_dir, out)  # fmt: skip
    assert summary["date"] == "2020-04-14"
    assert summary["drives"] == 2
    assert [r["rank"] for r in summary["top"]] == [1, 2]
    assert summary["top"][0]["score"] >= summary["top"][1]["score"]
    assert (out / "index.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert (out / "watchlist.csv").exists()
