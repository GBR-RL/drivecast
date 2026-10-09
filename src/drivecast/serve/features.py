"""Features for drives sent to the service, computed by the same SQL as the batch pipeline.

The request carries each drive's recent daily SMART readings. They are put in the shape of the
silver layer and run through ``features.build.window_sql``, the query that built the training
data, so a feature can never mean one thing in training and another in serving.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from drivecast.features.build import feature_names, window_sql
from drivecast.lake.silver import MANUFACTURER, SMART_NORMALIZED, SMART_RAW

READING_COLUMNS = frozenset(
    [f"smart_{i}_raw" for i in SMART_RAW] + [f"smart_{i}_normalized" for i in SMART_NORMALIZED]
)


def last_day_features(con: Any, drives: list[dict[str, Any]]) -> pd.DataFrame:
    """One feature row per drive, for the last day of its readings.

    Each drive is a dict with ``serial_number``, ``model``, ``capacity_bytes``, optionally
    ``first_seen`` (else its first reading), and ``readings``: dicts with ``date`` and SMART
    values named as in the data (``smart_5_raw``, ``smart_194_normalized``, ...).
    """
    rows = []
    spans = []
    for d in drives:
        days = [r["date"] for r in d["readings"]]
        spans.append({"serial_number": d["serial_number"],
                      "first_date": d.get("first_seen") or min(days),
                      "last_date": max(days), "failure_date": None})  # fmt: skip
        for r in d["readings"]:
            rows.append({"serial_number": d["serial_number"], "model": d["model"],
                         "capacity_bytes": d.get("capacity_bytes"), **r})  # fmt: skip
    history = pd.DataFrame(rows)
    missing = sorted(READING_COLUMNS - set(history.columns))
    history = history.reindex(columns=[*history.columns, *missing]).astype(
        dict.fromkeys(missing, "Float64")
    )
    history["date"] = pd.to_datetime(history["date"]).dt.date
    span_frame = pd.DataFrame(spans)
    span_frame["failure_date"] = pd.Series([None] * len(span_frame), dtype="object")
    con.register("history_frame", history)
    con.register("span_frame", span_frame)
    raw = ", ".join(f"CAST({c} AS BIGINT) AS {c}" for c in sorted(READING_COLUMNS))
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW silver AS SELECT date, serial_number, model, "
        f"{MANUFACTURER} AS manufacturer, CAST(capacity_bytes AS BIGINT) AS capacity_bytes, "
        f"0::TINYINT AS failure, NULL::VARCHAR AS datacenter, {raw} FROM history_frame"
    )
    con.execute(
        "CREATE OR REPLACE TEMP VIEW drives AS SELECT serial_number, "
        "CAST(first_date AS DATE) AS first_date, CAST(last_date AS DATE) AS last_date, "
        "CAST(failure_date AS DATE) AS failure_date FROM span_frame"
    )
    start = min(s["last_date"] for s in spans)
    end = max(s["last_date"] for s in spans) + timedelta(days=1)
    # Each drive's last day. (A QUALIFY on top of the window query trips a DuckDB internal
    # error, so the last day is joined instead.)
    frame: pd.DataFrame = con.execute(
        f"SELECT w.* FROM ({window_sql(start.isoformat(), end.isoformat(), str(date.max))}) w "
        f"JOIN drives d ON w.serial_number = d.serial_number AND w.date = d.last_date"
    ).df()
    con.unregister("history_frame")
    con.unregister("span_frame")
    keep = ["serial_number", "date", "model", "manufacturer", *feature_names()]
    return frame[keep].reset_index(drop=True)
