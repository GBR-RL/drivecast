"""Compare two feature outputs value by value, joined on drive and day."""

from __future__ import annotations

from typing import Any

from drivecast.features.build import feature_names

EXTRA = ("days_to_failure", "label", "label_complete")


def compare(con: Any, left: str, right: str, tolerance: float = 1e-9) -> dict[str, Any]:
    """Rows only on one side, and per column the joined rows whose values differ."""
    con.execute(f"CREATE OR REPLACE VIEW l AS SELECT * FROM read_parquet('{left}')")
    con.execute(f"CREATE OR REPLACE VIEW r AS SELECT * FROM read_parquet('{right}')")
    (left_rows,) = con.execute("SELECT count(*) FROM l").fetchone()
    (right_rows,) = con.execute("SELECT count(*) FROM r").fetchone()
    (only_left,) = con.execute(
        "SELECT count(*) FROM l ANTI JOIN r USING (serial_number, date)"
    ).fetchone()
    (only_right,) = con.execute(
        "SELECT count(*) FROM r ANTI JOIN l USING (serial_number, date)"
    ).fetchone()
    columns = [*feature_names(), *EXTRA]
    terms = []
    for c in columns:
        a, b = f"l.{c}", f"r.{c}"
        if c in ("label", "label_complete"):
            differs = f"{a} IS DISTINCT FROM {b}"
        else:
            differs = (
                f"({a} IS NULL) <> ({b} IS NULL) OR "
                f"abs({a}::DOUBLE - {b}::DOUBLE) > {tolerance} * greatest(1, abs({a}::DOUBLE))"
            )
        terms.append(f"count(*) FILTER (WHERE {differs}) AS {c}")
    row = con.execute(
        f"SELECT count(*), {', '.join(terms)} FROM l JOIN r USING (serial_number, date)"
    ).fetchone()
    assert row is not None
    joined, *diffs = row
    return {
        "left_rows": int(left_rows),
        "right_rows": int(right_rows),
        "only_left": int(only_left),
        "only_right": int(only_right),
        "joined": int(joined),
        "columns": len(columns),
        "differing_values": {c: int(n) for c, n in zip(columns, diffs, strict=True) if n},
    }
