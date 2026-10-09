from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from drivecast.lake.schema import COLUMNS

Row = dict[str, Any]


def write_bronze(path: Path, rows: list[Row]) -> Path:
    """A bronze Parquet file in the full schema from a few rows (missing columns are NULL)."""
    frame = pd.DataFrame(rows)
    con = duckdb.connect()
    con.register("frame", frame)
    select = ", ".join(
        f'CAST(frame."{c}" AS {t}) AS "{c}"' if c in frame else f'CAST(NULL AS {t}) AS "{c}"'
        for c, t in COLUMNS.items()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT {select} FROM frame) TO '{path.as_posix()}' (FORMAT parquet)")
    return path


@pytest.fixture
def bronze(tmp_path: Path) -> Callable[[list[Row]], Any]:
    """A DuckDB connection with the view "bronze" over the given rows, and the file behind it."""

    def make(rows: list[Row]) -> Any:
        path = write_bronze(tmp_path / "bronze" / "drive_stats_test.parquet", rows)
        con = duckdb.connect()
        con.execute(f"CREATE VIEW bronze AS SELECT * FROM read_parquet('{path.as_posix()}')")
        return con, [path.as_posix()]

    return make
