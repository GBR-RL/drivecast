"""DuckDB connections with the lake registered as views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

RELEASE_PREFIX = "release:"


def connect(
    *, threads: int | None = None, memory_limit: str | None = None, temp: Path | None = None
) -> Any:
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    if threads:
        con.execute(f"SET threads = {threads}")
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    if temp:
        temp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{temp.as_posix()}'")
    return con


def bronze_files(lake: str) -> list[str]:
    """Parquet files of the bronze lake: a local directory, or "release:<tag>" on GitHub."""
    if lake.startswith(RELEASE_PREFIX):
        from drivecast.lake.remote import asset_urls

        return asset_urls(lake.removeprefix(RELEASE_PREFIX))
    files = sorted(p.as_posix() for p in Path(lake).glob("drive_stats_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no drive_stats_*.parquet in {lake}")
    return files


def register_bronze(con: Any, lake: str, months: list[str] | None = None) -> list[str]:
    """Register the view "bronze", optionally over some months ("YYYY-MM") only."""
    files = bronze_files(lake)
    if months is not None:
        files = [f for f in files if f.rsplit("_", 1)[-1].removesuffix(".parquet") in months]
    if any(f.startswith("http") for f in files):
        con.execute("INSTALL httpfs; LOAD httpfs")
    listed = ", ".join(f"'{f}'" for f in files)
    con.execute(f"CREATE OR REPLACE VIEW bronze AS SELECT * FROM read_parquet([{listed}])")
    return files
