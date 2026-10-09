"""Bronze layer: one Parquet file per month, in one schema, from the published zip files.

A zip holds one CSV per day. The days of each month are extracted together, read by DuckDB as
text (the headers differ between eras), cast to the schema and written as zstd Parquet. Nothing
is filtered or corrected here; every value that does not cast is counted, so data problems show
up in the manifest instead of disappearing.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import httpx

from drivecast.lake.schema import COLUMNS, canonical, cast
from drivecast.lake.sources import Source

DAY_FILE = re.compile(r"(\d{4})-(\d{2})-(\d{2})\.csv$")
FILE_DATE = r"CAST(regexp_extract(filename, '(\d{4}-\d{2}-\d{2})\.csv$', 1) AS DATE)"
CHUNK = 1 << 20


@dataclass
class MonthStats:
    month: str
    file: str
    days: int
    rows: int
    drives: int
    failures: int
    bytes: int
    sha256: str
    cast_failures: dict[str, int] = field(default_factory=dict)
    unknown_columns: list[str] = field(default_factory=list)
    # Rows whose date column is not the date in the file's name, and rows with numbers in
    # scientific notation: both signs that a file was saved through a spreadsheet.
    date_disagrees: int = 0
    spreadsheet_rows: int = 0
    seconds: float = 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def download(source: Source, dest: Path, retries: int = 3) -> tuple[Path, str]:
    """Stream the zip to disk and return it with its SHA-256."""
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / source.url.rsplit("/", 1)[1]
    for attempt in range(1, retries + 1):
        digest = hashlib.sha256()
        try:
            with (
                httpx.stream("GET", source.url, follow_redirects=True, timeout=120) as r,
                path.open("wb") as f,
            ):
                r.raise_for_status()
                expected = int(r.headers.get("content-length", -1))
                for block in r.iter_bytes(CHUNK):
                    f.write(block)
                    digest.update(block)
            if expected not in (-1, path.stat().st_size):
                raise OSError(f"{path.name}: got {path.stat().st_size} of {expected} bytes")
            return path, digest.hexdigest()
        except (httpx.HTTPError, OSError):
            if attempt == retries:
                raise
            time.sleep(10 * attempt)
    raise AssertionError("unreachable")


def day_files(archive: zipfile.ZipFile) -> dict[str, list[zipfile.ZipInfo]]:
    """The daily CSVs of an archive grouped by month, skipping macOS resource files."""
    months: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    for info in archive.infolist():
        if "__MACOSX" in info.filename or info.is_dir():
            continue
        match = DAY_FILE.search(info.filename)
        if match:
            months[f"{match.group(1)}-{match.group(2)}"].append(info)
    return {m: sorted(v, key=lambda i: i.filename) for m, v in sorted(months.items())}


def _connect(work: Path, threads: int | None, memory_limit: str | None) -> Any:
    con = duckdb.connect()
    (work / "duckdb_tmp").mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{(work / 'duckdb_tmp').as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    if threads:
        con.execute(f"SET threads = {threads}")
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    return con


def convert_month(
    csv_files: list[Path],
    out: Path,
    work: Path,
    *,
    threads: int | None = None,
    memory_limit: str | None = None,
) -> MonthStats:
    """Cast one month of daily CSVs to the schema and write it as Parquet."""
    start = time.monotonic()
    con = _connect(work, threads, memory_limit)
    files = ", ".join(f"'{p.as_posix()}'" for p in csv_files)
    con.execute(
        f"CREATE VIEW raw AS SELECT * FROM read_csv([{files}], header = true, "
        "all_varchar = true, union_by_name = true, filename = true)"
    )
    present = [row[0] for row in con.execute("DESCRIBE raw").fetchall() if row[0] != "filename"]
    sources: dict[str, list[str]] = defaultdict(list)
    unknown = []
    for name in present:
        column = canonical(name)
        if column is None:
            unknown.append(name)
        else:
            sources[column].append(name)

    def expression(column: str, sql_type: str) -> str:
        if column == "date":  # one file per day, named by its date
            return f'{FILE_DATE} AS "date"'
        names = sources.get(column)
        if not names:
            return f'CAST(NULL AS {sql_type}) AS "{column}"'
        casts = [cast(n, sql_type) for n in names]
        value = casts[0] if len(casts) == 1 else f"COALESCE({', '.join(casts)})"
        return f'{value} AS "{column}"'

    select = ",\n".join(expression(c, t) for c, t in COLUMNS.items())
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY (SELECT {select} FROM raw) TO '{out.as_posix()}' "
        "(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 500000)"
    )
    # Values that were present as text but did not cast.
    checks = [
        f'count("{name}") - count({cast(name, COLUMNS[column])}) AS "{name}"'
        for column, names in sources.items()
        for name in names
    ]
    date_column = sources.get("date", ["date"])[0]
    written = (
        f'COALESCE(TRY_CAST("{date_column}" AS DATE), '
        f"TRY_STRPTIME(\"{date_column}\", '%m/%d/%y')::DATE)"
    )
    sheet = " OR ".join(f"\"{n}\" ILIKE '%E+%'" for names in sources.values() for n in names)
    checks += [
        f"count(*) FILTER (WHERE {written} IS DISTINCT FROM {FILE_DATE})",
        f"count(*) FILTER (WHERE {sheet})",
    ]
    row = con.execute(f"SELECT {', '.join(checks)} FROM raw").fetchone()
    assert row is not None
    *counts, date_disagrees, spreadsheet_rows = row
    labels = [name for names in sources.values() for name in names]
    failures = {
        name: int(n) for name, n in zip(labels, counts, strict=True) if n and name != date_column
    }
    rows, days, drives, failed = con.execute(
        f"SELECT count(*), count(DISTINCT date), count(DISTINCT serial_number), "
        f"coalesce(sum(failure), 0) FROM read_parquet('{out.as_posix()}')"
    ).fetchone() or (0, 0, 0, 0)
    con.close()
    return MonthStats(
        month=out.stem.rsplit("_", 1)[-1],
        file=out.name,
        days=int(days),
        rows=int(rows),
        drives=int(drives),
        failures=int(failed),
        bytes=out.stat().st_size,
        sha256=sha256(out),
        cast_failures=failures,
        unknown_columns=sorted(unknown),
        date_disagrees=int(date_disagrees),
        spreadsheet_rows=int(spreadsheet_rows),
        seconds=round(time.monotonic() - start, 1),
    )


def ingest(
    source: Source,
    out_dir: Path,
    work: Path,
    *,
    zip_path: Path | None = None,
    threads: int | None = None,
    memory_limit: str | None = None,
) -> dict[str, Any]:
    """Download (unless given) one published zip and write its months to ``out_dir``."""
    start = time.monotonic()
    if zip_path is None:
        zip_path, digest = download(source, work)
    else:
        digest = sha256(zip_path)
    entry: dict[str, Any] = {
        "source": source.name,
        "url": source.url,
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": digest,
        "download_seconds": round(time.monotonic() - start, 1),
        "months": [],
    }
    with zipfile.ZipFile(zip_path) as archive:
        for month, infos in day_files(archive).items():
            month_dir = work / f"csv_{month}"
            shutil.rmtree(month_dir, ignore_errors=True)
            month_dir.mkdir(parents=True)
            paths = []
            for info in infos:
                target = month_dir / Path(info.filename).name
                with archive.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                paths.append(target)
            stats = convert_month(
                paths,
                out_dir / f"drive_stats_{month}.parquet",
                work,
                threads=threads,
                memory_limit=memory_limit,
            )
            shutil.rmtree(month_dir)
            entry["months"].append(asdict(stats))
    entry["seconds"] = round(time.monotonic() - start, 1)
    return entry


def merge_manifests(paths: list[Path]) -> dict[str, Any]:
    """Join per-source manifest entries into one manifest, ordered by month."""
    entries = [json.loads(p.read_text()) for p in paths]
    entries.sort(key=lambda e: e["months"][0]["month"])
    months = [m for e in entries for m in e["months"]]
    return {
        "sources": len(entries),
        "months": len(months),
        "rows": sum(m["rows"] for m in months),
        "failures": sum(m["failures"] for m in months),
        "parquet_bytes": sum(m["bytes"] for m in months),
        "zip_bytes": sum(e["zip_bytes"] for e in entries),
        "entries": entries,
    }
