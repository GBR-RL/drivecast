"""Command line entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from drivecast import __version__
from drivecast.config import LAKE, RAW

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Predicting hard-drive failures in the Backblaze fleet."""


@app.command()
def sources() -> None:
    """Print the names of the published files as JSON (the ingest matrix)."""
    from drivecast.lake.sources import all_sources

    typer.echo(json.dumps([s.name for s in all_sources()]))


@app.command()
def ingest(
    name: str,
    out: Annotated[Path, typer.Option(help="Directory for the monthly Parquet files.")] = LAKE
    / "bronze",
    work: Annotated[Path, typer.Option(help="Scratch space for the zip and CSVs.")] = RAW,
    zip_path: Annotated[Path | None, typer.Option("--zip", help="Use a local zip.")] = None,
    manifest: Annotated[Path | None, typer.Option(help="Write the manifest entry here.")] = None,
    threads: int | None = None,
    memory_limit: str | None = None,
) -> None:
    """Download one published file and write its months as Parquet (bronze layer)."""
    from drivecast.lake.ingest import ingest as run
    from drivecast.lake.sources import by_name

    entry = run(
        by_name(name), out, work, zip_path=zip_path, threads=threads, memory_limit=memory_limit
    )
    manifest = manifest or out / f"manifest_{name}.json"
    manifest.write_text(json.dumps(entry, indent=2) + "\n")
    for m in entry["months"]:
        bad = sum(m["cast_failures"].values())
        typer.echo(
            f"{m['month']}: {m['rows']:,} rows, {m['drives']:,} drives, {m['failures']} failures, "
            f"{m['bytes'] / 2**20:.0f} MiB, {bad} cast failures, {m['seconds']} s"
        )
    typer.echo(f"{name}: {entry['seconds']} s, manifest {manifest}")


@app.command("manifest-merge")
def manifest_merge(
    paths: list[Path],
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("docs/lake/manifest.json"),
) -> None:
    """Join the per-source manifest entries into one manifest."""
    from drivecast.lake.ingest import merge_manifests

    merged = merge_manifests(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2) + "\n")
    typer.echo(
        f"{merged['months']} months, {merged['rows']:,} rows, {merged['failures']:,} failures, "
        f"{merged['parquet_bytes'] / 2**30:.1f} GiB Parquet -> {out}"
    )


@app.command()
def quality(
    lake: Annotated[str, typer.Option(help='Bronze directory, or "release:<tag>".')] = str(
        LAKE / "bronze"
    ),
    out: Annotated[Path, typer.Option(help="Markdown report.")] = Path("docs/quality.md"),
    json_out: Annotated[Path, typer.Option("--json")] = Path("docs/results/quality.json"),
    threads: int | None = None,
    memory_limit: str | None = None,
) -> None:
    """Run the data-quality checks and the reconciliation over the bronze lake."""
    from drivecast.lake.views import connect, register_bronze
    from drivecast.quality import report

    con = connect(threads=threads, memory_limit=memory_limit, temp=RAW / "duckdb_tmp")
    files = register_bronze(con, lake)
    typer.echo(f"{len(files)} monthly files")
    result = report.build(con)
    for path in (out, json_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2) + "\n")
    out.write_text(report.markdown(result), encoding="utf-8")
    typer.echo(f"wrote {out} and {json_out}")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
