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
    result = report.build(con, files, RAW / "quality_work")
    for path in (out, json_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2) + "\n")
    out.write_text(report.markdown(result), encoding="utf-8")
    typer.echo(f"wrote {out} and {json_out}")


@app.command()
def drives(
    lake: Annotated[str, typer.Option(help='Bronze directory, or "release:<tag>".')] = str(
        LAKE / "bronze"
    ),
    out: Path = LAKE / "silver" / "drives.parquet",
    threads: int | None = None,
    memory_limit: str | None = None,
) -> None:
    """Build the drive table (one row per serial number) from all of bronze."""
    from drivecast.lake.silver import build_drives
    from drivecast.lake.views import connect, register_bronze

    con = connect(threads=threads, memory_limit=memory_limit, temp=RAW / "duckdb_tmp")
    files = register_bronze(con, lake)
    if any(f.startswith("http") for f in files):
        con.execute("INSTALL httpfs; LOAD httpfs")
    typer.echo(f"{build_drives(con, files, out):,} drives -> {out}")


@app.command()
def quarters() -> None:
    """Print the quarters of the lake as JSON (the silver matrix)."""
    from drivecast.lake.silver import quarters as all_quarters

    typer.echo(json.dumps(all_quarters()))


@app.command()
def silver(
    quarter: str,
    lake: Annotated[str, typer.Option(help='Bronze directory, or "release:<tag>".')] = str(
        LAKE / "bronze"
    ),
    drives_path: Annotated[str, typer.Option("--drives")] = str(LAKE / "silver" / "drives.parquet"),
    out_dir: Annotated[Path, typer.Option("--out")] = LAKE / "silver",
    threads: int | None = None,
    memory_limit: str | None = None,
) -> None:
    """Write one quarter of the silver layer."""
    from drivecast.lake.silver import build_quarter
    from drivecast.lake.views import connect, register_bronze
    from drivecast.quality.published import quarter_months

    con = connect(threads=threads, memory_limit=memory_limit, temp=RAW / "duckdb_tmp")
    if drives_path.startswith("http"):
        con.execute("INSTALL httpfs; LOAD httpfs")
    register_bronze(con, lake, months=quarter_months(quarter))
    stats = build_quarter(con, quarter, drives_path, out_dir / f"silver_{quarter}.parquet")
    (out_dir / f"silver_{quarter}.json").write_text(json.dumps(stats, indent=2) + "\n")
    typer.echo(
        f"{quarter}: {stats['rows']:,} rows, {stats['drives']:,} drives, "
        f"{stats['failures']:,} failures, {stats['bytes'] / 2**20:.0f} MiB"
    )


@app.command()
def survival(
    drives_path: Annotated[str, typer.Option("--drives", help="Drive table (path or URL).")] = str(
        LAKE / "silver" / "drives.parquet"
    ),
    out: Path = Path("docs/results/survival.json"),
) -> None:
    """AFR with exact intervals, hazard by age, Kaplan-Meier and Cox models of drive lifetime."""
    import pandas as pd

    from drivecast.analysis.survival import analyse

    result = analyse(pd.read_parquet(drives_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=_plain) + "\n")
    typer.echo(f"{result['drives']:,} drives, {result['failures']:,} failures -> {out}")


def _plain(value: object) -> object:
    """JSON for numpy scalars and dates."""
    item = getattr(value, "item", None)
    return item() if callable(item) else str(value)


@app.command()
def features(
    quarter: str,
    silver_dir: Annotated[
        str, typer.Option("--silver", help="Silver directory or URL prefix.")
    ] = str(LAKE / "silver"),
    drives_path: Annotated[str, typer.Option("--drives")] = str(LAKE / "silver" / "drives.parquet"),
    out_dir: Annotated[Path, typer.Option("--out")] = LAKE / "gold",
    negative_rate: float = 0.01,
    threads: int | None = None,
    memory_limit: str | None = None,
) -> None:
    """Write one quarter of features and its training sample (gold layer)."""
    from drivecast.features.build import build_quarter, previous_quarter, sample_training_rows
    from drivecast.lake.views import connect

    con = connect(threads=threads, memory_limit=memory_limit, temp=RAW / "duckdb_tmp")
    if silver_dir.startswith("http") or drives_path.startswith("http"):
        con.execute("INSTALL httpfs; LOAD httpfs")
    names = [f"silver_{q}.parquet" for q in (previous_quarter(quarter), quarter)]
    files = [f"{silver_dir.rstrip('/')}/{n}" for n in names]
    if silver_dir.startswith("http"):
        import httpx

        files = [f for f in files if httpx.head(f, follow_redirects=True, timeout=60).is_success]
    else:
        files = [f for f in files if Path(f).exists()]
    stats = build_quarter(con, quarter, files, drives_path, out_dir / f"features_{quarter}.parquet")
    stats["training"] = sample_training_rows(
        con,
        out_dir / f"features_{quarter}.parquet",
        out_dir / f"train_{quarter}.parquet",
        negative_rate=negative_rate,
    )
    (out_dir / f"features_{quarter}.json").write_text(json.dumps(stats, indent=2) + "\n")
    typer.echo(
        f"{quarter}: {stats['rows']:,} rows, {stats['positive_rows']:,} positive, "
        f"{stats['training']['rows']:,} training rows, {stats['bytes'] / 2**20:.0f} MiB"
    )


@app.command("features-spark")
def features_spark(
    quarter: str,
    silver_dir: Annotated[str, typer.Option("--silver")] = str(LAKE / "silver"),
    drives_path: Annotated[str, typer.Option("--drives")] = str(LAKE / "silver" / "drives.parquet"),
    out_dir: Annotated[Path, typer.Option("--out")] = LAKE / "gold_spark",
    threads: int = 4,
    memory: str = "10g",
) -> None:
    """The same feature job on Apache Spark (local mode), for the engine comparison."""
    from drivecast.features import spark
    from drivecast.features.build import previous_quarter

    files = [
        f"{silver_dir}/silver_{q}.parquet"
        for q in (previous_quarter(quarter), quarter)
        if Path(f"{silver_dir}/silver_{q}.parquet").exists()
    ]
    session = spark.session(threads=threads, memory=memory)
    stats = spark.build_quarter(
        session, quarter, files, drives_path, out_dir / f"features_{quarter}"
    )
    session.stop()
    (out_dir / f"features_{quarter}.json").write_text(json.dumps(stats, indent=2) + "\n")
    typer.echo(f"{quarter} on Spark: {stats['rows']:,} rows in {stats['seconds']} s")


@app.command("features-compare")
def features_compare(
    left: str,
    right: str,
    out: Annotated[Path | None, typer.Option(help="Write the comparison as JSON.")] = None,
) -> None:
    """Compare two feature outputs value by value (paths or globs of Parquet files)."""
    import duckdb

    from drivecast.features.compare import compare

    result = compare(duckdb.connect(), left, right)
    if out:
        out.write_text(json.dumps(result, indent=2) + "\n")
    typer.echo(json.dumps(result, indent=2))


@app.command()
def backtest(
    quarter: str,
    gold: Annotated[str, typer.Option(help="Directory or URL prefix of the gold files.")] = str(
        LAKE / "gold"
    ),
    out_dir: Annotated[Path, typer.Option("--out")] = Path("runs/backtest"),
    scores: Annotated[bool, typer.Option(help="Also write every score as Parquet.")] = True,
    threads: int = 4,
) -> None:
    """Fit on the four quarters before QUARTER and evaluate every model on QUARTER."""
    import duckdb

    from drivecast.models.backtest import run_quarter

    con = duckdb.connect()
    if gold.startswith("http"):
        con.execute("INSTALL httpfs; LOAD httpfs")
    result = run_quarter(
        con,
        quarter,
        gold.rstrip("/"),
        threads=threads,
        scores_out=out_dir / f"scores_{quarter}.parquet" if scores else None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"backtest_{quarter}.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, m in result["models"].items():
        d = m["daily"]["25"]
        typer.echo(
            f"{quarter} {name:9s} AP {m['average_precision']:.3f}  "
            f"top-25/day precision {d['precision']:.2f} recall {d['recall']:.2f}"
        )


@app.command("backtest-summary")
def backtest_summary(
    paths: list[Path],
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("docs/results/backtest.json"),
) -> None:
    """Join the per-quarter backtest results: means, medians and paired wins per model."""
    from drivecast.models.backtest import summarise

    summary = summarise([json.loads(p.read_text()) for p in paths])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    for name, m in summary["models"].items():
        typer.echo(f"{name:9s} mean AP {m['mean']['average_precision']:.3f}")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
