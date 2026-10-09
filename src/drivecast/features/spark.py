"""The feature job of ``build.py`` written for Apache Spark, to compare the two engines.

Same inputs (silver quarters and the drive table), same definitions, same output columns. The
comparison in ``compare.py`` joins both outputs on drive and day and counts every value that
differs; the timing is taken on the same runner.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from drivecast.features.build import (
    COUNTERS,
    HORIZON_DAYS,
    LEVELS,
    LOOKBACK_DAYS,
    MANUFACTURERS,
    NORMALIZED,
)
from drivecast.quality.published import quarter_bounds


def session(threads: int = 4, memory: str = "10g") -> Any:
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master(f"local[{threads}]")
        .appName("drivecast-features")
        .config("spark.driver.memory", memory)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(threads * 8))
        .config("spark.sql.parquet.compression.codec", "zstd")
        .getOrCreate()
    )


def build_quarter(
    spark: Any,
    quarter: str,
    silver_files: list[str],
    drives: str,
    out: Path,
    *,
    lake_end: str | None = None,
) -> dict[str, Any]:
    from pyspark.sql import Window
    from pyspark.sql import functions as F  # noqa: N812

    start, end = quarter_bounds(quarter)
    begin = time.monotonic()
    day = F.datediff(F.col("date"), F.lit("1970-01-01"))
    s = (
        spark.read.parquet(*silver_files)
        .withColumn("day", day)
        .where(
            (F.col("date") >= F.date_sub(F.lit(start).cast("date"), LOOKBACK_DAYS))
            & (F.col("date") < F.lit(end).cast("date"))
        )
    )
    d = spark.read.parquet(drives).select(
        "serial_number", "first_date", "failure_date", "last_date"
    )
    if lake_end is None:
        lake_end = str(d.agg(F.max("last_date")).first()[0])
    by_drive = Window.partitionBy("serial_number").orderBy("day")
    w7 = by_drive.rangeBetween(-7, 0)
    w30 = by_drive.rangeBetween(-LOOKBACK_DAYS, 0)
    cols = [
        F.col("date"),
        F.col("serial_number"),
        F.col("model"),
        F.col("manufacturer"),
        (F.col("smart_9_raw") / 24.0).alias("age_days"),
        F.datediff(F.col("date"), F.col("first_date")).alias("days_seen"),
        F.round(F.col("capacity_bytes") / 1e12, 1).alias("capacity_tb"),
        F.max("smart_194_raw").over(w30).alias("temp_max_30d"),
    ]
    cols += [
        (F.col("manufacturer") == m).cast("int").alias(f"maker_{m.lower()}") for m in MANUFACTURERS
    ]
    for i in COUNTERS:
        v = F.col(f"smart_{i}_raw")
        cols += [
            v.alias(f"smart_{i}"),
            (v - F.first(v, ignorenulls=True).over(w7)).alias(f"smart_{i}_d7"),
            (v - F.first(v, ignorenulls=True).over(w30)).alias(f"smart_{i}_d30"),
        ]
    cols += [F.col(f"smart_{i}_raw").alias(f"smart_{i}") for i in LEVELS if i not in COUNTERS]
    cols += [F.col(f"smart_{i}_normalized").alias(f"smart_{i}_n") for i in NORMALIZED]
    cols.append(F.datediff(F.col("failure_date"), F.col("date")).alias("days_to_failure"))
    features = s.join(d, "serial_number").select(*cols)
    growing = sum((F.coalesce(F.col(f"smart_{i}_d30"), F.lit(0)) > 0).cast("int") for i in COUNTERS)
    features = (
        features.withColumn("counters_growing_30d", growing)
        .withColumn(
            "label",
            F.coalesce(F.col("days_to_failure").between(0, HORIZON_DAYS - 1), F.lit(False)),
        )
        .withColumn(
            "label_complete",
            (F.col("date") <= F.date_sub(F.lit(lake_end).cast("date"), HORIZON_DAYS))
            | F.col("days_to_failure").isNotNull(),
        )
        .where(
            (F.col("date") >= F.lit(start).cast("date")) & (F.col("date") < F.lit(end).cast("date"))
        )
    )
    features.write.mode("overwrite").parquet(out.as_posix())
    seconds = time.monotonic() - begin
    written = spark.read.parquet(out.as_posix())
    rows = written.count()
    positives = written.where(F.col("label")).count()
    return {
        "quarter": quarter,
        "engine": "spark",
        "rows": rows,
        "positive_rows": positives,
        "seconds": round(seconds, 1),
    }
