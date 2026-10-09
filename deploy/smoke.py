"""Smoke and load test of a running drivecast service.

Checks /health and the input validation, then sends concurrent /score requests with synthetic
drive histories and reports latency percentiles and throughput (as JSON, and as Markdown when
a summary file is given).

    python deploy/smoke.py http://127.0.0.1:8080 --requests 300 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx


def drive(i: int, days: int = 30) -> dict[str, Any]:
    rng = random.Random(i)
    start = date(2026, 6, 1)
    grow = rng.random() < 0.1
    readings = []
    for d in range(days):
        readings.append({
            "date": (start + timedelta(days=d)).isoformat(),
            "smart_5_raw": (d // 3) if grow else 0,
            "smart_9_raw": 24 * (900 + d),
            "smart_194_raw": 28 + rng.randint(0, 6),
            "smart_197_raw": int(grow and d > 20),
            "smart_187_raw": 0,
        })  # fmt: skip
    return {"serial_number": f"SMOKE{i:05d}", "model": "ST16000NM001G",
            "capacity_bytes": 16_000_900_661_248, "readings": readings}  # fmt: skip


async def load(url: str, requests: int, concurrency: int, drives: int) -> dict[str, Any]:
    payloads = [{"drives": [drive(r * drives + k) for k in range(drives)]} for r in range(requests)]
    latencies: list[float] = []
    errors = 0
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for p in payloads:
        queue.put_nowait(p)

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal errors
        while not queue.empty():
            payload = queue.get_nowait()
            t = time.perf_counter()
            r = await client.post(f"{url}/score", json=payload)
            latencies.append(1000 * (time.perf_counter() - t))
            errors += r.status_code != 200

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as client:
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
    seconds = time.perf_counter() - start
    q = statistics.quantiles(latencies, n=100)
    return {
        "requests": requests,
        "drives_per_request": drives,
        "concurrency": concurrency,
        "errors": errors,
        "seconds": round(seconds, 2),
        "requests_per_second": round(requests / seconds, 1),
        "drives_per_second": round(requests * drives / seconds, 1),
        "p50_ms": round(q[49], 1),
        "p95_ms": round(q[94], 1),
        "p99_ms": round(q[98], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--drives", type=int, default=10)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    url = args.url.rstrip("/")
    health = httpx.get(f"{url}/health", timeout=30).json()
    assert health["status"] == "ok", health
    one = httpx.post(f"{url}/score", json={"drives": [drive(1)]}, timeout=30)
    assert one.status_code == 200, one.text
    result = one.json()["results"][0]
    assert 0 <= result["score"] <= 1, result
    assert result["date"] == "2026-06-30", result
    bad = {"drives": [drive(2) | {"readings": [{"date": "2026-06-01", "nope": 1}]}]}
    assert httpx.post(f"{url}/score", json=bad, timeout=30).status_code == 422
    measured = asyncio.run(load(url, args.requests, args.concurrency, args.drives))
    report = {"model": health["model"], "load": measured}
    print(json.dumps(report, indent=2))
    assert report["load"]["errors"] == 0, "some requests failed"
    if args.summary:
        load_ = report["load"]
        lines = [
            "### Service on kind",
            "",
            f"Model: {health['model']['family']} fitted for {health['model']['quarter']}",
            "",
            "| requests | drives each | concurrency | req/s | drives/s | p50 ms | p95 ms "
            "| p99 ms |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {load_['requests']} | {load_['drives_per_request']} | {load_['concurrency']} | "
            f"{load_['requests_per_second']} | {load_['drives_per_second']} | {load_['p50_ms']} | "
            f"{load_['p95_ms']} | {load_['p99_ms']} |",
        ]
        with args.summary.open("a") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
