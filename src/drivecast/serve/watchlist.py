"""The fleet watchlist: every drive scored on the latest day in the lake, riskiest first.

Uses the silver files of the latest quarter (and the one before, for the 30-day windows), the
drive table and the served model, through the same feature query as everything else. Writes
the top drives as CSV and JSON, and a static HTML page.
"""

from __future__ import annotations

import html
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from drivecast.features.build import window_sql
from drivecast.models.estimators import RULE_ATTRIBUTES
from drivecast.serve.model import load

COLUMNS = ["rank", "serial_number", "model", "manufacturer", "capacity_tb", "age_years", "score",
           "warnings"]  # fmt: skip


def build(
    con: Any, silver_files: list[str], drives: str, model_dir: Path, out: Path, *, top: int = 100
) -> dict[str, Any]:
    listed = ", ".join(f"'{f}'" for f in silver_files)
    con.execute(f"CREATE OR REPLACE VIEW silver AS SELECT * FROM read_parquet([{listed}])")
    con.execute(f"CREATE OR REPLACE VIEW drives AS SELECT * FROM read_parquet('{drives}')")
    (last,) = con.execute("SELECT max(date) FROM silver").fetchone()
    day = pd.Timestamp(last).date()
    frame: pd.DataFrame = con.execute(
        window_sql(day.isoformat(), (day + timedelta(days=1)).isoformat(), day.isoformat())
    ).df()
    model = load(model_dir)
    frame["score"] = model.score(frame)
    frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
    frame["rank"] = frame.index + 1
    frame["age_years"] = (frame["age_days"] / 365).round(1)
    frame["warnings"] = [
        ", ".join(a for a in RULE_ATTRIBUTES if pd.notna(row[a]) and row[a] > 0)
        for _, row in frame[list(RULE_ATTRIBUTES)].iterrows()
    ]
    watch = frame.head(top)[COLUMNS]
    out.mkdir(parents=True, exist_ok=True)
    watch.to_csv(out / "watchlist.csv", index=False)
    summary = {
        "date": day.isoformat(),
        "model": model.describe(),
        "drives": len(frame),
        "with_warnings": int((frame["warnings"] != "").sum()),
        # The scores are roughly calibrated (the backtest scores sum to about 0.9 of the
        # positives), so their sum is the model's own expectation, not a forecast to rely on.
        "expected_failing_within_30_days": round(float(frame["score"].sum()), 1),
        "top": json.loads(watch.to_json(orient="records")),
    }
    (out / "watchlist.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "index.html").write_text(page(summary), encoding="utf-8")
    return summary


STYLE = """
:root { --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --muted:#5d6675; --line:#dfe3ea; --hot:#b42318; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171c; --card:#1d2128; --ink:#e7eaf0; --muted:#9aa3b2; --line:#2c323c;
          --hot:#f97066; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px; }
h1 { font-size:22px; margin:0 0 4px; } .muted { color:var(--muted); }
.stats { display:flex; flex-wrap:wrap; gap:12px; margin:16px 0; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:10px 14px; }
.stat b { display:block; font-size:20px; }
.wrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; background:var(--card);
        border:1px solid var(--line); }
td, th { text-align:left; padding:8px 10px; border-top:1px solid var(--line); white-space:nowrap; }
th { font-size:13px; color:var(--muted); border-top:none; }
td.num { text-align:right; font-variant-numeric: tabular-nums; }
td.warn { color:var(--hot); }
"""


def page(summary: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td class='num'>{r['rank']}</td><td>{html.escape(r['serial_number'])}</td>"
        f"<td>{html.escape(r['model'])}</td><td>{html.escape(r['manufacturer'])}</td>"
        f"<td class='num'>{r['capacity_tb']}</td><td class='num'>{r['age_years']}</td>"
        f"<td class='num'>{r['score']:.3f}</td>"
        f"<td class='warn'>{html.escape(r['warnings'])}</td></tr>"
        for r in summary["top"]
    )
    m = summary["model"]
    stats = (
        f"<div class='stat'>drives scored<b>{summary['drives']:,}</b></div>"
        f"<div class='stat'>with a SMART warning<b>{summary['with_warnings']:,}</b></div>"
        f"<div class='stat'>model's expected failures, next 30 days"
        f"<b>{summary['expected_failing_within_30_days']:,.0f}</b></div>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Drive watchlist</title><style>{STYLE}</style></head><body><main>"
        f"<h1>Drive watchlist, {summary['date']}</h1>"
        f"<p class='muted'>Every data drive in the Backblaze fleet on the latest day of the "
        f"published data, scored by the deployed model ({html.escape(m['family'])}, fitted for "
        f"{html.escape(m['quarter'])}): the chance of failing within 30 days. The "
        f"{len(summary['top'])} riskiest drives.</p>"
        f"<div class='stats'>{stats}</div><div class='wrap'><table><tr><th>#</th><th>serial</th>"
        "<th>model</th><th>maker</th><th>TB</th><th>age (y)</th><th>score</th>"
        f"<th>warnings</th></tr>{rows}</table></div></main></body></html>"
    )
