"""README charts from the committed result files, each in a light and a dark variant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from drivecast.config import ROOT

RESULTS = ROOT / "docs" / "results"
ASSETS = ROOT / "docs" / "assets"

# Categorical slots in fixed order (blue, orange, aqua, yellow), stepped per mode.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "muted": "#c3c2b7", "grid": "#33322f",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500"]},
}  # fmt: skip
MODELS = (("lightgbm", "LightGBM"), ("logreg", "logistic regression"), ("rule", "Backblaze rule"))
MAKERS = ("Seagate", "HGST", "Toshiba", "WDC")


def _json(name: str) -> Any:
    return json.loads((RESULTS / name).read_text())


def _style(ax: Any, t: dict[str, Any], grid: str = "y") -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.grid(axis=grid, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _title(fig: Any, t: dict[str, Any], text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", color=t["ink"], fontsize=11, fontweight="bold")


def _legend(ax: Any, t: dict[str, Any], **kw: Any) -> None:
    leg = ax.legend(frameon=False, fontsize=8.5, **kw)
    for text in leg.get_texts():
        text.set_color(t["ink"])


def _save(fig: Any, name: str, theme: str) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"{name}-{theme}.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return path


def backtest(theme: str) -> Path:
    """Average precision per test quarter for the three models."""
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    data = _json("backtest.json")["by_quarter"]
    quarters = sorted(data)
    fig, ax = plt.subplots(figsize=(9.6, 3.6), facecolor=t["surface"])
    _style(ax, t)
    x = range(len(quarters))
    for (key, label), color in zip(MODELS, t["series"], strict=False):
        ys = [data[q][key]["average_precision"] for q in quarters]
        ax.plot(x, ys, color=color, linewidth=2, label=label)
        ax.text(len(quarters) - 0.6, ys[-1], label, color=t["ink"], fontsize=8.5, va="center")
    ticks = [i for i, q in enumerate(quarters) if q.endswith("Q1") or i == 0]
    ax.set_xticks(ticks, [quarters[i][:4] for i in ticks])
    ax.set_xlim(-0.5, len(quarters) + 6)
    ax.set_ylabel("average precision", color=t["muted"], fontsize=9)
    _legend(ax, t, loc="upper left")
    _title(fig, t, "Failure within 30 days: each quarter predicted from the year before it")
    path = _save(fig, "backtest", theme)
    plt.close(fig)
    return path


def survival(theme: str) -> Path:
    """Failure rate by age (the bathtub) and survival by manufacturer."""
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    s = _json("survival.json")
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.4), facecolor=t["surface"])
    _style(left, t)
    hazard = [h for h in s["hazard_by_age"] if h["drive_years"] >= 5000]
    left.plot([h["age_from_years"] + 0.125 for h in hazard], [h["afr"] for h in hazard],
              color=t["series"][0], linewidth=2, marker="o", markersize=3.5)  # fmt: skip
    left.set_xlabel("age (years, from power-on hours)", color=t["muted"], fontsize=9)
    left.set_ylabel("annualized failure rate (%)", color=t["muted"], fontsize=9)
    left.set_ylim(0, None)
    left.set_title("Failure rate by age", loc="left", color=t["ink"], fontsize=10)
    _style(right, t)
    for maker, color in zip(MAKERS, t["series"], strict=True):
        points = s["curves"].get(maker)
        if not points:
            continue
        xs, ys = [p[0] for p in points], [100 * p[1] for p in points]
        right.plot(xs, ys, color=color, linewidth=2, label=maker)
        right.text(xs[-1] + 0.1, ys[-1], maker, color=t["ink"], fontsize=8, va="center")
    right.set_xlabel("age (years)", color=t["muted"], fontsize=9)
    right.set_ylabel("still running (%)", color=t["muted"], fontsize=9)
    right.set_xlim(0, 9.4)
    right.set_title("Survival by manufacturer (Kaplan-Meier, late entry)", loc="left",
                    color=t["ink"], fontsize=10)  # fmt: skip
    _legend(right, t, loc="lower left")
    fig.tight_layout()
    path = _save(fig, "survival", theme)
    plt.close(fig)
    return path


def leakage(theme: str) -> Path:
    """Mean AP of the same model under three ways of holding out data."""
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    lk = _json("leakage.json")
    rows = (("time", "by time (as deployed)"), ("random_drives", "random drives"),
            ("random_rows", "random rows"))  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.2, 2.4), facecolor=t["surface"])
    _style(ax, t, grid="x")
    ys = range(len(rows))
    means = [lk["mean"][k] for k, _ in rows]
    ax.barh(list(ys), means, height=0.55, color=t["series"][0])
    for y, (key, _) in zip(ys, rows, strict=True):
        values = [q[key] for q in lk["quarters"]]
        ax.scatter(values, [y] * len(values), s=14, color=t["muted"], zorder=3)
        ax.text(lk["mean"][key] - 0.003, y, f"{lk['mean'][key]:.3f}", va="center", ha="right",
                fontsize=9, color="#ffffff", fontweight="bold")  # fmt: skip
    ax.set_yticks(list(ys), [label for _, label in rows])
    ax.invert_yaxis()
    ax.set_xlabel("average precision (bar: mean; dots: the 11 test quarters)", color=t["muted"],
                  fontsize=9)  # fmt: skip
    _title(fig, t, "A random split doubles the score the deployed model actually gets")
    path = _save(fig, "leakage", theme)
    plt.close(fig)
    return path


def lake(theme: str) -> Path:
    """Rows in the lake per year."""
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    q = _json("quality.json")
    by_year = q["checks"]["rows"]["by_year"]
    years = sorted(by_year)
    fig, ax = plt.subplots(figsize=(7.6, 2.8), facecolor=t["surface"])
    _style(ax, t)
    values = [by_year[y] / 1e6 for y in years]
    ax.bar(years, values, color=t["series"][0], width=0.65)
    for x, v in zip(years, values, strict=True):
        ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=t["ink"])
    ax.set_ylabel("million drive-days", color=t["muted"], fontsize=9)
    ax.tick_params(axis="x", labelrotation=0)
    total = q["checks"]["rows"]["total"]
    _title(fig, t, f"{total / 1e6:.0f}M daily SMART reports from {q['drives']:,} drives "
                   "(2026: first half)")  # fmt: skip
    path = _save(fig, "lake", theme)
    plt.close(fig)
    return path


def engines(theme: str) -> Path:
    """Feature-job run time on DuckDB and Spark, same runner and data."""
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    runs = _json("engines.json")
    fig, ax = plt.subplots(figsize=(7.2, 2.8), facecolor=t["surface"])
    _style(ax, t)
    xs = range(len(runs))
    width = 0.36
    for offset, key, label, color in ((-width / 2, "duckdb_seconds", "DuckDB", t["series"][0]),
                                      (width / 2, "spark_seconds", "Spark (local mode)",
                                       t["series"][1])):  # fmt: skip
        values = [r[key] for r in runs]
        ax.bar([x + offset for x in xs], values, width=width, color=color, label=label)
        for x, v in zip(xs, values, strict=True):
            ax.text(x + offset, v + 4, f"{v:.0f}", ha="center", fontsize=8, color=t["ink"])
    ax.set_xticks(list(xs), [f"{r['quarter']}\n{r['rows'] / 1e6:.0f}M rows" for r in runs])
    ax.set_ylabel("seconds", color=t["muted"], fontsize=9)
    _legend(ax, t, loc="upper left")
    _title(fig, t, "The feature job on one 4-vCPU runner: identical output, run time")
    path = _save(fig, "engines", theme)
    plt.close(fig)
    return path


def staleness(theme: str) -> Path:
    """How a kept model decays with age, and what retraining policies buy."""
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    pol = _json("policies.json")
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.4), facecolor=t["surface"])
    _style(left, t)
    by_age = pol["drift_vs_decay"]["mean_relative_ap_by_age"]
    ages = sorted(by_age, key=int)
    values = [100 * by_age[a] for a in ages]
    left.bar(ages, values, color=t["series"][0], width=0.6)
    for x, v in zip(ages, values, strict=True):
        left.text(x, v - 1.2, f"{v:.0f}%", ha="center", va="top", fontsize=8.5, color=t["ink"])
    left.axhline(0, color=t["muted"], linewidth=0.8)
    left.set_ylim(min(values) * 1.25, 3)
    left.set_xlabel("model age (quarters since it was fitted)", color=t["muted"], fontsize=9)
    left.set_ylabel("AP against a fresh model", color=t["muted"], fontsize=9)
    left.set_title("A kept model loses its edge", loc="left", color=t["ink"], fontsize=10)
    _style(right, t, grid="both")
    names = {"never": "never", "yearly": "yearly", "quarterly": "every quarter"}
    offsets = {"never": (6, 2), "yearly": (-34, -3), "quarterly": (-62, 6)}
    quarterly = next(p for p in pol["policies"] if p["policy"] == "quarterly")
    for p in pol["policies"]:
        drift = p["policy"].startswith("drift")
        if drift and p["retrains"] >= quarterly["retrains"] - 2:
            continue  # retrains almost every quarter: the same point as quarterly
        point = (p["retrains"], p["mean"]["average_precision"])
        right.scatter(*point, s=46, zorder=3, color=t["series"][1] if drift else t["series"][0],
                      edgecolor=t["surface"], linewidth=1.2)  # fmt: skip
        label = f"PSI > {p['policy'].split('>')[1]}" if drift else names[p["policy"]]
        right.annotate(label, point, fontsize=8, color=t["ink"], textcoords="offset points",
                       xytext=(6, -11) if drift else offsets[p["policy"]])  # fmt: skip
    right.scatter([], [], color=t["series"][0], label="on a schedule")
    right.scatter([], [], color=t["series"][1], label="on drift")
    _legend(right, t, loc="lower right")
    right.set_xlabel("models fitted over the backtest", color=t["muted"], fontsize=9)
    right.set_ylabel("mean average precision", color=t["muted"], fontsize=9)
    right.set_title("Retraining policies: accuracy against cost", loc="left", color=t["ink"],
                    fontsize=10)  # fmt: skip
    fig.tight_layout()
    path = _save(fig, "staleness", theme)
    plt.close(fig)
    return path


def all_charts() -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    makers = [backtest, survival, leakage, lake, engines, staleness]
    return [fn(theme) for fn in makers for theme in THEMES]
