"""The published Backblaze Drive Stats files: one per year until 2015, one per quarter since."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx

BASE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data"
YEARLY = (2013, 2014, 2015)
FIRST_QUARTERLY_YEAR = 2016
# The newest quarter known to be published. A newer one can be passed as DRIVECAST_LATEST
# (e.g. "2026Q3"), which is how the refresh workflow runs the pipeline for a new quarter.
LATEST = (2026, 2)
NAME = re.compile(r"^(\d{4})(?:Q([1-4]))?$")


@dataclass(frozen=True)
class Source:
    name: str
    url: str


def parse_quarter(name: str) -> tuple[int, int]:
    match = NAME.match(name)
    if not match or not match.group(2):
        raise ValueError(f"not a quarter: {name}")
    return int(match.group(1)), int(match.group(2))


def latest() -> tuple[int, int]:
    """The newest quarter: DRIVECAST_LATEST if it is set and newer, else LATEST."""
    override = os.environ.get("DRIVECAST_LATEST", "").strip()
    return max(LATEST, parse_quarter(override)) if override else LATEST


def latest_name() -> str:
    year, quarter = latest()
    return f"{year}Q{quarter}"


def all_sources(newest: tuple[int, int] | None = None) -> list[Source]:
    newest = newest or latest()
    sources = [Source(str(y), f"{BASE_URL}/data_{y}.zip") for y in YEARLY]
    for year in range(FIRST_QUARTERLY_YEAR, newest[0] + 1):
        for quarter in range(1, 5):
            if (year, quarter) > newest:
                break
            sources.append(Source(f"{year}Q{quarter}", f"{BASE_URL}/data_Q{quarter}_{year}.zip"))
    return sources


def by_name(name: str) -> Source:
    """A source from its name ("2014" for the yearly files, "2026Q3" for quarters)."""
    match = NAME.match(name)
    if match and not match.group(2) and int(match.group(1)) in YEARLY:
        return Source(name, f"{BASE_URL}/data_{name}.zip")
    if match and match.group(2) and int(match.group(1)) >= FIRST_QUARTERLY_YEAR:
        year, quarter = int(match.group(1)), int(match.group(2))
        return Source(name, f"{BASE_URL}/data_Q{quarter}_{year}.zip")
    raise KeyError(f"no such source: {name}")


def next_quarter(year: int, quarter: int) -> tuple[int, int]:
    return (year + 1, 1) if quarter == 4 else (year, quarter + 1)


def probe(after: tuple[int, int] | None = None, client: httpx.Client | None = None) -> list[str]:
    """Quarters published after ``after`` (default: the newest known), in order."""
    year, quarter = after or latest()
    found: list[str] = []
    with client or httpx.Client(timeout=30, follow_redirects=True) as http:
        while True:
            year, quarter = next_quarter(year, quarter)
            name = f"{year}Q{quarter}"
            if http.head(by_name(name).url).status_code != 200:
                return found
            found.append(name)
