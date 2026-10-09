"""The published Backblaze Drive Stats files: one per year until 2015, one per quarter since."""

from __future__ import annotations

from dataclasses import dataclass

BASE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data"
FIRST_QUARTERLY_YEAR = 2016
# The newest quarter that is published. Raised when Backblaze releases the next one.
LATEST = (2026, 2)


@dataclass(frozen=True)
class Source:
    name: str
    url: str


def all_sources(latest: tuple[int, int] = LATEST) -> list[Source]:
    sources = [Source(str(y), f"{BASE_URL}/data_{y}.zip") for y in (2013, 2014, 2015)]
    for year in range(FIRST_QUARTERLY_YEAR, latest[0] + 1):
        for quarter in range(1, 5):
            if (year, quarter) > latest:
                break
            sources.append(Source(f"{year}Q{quarter}", f"{BASE_URL}/data_Q{quarter}_{year}.zip"))
    return sources


def by_name(name: str) -> Source:
    for source in all_sources():
        if source.name == name:
            return source
    raise KeyError(f"no such source: {name}")
