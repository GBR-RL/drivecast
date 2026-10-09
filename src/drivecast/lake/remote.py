"""The lake as stored on GitHub: the Parquet assets of a release, read in place over HTTP."""

from __future__ import annotations

import httpx

REPO = "GBR-RL/drivecast"
API = "https://api.github.com/repos"


def asset_urls(
    release: str = "lake-v1", prefix: str = "drive_stats_", repo: str = REPO
) -> list[str]:
    """Download URLs of the release's Parquet assets, in name (month) order."""
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        r = client.get(f"{API}/{repo}/releases/tags/{release}")
        r.raise_for_status()
        release_id = r.json()["id"]
        assets: list[dict[str, str]] = []
        page = 1
        while True:
            r = client.get(
                f"{API}/{repo}/releases/{release_id}/assets",
                params={"per_page": 100, "page": page},
            )
            r.raise_for_status()
            batch = r.json()
            assets += batch
            if len(batch) < 100:
                break
            page += 1
    return [
        a["browser_download_url"]
        for a in sorted(assets, key=lambda a: a["name"])
        if a["name"].startswith(prefix) and a["name"].endswith(".parquet")
    ]
