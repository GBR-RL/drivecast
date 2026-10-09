"""The one schema every month of the lake is written in.

Backblaze added columns over the years: 85 in 2013, 95 in 2015, 105 in 2018, 131 in 2020, 179 in
2022, 193 in late 2023 (datacenter, cluster, vault, pod) and 197 since Q3 2024. Every month is
cast to the current set (Drive_Stats_Schema_Current.csv, last updated Q2 2024); a column a month
does not have is NULL. Column names are matched by pattern, because the published schema file
itself spells two of them "normailized".
"""

from __future__ import annotations

import re

BASE: dict[str, str] = {
    "date": "DATE",
    "serial_number": "VARCHAR",
    "model": "VARCHAR",
    "capacity_bytes": "BIGINT",
    "failure": "TINYINT",
    # Identifiers with leading zeros ("0030"), so kept as text.
    "datacenter": "VARCHAR",
    "cluster_id": "VARCHAR",
    "vault_id": "VARCHAR",
    "pod_id": "VARCHAR",
    "pod_slot_num": "VARCHAR",
    "is_legacy_format": "BOOLEAN",
}

SMART_IDS: tuple[int, ...] = (
    1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 22, 23, 24, 27, 71, 82, 90, 160, 161,
    163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
    182, 183, 184, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
    206, 210, 211, 212, 218, 220, 222, 223, 224, 225, 226, 230, 231, 232, 233, 234, 235, 240, 241,
    242, 244, 245, 246, 247, 248, 250, 251, 252, 254, 255,
)  # fmt: skip

COLUMNS: dict[str, str] = {
    **BASE,
    **{
        f"smart_{i}_{kind}": sql_type
        for i in SMART_IDS
        for kind, sql_type in (("normalized", "SMALLINT"), ("raw", "BIGINT"))
    },
}

_SMART = re.compile(r"^smart_(\d+)_(norm\w*|raw)$")


def canonical(name: str) -> str | None:
    """The schema column a CSV header maps to, or None if it is not part of the schema."""
    name = name.strip().lstrip("﻿").lower()
    if name in BASE:
        return name
    match = _SMART.match(name)
    if not match:
        return None
    kind = "raw" if match.group(2) == "raw" else "normalized"
    column = f"smart_{int(match.group(1))}_{kind}"
    return column if column in COLUMNS else None


def cast(source: str, sql_type: str) -> str:
    """SQL that casts a text column; integers written as floats ("12.0") are accepted."""
    quoted = f'"{source}"'
    if sql_type in ("BIGINT", "SMALLINT", "TINYINT"):
        return (
            f"COALESCE(TRY_CAST({quoted} AS {sql_type}), "
            f"TRY_CAST(TRY_CAST({quoted} AS DOUBLE) AS {sql_type}))"
        )
    return f"TRY_CAST({quoted} AS {sql_type})"
