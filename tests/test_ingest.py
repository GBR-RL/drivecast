import json
import zipfile
from pathlib import Path

import duckdb
import pytest

from drivecast.lake import ingest, schema, sources


def _zip(path: Path) -> Path:
    old = "date,serial_number,model,capacity_bytes,failure,smart_5_normalized,smart_5_raw\n"
    new = (
        "date,serial_number,model,capacity_bytes,failure,datacenter,pod_slot_num,"
        "is_legacy_format,smart_5_normalized,smart_5_raw,smart_211_normailized,smart_999_raw\n"
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("2016/2016-01-30.csv", old + "2016-01-30,A,M1,4000,0,100,0\n")
        z.writestr(
            "2016/2016-01-31.csv",
            new + "2016-01-31,A,M1,4000,0,ams5,0030,false,100,abc,99,7\n"
            "2016-01-31,B,M2,8000,1,ams5,0031,true,90,12.0,,\n",
        )
        z.writestr("2016/2016-02-01.csv", old + "2016-02-01,A,M1,4000,0,100,3\n")
        # Saved through a spreadsheet: M/D/YY dates and numbers in scientific notation.
        z.writestr("2016/2016-02-02.csv", old + "2/2/16,A,M1,4.00079E+12,0,100,3\n")
        z.writestr("__MACOSX/2016/._2016-02-01.csv", "junk")
    return path


def test_header_names_map_to_the_schema() -> None:
    assert schema.canonical("﻿date") == "date"
    assert schema.canonical("smart_211_normailized") == "smart_211_normalized"
    assert schema.canonical("SMART_5_RAW") == "smart_5_raw"
    assert schema.canonical("smart_999_raw") is None
    assert len(schema.COLUMNS) == 11 + 2 * 93


def test_sources_cover_every_published_file() -> None:
    names = [s.name for s in sources.all_sources((2017, 2))]
    assert names == ["2013", "2014", "2015", "2016Q1", "2016Q2", "2016Q3", "2016Q4", "2017Q1",
                     "2017Q2"]  # fmt: skip
    assert sources.by_name("2016Q3").url.endswith("/data_Q3_2016.zip")
    with pytest.raises(KeyError):
        sources.by_name("1999")


def test_ingest_writes_one_file_per_month_in_one_schema(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "data_2016.zip")
    entry = ingest.ingest(
        sources.Source("test", "https://example.invalid/data_2016.zip"),
        tmp_path / "bronze",
        tmp_path / "work",
        zip_path=archive,
    )
    months = {m["month"]: m for m in entry["months"]}
    assert set(months) == {"2016-01", "2016-02"}
    feb = months["2016-02"]
    assert (feb["days"], feb["date_disagrees"], feb["spreadsheet_rows"]) == (2, 1, 1)
    assert feb["cast_failures"] == {}  # the date comes from the file name
    jan = months["2016-01"]
    assert (jan["rows"], jan["days"], jan["drives"], jan["failures"]) == (3, 2, 2, 1)
    assert jan["cast_failures"] == {"smart_5_raw": 1}  # "abc"; "12.0" casts to 12
    assert jan["unknown_columns"] == ["smart_999_raw"]
    con = duckdb.connect()
    path = (tmp_path / "bronze" / "drive_stats_2016-01.parquet").as_posix()
    described = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
    assert [(r[0], r[1]) for r in described] == list(schema.COLUMNS.items())
    rows = con.execute(
        f"SELECT serial_number, smart_5_raw, smart_211_normalized, pod_slot_num, is_legacy_format "
        f"FROM read_parquet('{path}') WHERE date = '2016-01-31' ORDER BY serial_number"
    ).fetchall()
    assert rows == [("A", None, 99, "0030", False), ("B", 12, None, "0031", True)]
    assert not (tmp_path / "work" / "csv_2016-01").exists()


def test_manifests_merge_in_month_order(tmp_path: Path) -> None:
    def entry(name: str, month: str, rows: int) -> Path:
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "source": name,
                    "zip_bytes": 10,
                    "months": [{"month": month, "rows": rows, "failures": 1, "bytes": 5}],
                }
            )
        )
        return path

    merged = ingest.merge_manifests([entry("b", "2016-04", 2), entry("a", "2016-01", 3)])
    assert [e["source"] for e in merged["entries"]] == ["a", "b"]
    assert (merged["months"], merged["rows"], merged["failures"]) == (2, 5, 2)


def test_newest_quarter_can_be_raised_and_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    monkeypatch.setenv("DRIVECAST_LATEST", "2026Q4")
    assert sources.latest_name() == "2026Q4"
    assert sources.all_sources()[-1].name == "2026Q4"
    monkeypatch.setenv("DRIVECAST_LATEST", "2020Q1")  # older than LATEST: ignored
    assert sources.latest() == sources.LATEST
    assert sources.by_name("2027Q1").url.endswith("/data_Q1_2027.zip")

    published = {"data_Q3_2026.zip", "data_Q4_2026.zip"}

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200 if name in published else 404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert sources.probe((2026, 2), client=client) == ["2026Q3", "2026Q4"]
