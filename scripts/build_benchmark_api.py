from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "docs" / "api" / "v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def latest_benchmark_file() -> Path:
    files = sorted(DATA.glob("quantitative_benchmarks_*.yaml"))
    if not files:
        raise FileNotFoundError("quantitative benchmark snapshot not found")
    return files[-1]


def load_snapshot(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = payload.get("benchmarks", [])
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate benchmark id")
    return payload


def dump_json(path: Path, payload: object) -> bytes:
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(data)
    return data


def build() -> dict:
    source = latest_benchmark_file()
    source_bytes = source.read_bytes()
    payload = load_snapshot(source)
    rows = sorted(payload["benchmarks"], key=lambda row: row["id"])
    OUT.mkdir(parents=True, exist_ok=True)

    benchmarks_payload = {
        "schema_version": 1,
        "as_of": payload["as_of"],
        "record_count": len(rows),
        "benchmarks": rows,
    }
    benchmarks_bytes = dump_json(OUT / "benchmarks.json", benchmarks_payload)

    fields = [
        "id", "source_id", "publisher", "title", "period", "published_at",
        "release_status", "aggregation", "metric", "value", "unit",
        "previous_value", "change_value", "change_unit", "source_url",
    ]
    csv_path = OUT / "benchmarks.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    csv_bytes = csv_path.read_bytes()

    facets = {
        "publisher": dict(sorted(Counter(row["publisher"] for row in rows).items())),
        "metric": dict(sorted(Counter(row["metric"] for row in rows).items())),
        "release_status": dict(sorted(Counter(row["release_status"] for row in rows).items())),
        "period": dict(sorted(Counter(str(row["period"]) for row in rows).items())),
    }
    facets_bytes = dump_json(OUT / "facets.json", {"schema_version": 1, "facets": facets})

    distributions = {}
    for name, data in [
        ("benchmarks.json", benchmarks_bytes),
        ("benchmarks.csv", csv_bytes),
        ("facets.json", facets_bytes),
    ]:
        distributions[name] = {"bytes": len(data), "sha256": sha256_bytes(data)}

    manifest = {
        "schema_version": 1,
        "api_version": "v1",
        "as_of": payload["as_of"],
        "record_count": len(rows),
        "source_snapshot": source.relative_to(ROOT).as_posix(),
        "source_sha256": sha256_bytes(source_bytes),
        "cache": {"strategy": "revalidate_manifest_then_fetch_changed_files"},
        "distributions": distributions,
    }
    dump_json(OUT / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    result = build()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
