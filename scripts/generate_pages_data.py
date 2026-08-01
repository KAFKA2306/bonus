#!/usr/bin/env python3
"""Build the public GitHub Pages JSON from the latest verified snapshot."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from generate_verified_bonus_summary import (
    BASE_DIR,
    DATA_DIR,
    UNIVERSE_FILE,
    latest_snapshot,
    load_universe_codes,
    load_yaml,
    point_month_values,
    validate_snapshot,
)

DEFAULT_OUTPUT = BASE_DIR / "docs" / "data" / "bonus.json"


def build_public_payload(
    snapshot: dict[str, Any],
    records: list[dict[str, Any]],
    input_path: Path,
    tracked_companies: int,
) -> dict[str, Any]:
    status_counts = Counter(item["evidence_status"] for item in records)
    classification_counts = Counter(
        item["classification"]
        for item in records
        if item.get("classification") is not None
    )
    point_values = point_month_values(records)

    public_records = []
    for item in sorted(records, key=lambda record: record["stock_code"]):
        public_records.append(
            {
                "stock_code": item["stock_code"],
                "company_name_ja": item["company_name_ja"],
                "subject": item["subject"],
                "employee_scope": item["employee_scope"],
                "classification": item.get("classification"),
                "evidence_status": item["evidence_status"],
                "as_of": item["as_of"],
                "bonus": item["bonus"],
                "notes": item.get("notes", []),
                "sources": item["sources"],
            }
        )

    try:
        generated_from = str(input_path.resolve().relative_to(BASE_DIR.resolve()))
    except ValueError:
        generated_from = str(input_path)

    return {
        "schema_version": 1,
        "as_of": snapshot["as_of"],
        "generated_from": generated_from,
        "universe": {
            "source_file": snapshot["universe"]["source_file"],
            "mutation_policy": snapshot["universe"]["mutation_policy"],
            "tracked_companies": tracked_companies,
        },
        "methodology": snapshot.get("methodology", {}),
        "summary": {
            "record_count": len(records),
            "confirmed_or_partial_count": sum(
                status_counts[name]
                for name in ("confirmed", "partially_confirmed")
            ),
            "evidence_status_counts": dict(sorted(status_counts.items())),
            "classification_counts": dict(sorted(classification_counts.items())),
            "explicit_point_months_count": len(point_values),
            "explicit_point_months_average": (
                round(statistics.mean(point_values), 2) if point_values else None
            ),
        },
        "records": public_records,
    }


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="verified snapshot YAML")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--universe", type=Path, default=UNIVERSE_FILE)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless the existing output exactly matches regenerated JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input or latest_snapshot(DATA_DIR)
    snapshot = load_yaml(input_path)
    universe_codes = load_universe_codes(args.universe)
    records = validate_snapshot(snapshot, universe_codes)
    expected = render_json(
        build_public_payload(
            snapshot,
            records,
            input_path,
            tracked_companies=len(universe_codes),
        )
    )

    if args.check:
        if not args.output.exists():
            raise SystemExit(f"Pages JSON is missing: {args.output}")
        actual = args.output.read_text(encoding="utf-8")
        if actual != expected:
            raise SystemExit(
                "Pages JSON is stale. Run: python scripts/generate_pages_data.py"
            )
        print(
            f"PASS: Pages JSON matches {input_path.name} "
            f"({len(records)} records)"
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    print(
        f"Wrote Pages JSON from {input_path.name}: "
        f"{args.output} ({len(records)} records)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
