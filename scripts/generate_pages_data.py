#!/usr/bin/env python3
"""Build public Pages JSON from verified facts and a separate hypothesis layer."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from bonus_hypotheses import latest_hypothesis, relative_path, validate_hypotheses
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
    hypotheses: dict[str, dict[str, Any]] | None = None,
    hypothesis_snapshot: dict[str, Any] | None = None,
    hypothesis_path: Path | None = None,
) -> dict[str, Any]:
    hypotheses = hypotheses or {}
    status_counts = Counter(item["evidence_status"] for item in records)
    classification_counts = Counter(
        item["classification"]
        for item in records
        if item.get("classification") is not None
    )
    point_values = point_month_values(records)
    hypothesis_central_values = [
        float(item["estimate"]["central"]) for item in hypotheses.values()
    ]

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
                "hypothesis": hypotheses.get(item["stock_code"]),
                "notes": item.get("notes", []),
                "sources": item["sources"],
            }
        )

    return {
        "schema_version": 1,
        "as_of": snapshot["as_of"],
        "generated_from": relative_path(input_path),
        "hypotheses_generated_from": (
            relative_path(hypothesis_path) if hypothesis_path is not None else None
        ),
        "universe": {
            "source_file": snapshot["universe"]["source_file"],
            "mutation_policy": snapshot["universe"]["mutation_policy"],
            "tracked_companies": tracked_companies,
        },
        "methodology": snapshot.get("methodology", {}),
        "hypothesis_methodology": (
            hypothesis_snapshot.get("methodology", {})
            if hypothesis_snapshot is not None
            else {}
        ),
        "summary": {
            "record_count": len(records),
            "confirmed_or_partial_count": sum(
                status_counts[name]
                for name in ("confirmed", "partially_confirmed")
            ),
            "hypothesis_count": len(hypotheses),
            "hypothesis_central_months_average": (
                round(statistics.mean(hypothesis_central_values), 2)
                if hypothesis_central_values
                else None
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
    parser.add_argument("--hypotheses", type=Path, help="hypothesis snapshot YAML")
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
    hypothesis_path = args.hypotheses or latest_hypothesis(DATA_DIR)
    snapshot = load_yaml(input_path)
    hypothesis_snapshot = load_yaml(hypothesis_path)
    universe_codes = load_universe_codes(args.universe)
    records = validate_snapshot(snapshot, universe_codes)
    hypotheses = validate_hypotheses(hypothesis_snapshot, universe_codes)
    expected = render_json(
        build_public_payload(
            snapshot,
            records,
            input_path,
            tracked_companies=len(universe_codes),
            hypotheses=hypotheses,
            hypothesis_snapshot=hypothesis_snapshot,
            hypothesis_path=hypothesis_path,
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
            f"PASS: Pages JSON matches {input_path.name} and "
            f"{hypothesis_path.name} ({len(records)} records)"
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    print(
        f"Wrote Pages JSON from {input_path.name} and {hypothesis_path.name}: "
        f"{args.output} ({len(records)} records, {len(hypotheses)} hypotheses)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
