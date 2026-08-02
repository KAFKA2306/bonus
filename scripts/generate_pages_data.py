#!/usr/bin/env python3
"""Build public GitHub Pages JSON from verified facts and full-universe hypotheses."""

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


def _placeholder_record(code: str, hypothesis: dict[str, Any], as_of: str) -> dict[str, Any]:
    return {
        "stock_code": code,
        "company_name_ja": hypothesis["company_name_ja"],
        "subject": "employees",
        "employee_scope": "固定Universe対象企業。一次情報による制度詳細は未監査で、以下は仮説推定。",
        "classification": None,
        "evidence_status": "unknown",
        "as_of": as_of,
        "bonus": {
            "frequency_per_year": None,
            "annual_months": None,
            "pool_basis": None,
            "allocation_logic": None,
            "base_salary_link": None,
        },
        "notes": [
            "一次情報の確認レコードは未整備。仮説値は旧調査またはセクター事前分布に基づく。",
            "仮説は確認済み集計へ含めない。",
        ],
        "sources": [],
    }


def build_public_payload(
    snapshot: dict[str, Any],
    records: list[dict[str, Any]],
    input_path: Path,
    tracked_companies: int,
    hypotheses: dict[str, dict[str, Any]],
    hypothesis_snapshot: dict[str, Any],
    hypothesis_path: Path,
) -> dict[str, Any]:
    verified_by_code = {item["stock_code"]: item for item in records}
    public_records = []

    for code in sorted(hypotheses):
        hypothesis = hypotheses[code]
        item = verified_by_code.get(code) or _placeholder_record(
            code, hypothesis, snapshot["as_of"]
        )
        public_records.append(
            {
                "stock_code": code,
                "company_name_ja": item["company_name_ja"],
                "subject": item["subject"],
                "employee_scope": item["employee_scope"],
                "classification": item.get("classification"),
                "evidence_status": item["evidence_status"],
                "as_of": item["as_of"],
                "bonus": item["bonus"],
                "hypothesis": hypothesis,
                "notes": item.get("notes", []),
                "sources": item.get("sources", []),
            }
        )

    status_counts = Counter(item["evidence_status"] for item in public_records)
    classification_counts = Counter(
        item["classification"]
        for item in public_records
        if item.get("classification") is not None
    )
    point_values = point_month_values(records)
    hypothesis_values = [
        float(item["estimate"]["central"]) for item in hypotheses.values()
    ]

    return {
        "schema_version": 1,
        "as_of": snapshot["as_of"],
        "generated_from": relative_path(input_path),
        "hypotheses_generated_from": relative_path(hypothesis_path),
        "universe": {
            "source_file": snapshot["universe"]["source_file"],
            "mutation_policy": snapshot["universe"]["mutation_policy"],
            "tracked_companies": tracked_companies,
            "covered_companies": len(public_records),
            "coverage_ratio": round(len(public_records) / tracked_companies, 4),
        },
        "methodology": snapshot.get("methodology", {}),
        "hypothesis_methodology": hypothesis_snapshot.get("methodology", {}),
        "summary": {
            "record_count": len(public_records),
            "verified_record_count": len(records),
            "confirmed_or_partial_count": sum(
                status_counts[name]
                for name in ("confirmed", "partially_confirmed")
            ),
            "hypothesis_count": len(hypotheses),
            "hypothesis_central_months_average": round(
                statistics.mean(hypothesis_values), 2
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
            f"PASS: Pages JSON covers all {len(universe_codes)} companies "
            f"with {len(records)} verified records and {len(hypotheses)} hypotheses"
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    print(
        f"Wrote Pages JSON: {len(universe_codes)} covered, "
        f"{len(records)} verified, {len(hypotheses)} hypotheses"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
