#!/usr/bin/env python3
"""Build the public source-first bonus research dashboard JSON."""

from __future__ import annotations

import argparse
import json
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
    normalise_code,
    validate_snapshot,
)
from quantitative_benchmarks import (
    latest_quantitative_benchmarks,
    relative_path as quantitative_relative_path,
    validate_quantitative_benchmarks,
)
from source_survey import latest_source_survey, relative_path, validate_source_survey

DEFAULT_OUTPUT = BASE_DIR / "docs" / "data" / "bonus.json"
SOURCE_TYPE_CHANNEL_MAP = {
    "company_official": "company_official",
    "union_official": "labor_union_official",
    "regulator_filing": "edinet",
}


def load_universe_companies(path: Path = UNIVERSE_FILE) -> dict[str, str]:
    payload = load_yaml(path)
    companies = payload.get("companies") if isinstance(payload, dict) else None
    if not isinstance(companies, list) and isinstance(payload, dict):
        companies = payload.get("nikkei225", {}).get("companies", [])
    result: dict[str, str] = {}
    if isinstance(companies, list):
        for index, item in enumerate(companies):
            if not isinstance(item, dict) or item.get("stock_code") is None:
                continue
            code = normalise_code(item.get("stock_code"))
            name = item.get("company_name_ja") or item.get("company_name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"universe company at index {index} has no name")
            if code in result:
                raise ValueError(f"duplicate universe stock code: {code}")
            result[code] = name.strip()
    if not result:
        raise ValueError(f"universe contains no named companies: {path}")
    return result


def _empty_bonus() -> dict[str, Any]:
    return {
        "frequency_per_year": None,
        "annual_months": None,
        "pool_basis": None,
        "allocation_logic": None,
        "base_salary_link": None,
    }


def _placeholder_record(code: str, company_name: str, as_of: str) -> dict[str, Any]:
    return {
        "stock_code": code,
        "company_name_ja": company_name,
        "subject": "employees",
        "employee_scope": "一次情報の調査対象。対象会社・雇用区分は未確定。",
        "classification": None,
        "evidence_status": "unknown",
        "as_of": as_of,
        "bonus": _empty_bonus(),
        "notes": ["一次情報レコードは未整備。推定値は公開しない。"],
        "sources": [],
    }


def _reviewed_channel_ids(record: dict[str, Any]) -> list[str]:
    result = {
        SOURCE_TYPE_CHANNEL_MAP[source.get("type")]
        for source in record.get("sources", [])
        if source.get("type") in SOURCE_TYPE_CHANNEL_MAP
    }
    return sorted(result)


def _open_questions(record: dict[str, Any]) -> list[str]:
    questions: list[str] = []
    bonus = record.get("bonus") or {}
    if not record.get("classification"):
        questions.append("賞与の算定方式を会社・労組の一次資料で特定する")
    if bonus.get("frequency_per_year") is None:
        questions.append("年間の支給回数・支給時期を確認する")
    if bonus.get("annual_months") is None:
        questions.append("標準者の年間月数または妥結額が明記されているか確認する")
    if not bonus.get("pool_basis"):
        questions.append("原資の決定式または業績連動条件を確認する")
    return questions


def _survey_state(
    record: dict[str, Any],
    required_channels: list[str],
    registry_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reviewed = _reviewed_channel_ids(record)
    reviewed_required = [item for item in required_channels if item in reviewed]
    remaining = [item for item in required_channels if item not in reviewed]
    next_channel_id = remaining[0] if remaining else None
    if record["evidence_status"] in {"confirmed", "partially_confirmed"}:
        stage = "evidence_found"
    elif record.get("sources"):
        stage = "source_reviewed"
    else:
        stage = "queued"
    return {
        "stage": stage,
        "reviewed_channel_ids": reviewed,
        "required_channel_count": len(required_channels),
        "reviewed_required_count": len(reviewed_required),
        "coverage_ratio": (
            round(len(reviewed_required) / len(required_channels), 4)
            if required_channels
            else 1.0
        ),
        "next_channel_id": next_channel_id,
        "next_channel_name_ja": (
            registry_by_id[next_channel_id]["name_ja"] if next_channel_id else None
        ),
        "open_questions": _open_questions(record),
    }


def build_public_payload(
    snapshot: dict[str, Any],
    records: list[dict[str, Any]],
    input_path: Path,
    universe_companies: dict[str, str],
    source_survey: dict[str, Any],
    source_survey_path: Path,
    quantitative: dict[str, Any],
    quantitative_path: Path,
) -> dict[str, Any]:
    verified_by_code = {item["stock_code"]: item for item in records}
    registry = source_survey["source_registry"]
    registry_by_id = {item["id"]: item for item in registry}
    required_channels = source_survey["required_channels"]
    public_records: list[dict[str, Any]] = []

    for code, company_name in sorted(universe_companies.items()):
        item = verified_by_code.get(code) or _placeholder_record(
            code, company_name, snapshot["as_of"]
        )
        survey = _survey_state(item, required_channels, registry_by_id)
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
                "notes": item.get("notes", []),
                "sources": item.get("sources", []),
                "survey": survey,
            }
        )

    status_counts = Counter(item["evidence_status"] for item in public_records)
    stage_counts = Counter(item["survey"]["stage"] for item in public_records)
    release_counts = Counter(item["release_status"] for item in quantitative["benchmarks"])
    primary_tiers = {"primary_company", "primary_collective", "official_disclosure"}
    total_reviewed_required = sum(
        item["survey"]["reviewed_required_count"] for item in public_records
    )
    total_required = len(public_records) * len(required_channels)

    return {
        "schema_version": 3,
        "as_of": max(source_survey["as_of"], quantitative["as_of"]),
        "generated_from": relative_path(input_path),
        "source_survey_generated_from": relative_path(source_survey_path),
        "quantitative_benchmarks_generated_from": quantitative_relative_path(quantitative_path),
        "universe": {
            "source_file": snapshot["universe"]["source_file"],
            "mutation_policy": snapshot["universe"]["mutation_policy"],
            "tracked_companies": len(universe_companies),
            "covered_companies": len(public_records),
            "coverage_ratio": round(len(public_records) / len(universe_companies), 4),
        },
        "methodology": source_survey["methodology"],
        "quantitative_methodology": quantitative["methodology"],
        "research_pipeline": source_survey["research_pipeline"],
        "required_channels": required_channels,
        "benchmark_channels": source_survey["benchmark_channels"],
        "discovery_channels": source_survey["discovery_channels"],
        "source_registry": registry,
        "quantitative_benchmarks": quantitative["benchmarks"],
        "summary": {
            "record_count": len(public_records),
            "verified_record_count": len(records),
            "confirmed_or_partial_count": sum(
                status_counts[name]
                for name in ("confirmed", "partially_confirmed")
            ),
            "source_channel_count": len(registry),
            "primary_channel_count": sum(
                1 for item in registry if item["tier"] in primary_tiers
            ),
            "required_channel_count": len(required_channels),
            "reviewed_required_channel_count": total_reviewed_required,
            "research_coverage_ratio": (
                round(total_reviewed_required / total_required, 4)
                if total_required
                else 1.0
            ),
            "quantitative_benchmark_count": len(quantitative["benchmarks"]),
            "quantitative_final_count": release_counts["final"],
            "quantitative_provisional_count": release_counts["first"],
            "evidence_status_counts": dict(sorted(status_counts.items())),
            "research_stage_counts": dict(sorted(stage_counts.items())),
        },
        "records": public_records,
    }


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="verified fact snapshot YAML")
    parser.add_argument("--source-survey", type=Path, help="source meta-survey YAML")
    parser.add_argument("--quantitative", type=Path, help="quantitative benchmark YAML")
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
    source_survey_path = args.source_survey or latest_source_survey(DATA_DIR)
    quantitative_path = args.quantitative or latest_quantitative_benchmarks(DATA_DIR)
    snapshot = load_yaml(input_path)
    survey_payload = validate_source_survey(load_yaml(source_survey_path))
    registry_ids = {item["id"] for item in survey_payload["source_registry"]}
    quantitative_payload = validate_quantitative_benchmarks(
        load_yaml(quantitative_path), registry_ids
    )
    universe_codes = load_universe_codes(args.universe)
    universe_companies = load_universe_companies(args.universe)
    if set(universe_companies) != universe_codes:
        raise SystemExit("named universe differs from the frozen code universe")
    records = validate_snapshot(snapshot, universe_codes)
    expected = render_json(
        build_public_payload(
            snapshot,
            records,
            input_path,
            universe_companies,
            survey_payload,
            source_survey_path,
            quantitative_payload,
            quantitative_path,
        )
    )

    if args.check:
        if not args.output.exists():
            raise SystemExit(f"Pages JSON is missing: {args.output}")
        if args.output.read_text(encoding="utf-8") != expected:
            raise SystemExit(
                "Pages JSON is stale. Run: python scripts/generate_pages_data.py"
            )
        print(
            f"PASS: source-first Pages JSON covers {len(universe_companies)} companies, "
            f"{len(survey_payload['source_registry'])} source channels, and "
            f"{len(quantitative_payload['benchmarks'])} quantitative benchmarks"
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    print(
        f"Wrote source-first Pages JSON: {len(universe_companies)} companies, "
        f"{len(records)} verified records, "
        f"{len(survey_payload['source_registry'])} source channels, "
        f"{len(quantitative_payload['benchmarks'])} quantitative benchmarks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
