#!/usr/bin/env python3
"""Validate source-backed employee bonus facts and build a conservative summary.

This pipeline deliberately does not infer bonus months from prose. A numeric value is
accepted only when the snapshot marks it as an explicit value from an eligible primary
source. The stock universe is read from ``nikkei225_companies.yaml`` and never modified.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
UNIVERSE_FILE = BASE_DIR / "nikkei225_companies.yaml"
DATA_DIR = BASE_DIR / "data"
SUMMARY_FILE = BASE_DIR / "analysis" / "summary" / "verified_bonus_overview.yaml"

ALLOWED_STATUSES = {"confirmed", "partially_confirmed", "unknown"}
ALLOWED_CLASSIFICATIONS = {
    "performance_linked",
    "base_salary_linked",
    "discretionary",
    "hybrid",
}
PRIMARY_SOURCE_TYPES = {"company_official", "union_official", "regulator_filing"}
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CODE_PATTERN = re.compile(r"^\d{4}$")


class ValidationError(ValueError):
    """Raised when a data snapshot violates the verification contract."""


def load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_iso_date(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not DATE_PATTERN.fullmatch(value):
        raise ValidationError(f"{field_name} must be YYYY-MM-DD")
    try:
        dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError(f"{field_name} is not a valid date: {value}") from exc
    return value


def normalise_code(value: Any) -> str:
    code = str(value or "").strip().zfill(4)
    if not CODE_PATTERN.fullmatch(code):
        raise ValidationError(f"invalid stock_code: {value!r}")
    return code


def load_universe_codes(path: Path = UNIVERSE_FILE) -> set[str]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError:
        payload = None

    codes: set[str] = set()
    if isinstance(payload, dict):
        companies = payload.get("nikkei225", {}).get("companies", [])
        if isinstance(companies, list):
            for item in companies:
                if isinstance(item, dict) and item.get("stock_code") is not None:
                    codes.add(normalise_code(item.get("stock_code")))

    if not codes:
        # The frozen historical file currently contains malformed YAML. Preserve it
        # unchanged and recover only explicit 4-digit stock_code entries.
        pattern = re.compile(
            r"(?m)^\s*-\s+stock_code:\s*[\"']?(\d{4})[\"']?\s*$"
        )
        codes = set(pattern.findall(text))

    if not codes:
        raise ValidationError(f"universe contains no recoverable stock codes: {path}")
    return codes


def is_https_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("https://")


def validate_sources(record: dict[str, Any], index: int) -> list[dict[str, Any]]:
    sources = record.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValidationError(f"records[{index}].sources must be a non-empty list")

    validated: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        prefix = f"records[{index}].sources[{source_index}]"
        if not isinstance(source, dict):
            raise ValidationError(f"{prefix} must be an object")
        source_type = source.get("type")
        if source_type not in PRIMARY_SOURCE_TYPES:
            raise ValidationError(
                f"{prefix}.type must be one of {sorted(PRIMARY_SOURCE_TYPES)}"
            )
        if not isinstance(source.get("title"), str) or not source["title"].strip():
            raise ValidationError(f"{prefix}.title is required")
        if not is_https_url(source.get("url")):
            raise ValidationError(f"{prefix}.url must be an https primary-source URL")
        for date_key in ("retrieved_at", "published_at", "updated_at"):
            date_value = source.get(date_key)
            if date_value is not None:
                parse_iso_date(date_value, f"{prefix}.{date_key}")
        validated.append(source)
    return validated


def validate_annual_months(
    annual_months: Any, status: str, index: int
) -> dict[str, Any] | None:
    if annual_months is None:
        return None
    prefix = f"records[{index}].bonus.annual_months"
    if not isinstance(annual_months, dict):
        raise ValidationError(f"{prefix} must be an object or null")
    if status != "confirmed":
        raise ValidationError(f"{prefix} requires evidence_status=confirmed")

    kind = annual_months.get("kind")
    if kind not in {"point", "minimum", "maximum", "range"}:
        raise ValidationError(f"{prefix}.kind is invalid")
    if annual_months.get("basis") != "explicit_source_value":
        raise ValidationError(
            f"{prefix}.basis must be explicit_source_value; inferred values are forbidden"
        )

    raw_values = (
        [annual_months.get("minimum"), annual_months.get("maximum")]
        if kind == "range"
        else [annual_months.get("value")]
    )
    values: list[float] = []
    for raw in raw_values:
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ValidationError(f"{prefix} contains a non-numeric value")
        value = float(raw)
        if not 0 < value <= 24:
            raise ValidationError(f"{prefix} values must be within (0, 24]")
        values.append(value)
    if kind == "range" and values[0] > values[1]:
        raise ValidationError(f"{prefix}.minimum cannot exceed maximum")
    return annual_months


def validate_record(
    record: Any, universe_codes: set[str], index: int
) -> dict[str, Any]:
    prefix = f"records[{index}]"
    if not isinstance(record, dict):
        raise ValidationError(f"{prefix} must be an object")

    code = normalise_code(record.get("stock_code"))
    if code not in universe_codes:
        raise ValidationError(
            f"{prefix}.stock_code {code} is outside the frozen universe"
        )
    if record.get("subject") != "employees":
        raise ValidationError(
            f"{prefix}.subject must be employees; executive compensation is excluded"
        )
    for key in ("company_name_ja", "employee_scope"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValidationError(f"{prefix}.{key} is required")

    status = record.get("evidence_status")
    if status not in ALLOWED_STATUSES:
        raise ValidationError(f"{prefix}.evidence_status is invalid")
    classification = record.get("classification")
    if classification is not None and classification not in ALLOWED_CLASSIFICATIONS:
        raise ValidationError(f"{prefix}.classification is invalid")
    if status == "unknown" and classification is not None:
        raise ValidationError(
            f"{prefix}.classification must be null when evidence_status is unknown"
        )

    parse_iso_date(record.get("as_of"), f"{prefix}.as_of")
    sources = validate_sources(record, index)

    bonus = record.get("bonus")
    if not isinstance(bonus, dict):
        raise ValidationError(f"{prefix}.bonus must be an object")
    frequency = bonus.get("frequency_per_year")
    if frequency is not None:
        if status == "unknown":
            raise ValidationError(
                f"{prefix}.bonus.frequency_per_year is forbidden for unknown evidence"
            )
        if not isinstance(frequency, int) or isinstance(frequency, bool):
            raise ValidationError(f"{prefix}.bonus.frequency_per_year must be an integer")
        if not 1 <= frequency <= 12:
            raise ValidationError(
                f"{prefix}.bonus.frequency_per_year must be between 1 and 12"
            )
    validate_annual_months(bonus.get("annual_months"), status, index)

    result = dict(record)
    result["stock_code"] = code
    result["sources"] = sources
    return result


def validate_snapshot(
    payload: Any, universe_codes: set[str]
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValidationError("snapshot must be an object")
    if payload.get("schema_version") != 1:
        raise ValidationError("schema_version must be 1")
    parse_iso_date(payload.get("as_of"), "as_of")

    universe = payload.get("universe")
    if not isinstance(universe, dict):
        raise ValidationError("universe metadata is required")
    if universe.get("source_file") != "nikkei225_companies.yaml":
        raise ValidationError("universe.source_file must be nikkei225_companies.yaml")
    if universe.get("mutation_policy") != "frozen":
        raise ValidationError("universe.mutation_policy must be frozen")

    records = payload.get("records")
    if not isinstance(records, list):
        raise ValidationError("records must be a list")

    validated: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    for index, record in enumerate(records):
        item = validate_record(record, universe_codes, index)
        code = item["stock_code"]
        if code in seen_codes:
            raise ValidationError(f"duplicate stock_code in snapshot: {code}")
        seen_codes.add(code)
        validated.append(item)
    return validated


def point_month_values(records: Iterable[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in records:
        if record.get("evidence_status") != "confirmed":
            continue
        annual = record.get("bonus", {}).get("annual_months")
        if not isinstance(annual, dict):
            continue
        if annual.get("kind") == "point" and annual.get("basis") == "explicit_source_value":
            values.append(float(annual["value"]))
    return values


def build_summary(
    snapshot: dict[str, Any], records: list[dict[str, Any]], input_path: Path
) -> dict[str, Any]:
    status_counts = Counter(record["evidence_status"] for record in records)
    classification_counts = Counter(
        record["classification"]
        for record in records
        if record.get("classification") is not None
    )
    point_values = point_month_values(records)

    record_rows = []
    for record in sorted(records, key=lambda item: item["stock_code"]):
        bonus = record["bonus"]
        record_rows.append(
            {
                "stock_code": record["stock_code"],
                "company_name_ja": record["company_name_ja"],
                "employee_scope": record["employee_scope"],
                "classification": record.get("classification"),
                "evidence_status": record["evidence_status"],
                "as_of": record["as_of"],
                "frequency_per_year": bonus.get("frequency_per_year"),
                "annual_months": bonus.get("annual_months"),
                "primary_source_count": len(record["sources"]),
            }
        )

    return {
        "summary_generated_on": snapshot["as_of"],
        "input_snapshot": str(input_path.relative_to(BASE_DIR)),
        "snapshot_as_of": snapshot["as_of"],
        "universe": {
            "source_file": "nikkei225_companies.yaml",
            "mutation_policy": "frozen",
            "universe_changed": False,
        },
        "record_count": len(records),
        "evidence_status_counts": dict(sorted(status_counts.items())),
        "classification_counts": dict(sorted(classification_counts.items())),
        "confirmed_or_partial_count": sum(
            status_counts[name] for name in ("confirmed", "partially_confirmed")
        ),
        "explicit_point_months_count": len(point_values),
        "explicit_point_months_average": (
            round(statistics.mean(point_values), 2) if point_values else None
        ),
        "records": record_rows,
    }


def latest_snapshot(data_dir: Path = DATA_DIR) -> Path:
    candidates = sorted(data_dir.glob("verified_bonus_facts_*.yaml"))
    if not candidates:
        raise ValidationError(f"no verified snapshot found under {data_dir}")
    return candidates[-1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="verified snapshot YAML")
    parser.add_argument("--output", type=Path, default=SUMMARY_FILE)
    parser.add_argument(
        "--check", action="store_true", help="validate only; do not write summary"
    )
    parser.add_argument(
        "--universe", type=Path, default=UNIVERSE_FILE, help="frozen universe YAML"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input or latest_snapshot()
    snapshot = load_yaml(input_path)
    universe_codes = load_universe_codes(args.universe)
    records = validate_snapshot(snapshot, universe_codes)
    summary = build_summary(snapshot, records, input_path)

    if args.check:
        print(
            f"PASS: {len(records)} records validated; "
            f"universe remains frozen ({len(universe_codes)} codes)"
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False, allow_unicode=True)
    print(f"Wrote verified summary for {len(records)} records: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
