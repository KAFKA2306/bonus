#!/usr/bin/env python3
"""Validate dated quantitative bonus benchmarks from official sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from generate_verified_bonus_summary import BASE_DIR, DATA_DIR, ValidationError, is_https_url, parse_iso_date

DEFAULT_PATTERN = "quantitative_benchmarks_*.yaml"
ALLOWED_RELEASE_STATUSES = {"first", "final"}
ALLOWED_AGGREGATIONS = {"worker_weighted_average", "company_average"}
ALLOWED_METRICS = {
    "annual_bonus_months",
    "annual_bonus_amount",
    "seasonal_bonus_months",
    "seasonal_bonus_amount",
    "settlement_amount",
    "request_amount",
}
ALLOWED_UNITS = {"yen", "months"}
ALLOWED_CHANGE_UNITS = {"yen", "months", "percent"}


def latest_quantitative_benchmarks(data_dir: Path = DATA_DIR) -> Path:
    candidates = sorted(data_dir.glob(DEFAULT_PATTERN))
    if not candidates:
        raise ValidationError(f"no quantitative benchmark snapshot found under {data_dir}")
    return candidates[-1]


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} is required")
    return value.strip()


def _number(value: Any, field_name: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{field_name} must be numeric")
    result = float(value)
    if positive and result <= 0:
        raise ValidationError(f"{field_name} must be positive")
    return result


def validate_quantitative_benchmarks(payload: Any, source_ids: set[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError("quantitative benchmarks must be an object")
    if payload.get("schema_version") != 1:
        raise ValidationError("quantitative benchmark schema_version must be 1")
    parse_iso_date(payload.get("as_of"), "quantitative_benchmarks.as_of")

    methodology = payload.get("methodology")
    if not isinstance(methodology, dict):
        raise ValidationError("quantitative_benchmarks.methodology is required")
    for key in ("purpose", "comparability_policy", "company_policy", "revision_policy"):
        _required_text(methodology.get(key), f"quantitative_benchmarks.methodology.{key}")

    rows = payload.get("benchmarks")
    if not isinstance(rows, list) or not rows:
        raise ValidationError("quantitative_benchmarks.benchmarks must be a non-empty list")

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(rows):
        prefix = f"quantitative_benchmarks.benchmarks[{index}]"
        if not isinstance(item, dict):
            raise ValidationError(f"{prefix} must be an object")
        row_id = _required_text(item.get("id"), f"{prefix}.id")
        if row_id in seen:
            raise ValidationError(f"duplicate quantitative benchmark id: {row_id}")
        seen.add(row_id)
        source_id = _required_text(item.get("source_id"), f"{prefix}.source_id")
        if source_id not in source_ids:
            raise ValidationError(f"{prefix}.source_id references unknown source: {source_id}")
        for key in ("publisher", "title", "period", "scope", "note"):
            _required_text(item.get(key), f"{prefix}.{key}")
        parse_iso_date(item.get("published_at"), f"{prefix}.published_at")
        if item.get("release_status") not in ALLOWED_RELEASE_STATUSES:
            raise ValidationError(f"{prefix}.release_status is invalid")
        if item.get("aggregation") not in ALLOWED_AGGREGATIONS:
            raise ValidationError(f"{prefix}.aggregation is invalid")
        if item.get("metric") not in ALLOWED_METRICS:
            raise ValidationError(f"{prefix}.metric is invalid")
        unit = item.get("unit")
        if unit not in ALLOWED_UNITS:
            raise ValidationError(f"{prefix}.unit is invalid")
        change_unit = item.get("change_unit")
        if change_unit not in ALLOWED_CHANGE_UNITS:
            raise ValidationError(f"{prefix}.change_unit is invalid")
        value = _number(item.get("value"), f"{prefix}.value", positive=True)
        previous = _number(item.get("previous_value"), f"{prefix}.previous_value", positive=True)
        change = _number(item.get("change_value"), f"{prefix}.change_value")
        if change_unit == "percent":
            expected = round((value / previous - 1) * 100, 2)
            if abs(expected - change) > 0.02:
                raise ValidationError(f"{prefix}.change_value does not match percentage change")
        else:
            if change_unit != unit:
                raise ValidationError(f"{prefix}.change_unit must match unit unless percent")
            if abs((value - previous) - change) > 0.01:
                raise ValidationError(f"{prefix}.change_value does not match value difference")
        sample = item.get("sample")
        if not isinstance(sample, dict):
            raise ValidationError(f"{prefix}.sample must be an object")
        organizations = sample.get("organizations")
        if not isinstance(organizations, int) or isinstance(organizations, bool) or organizations < 1:
            raise ValidationError(f"{prefix}.sample.organizations must be a positive integer")
        workers = sample.get("workers")
        if workers is not None and (not isinstance(workers, int) or isinstance(workers, bool) or workers < 1):
            raise ValidationError(f"{prefix}.sample.workers must be null or a positive integer")
        if not is_https_url(item.get("source_url")):
            raise ValidationError(f"{prefix}.source_url must be https")
        validated.append(dict(item))

    return {**payload, "methodology": dict(methodology), "benchmarks": validated}


def relative_path(path: Path, root: Path = BASE_DIR) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
