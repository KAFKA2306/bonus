#!/usr/bin/env python3
"""Validate the source-first meta-survey used by the bonus research pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from generate_verified_bonus_summary import (
    BASE_DIR,
    DATA_DIR,
    ValidationError,
    is_https_url,
    parse_iso_date,
)

DEFAULT_PATTERN = "source_survey_*.yaml"
ALLOWED_TIERS = {
    "primary_company",
    "primary_collective",
    "official_disclosure",
    "official_benchmark",
    "discovery_only",
}
ALLOWED_SCOPES = {
    "company_specific",
    "collective_bargaining",
    "legal_disclosure",
    "timely_disclosure",
    "industry_benchmark",
    "national_benchmark",
    "discovery",
}
ALLOWED_VERIFICATION_FIELDS = {
    "employee_scope",
    "classification",
    "frequency_per_year",
    "payment_timing",
    "annual_months",
    "pool_basis",
    "allocation_logic",
    "base_salary_link",
    "settlement_amount",
    "sector_context",
    "national_context",
    "discovery_lead",
}


def latest_source_survey(data_dir: Path = DATA_DIR) -> Path:
    candidates = sorted(data_dir.glob(DEFAULT_PATTERN))
    if not candidates:
        raise ValidationError(f"no source survey found under {data_dir}")
    return candidates[-1]


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} is required")
    return value.strip()


def _text_list(value: Any, field_name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValidationError(f"{field_name} must be {qualifier}")
    return [
        _required_text(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    ]


def validate_source_survey(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError("source survey must be an object")
    if payload.get("schema_version") != 1:
        raise ValidationError("source survey schema_version must be 1")
    parse_iso_date(payload.get("as_of"), "source_survey.as_of")

    methodology = payload.get("methodology")
    if not isinstance(methodology, dict):
        raise ValidationError("source_survey.methodology is required")
    for key in (
        "purpose",
        "source_first_policy",
        "verification_policy",
        "benchmark_policy",
        "discovery_policy",
        "freshness_policy",
    ):
        _required_text(methodology.get(key), f"source_survey.methodology.{key}")

    pipeline = _text_list(payload.get("research_pipeline"), "source_survey.research_pipeline")
    if len(set(pipeline)) != len(pipeline):
        raise ValidationError("source_survey.research_pipeline contains duplicates")

    registry = payload.get("source_registry")
    if not isinstance(registry, list) or not registry:
        raise ValidationError("source_survey.source_registry must be a non-empty list")

    validated_registry: list[dict[str, Any]] = []
    registry_by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(registry):
        prefix = f"source_survey.source_registry[{index}]"
        if not isinstance(item, dict):
            raise ValidationError(f"{prefix} must be an object")
        source_id = _required_text(item.get("id"), f"{prefix}.id")
        if source_id in registry_by_id:
            raise ValidationError(f"duplicate source registry id: {source_id}")
        name = _required_text(item.get("name_ja"), f"{prefix}.name_ja")
        tier = item.get("tier")
        if tier not in ALLOWED_TIERS:
            raise ValidationError(f"{prefix}.tier is invalid")
        scope = item.get("scope")
        if scope not in ALLOWED_SCOPES:
            raise ValidationError(f"{prefix}.scope is invalid")
        priority = item.get("priority")
        if not isinstance(priority, int) or isinstance(priority, bool) or priority < 1:
            raise ValidationError(f"{prefix}.priority must be a positive integer")
        url = item.get("url")
        if url is not None and not is_https_url(url):
            raise ValidationError(f"{prefix}.url must be null or https")
        verifies = _text_list(item.get("verifies"), f"{prefix}.verifies", allow_empty=True)
        invalid_fields = sorted(set(verifies) - ALLOWED_VERIFICATION_FIELDS)
        if invalid_fields:
            raise ValidationError(f"{prefix}.verifies contains invalid fields: {invalid_fields}")
        limitations = _required_text(item.get("limitations"), f"{prefix}.limitations")
        use_when = _required_text(item.get("use_when"), f"{prefix}.use_when")
        result = {
            **item,
            "id": source_id,
            "name_ja": name,
            "verifies": verifies,
            "limitations": limitations,
            "use_when": use_when,
        }
        validated_registry.append(result)
        registry_by_id[source_id] = result

    required_channels = _text_list(
        payload.get("required_channels"), "source_survey.required_channels"
    )
    benchmark_channels = _text_list(
        payload.get("benchmark_channels"), "source_survey.benchmark_channels"
    )
    discovery_channels = _text_list(
        payload.get("discovery_channels"), "source_survey.discovery_channels"
    )
    referenced = set(required_channels + benchmark_channels + discovery_channels)
    missing = sorted(referenced - set(registry_by_id))
    if missing:
        raise ValidationError(f"source survey references unknown channel ids: {missing}")
    if len(set(required_channels)) != len(required_channels):
        raise ValidationError("source_survey.required_channels contains duplicates")

    return {
        **payload,
        "methodology": dict(methodology),
        "research_pipeline": pipeline,
        "source_registry": sorted(
            validated_registry, key=lambda item: (item["priority"], item["id"])
        ),
        "required_channels": required_channels,
        "benchmark_channels": benchmark_channels,
        "discovery_channels": discovery_channels,
    }


def relative_path(path: Path, root: Path = BASE_DIR) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
