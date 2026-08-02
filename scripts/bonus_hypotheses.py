#!/usr/bin/env python3
"""Validate bonus hypotheses kept separate from verified facts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from generate_verified_bonus_summary import (
    ALLOWED_CLASSIFICATIONS,
    BASE_DIR,
    DATA_DIR,
    ValidationError,
    normalise_code,
    parse_iso_date,
)

ALLOWED_CONFIDENCE_LEVELS = {"low", "medium", "high"}
ALLOWED_BASIS_TYPES = {
    "verified_fact",
    "legacy_prior",
    "sector_prior",
    "calculation",
}
DEFAULT_PATTERN = "bonus_hypotheses_*.yaml"


def latest_hypothesis(data_dir: Path = DATA_DIR) -> Path:
    candidates = sorted(data_dir.glob(DEFAULT_PATTERN))
    if not candidates:
        raise ValidationError(f"no hypothesis snapshot found under {data_dir}")
    return candidates[-1]


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} is required")
    return value.strip()


def _text_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{field_name} must be a non-empty list")
    return [_required_text(item, f"{field_name}[{index}]") for index, item in enumerate(value)]


def validate_hypotheses(
    payload: Any, universe_codes: set[str]
) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValidationError("hypothesis snapshot must be an object")
    if payload.get("schema_version") != 1:
        raise ValidationError("hypothesis schema_version must be 1")
    parse_iso_date(payload.get("as_of"), "hypothesis.as_of")

    methodology = payload.get("methodology")
    if not isinstance(methodology, dict):
        raise ValidationError("hypothesis methodology is required")
    for key in (
        "purpose",
        "separation_policy",
        "interval_policy",
        "confidence_policy",
        "aggregation_policy",
    ):
        _required_text(methodology.get(key), f"hypothesis.methodology.{key}")

    estimates = payload.get("estimates")
    if not isinstance(estimates, list) or not estimates:
        raise ValidationError("hypothesis estimates must be a non-empty list")

    validated: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(estimates):
        prefix = f"hypothesis.estimates[{index}]"
        if not isinstance(item, dict):
            raise ValidationError(f"{prefix} must be an object")

        code = normalise_code(item.get("stock_code"))
        if code not in universe_codes:
            raise ValidationError(f"{prefix}.stock_code {code} is outside the frozen universe")
        if code in validated:
            raise ValidationError(f"duplicate hypothesis stock_code: {code}")
        company_name = _required_text(item.get("company_name_ja"), f"{prefix}.company_name_ja")
        if item.get("target") != "annual_bonus_months":
            raise ValidationError(f"{prefix}.target must be annual_bonus_months")

        estimate = item.get("estimate")
        if not isinstance(estimate, dict):
            raise ValidationError(f"{prefix}.estimate must be an object")
        if estimate.get("unit") != "base_salary_months":
            raise ValidationError(f"{prefix}.estimate.unit must be base_salary_months")
        values = []
        for key in ("minimum", "central", "maximum"):
            value = estimate.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValidationError(f"{prefix}.estimate.{key} must be numeric")
            value = float(value)
            if not 0 < value <= 24:
                raise ValidationError(f"{prefix}.estimate.{key} must be within (0, 24]")
            values.append(value)
        if not values[0] <= values[1] <= values[2]:
            raise ValidationError(
                f"{prefix}.estimate must satisfy minimum <= central <= maximum"
            )

        classification = item.get("classification_hypothesis")
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise ValidationError(f"{prefix}.classification_hypothesis is invalid")
        frequency = item.get("frequency_per_year_hypothesis")
        if not isinstance(frequency, int) or isinstance(frequency, bool):
            raise ValidationError(f"{prefix}.frequency_per_year_hypothesis must be integer")
        if not 1 <= frequency <= 12:
            raise ValidationError(
                f"{prefix}.frequency_per_year_hypothesis must be between 1 and 12"
            )

        confidence = item.get("confidence")
        if not isinstance(confidence, dict):
            raise ValidationError(f"{prefix}.confidence must be an object")
        if confidence.get("level") not in ALLOWED_CONFIDENCE_LEVELS:
            raise ValidationError(f"{prefix}.confidence.level is invalid")
        score = confidence.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValidationError(f"{prefix}.confidence.score must be numeric")
        if not 0 <= float(score) <= 1:
            raise ValidationError(f"{prefix}.confidence.score must be within [0, 1]")

        _required_text(item.get("method"), f"{prefix}.method")
        basis = item.get("basis")
        if not isinstance(basis, list) or not basis:
            raise ValidationError(f"{prefix}.basis must be a non-empty list")
        for basis_index, entry in enumerate(basis):
            basis_prefix = f"{prefix}.basis[{basis_index}]"
            if not isinstance(entry, dict):
                raise ValidationError(f"{basis_prefix} must be an object")
            if entry.get("type") not in ALLOWED_BASIS_TYPES:
                raise ValidationError(f"{basis_prefix}.type is invalid")
            _required_text(entry.get("statement"), f"{basis_prefix}.statement")
            _required_text(entry.get("reference"), f"{basis_prefix}.reference")

        _text_list(item.get("assumptions"), f"{prefix}.assumptions")
        _text_list(item.get("falsifiers"), f"{prefix}.falsifiers")
        if item.get("not_for_verified_aggregate") is not True:
            raise ValidationError(f"{prefix}.not_for_verified_aggregate must be true")

        result = dict(item)
        result["stock_code"] = code
        result["company_name_ja"] = company_name
        validated[code] = result

    missing = sorted(universe_codes - set(validated))
    extras = sorted(set(validated) - universe_codes)
    if missing or extras:
        raise ValidationError(
            "hypothesis coverage must exactly match frozen universe; "
            f"missing={missing}, extras={extras}"
        )
    return validated


def relative_path(path: Path, root: Path = BASE_DIR) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
