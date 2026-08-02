#!/usr/bin/env python3
"""Validate and apply the quantified company bonus estimation model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from generate_verified_bonus_summary import (
    BASE_DIR,
    DATA_DIR,
    ValidationError,
    is_https_url,
    normalise_code,
    parse_iso_date,
)

DEFAULT_PATTERN = "company_estimation_model_*.yaml"
REQUIRED_METHOD_KEYS = (
    "purpose",
    "model_type",
    "central_formula",
    "weight_formula",
    "interval_formula",
    "amount_formula",
    "verified_override_policy",
    "disclosure_policy",
)
REQUIRED_PARAMETER_KEYS = (
    "base_company_weight",
    "confidence_multiplier",
    "verified_evidence_bonus",
    "sector_only_penalty",
    "minimum_company_weight",
    "maximum_company_weight",
    "minimum_sector_band_months",
)


def latest_company_estimation_model(data_dir: Path = DATA_DIR) -> Path:
    candidates = sorted(data_dir.glob(DEFAULT_PATTERN))
    if not candidates:
        raise ValidationError(f"no company estimation model found under {data_dir}")
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


def _positive_sample(value: Any, field_name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    result: dict[str, int] = {}
    for key in ("organizations", "workers"):
        item = value.get(key)
        if not isinstance(item, int) or isinstance(item, bool) or item < 1:
            raise ValidationError(f"{field_name}.{key} must be a positive integer")
        result[key] = item
    return result


def validate_company_estimation_model(
    payload: Any, universe_codes: set[str]
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError("company estimation model must be an object")
    if payload.get("schema_version") != 1:
        raise ValidationError("company estimation model schema_version must be 1")
    parse_iso_date(payload.get("as_of"), "company_estimation_model.as_of")

    methodology = payload.get("methodology")
    if not isinstance(methodology, dict):
        raise ValidationError("company_estimation_model.methodology is required")
    for key in REQUIRED_METHOD_KEYS:
        _required_text(methodology.get(key), f"company_estimation_model.methodology.{key}")

    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValidationError("company_estimation_model.parameters is required")
    missing_parameters = set(REQUIRED_PARAMETER_KEYS) - set(parameters)
    if missing_parameters:
        raise ValidationError(f"missing company estimation parameters: {sorted(missing_parameters)}")
    validated_parameters = {
        key: _number(parameters.get(key), f"company_estimation_model.parameters.{key}")
        for key in REQUIRED_PARAMETER_KEYS
    }
    if not 0 <= validated_parameters["minimum_company_weight"] <= 1:
        raise ValidationError("minimum_company_weight must be within [0, 1]")
    if not 0 <= validated_parameters["maximum_company_weight"] <= 1:
        raise ValidationError("maximum_company_weight must be within [0, 1]")
    if validated_parameters["minimum_company_weight"] > validated_parameters["maximum_company_weight"]:
        raise ValidationError("minimum_company_weight cannot exceed maximum_company_weight")
    if validated_parameters["minimum_sector_band_months"] <= 0:
        raise ValidationError("minimum_sector_band_months must be positive")

    sectors = payload.get("sectors")
    if not isinstance(sectors, dict) or not sectors:
        raise ValidationError("company_estimation_model.sectors must be a non-empty object")

    validated_sectors: dict[str, dict[str, Any]] = {}
    code_to_sector: dict[str, str] = {}
    for sector_id, item in sectors.items():
        prefix = f"company_estimation_model.sectors.{sector_id}"
        if not isinstance(sector_id, str) or not sector_id:
            raise ValidationError("sector id must be non-empty text")
        if not isinstance(item, dict):
            raise ValidationError(f"{prefix} must be an object")
        name = _required_text(item.get("name_ja"), f"{prefix}.name_ja")
        response_months = _number(item.get("response_months"), f"{prefix}.response_months", positive=True)
        demand_months = _number(item.get("demand_months"), f"{prefix}.demand_months", positive=True)
        previous_months = _number(item.get("previous_months"), f"{prefix}.previous_months", positive=True)
        response_amount = _number(item.get("response_amount_yen"), f"{prefix}.response_amount_yen", positive=True)
        previous_amount = _number(item.get("previous_amount_yen"), f"{prefix}.previous_amount_yen", positive=True)
        sample_months = _positive_sample(item.get("sample_months"), f"{prefix}.sample_months")
        sample_amount = _positive_sample(item.get("sample_amount"), f"{prefix}.sample_amount")
        source_url = item.get("source_url")
        if not is_https_url(source_url):
            raise ValidationError(f"{prefix}.source_url must be https")
        codes = item.get("company_codes")
        if not isinstance(codes, list) or not codes:
            raise ValidationError(f"{prefix}.company_codes must be a non-empty list")
        normalized_codes: list[str] = []
        for index, code in enumerate(codes):
            normalized = normalise_code(code)
            if normalized not in universe_codes:
                raise ValidationError(f"{prefix}.company_codes[{index}] is outside the frozen universe")
            if normalized in code_to_sector:
                raise ValidationError(
                    f"company {normalized} is assigned to both {code_to_sector[normalized]} and {sector_id}"
                )
            code_to_sector[normalized] = sector_id
            normalized_codes.append(normalized)
        validated_sectors[sector_id] = {
            **item,
            "name_ja": name,
            "response_months": response_months,
            "demand_months": demand_months,
            "previous_months": previous_months,
            "response_amount_yen": response_amount,
            "previous_amount_yen": previous_amount,
            "sample_months": sample_months,
            "sample_amount": sample_amount,
            "company_codes": normalized_codes,
        }

    missing = sorted(universe_codes - set(code_to_sector))
    extras = sorted(set(code_to_sector) - universe_codes)
    if missing or extras:
        raise ValidationError(
            "company estimation sectors must exactly cover the frozen universe; "
            f"missing={missing}, extras={extras}"
        )

    return {
        **payload,
        "methodology": dict(methodology),
        "parameters": validated_parameters,
        "sectors": validated_sectors,
        "code_to_sector": code_to_sector,
    }


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _confidence_level(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def _verified_numeric_override(
    estimate: dict[str, float], record: dict[str, Any] | None
) -> tuple[dict[str, float], str | None]:
    if not record:
        return estimate, None
    annual = (record.get("bonus") or {}).get("annual_months")
    if not isinstance(annual, dict):
        return estimate, None
    kind = annual.get("kind")
    result = dict(estimate)
    if kind == "minimum" and isinstance(annual.get("value"), (int, float)):
        floor = float(annual["value"])
        result["minimum"] = max(result["minimum"], floor)
        result["central"] = max(result["central"], result["minimum"])
        result["maximum"] = max(result["maximum"], result["central"])
        return result, f"一次資料の最低{floor:g}か月を下限へ適用"
    if kind == "maximum" and isinstance(annual.get("value"), (int, float)):
        ceiling = float(annual["value"])
        result["maximum"] = min(result["maximum"], ceiling)
        result["central"] = min(result["central"], result["maximum"])
        result["minimum"] = min(result["minimum"], result["central"])
        return result, f"一次資料の最大{ceiling:g}か月を上限へ適用"
    if kind == "range" and all(isinstance(annual.get(key), (int, float)) for key in ("minimum", "maximum")):
        low, high = float(annual["minimum"]), float(annual["maximum"])
        return {"minimum": low, "central": round((low + high) / 2, 2), "maximum": high}, "一次資料の明示レンジを優先"
    if kind == "exact" and isinstance(annual.get("value"), (int, float)):
        value = float(annual["value"])
        return {"minimum": value, "central": value, "maximum": value}, "一次資料の明示値を優先"
    return estimate, None


def build_company_estimates(
    hypotheses_by_code: dict[str, dict[str, Any]],
    verified_records: list[dict[str, Any]],
    model: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    records_by_code = {item["stock_code"]: item for item in verified_records}
    parameters = model["parameters"]
    result: dict[str, dict[str, Any]] = {}

    for code, hypothesis in hypotheses_by_code.items():
        sector_id = model["code_to_sector"][code]
        sector = model["sectors"][sector_id]
        prior = hypothesis["estimate"]
        prior_score = float(hypothesis["confidence"]["score"])
        verified_structure = hypothesis["method"] == "verified_and_legacy_prior"
        sector_only = hypothesis["method"] == "sector_prior_low_confidence"
        weight = (
            parameters["base_company_weight"]
            + parameters["confidence_multiplier"] * prior_score
            + (parameters["verified_evidence_bonus"] if verified_structure else 0.0)
            - (parameters["sector_only_penalty"] if sector_only else 0.0)
        )
        weight = _clamp(
            weight,
            parameters["minimum_company_weight"],
            parameters["maximum_company_weight"],
        )
        sector_band = max(
            parameters["minimum_sector_band_months"],
            abs(sector["demand_months"] - sector["response_months"]),
        )
        estimate = {
            "minimum": round(
                weight * float(prior["minimum"])
                + (1 - weight) * (sector["response_months"] - sector_band),
                2,
            ),
            "central": round(
                weight * float(prior["central"])
                + (1 - weight) * sector["response_months"],
                2,
            ),
            "maximum": round(
                weight * float(prior["maximum"])
                + (1 - weight) * (sector["response_months"] + sector_band),
                2,
            ),
        }
        estimate, override_note = _verified_numeric_override(estimate, records_by_code.get(code))
        estimate = {key: round(max(0.1, value), 2) for key, value in estimate.items()}

        implied_monthly_base = sector["response_amount_yen"] / sector["response_months"]
        amount = {
            key: int(round(implied_monthly_base * estimate[key] / 1000) * 1000)
            for key in ("minimum", "central", "maximum")
        }
        record = records_by_code.get(code)
        verified_record = bool(record and record.get("evidence_status") in {"confirmed", "partially_confirmed"})
        confidence_score = prior_score * 0.75 + 0.20
        if verified_structure or verified_record:
            confidence_score += 0.10
        if sector_only:
            confidence_score -= 0.05
        confidence_score = round(_clamp(confidence_score, 0.25, 0.82), 2)
        amount_score = confidence_score - 0.15
        if sector["sample_amount"]["organizations"] < 10:
            amount_score -= 0.12
        elif sector["sample_amount"]["organizations"] < 50:
            amount_score -= 0.05
        amount_score = round(_clamp(amount_score, 0.15, 0.75), 2)

        status = "estimated"
        if override_note and "明示値" in override_note:
            status = "verified_numeric"
        elif override_note:
            status = "estimated_with_verified_bound"
        elif verified_record:
            status = "estimated_with_verified_structure"

        classification = (
            record.get("classification")
            if record and record.get("classification")
            else hypothesis["classification_hypothesis"]
        )
        verified_frequency = (record.get("bonus") or {}).get("frequency_per_year") if record else None
        frequency = (
            verified_frequency
            if verified_frequency is not None
            else hypothesis["frequency_per_year_hypothesis"]
        )
        basis = list(hypothesis["basis"])
        basis.append(
            {
                "type": "official_sector_actual",
                "statement": (
                    f"連合2026最終集計の{sector['name_ja']}年間一時金は"
                    f"{sector['response_months']:.2f}か月・{int(sector['response_amount_yen']):,}円。"
                ),
                "reference": sector["source_url"],
            }
        )
        assumptions = list(hypothesis["assumptions"])
        assumptions.append("旧個社調査の相対的な高低が2026年にも一定程度残る。")
        assumptions.append("参考換算額では個社の基本月額を業種平均と同じと仮定する。")
        falsifiers = list(hypothesis["falsifiers"])
        falsifiers.append("会社・労組の一次資料で年間月数または金額が推定レンジ外と確認される。")

        result[code] = {
            "status": status,
            "method": model["methodology"]["model_type"],
            "sector_id": sector_id,
            "sector_name_ja": sector["name_ja"],
            "months": estimate,
            "amount_yen": amount,
            "frequency_per_year": frequency,
            "classification": classification,
            "confidence": {
                "score": confidence_score,
                "level": _confidence_level(confidence_score),
                "amount_score": amount_score,
            },
            "weights": {
                "company_prior": round(weight, 3),
                "sector_actual": round(1 - weight, 3),
            },
            "anchors": {
                "company_prior_months": {
                    "minimum": float(prior["minimum"]),
                    "central": float(prior["central"]),
                    "maximum": float(prior["maximum"]),
                },
                "sector_actual_months": sector["response_months"],
                "sector_demand_months": sector["demand_months"],
                "sector_previous_months": sector["previous_months"],
                "sector_actual_amount_yen": int(sector["response_amount_yen"]),
                "sector_implied_monthly_base_yen": int(round(implied_monthly_base)),
                "sector_sample_months": sector["sample_months"],
                "sector_sample_amount": sector["sample_amount"],
                "source_url": sector["source_url"],
            },
            "override_note": override_note,
            "basis": basis,
            "assumptions": assumptions,
            "falsifiers": falsifiers,
            "amount_caution": "業種平均基本月額を仮定した参考換算であり、個社の実支給額ではない。月数と金額の標本も一致しない。",
        }

    return result


def relative_path(path: Path, root: Path = BASE_DIR) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
