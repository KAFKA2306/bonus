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
REQUIRED_MECHANISM_KEYS = (
    "label_ja",
    "broad_classification",
    "upside_profile",
    "upside_score",
    "minimum_adjustment_months",
    "central_adjustment_months",
    "maximum_adjustment_months",
)
ALLOWED_UPSIDE_PROFILES = {"very_high", "high", "medium", "low"}
ALLOWED_FORMULA_DISCLOSURE = {"explicit", "not_disclosed", "not_applicable", "unknown"}
ALLOWED_AMOUNT_POLICIES = {"sector_implied", "project_from_official_seasonal_base"}


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


def _validate_amount_conversion(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    amount_sample_id = _required_text(value.get("amount_sample_id"), f"{field_name}.amount_sample_id")
    months_sample_id = _required_text(value.get("months_sample_id"), f"{field_name}.months_sample_id")
    aggregation = _required_text(value.get("aggregation"), f"{field_name}.aggregation")
    reason = _required_text(value.get("reason"), f"{field_name}.reason")
    matched_population = value.get("matched_population")
    if not isinstance(matched_population, bool):
        raise ValidationError(f"{field_name}.matched_population must be boolean")
    if matched_population and amount_sample_id != months_sample_id:
        raise ValidationError(
            f"{field_name} cannot be matched when amount_sample_id and months_sample_id differ"
        )
    status = "matched_sample" if matched_population else "unavailable"
    if value.get("status") != status:
        raise ValidationError(f"{field_name}.status must be {status}")
    return {
        **value,
        "status": status,
        "amount_sample_id": amount_sample_id,
        "months_sample_id": months_sample_id,
        "matched_population": matched_population,
        "aggregation": aggregation,
        "reason": reason,
    }


def _validate_official_months(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    period = _required_text(value.get("period"), f"{field_name}.period")
    months = _number(value.get("value"), f"{field_name}.value", positive=True)
    if months > 24:
        raise ValidationError(f"{field_name}.value must be within (0, 24]")
    source_url = value.get("source_url")
    if not is_https_url(source_url):
        raise ValidationError(f"{field_name}.source_url must be https")
    note = _required_text(value.get("note"), f"{field_name}.note")
    return {
        **value,
        "period": period,
        "value": months,
        "source_url": source_url,
        "note": note,
    }


def _validate_seasonal_observation(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    period = _required_text(value.get("period"), f"{field_name}.period")
    months = _number(value.get("months"), f"{field_name}.months", positive=True)
    amount_yen = _number(value.get("amount_yen"), f"{field_name}.amount_yen", positive=True)
    model_age = value.get("model_age")
    if model_age is not None and (
        not isinstance(model_age, int) or isinstance(model_age, bool) or model_age < 18
    ):
        raise ValidationError(f"{field_name}.model_age must be null or an adult age")
    source_url = value.get("source_url")
    if not is_https_url(source_url):
        raise ValidationError(f"{field_name}.source_url must be https")
    note = _required_text(value.get("note"), f"{field_name}.note")
    return {
        **value,
        "period": period,
        "months": months,
        "amount_yen": int(amount_yen),
        "model_age": model_age,
        "source_url": source_url,
        "note": note,
    }


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

    mechanisms = payload.get("mechanisms")
    if not isinstance(mechanisms, dict) or not mechanisms:
        raise ValidationError("company_estimation_model.mechanisms must be a non-empty object")
    validated_mechanisms: dict[str, dict[str, Any]] = {}
    for mechanism_id, item in mechanisms.items():
        prefix = f"company_estimation_model.mechanisms.{mechanism_id}"
        if not isinstance(item, dict):
            raise ValidationError(f"{prefix} must be an object")
        for key in REQUIRED_MECHANISM_KEYS:
            if key in {"upside_score", "minimum_adjustment_months", "central_adjustment_months", "maximum_adjustment_months"}:
                continue
            _required_text(item.get(key), f"{prefix}.{key}")
        if item.get("upside_profile") not in ALLOWED_UPSIDE_PROFILES:
            raise ValidationError(f"{prefix}.upside_profile is invalid")
        upside_score = _number(item.get("upside_score"), f"{prefix}.upside_score")
        if not 0 <= upside_score <= 1:
            raise ValidationError(f"{prefix}.upside_score must be within [0, 1]")
        validated_mechanisms[mechanism_id] = {
            **item,
            "upside_score": upside_score,
            "minimum_adjustment_months": _number(
                item.get("minimum_adjustment_months"),
                f"{prefix}.minimum_adjustment_months",
            ),
            "central_adjustment_months": _number(
                item.get("central_adjustment_months"),
                f"{prefix}.central_adjustment_months",
            ),
            "maximum_adjustment_months": _number(
                item.get("maximum_adjustment_months"),
                f"{prefix}.maximum_adjustment_months",
            ),
        }

    defaults = payload.get("default_mechanism_by_classification")
    if not isinstance(defaults, dict) or not defaults:
        raise ValidationError("default_mechanism_by_classification is required")
    for classification, mechanism_id in defaults.items():
        if mechanism_id not in validated_mechanisms:
            raise ValidationError(
                f"default mechanism for {classification} references unknown mechanism {mechanism_id}"
            )

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
        amount_conversion = _validate_amount_conversion(
            item.get("amount_conversion"), f"{prefix}.amount_conversion"
        )
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
            "amount_conversion": amount_conversion,
            "company_codes": normalized_codes,
        }

    missing = sorted(universe_codes - set(code_to_sector))
    extras = sorted(set(code_to_sector) - universe_codes)
    if missing or extras:
        raise ValidationError(
            "company estimation sectors must exactly cover the frozen universe; "
            f"missing={missing}, extras={extras}"
        )

    overrides = payload.get("company_overrides", {})
    if not isinstance(overrides, dict):
        raise ValidationError("company_overrides must be an object")
    validated_overrides: dict[str, dict[str, Any]] = {}
    for raw_code, item in overrides.items():
        code = normalise_code(raw_code)
        prefix = f"company_estimation_model.company_overrides.{code}"
        if code not in universe_codes:
            raise ValidationError(f"{prefix} is outside the frozen universe")
        if not isinstance(item, dict):
            raise ValidationError(f"{prefix} must be an object")
        mechanism_id = item.get("mechanism")
        if mechanism_id not in validated_mechanisms:
            raise ValidationError(f"{prefix}.mechanism references unknown mechanism")
        formula_disclosure = item.get("formula_disclosure", "unknown")
        if formula_disclosure not in ALLOWED_FORMULA_DISCLOSURE:
            raise ValidationError(f"{prefix}.formula_disclosure is invalid")
        amount_policy = item.get("amount_policy", "sector_implied")
        if amount_policy not in ALLOWED_AMOUNT_POLICIES:
            raise ValidationError(f"{prefix}.amount_policy is invalid")
        source_url = item.get("source_url")
        if source_url is not None and not is_https_url(source_url):
            raise ValidationError(f"{prefix}.source_url must be https or null")
        annual = item.get("official_annual_months")
        seasonal = item.get("latest_seasonal")
        validated_overrides[code] = {
            **item,
            "mechanism": mechanism_id,
            "formula_disclosure": formula_disclosure,
            "amount_policy": amount_policy,
            "official_annual_months": (
                _validate_official_months(annual, f"{prefix}.official_annual_months")
                if annual is not None
                else None
            ),
            "latest_seasonal": (
                _validate_seasonal_observation(seasonal, f"{prefix}.latest_seasonal")
                if seasonal is not None
                else None
            ),
        }
        if amount_policy == "project_from_official_seasonal_base" and (
            annual is None or seasonal is None
        ):
            raise ValidationError(
                f"{prefix} requires official_annual_months and latest_seasonal for amount projection"
            )

    return {
        **payload,
        "methodology": dict(methodology),
        "parameters": validated_parameters,
        "mechanisms": validated_mechanisms,
        "default_mechanism_by_classification": dict(defaults),
        "sectors": validated_sectors,
        "code_to_sector": code_to_sector,
        "company_overrides": validated_overrides,
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
    if kind == "range" and all(
        isinstance(annual.get(key), (int, float)) for key in ("minimum", "maximum")
    ):
        low, high = float(annual["minimum"]), float(annual["maximum"])
        return {
            "minimum": low,
            "central": round((low + high) / 2, 2),
            "maximum": high,
        }, "一次資料の明示レンジを優先"
    if kind in {"point", "exact"} and isinstance(annual.get("value"), (int, float)):
        value = float(annual["value"])
        return {
            "minimum": value,
            "central": value,
            "maximum": value,
        }, "一次資料の明示値を優先"
    return estimate, None


def _apply_model_override(
    estimate: dict[str, float], override: dict[str, Any]
) -> tuple[dict[str, float], str | None]:
    annual = override.get("official_annual_months")
    if not annual:
        return estimate, None
    value = float(annual["value"])
    return {
        "minimum": value,
        "central": value,
        "maximum": value,
    }, f"{annual['period']}の会社公式年間{value:g}か月を優先"


def _amount_from_sector(
    estimate: dict[str, float], sector: dict[str, Any]
) -> tuple[dict[str, int] | None, str, str, dict[str, Any]]:
    conversion = dict(sector["amount_conversion"])
    if not conversion["matched_population"]:
        return (
            None,
            "not_estimable_from_available_samples",
            "業種の金額平均と月数平均は回答標本が異なるため、比から個社金額を算定しない。",
            {**conversion, "status": "unavailable", "monthly_base_yen": None},
        )
    if conversion["amount_sample_id"] != conversion["months_sample_id"]:
        raise ValidationError("matched amount conversion requires one shared sample id")
    implied_monthly_base = sector["response_amount_yen"] / sector["response_months"]
    amount = {
        key: int(round(implied_monthly_base * estimate[key] / 1000) * 1000)
        for key in ("minimum", "central", "maximum")
    }
    return (
        amount,
        "matched_sector_sample",
        "同一回答標本の年間金額と年間月数から得た基礎月額による参考換算。",
        {
            **conversion,
            "status": "matched_sample",
            "monthly_base_yen": int(round(implied_monthly_base)),
        },
    )


def _amount_from_official_seasonal_base(
    annual_months: dict[str, float], seasonal: dict[str, Any]
) -> tuple[dict[str, int], str, str, dict[str, Any]]:
    monthly_base = seasonal["amount_yen"] / seasonal["months"]
    central = monthly_base * annual_months["central"]
    uncertainty_rate = 0.05
    sample_id = f"company-official:{seasonal['source_url']}:{seasonal['period']}"
    return (
        {
            "minimum": int(round(central * (1 - uncertainty_rate) / 1000) * 1000),
            "central": int(round(central / 1000) * 1000),
            "maximum": int(round(central * (1 + uncertainty_rate) / 1000) * 1000),
        },
        "official_company_base_projection",
        (
            f"{seasonal['period']}の会社公式モデル額÷月数で基礎月額を推定し、別期間の公式年間月数を乗じた参考値。"
            "期間差を含むため実支給額ではない。"
        ),
        {
            "status": "company_official",
            "amount_sample_id": sample_id,
            "months_sample_id": sample_id,
            "matched_population": True,
            "aggregation": "company_model_employee",
            "reason": "company published amount and months for the same seasonal model employee",
            "monthly_base_yen": int(round(monthly_base)),
        },
    )


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
        record = records_by_code.get(code)
        override = model["company_overrides"].get(code, {})

        classification = (
            override.get("classification")
            or (record.get("classification") if record else None)
            or hypothesis["classification_hypothesis"]
        )
        mechanism_id = override.get("mechanism") or model[
            "default_mechanism_by_classification"
        ].get(classification)
        if mechanism_id not in model["mechanisms"]:
            raise ValidationError(
                f"no valid mechanism for company {code} classification {classification}"
            )
        mechanism = model["mechanisms"][mechanism_id]

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
                + (1 - weight) * (sector["response_months"] - sector_band)
                + mechanism["minimum_adjustment_months"],
                2,
            ),
            "central": round(
                weight * float(prior["central"])
                + (1 - weight) * sector["response_months"]
                + mechanism["central_adjustment_months"],
                2,
            ),
            "maximum": round(
                weight * float(prior["maximum"])
                + (1 - weight) * (sector["response_months"] + sector_band)
                + mechanism["maximum_adjustment_months"],
                2,
            ),
        }
        estimate, verified_note = _verified_numeric_override(estimate, record)
        estimate, model_note = _apply_model_override(estimate, override)
        override_note = model_note or verified_note
        estimate = {key: round(max(0.1, value), 2) for key, value in estimate.items()}

        if override.get("amount_policy") == "project_from_official_seasonal_base":
            amount, amount_method, amount_caution, amount_conversion = _amount_from_official_seasonal_base(
                estimate, override["latest_seasonal"]
            )
        else:
            amount, amount_method, amount_caution, amount_conversion = _amount_from_sector(
                estimate, sector
            )
        amount_status = "available" if amount is not None else "unavailable"

        verified_record = bool(
            record
            and record.get("evidence_status") in {"confirmed", "partially_confirmed"}
        )
        confidence_score = prior_score * 0.75 + 0.20
        if verified_structure or verified_record:
            confidence_score += 0.10
        if sector_only:
            confidence_score -= 0.05
        if override.get("source_url"):
            confidence_score += 0.05
        if override.get("official_annual_months"):
            confidence_score = max(confidence_score, 0.82)
        confidence_score = round(_clamp(confidence_score, 0.25, 0.90), 2)

        if amount is None:
            amount_score = None
        else:
            amount_score = confidence_score - 0.15
            if amount_method == "matched_sector_sample":
                if sector["sample_amount"]["organizations"] < 10:
                    amount_score -= 0.12
                elif sector["sample_amount"]["organizations"] < 50:
                    amount_score -= 0.05
            else:
                amount_score = max(amount_score, 0.65)
            amount_score = round(_clamp(amount_score, 0.15, 0.82), 2)

        status = "estimated"
        if override.get("official_annual_months") or (
            override_note and "明示値" in override_note
        ):
            status = "verified_numeric"
        elif override_note:
            status = "estimated_with_verified_bound"
        elif verified_record or override.get("source_url"):
            status = "estimated_with_verified_structure"

        verified_frequency = (
            (record.get("bonus") or {}).get("frequency_per_year") if record else None
        )
        frequency = (
            override.get("frequency_per_year")
            or verified_frequency
            or hypothesis["frequency_per_year_hypothesis"]
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
        if override.get("source_url"):
            basis.append(
                {
                    "type": "company_mechanism",
                    "statement": override.get("source_note")
                    or f"会社公式資料で{mechanism['label_ja']}の制度構造を確認。",
                    "reference": override["source_url"],
                }
            )
        if override.get("official_annual_months"):
            annual = override["official_annual_months"]
            basis.append(
                {
                    "type": "company_official_numeric",
                    "statement": annual["note"],
                    "reference": annual["source_url"],
                }
            )
        if override.get("latest_seasonal"):
            seasonal = override["latest_seasonal"]
            basis.append(
                {
                    "type": "company_official_numeric",
                    "statement": seasonal["note"],
                    "reference": seasonal["source_url"],
                }
            )

        assumptions = list(hypothesis["assumptions"])
        assumptions.append("旧個社調査の相対的な高低が2026年にも一定程度残る。")
        if amount_method == "not_estimable_from_available_samples":
            assumptions.append("金額標本と月数標本が異なるため、個社参考金額を算定しない。")
        elif amount_method == "matched_sector_sample":
            assumptions.append("同一標本から得た業種基礎月額を個社月数へ参考適用する。")
        else:
            assumptions.append("会社公式の季別モデル基礎額が年間換算にも近似利用できる。")
        falsifiers = list(hypothesis["falsifiers"])
        falsifiers.append("会社・労組の一次資料で年間月数または金額が推定レンジ外と確認される。")

        result[code] = {
            "status": status,
            "method": model["methodology"]["model_type"],
            "sector_id": sector_id,
            "sector_name_ja": sector["name_ja"],
            "months": estimate,
            "amount_yen": amount,
            "amount_status": amount_status,
            "amount_method": amount_method,
            "amount_conversion": amount_conversion,
            "frequency_per_year": frequency,
            "classification": classification,
            "mechanism": {
                "id": mechanism_id,
                "label_ja": mechanism["label_ja"],
                "upside_profile": mechanism["upside_profile"],
                "upside_score": mechanism["upside_score"],
                "formula_disclosure": override.get("formula_disclosure", "unknown"),
                "source_url": override.get("source_url"),
                "source_note": override.get("source_note"),
            },
            "official_observations": {
                "annual_months": override.get("official_annual_months"),
                "latest_seasonal": override.get("latest_seasonal"),
            },
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
                "sector_implied_monthly_base_yen": amount_conversion.get("monthly_base_yen"),
                "sector_sample_months": sector["sample_months"],
                "sector_sample_amount": sector["sample_amount"],
                "source_url": sector["source_url"],
            },
            "override_note": override_note,
            "basis": basis,
            "assumptions": assumptions,
            "falsifiers": falsifiers,
            "amount_caution": amount_caution,
        }

    return result


def relative_path(path: Path, root: Path = BASE_DIR) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
