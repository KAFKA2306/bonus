from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from company_estimates import (  # noqa: E402
    build_company_estimates,
    validate_company_estimation_model,
)
from generate_verified_bonus_summary import ValidationError  # noqa: E402


def mechanism_config():
    return {
        "formula_linked_performance": {
            "label_ja": "算式明示型・業績連動",
            "broad_classification": "performance_linked",
            "upside_profile": "very_high",
            "upside_score": 1.0,
            "minimum_adjustment_months": 0.0,
            "central_adjustment_months": 0.35,
            "maximum_adjustment_months": 1.3,
        },
        "nonformula_performance": {
            "label_ja": "算式非開示型・業績連動",
            "broad_classification": "performance_linked",
            "upside_profile": "high",
            "upside_score": 0.8,
            "minimum_adjustment_months": 0.0,
            "central_adjustment_months": 0.2,
            "maximum_adjustment_months": 0.8,
        },
        "hybrid": {
            "label_ja": "固定＋業績連動ハイブリッド",
            "broad_classification": "hybrid",
            "upside_profile": "medium",
            "upside_score": 0.55,
            "minimum_adjustment_months": 0.0,
            "central_adjustment_months": 0.05,
            "maximum_adjustment_months": 0.3,
        },
        "discretionary": {
            "label_ja": "労使妥結・総合判断",
            "broad_classification": "discretionary",
            "upside_profile": "low",
            "upside_score": 0.25,
            "minimum_adjustment_months": 0.0,
            "central_adjustment_months": 0.0,
            "maximum_adjustment_months": -0.2,
        },
        "base_salary_linked": {
            "label_ja": "固定・基本給連動",
            "broad_classification": "base_salary_linked",
            "upside_profile": "low",
            "upside_score": 0.15,
            "minimum_adjustment_months": 0.0,
            "central_adjustment_months": 0.0,
            "maximum_adjustment_months": -0.3,
        },
    }


def model(codes=None):
    codes = codes or ["6146"]
    return {
        "schema_version": 1,
        "as_of": "2026-08-02",
        "methodology": {
            "purpose": "quantify",
            "model_type": "empirical_bayes_shrinkage_with_mechanism_upside",
            "central_formula": "weighted plus mechanism",
            "weight_formula": "confidence",
            "interval_formula": "weighted interval plus mechanism",
            "amount_formula": "sector or company-base projection",
            "verified_override_policy": "verified values override model",
            "disclosure_policy": "show assumptions",
        },
        "parameters": {
            "base_company_weight": 0.45,
            "confidence_multiplier": 0.50,
            "verified_evidence_bonus": 0.15,
            "sector_only_penalty": 0.05,
            "minimum_company_weight": 0.55,
            "maximum_company_weight": 0.90,
            "minimum_sector_band_months": 0.35,
        },
        "mechanisms": mechanism_config(),
        "default_mechanism_by_classification": {
            "performance_linked": "nonformula_performance",
            "hybrid": "hybrid",
            "discretionary": "discretionary",
            "base_salary_linked": "base_salary_linked",
        },
        "company_overrides": {},
        "sectors": {
            "manufacturing": {
                "name_ja": "製造業",
                "response_months": 5.44,
                "demand_months": 5.62,
                "previous_months": 5.45,
                "response_amount_yen": 1854847,
                "previous_amount_yen": 1739443,
                "sample_months": {"organizations": 1928, "workers": 1449045},
                "sample_amount": {"organizations": 1003, "workers": 739115},
                "source_url": "https://example.com/rengo.pdf",
                "company_codes": codes,
            }
        },
    }


def hypothesis(code="6146", classification="hybrid", central=10.5):
    return {
        "stock_code": code,
        "company_name_ja": f"会社{code}",
        "target": "annual_bonus_months",
        "estimate": {
            "minimum": central - 1.5,
            "central": central,
            "maximum": central + 1.5,
            "unit": "base_salary_months",
        },
        "classification_hypothesis": classification,
        "frequency_per_year_hypothesis": 2,
        "confidence": {"level": "medium", "score": 0.58},
        "method": "verified_and_legacy_prior",
        "basis": [
            {
                "type": "legacy_prior",
                "statement": "legacy",
                "reference": f"data/legacy#{code}",
            }
        ],
        "assumptions": ["assumption"],
        "falsifiers": ["falsifier"],
        "not_for_verified_aggregate": True,
    }


class CompanyEstimateTests(unittest.TestCase):
    def test_builds_quantified_estimate_with_verified_floor(self):
        validated = validate_company_estimation_model(model(), {"6146"})
        records = [
            {
                "stock_code": "6146",
                "evidence_status": "confirmed",
                "classification": "hybrid",
                "bonus": {
                    "frequency_per_year": 4,
                    "annual_months": {"kind": "minimum", "value": 4.0},
                },
            }
        ]
        result = build_company_estimates(
            {"6146": hypothesis()}, records, validated
        )["6146"]
        self.assertEqual(result["status"], "estimated_with_verified_bound")
        self.assertGreater(result["months"]["central"], 5.44)
        self.assertGreaterEqual(result["months"]["minimum"], 4.0)
        self.assertEqual(result["frequency_per_year"], 4)
        self.assertEqual(result["mechanism"]["upside_profile"], "medium")
        self.assertAlmostEqual(
            result["weights"]["company_prior"]
            + result["weights"]["sector_actual"],
            1.0,
            places=3,
        )

    def test_point_value_is_treated_as_verified_numeric(self):
        validated = validate_company_estimation_model(model(), {"6146"})
        records = [
            {
                "stock_code": "6146",
                "evidence_status": "confirmed",
                "classification": "hybrid",
                "bonus": {
                    "frequency_per_year": 2,
                    "annual_months": {"kind": "point", "value": 6.15},
                },
            }
        ]
        result = build_company_estimates(
            {"6146": hypothesis()}, records, validated
        )["6146"]
        self.assertEqual(result["status"], "verified_numeric")
        self.assertEqual(result["months"], {"minimum": 6.15, "central": 6.15, "maximum": 6.15})

    def test_official_company_override_beats_sector_model(self):
        payload = model(["9022"])
        payload["company_overrides"] = {
            "9022": {
                "classification": "discretionary",
                "mechanism": "discretionary",
                "formula_disclosure": "not_applicable",
                "amount_policy": "project_from_official_seasonal_base",
                "frequency_per_year": 2,
                "source_url": "https://example.com/jr",
                "source_note": "union settlement",
                "official_annual_months": {
                    "period": "2025",
                    "value": 6.15,
                    "source_url": "https://example.com/jr-year",
                    "note": "summer plus year-end",
                },
                "latest_seasonal": {
                    "period": "2026-summer",
                    "months": 3.1,
                    "amount_yen": 1082800,
                    "model_age": 35,
                    "source_url": "https://example.com/jr-summer",
                    "note": "official model amount",
                },
            }
        }
        validated = validate_company_estimation_model(payload, {"9022"})
        result = build_company_estimates(
            {"9022": hypothesis("9022", "base_salary_linked", 3.0)},
            [],
            validated,
        )["9022"]
        self.assertEqual(result["status"], "verified_numeric")
        self.assertEqual(result["classification"], "discretionary")
        self.assertEqual(result["mechanism"]["label_ja"], "労使妥結・総合判断")
        self.assertEqual(result["mechanism"]["upside_profile"], "low")
        self.assertEqual(result["months"]["central"], 6.15)
        self.assertEqual(result["amount_method"], "official_company_base_projection")
        self.assertGreater(result["amount_yen"]["central"], 2_000_000)

    def test_nonformula_performance_expands_upside_without_salary_inference(self):
        payload = model(["8035"])
        payload["company_overrides"] = {
            "8035": {
                "classification": "performance_linked",
                "mechanism": "nonformula_performance",
                "formula_disclosure": "not_disclosed",
                "amount_policy": "sector_implied",
                "source_url": "https://example.com/tel",
                "source_note": "performance bonus",
            }
        }
        validated = validate_company_estimation_model(payload, {"8035"})
        result = build_company_estimates(
            {"8035": hypothesis("8035", "performance_linked", 6.0)},
            [],
            validated,
        )["8035"]
        self.assertEqual(result["mechanism"]["upside_profile"], "high")
        self.assertEqual(result["mechanism"]["formula_disclosure"], "not_disclosed")
        self.assertGreater(result["months"]["maximum"], 7.0)
        self.assertEqual(result["amount_method"], "sector_implied")
        self.assertIn("業種平均基本月額", result["amount_caution"])

    def test_rejects_incomplete_sector_coverage(self):
        with self.assertRaises(ValidationError):
            validate_company_estimation_model(model(), {"6146", "7203"})

    def test_rejects_duplicate_sector_assignment(self):
        payload = model()
        payload["sectors"]["other"] = {
            **payload["sectors"]["manufacturing"],
            "company_codes": ["6146"],
        }
        with self.assertRaises(ValidationError):
            validate_company_estimation_model(payload, {"6146"})

    def test_rejects_unknown_override_mechanism(self):
        payload = model()
        payload["company_overrides"] = {
            "6146": {"mechanism": "unknown", "amount_policy": "sector_implied"}
        }
        with self.assertRaises(ValidationError):
            validate_company_estimation_model(payload, {"6146"})


if __name__ == "__main__":
    unittest.main()
