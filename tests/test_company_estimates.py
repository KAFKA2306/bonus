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


def model():
    return {
        "schema_version": 1,
        "as_of": "2026-08-02",
        "methodology": {
            "purpose": "quantify",
            "model_type": "empirical_bayes_shrinkage",
            "central_formula": "weighted",
            "weight_formula": "confidence",
            "interval_formula": "weighted interval",
            "amount_formula": "sector amount per month",
            "verified_override_policy": "verified floor",
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
                "company_codes": ["6146"],
            }
        },
    }


def hypothesis():
    return {
        "stock_code": "6146",
        "company_name_ja": "ディスコ",
        "target": "annual_bonus_months",
        "estimate": {
            "minimum": 9.0,
            "central": 10.5,
            "maximum": 12.0,
            "unit": "base_salary_months",
        },
        "classification_hypothesis": "hybrid",
        "frequency_per_year_hypothesis": 4,
        "confidence": {"level": "medium", "score": 0.58},
        "method": "verified_and_legacy_prior",
        "basis": [
            {
                "type": "legacy_prior",
                "statement": "legacy",
                "reference": "data/legacy#6146",
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
        result = build_company_estimates({"6146": hypothesis()}, records, validated)["6146"]
        self.assertEqual(result["status"], "estimated_with_verified_bound")
        self.assertGreater(result["months"]["central"], 5.44)
        self.assertLess(result["months"]["central"], 10.5)
        self.assertGreaterEqual(result["months"]["minimum"], 4.0)
        self.assertEqual(result["frequency_per_year"], 4)
        self.assertGreater(result["amount_yen"]["central"], 0)
        self.assertAlmostEqual(
            result["weights"]["company_prior"] + result["weights"]["sector_actual"],
            1.0,
            places=3,
        )

    def test_rejects_incomplete_sector_coverage(self):
        payload = model()
        with self.assertRaises(ValidationError):
            validate_company_estimation_model(payload, {"6146", "7203"})

    def test_rejects_duplicate_sector_assignment(self):
        payload = model()
        payload["sectors"]["other"] = {
            **payload["sectors"]["manufacturing"],
            "company_codes": ["6146"],
        }
        with self.assertRaises(ValidationError):
            validate_company_estimation_model(payload, {"6146"})


if __name__ == "__main__":
    unittest.main()
