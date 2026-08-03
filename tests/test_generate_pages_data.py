from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from company_estimates import validate_company_estimation_model  # noqa: E402
from generate_pages_data import build_public_payload, render_json  # noqa: E402


def survey():
    return {
        "as_of": "2026-08-02",
        "methodology": {"source_first_policy": "first"},
        "research_pipeline": ["discover", "publish"],
        "required_channels": [
            "company_official",
            "labor_union_official",
            "edinet",
            "jpx_tdnet",
        ],
        "benchmark_channels": ["benchmark"],
        "discovery_channels": ["discovery"],
        "source_registry": [
            {"id": "company_official", "name_ja": "会社公式", "tier": "primary_company", "priority": 1},
            {"id": "labor_union_official", "name_ja": "労組公式", "tier": "primary_collective", "priority": 2},
            {"id": "edinet", "name_ja": "EDINET", "tier": "official_disclosure", "priority": 3},
            {"id": "jpx_tdnet", "name_ja": "TDnet", "tier": "official_disclosure", "priority": 4},
            {"id": "benchmark", "name_ja": "統計", "tier": "official_benchmark", "priority": 5},
            {"id": "discovery", "name_ja": "探索", "tier": "discovery_only", "priority": 6},
        ],
    }


def quantitative():
    return {
        "as_of": "2026-08-02",
        "methodology": {"comparability_policy": "separate populations"},
        "benchmarks": [
            {
                "id": "benchmark-final",
                "source_id": "benchmark",
                "publisher": "公的機関",
                "title": "最終集計",
                "period": "2026-summer",
                "published_at": "2026-07-01",
                "release_status": "final",
                "aggregation": "company_average",
                "scope": "対象企業",
                "metric": "settlement_amount",
                "value": 100,
                "unit": "yen",
                "previous_value": 90,
                "change_value": 11.11,
                "change_unit": "percent",
                "sample": {"organizations": 10, "workers": None},
                "source_url": "https://example.com/final",
                "note": "final",
            },
            {
                "id": "benchmark-first",
                "source_id": "benchmark",
                "publisher": "公的機関",
                "title": "第1回集計",
                "period": "2026-summer",
                "published_at": "2026-06-01",
                "release_status": "first",
                "aggregation": "worker_weighted_average",
                "scope": "対象企業",
                "metric": "settlement_amount",
                "value": 95,
                "unit": "yen",
                "previous_value": 90,
                "change_value": 5.56,
                "change_unit": "percent",
                "sample": {"organizations": 5, "workers": 1000},
                "source_url": "https://example.com/first",
                "note": "first",
            },
        ],
    }


def record(code: str):
    return {
        "stock_code": code,
        "company_name_ja": f"会社{code}",
        "subject": "employees",
        "employee_scope": "対象",
        "classification": "hybrid",
        "evidence_status": "confirmed",
        "as_of": "2026-08-02",
        "bonus": {
            "frequency_per_year": 2,
            "annual_months": None,
            "pool_basis": "業績",
            "allocation_logic": "評価",
            "base_salary_link": None,
        },
        "notes": [],
        "sources": [
            {"type": "company_official", "title": "公式", "url": "https://example.com"}
        ],
    }


def hypothesis(code: str, central: float):
    return {
        "stock_code": code,
        "company_name_ja": f"会社{code}",
        "target": "annual_bonus_months",
        "estimate": {
            "minimum": central - 0.5,
            "central": central,
            "maximum": central + 0.5,
            "unit": "base_salary_months",
        },
        "classification_hypothesis": "hybrid",
        "frequency_per_year_hypothesis": 2,
        "confidence": {"level": "medium", "score": 0.5},
        "method": "legacy_prior_with_uncertainty",
        "basis": [{"type": "legacy_prior", "statement": "prior", "reference": f"data#{code}"}],
        "assumptions": ["assumption"],
        "falsifiers": ["falsifier"],
        "not_for_verified_aggregate": True,
    }


def model(codes):
    payload = {
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
        "mechanisms": {
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
        },
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
                "amount_conversion": {
                    "status": "unavailable",
                    "amount_sample_id": "fixture-sector-1:amount",
                    "months_sample_id": "fixture-sector-1:months",
                    "matched_population": False,
                    "aggregation": "worker_weighted_average",
                    "reason": "different respondent samples",
                },
                "source_url": "https://example.com/rengo.pdf",
                "company_codes": list(codes),
            }
        },
    }
    return validate_company_estimation_model(payload, set(codes))


def build(snapshot, records, companies):
    codes = set(companies)
    hypotheses = {
        code: hypothesis(code, 8.0 if code == "6146" else 6.0)
        for code in codes
    }
    return build_public_payload(
        snapshot,
        records,
        ROOT / "data" / "verified_bonus_facts_2026-08-02.yaml",
        companies,
        survey(),
        ROOT / "data" / "source_survey_2026-08-02.yaml",
        quantitative(),
        ROOT / "data" / "quantitative_benchmarks_2026-08-02.yaml",
        hypotheses,
        ROOT / "data" / "bonus_hypotheses_2026-08-02.yaml",
        model(codes),
        ROOT / "data" / "company_estimation_model_2026-08-02.yaml",
    )


class PagesTests(unittest.TestCase):
    def test_projection_quantifies_every_company(self):
        snapshot = {
            "as_of": "2026-08-02",
            "universe": {
                "source_file": "nikkei225_bonus_survey_2024_en.yaml",
                "mutation_policy": "frozen",
            },
        }
        payload = build(
            snapshot,
            [record("6146")],
            {"6146": "ディスコ", "7203": "トヨタ"},
        )
        self.assertEqual(payload["schema_version"], 5)
        self.assertEqual(payload["summary"]["record_count"], 2)
        self.assertEqual(payload["summary"]["quantified_company_count"], 2)
        self.assertEqual(payload["summary"]["quantitative_benchmark_count"], 2)
        self.assertEqual(len(payload["sector_anchors"]), 1)
        first = payload["records"][0]
        self.assertIn("estimate", first)
        self.assertGreater(first["estimate"]["months"]["central"], 5.44)
        self.assertIn("mechanism", first["estimate"])
        queued = payload["records"][1]
        self.assertEqual(queued["survey"]["stage"], "queued")
        self.assertEqual(queued["estimate"]["amount_status"], "unavailable")
        self.assertIsNone(queued["estimate"]["amount_yen"])
        self.assertTrue(render_json(payload).endswith("\n"))

    def test_research_coverage_remains_separate_from_quantification(self):
        snapshot = {
            "as_of": "2026-08-02",
            "universe": {
                "source_file": "nikkei225_bonus_survey_2024_en.yaml",
                "mutation_policy": "frozen",
            },
        }
        payload = build(snapshot, [], {"7203": "トヨタ"})
        self.assertEqual(payload["summary"]["required_channel_count"], 4)
        self.assertEqual(payload["summary"]["research_coverage_ratio"], 0)
        self.assertEqual(payload["summary"]["quantified_company_count"], 1)


if __name__ == "__main__":
    unittest.main()
