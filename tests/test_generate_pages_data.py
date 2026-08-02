from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

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


def build(snapshot, records, companies):
    return build_public_payload(
        snapshot,
        records,
        ROOT / "data" / "verified_bonus_facts_2026-08-02.yaml",
        companies,
        survey(),
        ROOT / "data" / "source_survey_2026-08-02.yaml",
        quantitative(),
        ROOT / "data" / "quantitative_benchmarks_2026-08-02.yaml",
    )


class PagesTests(unittest.TestCase):
    def test_source_first_projection_includes_quantitative_layer(self):
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
        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(payload["summary"]["record_count"], 2)
        self.assertEqual(payload["summary"]["source_channel_count"], 6)
        self.assertEqual(payload["summary"]["quantitative_benchmark_count"], 2)
        self.assertEqual(payload["summary"]["quantitative_final_count"], 1)
        self.assertEqual(payload["summary"]["quantitative_provisional_count"], 1)
        self.assertEqual(len(payload["quantitative_benchmarks"]), 2)
        first = payload["records"][0]
        self.assertNotIn("hypothesis", first)
        self.assertEqual(first["survey"]["reviewed_required_count"], 1)
        self.assertEqual(first["survey"]["next_channel_id"], "labor_union_official")
        queued = payload["records"][1]
        self.assertEqual(queued["survey"]["stage"], "queued")
        self.assertTrue(render_json(payload).endswith("\n"))

    def test_benchmark_not_counted_as_required_channel(self):
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
        self.assertEqual(payload["summary"]["quantitative_benchmark_count"], 2)


if __name__ == "__main__":
    unittest.main()
