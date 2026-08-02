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


class PagesTests(unittest.TestCase):
    def test_source_first_projection(self):
        snapshot = {
            "as_of": "2026-08-02",
            "universe": {
                "source_file": "nikkei225_bonus_survey_2024_en.yaml",
                "mutation_policy": "frozen",
            },
        }
        payload = build_public_payload(
            snapshot,
            [record("6146")],
            ROOT / "data" / "verified_bonus_facts_2026-08-02.yaml",
            {"6146": "ディスコ", "7203": "トヨタ"},
            survey(),
            ROOT / "data" / "source_survey_2026-08-02.yaml",
        )
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["summary"]["record_count"], 2)
        self.assertEqual(payload["summary"]["source_channel_count"], 6)
        first = payload["records"][0]
        self.assertNotIn("hypothesis", first)
        self.assertEqual(first["survey"]["reviewed_required_count"], 1)
        self.assertEqual(first["survey"]["next_channel_id"], "labor_union_official")
        queued = payload["records"][1]
        self.assertEqual(queued["survey"]["stage"], "queued")
        self.assertTrue(render_json(payload).endswith("\n"))

    def test_benchmark_not_counted_as_required(self):
        snapshot = {
            "as_of": "2026-08-02",
            "universe": {
                "source_file": "nikkei225_bonus_survey_2024_en.yaml",
                "mutation_policy": "frozen",
            },
        }
        payload = build_public_payload(
            snapshot,
            [],
            Path("/tmp/facts.yaml"),
            {"7203": "トヨタ"},
            survey(),
            Path("/tmp/source_survey.yaml"),
        )
        self.assertEqual(payload["summary"]["required_channel_count"], 4)
        self.assertEqual(payload["summary"]["research_coverage_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
