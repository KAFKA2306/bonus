from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_pages_data import build_public_payload, render_json  # noqa: E402
from generate_verified_bonus_summary import latest_snapshot  # noqa: E402


def source():
    return {
        "type": "company_official",
        "title": "Official source",
        "url": "https://example.com/source",
        "retrieved_at": "2026-08-02",
        "page_or_section": "Bonus",
    }


def record(code: str, *, status: str = "confirmed", months=None):
    return {
        "stock_code": code,
        "company_name_ja": f"会社{code}",
        "subject": "employees",
        "employee_scope": "対象従業員",
        "classification": "hybrid" if status != "unknown" else None,
        "evidence_status": status,
        "as_of": "2026-08-02",
        "bonus": {
            "frequency_per_year": 2 if status != "unknown" else None,
            "annual_months": months,
            "pool_basis": "会社業績",
            "allocation_logic": "個人評価",
            "base_salary_link": None,
        },
        "notes": ["監査済み"],
        "sources": [source()],
    }


def hypothesis(code: str, central: float = 5.0):
    return {
        "stock_code": code,
        "target": "annual_bonus_months",
        "estimate": {
            "minimum": central - 0.5,
            "central": central,
            "maximum": central + 0.5,
            "unit": "base_salary_months",
        },
        "classification_hypothesis": "hybrid",
        "frequency_per_year_hypothesis": 2,
        "confidence": {"level": "medium", "score": 0.55},
        "method": "test_prior",
        "basis": [
            {"type": "legacy_prior", "statement": "旧値", "reference": "legacy"}
        ],
        "assumptions": ["前提"],
        "falsifiers": ["反証"],
        "not_for_verified_aggregate": True,
    }


class PagesDataTests(unittest.TestCase):
    def test_latest_snapshot_selects_newest_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            older = data_dir / "verified_bonus_facts_2026-07-31.yaml"
            newer = data_dir / "verified_bonus_facts_2026-08-02.yaml"
            older.write_text("old", encoding="utf-8")
            newer.write_text("new", encoding="utf-8")
            self.assertEqual(latest_snapshot(data_dir), newer)

    def test_projection_preserves_verified_fields_and_hypothesis(self):
        snapshot = {
            "as_of": "2026-08-02",
            "universe": {
                "source_file": "nikkei225_bonus_survey_2024_en.yaml",
                "mutation_policy": "frozen",
            },
            "methodology": {"primary_sources_only": True},
        }
        hypothesis_snapshot = {
            "methodology": {"separation_policy": "事実と推定を分離"}
        }
        item = record("6146")
        estimate = hypothesis("6146", 10.5)
        payload = build_public_payload(
            snapshot,
            [item],
            ROOT / "data" / "verified_bonus_facts_2026-08-02.yaml",
            tracked_companies=30,
            hypotheses={"6146": estimate},
            hypothesis_snapshot=hypothesis_snapshot,
            hypothesis_path=ROOT / "data" / "bonus_hypotheses_2026-08-02.yaml",
        )
        public = payload["records"][0]
        self.assertEqual(public["employee_scope"], item["employee_scope"])
        self.assertEqual(public["bonus"], item["bonus"])
        self.assertEqual(public["hypothesis"], estimate)
        self.assertEqual(public["notes"], item["notes"])
        self.assertEqual(public["sources"], item["sources"])
        self.assertEqual(
            payload["generated_from"],
            "data/verified_bonus_facts_2026-08-02.yaml",
        )
        self.assertEqual(
            payload["hypotheses_generated_from"],
            "data/bonus_hypotheses_2026-08-02.yaml",
        )
        self.assertEqual(payload["summary"]["hypothesis_count"], 1)
        self.assertEqual(payload["summary"]["hypothesis_central_months_average"], 10.5)
        self.assertEqual(payload["universe"]["tracked_companies"], 30)
        self.assertTrue(render_json(payload).endswith("\n"))

    def test_verified_average_excludes_hypotheses_and_minimums(self):
        point = record(
            "6146",
            months={
                "kind": "point",
                "value": 6.0,
                "basis": "explicit_source_value",
            },
        )
        minimum = record(
            "6503",
            months={
                "kind": "minimum",
                "value": 4.0,
                "basis": "explicit_source_value",
            },
        )
        partial = record("6758", status="partially_confirmed", months=None)
        snapshot = {
            "as_of": "2026-08-02",
            "universe": {
                "source_file": "nikkei225_bonus_survey_2024_en.yaml",
                "mutation_policy": "frozen",
            },
        }
        hypotheses = {
            "6146": hypothesis("6146", 10.0),
            "6503": hypothesis("6503", 5.5),
        }
        payload = build_public_payload(
            snapshot,
            [point, minimum, partial],
            ROOT / "data" / "x.yaml",
            tracked_companies=30,
            hypotheses=hypotheses,
        )
        self.assertEqual(payload["summary"]["explicit_point_months_count"], 1)
        self.assertEqual(payload["summary"]["explicit_point_months_average"], 6.0)
        self.assertEqual(payload["summary"]["hypothesis_count"], 2)
        self.assertEqual(payload["summary"]["hypothesis_central_months_average"], 7.75)


if __name__ == "__main__":
    unittest.main()
