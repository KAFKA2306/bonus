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


class PagesDataTests(unittest.TestCase):
    def test_latest_snapshot_selects_newest_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            older = data_dir / "verified_bonus_facts_2026-07-31.yaml"
            newer = data_dir / "verified_bonus_facts_2026-08-02.yaml"
            older.write_text("old", encoding="utf-8")
            newer.write_text("new", encoding="utf-8")
            self.assertEqual(latest_snapshot(data_dir), newer)

    def test_projection_preserves_verified_fields(self):
        snapshot = {
            "as_of": "2026-08-02",
            "universe": {
                "source_file": "nikkei225_bonus_survey_2024_en.yaml",
                "mutation_policy": "frozen",
            },
            "methodology": {"primary_sources_only": True},
        }
        item = record("6146")
        payload = build_public_payload(
            snapshot,
            [item],
            ROOT / "data" / "verified_bonus_facts_2026-08-02.yaml",
            tracked_companies=30,
        )
        public = payload["records"][0]
        self.assertEqual(public["employee_scope"], item["employee_scope"])
        self.assertEqual(public["bonus"], item["bonus"])
        self.assertEqual(public["notes"], item["notes"])
        self.assertEqual(public["sources"], item["sources"])
        self.assertEqual(
            payload["generated_from"],
            "data/verified_bonus_facts_2026-08-02.yaml",
        )
        self.assertEqual(payload["universe"]["tracked_companies"], 30)
        self.assertTrue(render_json(payload).endswith("\n"))

    def test_average_excludes_minimum_and_partial_records(self):
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
        payload = build_public_payload(
            snapshot,
            [point, minimum, partial],
            ROOT / "data" / "x.yaml",
            tracked_companies=30,
        )
        self.assertEqual(payload["summary"]["explicit_point_months_count"], 1)
        self.assertEqual(payload["summary"]["explicit_point_months_average"], 6.0)


if __name__ == "__main__":
    unittest.main()
