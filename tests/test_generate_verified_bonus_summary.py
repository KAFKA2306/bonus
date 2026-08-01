from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_verified_bonus_summary import (  # noqa: E402
    ValidationError,
    build_summary,
    validate_snapshot,
)


def source():
    return {
        "type": "company_official",
        "title": "Official policy",
        "url": "https://example.com/policy",
        "retrieved_at": "2026-08-02",
    }


def record(code="6146", status="confirmed"):
    return {
        "stock_code": code,
        "company_name_ja": "テスト会社",
        "subject": "employees",
        "employee_scope": "社員",
        "classification": "hybrid" if status != "unknown" else None,
        "evidence_status": status,
        "as_of": "2026-08-02",
        "bonus": {
            "frequency_per_year": None,
            "annual_months": None,
            "pool_basis": None,
            "allocation_logic": None,
            "base_salary_link": None,
        },
        "sources": [source()],
    }


def snapshot(records):
    return {
        "schema_version": 1,
        "as_of": "2026-08-02",
        "universe": {
            "source_file": "nikkei225_companies.yaml",
            "mutation_policy": "frozen",
        },
        "records": records,
    }


class VerificationTests(unittest.TestCase):
    def test_rejects_code_outside_frozen_universe(self):
        with self.assertRaisesRegex(ValidationError, "outside the frozen universe"):
            validate_snapshot(snapshot([record("9999")]), {"6146"})

    def test_unknown_record_cannot_carry_numeric_months(self):
        item = record(status="unknown")
        item["bonus"]["annual_months"] = {
            "kind": "point",
            "value": 6.0,
            "basis": "explicit_source_value",
        }
        with self.assertRaisesRegex(ValidationError, "requires evidence_status=confirmed"):
            validate_snapshot(snapshot([item]), {"6146"})

    def test_inferred_months_are_rejected(self):
        item = record()
        item["bonus"]["annual_months"] = {
            "kind": "point",
            "value": 6.0,
            "basis": "parsed_from_notes",
        }
        with self.assertRaisesRegex(ValidationError, "inferred values are forbidden"):
            validate_snapshot(snapshot([item]), {"6146"})

    def test_summary_averages_only_explicit_point_values(self):
        point = record("6146")
        point["bonus"]["annual_months"] = {
            "kind": "point",
            "value": 6.0,
            "basis": "explicit_source_value",
        }
        minimum = record("6503")
        minimum["bonus"]["annual_months"] = {
            "kind": "minimum",
            "value": 4.0,
            "basis": "explicit_source_value",
        }
        records = validate_snapshot(snapshot([point, minimum]), {"6146", "6503"})
        result = build_summary(
            snapshot([point, minimum]), records, ROOT / "data" / "x.yaml"
        )
        self.assertEqual(result["explicit_point_months_count"], 1)
        self.assertEqual(result["explicit_point_months_average"], 6.0)


if __name__ == "__main__":
    unittest.main()
