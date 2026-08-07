from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_verified_bonus_summary import ValidationError  # noqa: E402
from quantitative_benchmarks import validate_quantitative_benchmarks  # noqa: E402


def payload():
    return {
        "schema_version": 1,
        "as_of": "2026-08-02",
        "methodology": {
            "purpose": "official numbers",
            "comparability_policy": "separate populations",
            "company_policy": "no company inference",
            "revision_policy": "replace provisional with final",
        },
        "benchmarks": [
            {
                "id": "official-row",
                "source_id": "official",
                "publisher": "公的機関",
                "title": "平均妥結額",
                "period": "2026-summer",
                "published_at": "2026-07-01",
                "release_status": "final",
                "aggregation": "company_average",
                "scope": "対象企業",
                "metric": "settlement_amount",
                "value": 110.0,
                "unit": "yen",
                "previous_value": 100.0,
                "change_value": 10.0,
                "change_unit": "percent",
                "sample": {"organizations": 12, "workers": None},
                "previous_sample": {"organizations": 10, "workers": None},
                "source_url": "https://example.com/data",
                "note": "final",
            }
        ],
    }


class QuantitativeBenchmarkTests(unittest.TestCase):
    def test_validates_official_quantitative_row(self):
        result = validate_quantitative_benchmarks(payload(), {"official"})
        row = result["benchmarks"][0]
        self.assertEqual(row["value"], 110.0)
        self.assertEqual(row["yoy_audit"]["target_year"], 2026)
        self.assertEqual(row["yoy_audit"]["aggregate_organizations"], 12)
        self.assertEqual(row["yoy_audit"]["previous_aggregate_organizations"], 10)

    def test_recalculates_yoy_and_warns_on_ten_percent_denominator_shift(self):
        result = validate_quantitative_benchmarks(payload(), {"official"})
        audit = result["benchmarks"][0]["yoy_audit"]
        self.assertEqual(audit["formula"], "(current_value / previous_value - 1) * 100")
        self.assertEqual(audit["calculated_percent"], 10.0)
        self.assertEqual(audit["denominator_change_ratio"], 0.2)
        self.assertTrue(audit["denominator_warning"])
        self.assertIn("10%以上の分母変更警告", result["benchmarks"][0]["note"])

    def test_denominator_warning_is_unknown_without_previous_sample(self):
        candidate = payload()
        candidate["benchmarks"][0].pop("previous_sample")
        result = validate_quantitative_benchmarks(candidate, {"official"})
        audit = result["benchmarks"][0]["yoy_audit"]
        self.assertIsNone(audit["denominator_warning"])
        self.assertEqual(audit["denominator_status"], "previous_sample_unavailable")
        self.assertIn("警告判定不可", result["benchmarks"][0]["note"])

    def test_rejects_unknown_source(self):
        with self.assertRaisesRegex(ValidationError, "unknown source"):
            validate_quantitative_benchmarks(payload(), {"other"})

    def test_rejects_inconsistent_percentage(self):
        invalid = deepcopy(payload())
        invalid["benchmarks"][0]["change_value"] = 9.0
        with self.assertRaisesRegex(ValidationError, "percentage change"):
            validate_quantitative_benchmarks(invalid, {"official"})

    def test_rejects_non_https_source(self):
        invalid = deepcopy(payload())
        invalid["benchmarks"][0]["source_url"] = "http://example.com/data"
        with self.assertRaisesRegex(ValidationError, "must be https"):
            validate_quantitative_benchmarks(invalid, {"official"})

    def test_rejects_invalid_previous_sample(self):
        invalid = deepcopy(payload())
        invalid["benchmarks"][0]["previous_sample"]["organizations"] = 0
        with self.assertRaisesRegex(ValidationError, "previous_sample.organizations"):
            validate_quantitative_benchmarks(invalid, {"official"})


if __name__ == "__main__":
    unittest.main()
