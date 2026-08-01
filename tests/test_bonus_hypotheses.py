from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bonus_hypotheses import latest_hypothesis, validate_hypotheses  # noqa: E402
from generate_verified_bonus_summary import ValidationError  # noqa: E402


def payload():
    return {
        "schema_version": 1,
        "as_of": "2026-08-02",
        "methodology": {
            "purpose": "推定する",
            "separation_policy": "事実と分離する",
            "interval_policy": "レンジを使う",
            "confidence_policy": "確度を示す",
            "aggregation_policy": "別集計にする",
        },
        "estimates": [
            {
                "stock_code": "6861",
                "target": "annual_bonus_months",
                "estimate": {
                    "minimum": 6.0,
                    "central": 7.0,
                    "maximum": 8.0,
                    "unit": "base_salary_months",
                },
                "classification_hypothesis": "performance_linked",
                "frequency_per_year_hypothesis": 4,
                "confidence": {"level": "low", "score": 0.4},
                "method": "legacy_prior_with_high_uncertainty",
                "basis": [
                    {
                        "type": "legacy_prior",
                        "statement": "旧調査値",
                        "reference": "legacy#6861",
                    }
                ],
                "assumptions": ["賞与比率が維持される"],
                "falsifiers": ["公式値がレンジ外"],
                "not_for_verified_aggregate": True,
            }
        ],
    }


class HypothesisTests(unittest.TestCase):
    def test_valid_hypothesis_is_indexed_by_code(self):
        result = validate_hypotheses(payload(), {"6861"})
        self.assertEqual(result["6861"]["estimate"]["central"], 7.0)

    def test_rejects_inverted_range(self):
        data = copy.deepcopy(payload())
        data["estimates"][0]["estimate"]["minimum"] = 9.0
        with self.assertRaises(ValidationError):
            validate_hypotheses(data, {"6861"})

    def test_rejects_verified_aggregate_mixing(self):
        data = copy.deepcopy(payload())
        data["estimates"][0]["not_for_verified_aggregate"] = False
        with self.assertRaises(ValidationError):
            validate_hypotheses(data, {"6861"})

    def test_rejects_code_outside_universe(self):
        with self.assertRaises(ValidationError):
            validate_hypotheses(payload(), {"6146"})

    def test_latest_hypothesis_selects_newest_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            older = data_dir / "bonus_hypotheses_2026-08-01.yaml"
            newer = data_dir / "bonus_hypotheses_2026-08-02.yaml"
            older.write_text("old", encoding="utf-8")
            newer.write_text("new", encoding="utf-8")
            self.assertEqual(latest_hypothesis(data_dir), newer)


if __name__ == "__main__":
    unittest.main()
