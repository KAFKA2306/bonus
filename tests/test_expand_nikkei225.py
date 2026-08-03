from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from expand_nikkei225 import synthetic_record  # noqa: E402


class Nikkei225ExpansionTests(unittest.TestCase):
    def test_sector_prior_does_not_convert_cross_sample_amounts(self) -> None:
        record = synthetic_record(
            {
                "stock_code": "9999",
                "company_name_ja": "合成サービス社",
                "display_name_ja": "合成サービス",
                "nikkei_industry": "サービス",
            }
        )
        estimate = record["estimate"]

        self.assertEqual(estimate["status"], "sector_prior")
        self.assertGreater(estimate["months"]["central"], 0)
        self.assertEqual(estimate["amount_status"], "unavailable")
        self.assertEqual(
            estimate["amount_method"],
            "not_estimable_from_available_samples",
        )
        self.assertIsNone(estimate["amount_yen"])
        self.assertIsNone(estimate["confidence"]["amount_score"])
        self.assertIsNone(
            estimate["amount_conversion"]["monthly_base_yen"]
        )
        self.assertFalse(
            estimate["amount_conversion"]["matched_population"]
        )
        self.assertNotEqual(
            estimate["amount_conversion"]["amount_sample_id"],
            estimate["amount_conversion"]["months_sample_id"],
        )
        self.assertEqual(
            estimate["anchors"]["sector_actual_amount_yen"],
            890000,
        )
        self.assertIsNone(
            estimate["anchors"]["sector_implied_monthly_base_yen"]
        )


if __name__ == "__main__":
    unittest.main()
