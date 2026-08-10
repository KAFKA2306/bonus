from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_media_widget_contract import SCHEMA_VERSION, build_contract  # noqa: E402


def record(*, code="7203", annual_months=None, amount_status="unavailable"):
    return {
        "stock_code": code,
        "company_name_ja": f"会社{code}",
        "as_of": "2026-08-02",
        "evidence_status": "confirmed",
        "classification": "hybrid",
        "bonus": {"annual_months": annual_months},
        "estimate": {
            "months": {"minimum": 4.5, "central": 5.0, "maximum": 5.5},
            "amount_status": amount_status,
            "amount_yen": (
                {"minimum": 800000, "central": 900000, "maximum": 1000000}
                if amount_status == "available"
                else None
            ),
            "confidence": {"level": "medium", "score": 0.71},
        },
        "sources": [
            {"type": "company_official", "title": "一次資料", "url": "https://example.com/source"},
            {"type": "bad", "title": "invalid", "url": "javascript:alert(1)"},
        ],
    }


def payload(records):
    return {"schema_version": 5, "as_of": "2026-08-02", "records": records}


class MediaWidgetContractTests(unittest.TestCase):
    def test_estimated_record_keeps_unavailable_amount_null(self):
        output = build_contract(payload([record()]))
        item = output["records"][0]
        self.assertEqual(output["schema_version"], SCHEMA_VERSION)
        self.assertEqual(item["status"], "estimated")
        self.assertEqual(item["months"]["central"], 5.0)
        self.assertEqual(item["amount"]["status"], "unavailable")
        self.assertIsNone(item["amount"]["central_yen"])
        self.assertEqual(item["sources"], [{"type": "company_official", "title": "一次資料", "url": "https://example.com/source"}])

    def test_verified_months_override_model_estimate(self):
        item = build_contract(payload([record(annual_months=6.2, amount_status="available")]))["records"][0]
        self.assertEqual(item["status"], "verified")
        self.assertEqual(item["months"], {"minimum": 6.2, "central": 6.2, "maximum": 6.2})
        self.assertEqual(item["amount"]["status"], "available")
        self.assertEqual(item["amount"]["central_yen"], 900000)

    def test_duplicate_company_ids_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate stock_code"):
            build_contract(payload([record(), record()]))

    def test_empty_records_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "non-empty records"):
            build_contract(payload([]))


if __name__ == "__main__":
    unittest.main()
