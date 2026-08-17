from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_live_pages import validate_payload  # noqa: E402


def valid_payload() -> dict:
    return {
        "summary": {
            "record_count": 225,
            "quantified_company_count": 225,
            "quantitative_benchmark_count": 11,
            "median_estimated_months": 4.5,
        },
        "sector_anchors": [{} for _ in range(6)],
        "universe": {
            "tracked_companies": 225,
            "covered_companies": 225,
            "coverage_ratio": 1.0,
        },
        "records": [{"estimate": {}} for _ in range(225)],
    }


class LiveAuditTests(unittest.TestCase):
    def test_published_payload_contract(self):
        validate_payload(valid_payload())

    def test_missing_company_fails(self):
        payload = valid_payload()
        payload["summary"]["record_count"] = 224
        with self.assertRaisesRegex(RuntimeError, "225 Nikkei 225 companies"):
            validate_payload(payload)


if __name__ == "__main__":
    unittest.main()
