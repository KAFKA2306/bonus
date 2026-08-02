from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_verified_bonus_summary import ValidationError  # noqa: E402
from source_survey import latest_source_survey, validate_source_survey  # noqa: E402


def payload():
    return {
        "schema_version": 1,
        "as_of": "2026-08-02",
        "methodology": {
            key: key
            for key in (
                "purpose",
                "source_first_policy",
                "verification_policy",
                "benchmark_policy",
                "discovery_policy",
                "freshness_policy",
            )
        },
        "research_pipeline": ["discover", "qualify", "publish"],
        "source_registry": [
            {
                "id": "company_official",
                "name_ja": "会社公式",
                "tier": "primary_company",
                "scope": "company_specific",
                "priority": 1,
                "url": None,
                "verifies": ["annual_months"],
                "use_when": "個社確認",
                "limitations": "対象範囲を確認",
            },
            {
                "id": "benchmark",
                "name_ja": "統計",
                "tier": "official_benchmark",
                "scope": "national_benchmark",
                "priority": 2,
                "url": "https://example.com",
                "verifies": ["national_context"],
                "use_when": "比較",
                "limitations": "個社に使わない",
            },
            {
                "id": "discovery",
                "name_ja": "探索",
                "tier": "discovery_only",
                "scope": "discovery",
                "priority": 3,
                "url": None,
                "verifies": ["discovery_lead"],
                "use_when": "探索",
                "limitations": "確認根拠にしない",
            },
        ],
        "required_channels": ["company_official"],
        "benchmark_channels": ["benchmark"],
        "discovery_channels": ["discovery"],
    }


class SourceSurveyTests(unittest.TestCase):
    def test_valid(self):
        result = validate_source_survey(payload())
        self.assertEqual(result["source_registry"][0]["id"], "company_official")

    def test_unknown_channel_rejected(self):
        item = payload()
        item["required_channels"] = ["missing"]
        with self.assertRaises(ValidationError):
            validate_source_survey(item)

    def test_non_https_url_rejected(self):
        item = payload()
        item["source_registry"][1]["url"] = "http://example.com"
        with self.assertRaises(ValidationError):
            validate_source_survey(item)

    def test_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source_survey_2026-08-01.yaml").write_text("x")
            newest = root / "source_survey_2026-08-02.yaml"
            newest.write_text("y")
            self.assertEqual(latest_source_survey(root), newest)


if __name__ == "__main__":
    unittest.main()
