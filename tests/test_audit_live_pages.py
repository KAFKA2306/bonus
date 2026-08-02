from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_live_pages import expected_generated_from, expected_source_survey_from  # noqa: E402


class LiveAuditTests(unittest.TestCase):
    def test_expected_sources_advance_with_newer_snapshots(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            data_dir = Path(tmp)
            fact = data_dir / "verified_bonus_facts_2026-08-03.yaml"
            fact.write_text("new", encoding="utf-8")
            survey = data_dir / "source_survey_2026-08-03.yaml"
            survey.write_text("new", encoding="utf-8")
            self.assertEqual(
                expected_generated_from(data_dir),
                str(fact.resolve().relative_to(ROOT.resolve())),
            )
            self.assertEqual(
                expected_source_survey_from(data_dir),
                str(survey.resolve().relative_to(ROOT.resolve())),
            )


if __name__ == "__main__":
    unittest.main()
