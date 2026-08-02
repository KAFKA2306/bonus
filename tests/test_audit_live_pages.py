from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_live_pages import (  # noqa: E402
    expected_company_model_from,
    expected_generated_from,
    expected_hypotheses_from,
    expected_quantitative_benchmarks_from,
    expected_source_survey_from,
)


class LiveAuditTests(unittest.TestCase):
    def test_expected_sources_advance_with_newer_snapshots(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            data_dir = Path(tmp)
            fact = data_dir / "verified_bonus_facts_2026-08-03.yaml"
            fact.write_text("new", encoding="utf-8")
            survey = data_dir / "source_survey_2026-08-03.yaml"
            survey.write_text("new", encoding="utf-8")
            quantitative = data_dir / "quantitative_benchmarks_2026-08-03.yaml"
            quantitative.write_text("new", encoding="utf-8")
            hypotheses = data_dir / "bonus_hypotheses_2026-08-03.yaml"
            hypotheses.write_text("new", encoding="utf-8")
            company_model = data_dir / "company_estimation_model_2026-08-03.yaml"
            company_model.write_text("new", encoding="utf-8")
            self.assertEqual(
                expected_generated_from(data_dir),
                str(fact.resolve().relative_to(ROOT.resolve())),
            )
            self.assertEqual(
                expected_source_survey_from(data_dir),
                str(survey.resolve().relative_to(ROOT.resolve())),
            )
            self.assertEqual(
                expected_quantitative_benchmarks_from(data_dir),
                str(quantitative.resolve().relative_to(ROOT.resolve())),
            )
            self.assertEqual(
                expected_hypotheses_from(data_dir),
                str(hypotheses.resolve().relative_to(ROOT.resolve())),
            )
            self.assertEqual(
                expected_company_model_from(data_dir),
                str(company_model.resolve().relative_to(ROOT.resolve())),
            )


if __name__ == "__main__":
    unittest.main()
