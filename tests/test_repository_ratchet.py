from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryRatchetTests(unittest.TestCase):
    def test_canonical_pipeline_paths_exist(self):
        required_files = [
            "scripts/company_estimates.py",
            "scripts/generate_pages_data.py",
            "scripts/validate_pages.py",
            "docs/data/bonus.json",
            "docs/index.html",
            "docs/app.js",
            "docs/architecture/canonical-pipeline.md",
        ]
        for relative_path in required_files:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file(), relative_path)

        required_input_patterns = [
            "verified_bonus_facts_*.yaml",
            "bonus_hypotheses_*.yaml",
            "company_estimation_model_*.yaml",
            "source_survey_*.yaml",
            "quantitative_benchmarks_*.yaml",
        ]
        for pattern in required_input_patterns:
            with self.subTest(pattern=pattern):
                self.assertTrue(list((ROOT / "data").glob(pattern)), pattern)

    def test_scripts_contains_no_generated_png_artifacts(self):
        pngs = sorted(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "scripts").glob("*.png")
        )
        self.assertEqual([], pngs, f"generated PNGs must not live under scripts/: {pngs}")


if __name__ == "__main__":
    unittest.main()
