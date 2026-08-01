from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_live_pages import expected_generated_from  # noqa: E402
from generate_verified_bonus_summary import DATA_DIR, latest_snapshot  # noqa: E402


class LiveAuditTests(unittest.TestCase):
    def test_expected_source_tracks_repository_latest_snapshot(self):
        expected = str(latest_snapshot(DATA_DIR).resolve().relative_to(ROOT.resolve()))
        self.assertEqual(expected_generated_from(), expected)

    def test_expected_source_advances_with_newer_snapshot(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            data_dir = Path(tmp)
            (data_dir / "verified_bonus_facts_2026-08-02.yaml").write_text(
                "old", encoding="utf-8"
            )
            newest = data_dir / "verified_bonus_facts_2026-08-03.yaml"
            newest.write_text("new", encoding="utf-8")
            expected = str(newest.resolve().relative_to(ROOT.resolve()))
            self.assertEqual(expected_generated_from(data_dir=data_dir), expected)


if __name__ == "__main__":
    unittest.main()
