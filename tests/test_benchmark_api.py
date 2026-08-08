import csv
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_benchmark_api", ROOT / "scripts" / "build_benchmark_api.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BenchmarkApiTest(unittest.TestCase):
    def test_build_distribution_matches_snapshot(self):
        manifest = MODULE.build()
        benchmark_path = ROOT / "docs" / "api" / "v1" / "benchmarks.json"
        csv_path = ROOT / "docs" / "api" / "v1" / "benchmarks.csv"
        facets_path = ROOT / "docs" / "api" / "v1" / "facets.json"

        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["record_count"], len(payload["benchmarks"]))
        self.assertEqual(len({row["id"] for row in payload["benchmarks"]}), manifest["record_count"])
        self.assertEqual(manifest["record_count"], 11)

        with csv_path.open(encoding="utf-8", newline="") as handle:
            self.assertEqual(sum(1 for _ in csv.DictReader(handle)), 11)

        facets = json.loads(facets_path.read_text(encoding="utf-8"))["facets"]
        self.assertEqual(sum(facets["publisher"].values()), 11)
        self.assertEqual(sum(facets["metric"].values()), 11)

    def test_rengo_final_snapshot_is_linked(self):
        snapshot = json.loads(
            (ROOT / "data" / "source_snapshots" / "rengo-2026-final-index-2026-08-08.json").read_text(encoding="utf-8")
        )
        self.assertEqual(snapshot["observed"]["final_publication_date"], "2026-07-03")
        self.assertEqual(len(snapshot["linked_benchmark_ids"]), 4)


if __name__ == "__main__":
    unittest.main()
