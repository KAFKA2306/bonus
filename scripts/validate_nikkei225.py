#!/usr/bin/env python3
"""Validate the expanded Nikkei 225 public payload."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "docs" / "data" / "bonus.json"
SOURCE_URL = "https://indexes.nikkei.co.jp/nkave/index/component?idx=nk225"


def main() -> int:
    public = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    records = public["records"]
    codes = [row["stock_code"] for row in records]
    assert public["universe"]["source_file"] == SOURCE_URL
    assert public["universe"]["tracked_companies"] == 225
    assert public["universe"]["covered_companies"] == 225
    assert public["summary"]["record_count"] == 225
    assert public["summary"]["quantified_company_count"] == 225
    assert len(records) == 225
    assert len(set(codes)) == 225
    assert public["universe"]["company_specific_prior_count"] >= 30
    assert public["universe"]["sector_prior_count"] <= 195
    assert sum(public["summary"]["mechanism_counts"].values()) == 225
    status_counts = Counter(row["estimate"]["status"] for row in records)
    assert status_counts["sector_prior"] > 0
    for row in records:
        estimate = row["estimate"]
        months = estimate["months"]
        amount = estimate["amount_yen"]
        assert 0 < months["minimum"] <= months["central"] <= months["maximum"] <= 24
        assert 0 < amount["minimum"] <= amount["central"] <= amount["maximum"]
        assert estimate["mechanism"]["upside_profile"] in {"very_high", "high", "medium", "low"}
        assert 0 <= estimate["mechanism"]["upside_score"] <= 1
        assert 0 <= estimate["confidence"]["score"] <= 1
        assert estimate["basis"] and estimate["assumptions"] and estimate["falsifiers"]
    assert any(row["stock_code"] == "285A" for row in records), "Kioxia must be in the current official universe"
    assert any(row["stock_code"] == "9022" and row["estimate"]["months"]["central"] == 6.15 for row in records)
    assert any(
        row["stock_code"] == "8035"
        and row["estimate"]["classification"] == "performance_linked"
        and row["estimate"]["mechanism"]["upside_profile"] == "high"
        and row["estimate"]["mechanism"]["formula_disclosure"] == "not_disclosed"
        for row in records
    )
    print(f"PASS: official Nikkei 225 expansion covers {len(records)} companies; sector priors={status_counts['sector_prior']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
