#!/usr/bin/env python3
"""Validate the expanded Nikkei 225 public payload and responsive layout contract."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "docs" / "data" / "bonus.json"
STYLES = ROOT / "docs" / "styles.css"
SOURCE_URL = "https://indexes.nikkei.co.jp/nkave/index/component?idx=nk225"


def validate_no_horizontal_scroll() -> None:
    css = STYLES.read_text(encoding="utf-8")
    required = (
        "html { scroll-behavior: smooth; overflow-x: hidden; }",
        "body { margin: 0; overflow-x: hidden;",
        ".table-wrap { max-width: 100%;",
        "overflow-x: hidden; overflow-y: auto;",
        ".comparison-table { width: 100%; table-layout: fixed;",
        "@media (max-width: 1120px)",
        ".comparison-table tbody tr { display: grid; grid-template-columns: repeat(2,minmax(0,1fr));",
        "@media (max-width: 680px)",
        ".comparison-table tbody tr { grid-template-columns: 1fr; }",
    )
    for marker in required:
        assert marker in css, f"missing no-horizontal-scroll contract: {marker}"

    forbidden = (
        "overflow-x: auto",
        "min-width: 1060px",
        "min-width: 1510px",
        "min-width: 1240px",
        "min-width: 1460px",
    )
    for marker in forbidden:
        assert marker not in css, f"horizontal-scroll layout returned: {marker}"


def validate_amount(estimate: dict) -> str:
    amount_status = estimate.get("amount_status")
    amount = estimate.get("amount_yen")
    amount_score = estimate["confidence"].get("amount_score")
    conversion = estimate.get("amount_conversion")
    assert isinstance(conversion, dict), "amount_conversion must be explicit"

    if amount_status == "available":
        assert isinstance(amount, dict)
        assert 0 < amount["minimum"] <= amount["central"] <= amount["maximum"]
        assert isinstance(amount_score, (int, float))
        assert 0 <= amount_score <= 1
        assert conversion["status"] in {"matched_sample", "company_official"}
        assert conversion["matched_population"] is True
        assert conversion["amount_sample_id"] == conversion["months_sample_id"]
        assert conversion["monthly_base_yen"] > 0
        return "available"

    assert amount_status == "unavailable"
    assert amount is None
    assert amount_score is None
    assert estimate["amount_method"] == "not_estimable_from_available_samples"
    assert conversion["status"] == "unavailable"
    assert conversion["matched_population"] is False
    assert conversion["amount_sample_id"] != conversion["months_sample_id"]
    assert conversion["monthly_base_yen"] is None
    return "unavailable"


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
    amount_counts: Counter[str] = Counter()
    available_central_amounts: list[int] = []

    for row in records:
        estimate = row["estimate"]
        months = estimate["months"]
        assert 0 < months["minimum"] <= months["central"] <= months["maximum"] <= 24
        amount_status = validate_amount(estimate)
        amount_counts[amount_status] += 1
        if amount_status == "available":
            available_central_amounts.append(estimate["amount_yen"]["central"])
        if estimate["status"] == "sector_prior":
            assert amount_status == "unavailable"
        assert estimate["mechanism"]["upside_profile"] in {
            "very_high",
            "high",
            "medium",
            "low",
        }
        assert 0 <= estimate["mechanism"]["upside_score"] <= 1
        assert 0 <= estimate["confidence"]["score"] <= 1
        assert estimate["basis"] and estimate["assumptions"] and estimate["falsifiers"]

    assert public["summary"]["amount_available_company_count"] == amount_counts["available"]
    assert public["summary"]["amount_unavailable_company_count"] == amount_counts["unavailable"]
    assert sum(amount_counts.values()) == 225
    if available_central_amounts:
        assert public["summary"]["median_estimated_amount_yen"] > 0
    else:
        assert public["summary"]["median_estimated_amount_yen"] is None

    assert any(
        row["stock_code"] == "285A" for row in records
    ), "Kioxia must be in the current official universe"
    assert any(
        row["stock_code"] == "9022"
        and row["estimate"]["months"]["central"] == 6.15
        for row in records
    )
    assert any(
        row["stock_code"] == "8035"
        and row["estimate"]["classification"] == "performance_linked"
        and row["estimate"]["mechanism"]["upside_profile"] == "high"
        and row["estimate"]["mechanism"]["formula_disclosure"] == "not_disclosed"
        for row in records
    )
    validate_no_horizontal_scroll()
    print(
        f"PASS: official Nikkei 225 expansion covers {len(records)} companies; "
        f"sector priors={status_counts['sector_prior']}; "
        f"amount available={amount_counts['available']} "
        f"unavailable={amount_counts['unavailable']}; horizontal scrolling disabled"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
