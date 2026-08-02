#!/usr/bin/env python3
"""Fail closed when the quantified static Pages bundle is stale or incomplete."""

from __future__ import annotations

import json
import re
from pathlib import Path

from bonus_hypotheses import latest_hypothesis, validate_hypotheses
from company_estimates import latest_company_estimation_model, validate_company_estimation_model
from generate_pages_data import DEFAULT_OUTPUT, build_public_payload, load_universe_companies, render_json
from generate_verified_bonus_summary import DATA_DIR, UNIVERSE_FILE, latest_snapshot, load_universe_codes, load_yaml, validate_snapshot
from quantitative_benchmarks import latest_quantitative_benchmarks, validate_quantitative_benchmarks
from source_survey import latest_source_survey, validate_source_survey

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[index:index+2], 16) / 255 for index in (1, 3, 5)]
    linear = [channel / 12.92 if channel <= .04045 else ((channel + .055) / 1.055) ** 2.4 for channel in channels]
    return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    first, second = relative_luminance(foreground), relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + .05) / (darker + .05)


def css_variable(css: str, name: str) -> str:
    match = re.search(rf"--{re.escape(name)}:\s*(#[0-9a-fA-F]{{6}})", css)
    if not match:
        raise AssertionError(f"missing CSS color variable: --{name}")
    return match.group(1)


def main() -> int:
    input_path = latest_snapshot(DATA_DIR)
    survey_path = latest_source_survey(DATA_DIR)
    quantitative_path = latest_quantitative_benchmarks(DATA_DIR)
    hypothesis_path = latest_hypothesis(DATA_DIR)
    company_model_path = latest_company_estimation_model(DATA_DIR)
    snapshot = load_yaml(input_path)
    source_survey = validate_source_survey(load_yaml(survey_path))
    source_ids = {item["id"] for item in source_survey["source_registry"]}
    quantitative = validate_quantitative_benchmarks(load_yaml(quantitative_path), source_ids)
    universe_codes = load_universe_codes(UNIVERSE_FILE)
    universe_companies = load_universe_companies(UNIVERSE_FILE)
    assert set(universe_companies) == universe_codes
    records = validate_snapshot(snapshot, universe_codes)
    hypotheses = validate_hypotheses(load_yaml(hypothesis_path), universe_codes)
    company_model = validate_company_estimation_model(
        load_yaml(company_model_path), universe_codes
    )
    expected = render_json(
        build_public_payload(
            snapshot,
            records,
            input_path,
            universe_companies,
            source_survey,
            survey_path,
            quantitative,
            quantitative_path,
            hypotheses,
            hypothesis_path,
            company_model,
            company_model_path,
        )
    )
    assert DEFAULT_OUTPUT.exists(), "generated Pages JSON is missing"
    actual = DEFAULT_OUTPUT.read_text(encoding="utf-8")
    assert actual == expected, "Pages JSON is stale; run python scripts/generate_pages_data.py"
    public = json.loads(actual)
    assert public["schema_version"] == 4
    assert public["generated_from"] == str(input_path.relative_to(ROOT))
    assert public["source_survey_generated_from"] == str(survey_path.relative_to(ROOT))
    assert public["quantitative_benchmarks_generated_from"] == str(quantitative_path.relative_to(ROOT))
    assert public["hypotheses_generated_from"] == str(hypothesis_path.relative_to(ROOT))
    assert public["company_estimation_model_generated_from"] == str(company_model_path.relative_to(ROOT))
    assert public["summary"]["record_count"] == len(universe_codes)
    assert public["summary"]["quantified_company_count"] == len(universe_codes)
    assert public["summary"]["source_channel_count"] == len(source_survey["source_registry"])
    assert public["summary"]["required_channel_count"] == len(source_survey["required_channels"])
    assert public["summary"]["quantitative_benchmark_count"] == len(quantitative["benchmarks"])
    assert public["summary"]["quantitative_final_count"] > 0
    assert public["summary"]["quantitative_provisional_count"] > 0
    assert public["summary"]["median_estimated_months"] > 0
    assert public["summary"]["median_estimated_amount_yen"] > 0
    assert 0 < public["summary"]["average_estimate_confidence"] <= 1
    assert public["universe"]["coverage_ratio"] == 1.0
    assert {item["stock_code"] for item in public["records"]} == universe_codes
    assert len(public["sector_anchors"]) == len(company_model["sectors"])
    assert all("estimate" in item for item in public["records"])
    assert all("survey" in item for item in public["records"])
    assert all("hypothesis" not in item for item in public["records"])
    assert all(item["source_url"].startswith("https://") for item in public["quantitative_benchmarks"])
    assert all(item["source_url"].startswith("https://") for item in public["sector_anchors"])
    for item in public["records"]:
        estimate = item["estimate"]
        months = estimate["months"]
        amount = estimate["amount_yen"]
        assert 0 < months["minimum"] <= months["central"] <= months["maximum"] <= 24
        assert 0 < amount["minimum"] <= amount["central"] <= amount["maximum"]
        assert 0 <= estimate["confidence"]["score"] <= 1
        assert 0 <= estimate["confidence"]["amount_score"] <= 1
        assert abs(
            estimate["weights"]["company_prior"]
            + estimate["weights"]["sector_actual"]
            - 1
        ) <= .002
        assert estimate["basis"] and estimate["assumptions"] and estimate["falsifiers"]

    html = (DOCS / "index.html").read_text(encoding="utf-8")
    app = (DOCS / "app.js").read_text(encoding="utf-8")
    css = (DOCS / "styles.css").read_text(encoding="utf-8")

    for marker in (
        'name="bonus-build" content="quantified-v7"',
        "<title>主要30社 賞与定量モデル</title>",
        'id="companies"', 'id="benchmarks"', 'id="sources"', 'id="rules"',
        'id="benchmark-body"', 'id="source-body"', 'id="company-body"', 'id="company-table"',
        'id="metric-quantified"', 'id="metric-median-months"', 'id="metric-median-amount"',
        'id="metric-confidence"', 'id="metric-verified"',
        'data-confidence="all"', 'data-sort="months"', 'data-sort="amount"',
        "company-table", "benchmark-table", 'class="sort-button"', 'aria-sort="none"',
        'role="status"', 'aria-live="polite"', 'aria-pressed="true"',
        '<script src="./app.js" defer></script>',
    ):
        assert marker in html, f"missing HTML marker: {marker}"
    for forbidden in (
        "企業別の調査キュー",
        "推定せず未確認",
        'data-stage="queued"',
        "推定レンジ</button>",
    ):
        assert forbidden not in html, f"legacy queue UI must be removed: {forbidden}"

    for marker in (
        "./data/bonus.json", "quantitative_benchmarks", "function benchmarkRow",
        "function sourceRow", "function companyRow", "function estimateDetails",
        "record.estimate", "estimate.months", "estimate.amount_yen", "estimate.weights",
        "quantified_company_count", "median_estimated_months", "average_estimate_confidence",
        "setAttribute('aria-sort'", "setAttribute('aria-pressed'", "#benchmark-body",
        "#source-body", "#company-body",
    ):
        assert marker in app, f"missing app behavior: {marker}"
    for forbidden in ("state.stage", "data-stage", "推定せず未確認"):
        assert forbidden not in app, f"legacy queue behavior must be removed: {forbidden}"

    for marker in (
        ":focus-visible", "scroll-margin-top", "@media (prefers-reduced-motion: reduce)",
        ".model-note", ".company-wrap", ".company-table", ".estimate-main",
        ".confidence-high", ".confidence-medium", ".confidence-low", ".estimate-panel",
        ".formula-grid", ".override-note", ".basis-list", ".benchmark-note",
        ".benchmark-wrap", ".benchmark-table", ".release-first", ".release-final",
        ".numeric", ".table-wrap", ".comparison-table", ".company-cell", ".sort-button",
        "position: sticky", 'th[aria-sort="ascending"]', ".row-details", ".detail-panel",
        ".tier-primary_company", ".tier-official_benchmark", "@media (max-width: 920px)",
    ):
        assert marker in css, f"missing CSS behavior: {marker}"

    background = css_variable(css, "bg")
    for name in ("muted", "link"):
        ratio = contrast_ratio(css_variable(css, name), background)
        assert ratio >= 4.5, f"--{name} contrast is only {ratio:.2f}:1"

    print(
        f"PASS: quantified dashboard covers {len(universe_codes)} companies, "
        f"{len(records)} verified records, {len(company_model['sectors'])} sector anchors, "
        f"and {len(quantitative['benchmarks'])} public benchmarks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
