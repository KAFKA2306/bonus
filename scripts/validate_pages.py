#!/usr/bin/env python3
"""Fail closed when the source-first static Pages bundle is stale or incomplete."""

from __future__ import annotations

import json
import re
from pathlib import Path

from generate_pages_data import DEFAULT_OUTPUT, build_public_payload, load_universe_companies, render_json
from generate_verified_bonus_summary import DATA_DIR, UNIVERSE_FILE, latest_snapshot, load_universe_codes, load_yaml, validate_snapshot
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
    snapshot = load_yaml(input_path)
    source_survey = validate_source_survey(load_yaml(survey_path))
    universe_codes = load_universe_codes(UNIVERSE_FILE)
    universe_companies = load_universe_companies(UNIVERSE_FILE)
    assert set(universe_companies) == universe_codes
    records = validate_snapshot(snapshot, universe_codes)
    expected = render_json(build_public_payload(snapshot, records, input_path, universe_companies, source_survey, survey_path))
    assert DEFAULT_OUTPUT.exists(), "generated Pages JSON is missing"
    actual = DEFAULT_OUTPUT.read_text(encoding="utf-8")
    assert actual == expected, "Pages JSON is stale; run python scripts/generate_pages_data.py"
    public = json.loads(actual)
    assert public["schema_version"] == 2
    assert public["generated_from"] == str(input_path.relative_to(ROOT))
    assert public["source_survey_generated_from"] == str(survey_path.relative_to(ROOT))
    assert public["summary"]["record_count"] == len(universe_codes)
    assert public["summary"]["source_channel_count"] == len(source_survey["source_registry"])
    assert public["summary"]["required_channel_count"] == len(source_survey["required_channels"])
    assert public["universe"]["coverage_ratio"] == 1.0
    assert {item["stock_code"] for item in public["records"]} == universe_codes
    assert all("hypothesis" not in item for item in public["records"])
    assert all("survey" in item for item in public["records"])

    html = (DOCS / "index.html").read_text(encoding="utf-8")
    app = (DOCS / "app.js").read_text(encoding="utf-8")
    css = (DOCS / "styles.css").read_text(encoding="utf-8")

    for marker in (
        'name="bonus-build" content="source-survey-v5"',
        "<title>主要30社 賞与ソース・メタサーベイ</title>",
        'id="sources"', 'id="companies"', 'id="rules"',
        'id="source-body"', 'id="company-body"', 'id="company-table"',
        'id="metric-channels"', 'id="metric-primary"', 'id="metric-coverage"',
        'class="sort-button"', 'aria-sort="none"', 'role="status"',
        'aria-live="polite"', 'aria-pressed="true"', '<script src="./app.js" defer></script>',
    ):
        assert marker in html, f"missing HTML marker: {marker}"
    for forbidden in ("推定レンジ</button>", "確度</button>", "仮説推定比較表"):
        assert forbidden not in html, f"legacy estimate UI must be removed: {forbidden}"

    for marker in (
        "./data/bonus.json", "source_registry", "function sourceRow", "function companyRow",
        "record.survey", "next_channel_name_ja", "research_coverage_ratio",
        "setAttribute('aria-sort'", "setAttribute('aria-pressed'", "#source-body", "#company-body",
    ):
        assert marker in app, f"missing app behavior: {marker}"
    for forbidden in ("hypothesisRange", "confidenceScore", "record.hypothesis"):
        assert forbidden not in app, f"legacy hypothesis behavior must be removed: {forbidden}"

    for marker in (
        ":focus-visible", "scroll-margin-top", "@media (prefers-reduced-motion: reduce)",
        ".table-wrap", ".comparison-table", ".company-cell", ".sort-button",
        "position: sticky", 'th[aria-sort="ascending"]', ".row-details", ".detail-panel",
        ".tier-primary_company", ".tier-official_benchmark", ".status-evidence_found",
        "@media (max-width: 920px)",
    ):
        assert marker in css, f"missing CSS behavior: {marker}"

    background = css_variable(css, "bg")
    for name in ("muted", "link"):
        ratio = contrast_ratio(css_variable(css, name), background)
        assert ratio >= 4.5, f"--{name} contrast is only {ratio:.2f}:1"

    print(f"PASS: source survey covers {len(universe_codes)} companies, {len(source_survey['source_registry'])} channels, {len(records)} verified records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
