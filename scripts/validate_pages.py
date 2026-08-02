#!/usr/bin/env python3
"""Fail closed when the static Pages bundle is stale, inaccessible, or incomplete."""

from __future__ import annotations

import json
import re
from pathlib import Path

from bonus_hypotheses import latest_hypothesis, validate_hypotheses
from generate_pages_data import DEFAULT_OUTPUT, build_public_payload, render_json
from generate_verified_bonus_summary import (
    DATA_DIR,
    UNIVERSE_FILE,
    latest_snapshot,
    load_universe_codes,
    load_yaml,
    validate_snapshot,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    first = relative_luminance(foreground)
    second = relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def css_variable(css: str, name: str) -> str:
    match = re.search(rf"--{re.escape(name)}:\s*(#[0-9a-fA-F]{{6}})", css)
    if not match:
        raise AssertionError(f"missing CSS color variable: --{name}")
    return match.group(1)


def main() -> int:
    input_path = latest_snapshot(DATA_DIR)
    hypothesis_path = latest_hypothesis(DATA_DIR)
    snapshot = load_yaml(input_path)
    hypothesis_snapshot = load_yaml(hypothesis_path)
    universe_codes = load_universe_codes(UNIVERSE_FILE)
    records = validate_snapshot(snapshot, universe_codes)
    hypotheses = validate_hypotheses(hypothesis_snapshot, universe_codes)
    expected = render_json(
        build_public_payload(
            snapshot,
            records,
            input_path,
            tracked_companies=len(universe_codes),
            hypotheses=hypotheses,
            hypothesis_snapshot=hypothesis_snapshot,
            hypothesis_path=hypothesis_path,
        )
    )
    assert DEFAULT_OUTPUT.exists(), "generated Pages JSON is missing"
    actual = DEFAULT_OUTPUT.read_text(encoding="utf-8")
    assert actual == expected, (
        "Pages JSON differs from latest facts or hypotheses; "
        "run python scripts/generate_pages_data.py"
    )
    public = json.loads(actual)
    assert public["generated_from"] == str(input_path.relative_to(ROOT))
    assert public["hypotheses_generated_from"] == str(hypothesis_path.relative_to(ROOT))
    assert public["summary"]["record_count"] == len(universe_codes)
    assert public["summary"]["verified_record_count"] == len(records)
    assert public["summary"]["hypothesis_count"] == len(universe_codes)
    assert public["universe"]["covered_companies"] == len(universe_codes)
    assert public["universe"]["coverage_ratio"] == 1.0
    assert {item["stock_code"] for item in public["records"]} == universe_codes
    assert all(item["hypothesis"] is not None for item in public["records"])
    assert all(
        item["hypothesis"]["not_for_verified_aggregate"] is True
        for item in public["records"]
    )

    html = (DOCS / "index.html").read_text(encoding="utf-8")
    app = (DOCS / "app.js").read_text(encoding="utf-8")
    css = (DOCS / "styles.css").read_text(encoding="utf-8")

    html_markers = (
        'name="bonus-build" content="verified-pages-v4"',
        "<title>主要30社 賞与制度・推定比較表</title>",
        'id="comparison"',
        'id="method"',
        'id="metric-verified"',
        'id="metric-coverage"',
        'id="bonus-table"',
        'id="table-body"',
        'class="comparison-table"',
        'class="sort-button"',
        'aria-sort="none"',
        'data-status="unknown"',
        'id="result-count"',
        'role="status"',
        'aria-live="polite"',
        'aria-pressed="true"',
        'aria-controls="bonus-table"',
        "<noscript>",
        "./app.js",
        "./styles.css",
    )
    for marker in html_markers:
        assert marker in html, f"missing HTML marker: {marker}"
    for forbidden in ('id="cards"', 'class="cards"', 'class="card"'):
        assert forbidden not in html, f"card UI must be removed: {forbidden}"

    app_markers = (
        "./data/bonus.json",
        "function searchText",
        "function tableRow",
        "function sortRecords",
        "function updateSortHeaders",
        "function detailsCell",
        "record.hypothesis",
        "hypothesis.falsifiers",
        "'aria-sort'",
        "setAttribute('aria-pressed'",
        "#table-body",
        "#result-count",
        "#metric-verified",
        "#metric-coverage",
    )
    for marker in app_markers:
        assert marker in app, f"missing app behavior: {marker}"
    for forbidden in ("function card(", "#cards", "hypothesisPanel"):
        assert forbidden not in app, f"legacy card behavior must be removed: {forbidden}"

    css_markers = (
        ":focus-visible",
        "scroll-margin-top",
        "@media (prefers-reduced-motion: reduce)",
        ".table-wrap",
        ".comparison-table",
        ".company-cell",
        ".sort-button",
        "position: sticky",
        'th[aria-sort="ascending"]',
        ".row-details",
        ".detail-panel",
        ".status-confirmed",
        ".status-partially_confirmed",
        ".status-unknown",
        ".confidence-low",
        "@media (max-width: 920px)",
    )
    for marker in css_markers:
        assert marker in css, f"missing CSS behavior: {marker}"
    for forbidden in (".cards {", ".card {"):
        assert forbidden not in css, f"legacy card CSS must be removed: {forbidden}"

    background = css_variable(css, "bg")
    for name in ("muted", "link"):
        ratio = contrast_ratio(css_variable(css, name), background)
        assert ratio >= 4.5, f"--{name} contrast is only {ratio:.2f}:1"

    print(
        f"PASS: table view covers all {len(universe_codes)} companies; "
        f"{len(records)} verified facts and {len(hypotheses)} hypotheses"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
