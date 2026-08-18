#!/usr/bin/env python3
"""Validate the 2026 responsive comparison UI without network access."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def require(text: str, markers: tuple[str, ...], label: str) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise AssertionError(f"{label} is missing: {', '.join(missing)}")


def main() -> int:
    html_path = DOCS / "index.html"
    app_path = DOCS / "app.js"
    css_path = DOCS / "comparison.css"
    for path in (html_path, app_path, css_path):
        assert path.is_file(), f"missing comparison UI file: {path.relative_to(ROOT)}"

    html = html_path.read_text(encoding="utf-8")
    app = app_path.read_text(encoding="utf-8")
    css = css_path.read_text(encoding="utf-8")

    require(
        html,
        (
            'name="bonus-ui" content="comparison-v9"',
            'id="company-controls"',
            'id="confidence-filter"',
            'id="status-filter"',
            'id="sector-filter"',
            'id="sort-select"',
            'data-view="cards"',
            'data-view="table"',
            'id="compare-tray"',
            'id="copy-comparison-url"',
            'navigator.clipboard.writeText(location.href)',
            'id="comparison"',
            'id="comparison-table-wrap"',
            'id="comparison-cards"',
            'id="company-card-list"',
            'id="load-more"',
            'aria-label="企業比較表。横方向にスクロールできます。"',
            '<link rel="stylesheet" href="./comparison.css">',
        ),
        "docs/index.html",
    )

    require(
        app,
        (
            "const PAGE_SIZE = 60",
            "const MAX_COMPARE = 5",
            "function estimateKind",
            "function companyCard",
            "function comparisonTable",
            "function comparisonCard",
            "function readUrlState",
            "function writeUrlState",
            "new URLSearchParams",
            "params.set('compare'",
            "params.set('view'",
            "params.set('limit'",
            "state.visibleLimit += PAGE_SIZE",
            "state.compare.size >= MAX_COMPARE",
            "window.matchMedia('(max-width: 760px)')",
            "setAttribute('aria-pressed'",
            "data-compare-code",
        ),
        "docs/app.js",
    )

    require(
        css,
        (
            ".comparison-toolbar",
            ".compare-tray",
            ".comparison-panel",
            ".company-card-list",
            ".company-card",
            ".estimate-state-verified",
            ".estimate-state-company_estimate",
            ".estimate-state-sector_initial",
            ".amount-unavailable",
            ".selected-comparison-table",
            ".comparison-mobile",
            "overflow-x: auto !important",
            "position: sticky",
            "min-height: 44px",
            "@media (max-width: 760px)",
        ),
        "docs/comparison.css",
    )

    assert "overflow-x: hidden !important" not in css
    assert html.count('id="comparison"') == 1
    assert html.count('id="company-card-list"') == 1
    assert html.count('id="copy-comparison-url"') == 1

    print("PASS: responsive search, card, URL-state and 2-5 company comparison contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
