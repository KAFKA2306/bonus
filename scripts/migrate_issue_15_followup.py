#!/usr/bin/env python3
"""Apply the two fail-closed follow-up edits found by issue #15 diagnostics."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def update_generate_pages_fixture() -> None:
    path = ROOT / "tests" / "test_generate_pages_data.py"
    text = path.read_text(encoding="utf-8")
    if '"amount_conversion"' in text:
        return

    pattern = re.compile(
        r'(?P<indent>^[ \t]+)"sample_amount":\s*'
        r'\{"organizations":\s*\d+,\s*"workers":\s*\d+\},\n',
        flags=re.MULTILINE,
    )
    counter = 0

    def replacement(match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        indent = match.group("indent")
        original = match.group(0)
        return (
            original
            + f'{indent}"amount_conversion": {{\n'
            + f'{indent}    "status": "unavailable",\n'
            + f'{indent}    "amount_sample_id": "fixture-sector-{counter}:amount",\n'
            + f'{indent}    "months_sample_id": "fixture-sector-{counter}:months",\n'
            + f'{indent}    "matched_population": False,\n'
            + f'{indent}    "aggregation": "worker_weighted_average",\n'
            + f'{indent}    "reason": "different respondent samples",\n'
            + f'{indent}}},\n'
        )

    updated = pattern.sub(replacement, text)
    if counter < 1:
        raise RuntimeError(
            "tests/test_generate_pages_data.py did not contain the expected sample_amount fixture"
        )
    path.write_text(updated, encoding="utf-8")


def update_html_schema_marker() -> None:
    path = ROOT / "docs" / "index.html"
    text = path.read_text(encoding="utf-8")
    old = 'data-schema-version="4"'
    new = 'data-schema-version="5"'
    count = text.count(old)
    if count == 0:
        if new in text:
            return
        raise RuntimeError("docs/index.html does not contain the expected schema marker")
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> int:
    update_generate_pages_fixture()
    update_html_schema_marker()
    print("PASS: issue 15 follow-up migration applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
