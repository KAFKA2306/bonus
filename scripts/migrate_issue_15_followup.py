#!/usr/bin/env python3
"""Apply the fail-closed follow-up edits found by issue #15 diagnostics."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        if new in text:
            return
        raise RuntimeError(f"expected source fragment not found in {path}: {old!r}")
    if text.count(old) != 1:
        raise RuntimeError(f"expected one source fragment in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def update_generate_pages_fixture() -> None:
    path = ROOT / "tests" / "test_generate_pages_data.py"
    text = path.read_text(encoding="utf-8")
    if '"amount_conversion"' not in text:
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

        text = pattern.sub(replacement, text)
        if counter < 1:
            raise RuntimeError(
                "tests/test_generate_pages_data.py did not contain the expected sample_amount fixture"
            )
        path.write_text(text, encoding="utf-8")

    replace_once(
        path,
        '        self.assertEqual(payload["schema_version"], 4)\n',
        '        self.assertEqual(payload["schema_version"], 5)\n',
    )
    replace_once(
        path,
        '        self.assertGreater(queued["estimate"]["amount_yen"]["central"], 0)\n',
        '        self.assertEqual(queued["estimate"]["amount_status"], "unavailable")\n'
        '        self.assertIsNone(queued["estimate"]["amount_yen"])\n',
    )


def update_html_build_marker() -> None:
    html_path = ROOT / "docs" / "index.html"
    validator_path = ROOT / "scripts" / "validate_pages.py"
    html_old = 'name="bonus-build" content="nikkei225-v1"'
    validator_old = 'name="bonus-build" content="quantified-v7"'
    new = 'name="bonus-build" content="quantified-v8"'

    replace_once(html_path, html_old, new)
    replace_once(validator_path, validator_old, new)
    replace_once(
        validator_path,
        '        "<title>主要30社 賞与定量モデル</title>",\n',
        '        "<title>日経225 賞与定量モデル</title>",\n',
    )


def main() -> int:
    update_generate_pages_fixture()
    update_html_build_marker()
    print("PASS: issue 15 follow-up migration applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
