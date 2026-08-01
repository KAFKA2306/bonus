#!/usr/bin/env python3
"""Validate the static Pages bundle against the verified YAML snapshot."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SOURCE = ROOT / "data" / "verified_bonus_facts_2026-08-02.yaml"
PUBLIC = DOCS / "data" / "bonus.json"


def main() -> int:
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    public = json.loads(PUBLIC.read_text(encoding="utf-8"))
    source_records = {str(item["stock_code"]): item for item in source["records"]}
    public_records = {str(item["stock_code"]): item for item in public["records"]}

    assert source["as_of"] == public["as_of"], "Pages as_of differs from verified snapshot"
    assert source_records.keys() == public_records.keys(), "Pages company set differs from verified snapshot"
    for code, verified in source_records.items():
        page = public_records[code]
        for key in ("company_name_ja", "classification", "evidence_status", "as_of"):
            assert verified.get(key) == page.get(key), f"{code}: {key} drift"
        assert verified["bonus"].get("frequency_per_year") == page["bonus"].get("frequency_per_year"), f"{code}: frequency drift"
        assert verified["bonus"].get("annual_months") == page["bonus"].get("annual_months"), f"{code}: annual_months drift"
        verified_urls = {item["url"] for item in verified["sources"]}
        page_urls = {item["url"] for item in page["sources"]}
        assert verified_urls == page_urls, f"{code}: source URL drift"

    html = (DOCS / "index.html").read_text(encoding="utf-8")
    app = (DOCS / "app.js").read_text(encoding="utf-8")
    css = (DOCS / "styles.css").read_text(encoding="utf-8")
    for required in ("id=\"companies\"", "id=\"method\"", "./app.js", "./styles.css"):
        assert required in html, f"missing HTML marker: {required}"
    assert "./data/bonus.json" in app, "app does not load published data"
    assert "@media" in css, "responsive CSS is missing"
    print(f"PASS: Pages bundle matches {len(source_records)} verified records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
