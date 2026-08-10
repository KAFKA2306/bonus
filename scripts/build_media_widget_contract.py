#!/usr/bin/env python3
"""Build a minimal, versioned media-widget contract from the canonical Pages payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "data" / "bonus.json"
DEFAULT_OUTPUT = ROOT / "docs" / "data" / "media-widget-v1.json"
SCHEMA_VERSION = "bonus.media-widget.v1"


def _number(value: object) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _source(source: object) -> dict[str, Any] | None:
    if not isinstance(source, dict):
        return None
    url = source.get("url")
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return {
        "type": source.get("type") if isinstance(source.get("type"), str) else None,
        "title": source.get("title") if isinstance(source.get("title"), str) else None,
        "url": url,
    }


def _months(record: dict[str, Any]) -> tuple[str, dict[str, float | int | None]]:
    bonus = record.get("bonus") if isinstance(record.get("bonus"), dict) else {}
    verified = _number(bonus.get("annual_months"))
    if verified is not None and record.get("evidence_status") in {"confirmed", "partially_confirmed"}:
        return "verified", {"minimum": verified, "central": verified, "maximum": verified}

    estimate = record.get("estimate") if isinstance(record.get("estimate"), dict) else {}
    estimate_months = estimate.get("months") if isinstance(estimate.get("months"), dict) else {}
    central = _number(estimate_months.get("central"))
    if central is not None:
        return "estimated", {
            "minimum": _number(estimate_months.get("minimum")),
            "central": central,
            "maximum": _number(estimate_months.get("maximum")),
        }
    return "unavailable", {"minimum": None, "central": None, "maximum": None}


def _amount(record: dict[str, Any]) -> dict[str, Any]:
    estimate = record.get("estimate") if isinstance(record.get("estimate"), dict) else {}
    amount = estimate.get("amount_yen") if isinstance(estimate.get("amount_yen"), dict) else {}
    available = estimate.get("amount_status") == "available" and _number(amount.get("central")) is not None
    if not available:
        return {"status": "unavailable", "minimum_yen": None, "central_yen": None, "maximum_yen": None}
    return {
        "status": "available",
        "minimum_yen": _number(amount.get("minimum")),
        "central_yen": _number(amount.get("central")),
        "maximum_yen": _number(amount.get("maximum")),
    }


def build_contract(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("canonical payload must contain non-empty records")

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise ValueError(f"record {index} must be an object")
        code = raw.get("stock_code")
        name = raw.get("company_name_ja")
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"record {index} has no stock_code")
        if code in seen:
            raise ValueError(f"duplicate stock_code: {code}")
        seen.add(code)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"record {code} has no company_name_ja")

        status, months = _months(raw)
        estimate = raw.get("estimate") if isinstance(raw.get("estimate"), dict) else {}
        confidence = estimate.get("confidence") if isinstance(estimate.get("confidence"), dict) else {}
        sources = [item for item in (_source(source) for source in raw.get("sources", [])) if item]
        result.append(
            {
                "company_id": code,
                "company_name_ja": name,
                "as_of": raw.get("as_of") if isinstance(raw.get("as_of"), str) else payload.get("as_of"),
                "status": status,
                "evidence_status": raw.get("evidence_status"),
                "classification": raw.get("classification"),
                "months": months,
                "amount": _amount(raw),
                "confidence": {
                    "level": confidence.get("level") if isinstance(confidence.get("level"), str) else None,
                    "score": _number(confidence.get("score")),
                },
                "sources": sources,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": payload.get("schema_version"),
        "as_of": payload.get("as_of"),
        "record_count": len(result),
        "contract": {
            "status_values": ["verified", "estimated", "unavailable"],
            "amount_rule": "unavailable amounts remain null and must never be rendered as zero or inferred yen",
            "provenance_rule": "sources are projected only from the canonical bonus.json record",
        },
        "records": result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    contract = build_contract(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
