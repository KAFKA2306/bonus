#!/usr/bin/env python3
"""Compare retired cross-sample amount projections with publishable estimates."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = ROOT / "docs" / "data" / "bonus.json"
OUTPUT = ROOT / "audit" / "amount_conversion_diff.json"


def old_sector_projection(record: dict[str, Any]) -> dict[str, int] | None:
    estimate = record["estimate"]
    anchors = estimate["anchors"]
    months = estimate["months"]
    response_months = anchors.get("sector_actual_months")
    response_amount = anchors.get("sector_actual_amount_yen")
    if not isinstance(response_months, (int, float)) or response_months <= 0:
        return None
    if not isinstance(response_amount, (int, float)) or response_amount <= 0:
        return None
    base = response_amount / response_months
    return {
        key: int(round(base * float(months[key]) / 1000) * 1000)
        for key in ("minimum", "central", "maximum")
    }


def build_audit(payload: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in payload["records"]:
        estimate = record["estimate"]
        retired = old_sector_projection(record)
        current = estimate.get("amount_yen")
        rows.append(
            {
                "stock_code": record["stock_code"],
                "company_name_ja": record["company_name_ja"],
                "amount_status": estimate["amount_status"],
                "amount_method": estimate["amount_method"],
                "amount_conversion": estimate["amount_conversion"],
                "retired_cross_sample_projection_yen": retired,
                "current_publishable_amount_yen": current,
                "central_change_yen": (
                    int(current["central"] - retired["central"])
                    if isinstance(current, dict) and isinstance(retired, dict)
                    else None
                ),
            }
        )
    unavailable = sum(row["amount_status"] == "unavailable" for row in rows)
    available = len(rows) - unavailable
    if len(rows) != 30:
        raise AssertionError(f"expected 30 companies, got {len(rows)}")
    if unavailable == 0:
        raise AssertionError("cross-sample projections were not retired")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(PUBLIC_DATA.relative_to(ROOT)),
        "policy": "do_not_divide_amount_and_month_averages_from_different_samples",
        "company_count": len(rows),
        "amount_available_company_count": available,
        "amount_unavailable_company_count": unavailable,
        "records": rows,
    }


def main() -> int:
    payload = json.loads(PUBLIC_DATA.read_text(encoding="utf-8"))
    audit = build_audit(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "PASS: amount conversion audit "
        f"available={audit['amount_available_company_count']} "
        f"unavailable={audit['amount_unavailable_company_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
