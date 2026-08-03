#!/usr/bin/env python3
"""Expand the source-backed payload to the official Nikkei 225 universe."""
from __future__ import annotations

import html
import json
import re
import statistics
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "docs" / "data" / "bonus.json"
SOURCE_URL = "https://indexes.nikkei.co.jp/nkave/index/component?idx=nk225"
RENGo_SOURCE_URL = "https://www.jtuc-rengo.or.jp/activity/roudou/shuntou/2026/yokyu_kaito/kaito/kaito_no7_ichiji.pdf"

# The amount value remains an observed sector anchor. It must not be divided by
# the months value because the two aggregates use different respondent samples.
SECTOR_RULES = {
    "医薬品": ("manufacturing", "製造業", 5.44, 1854847, "hybrid"),
    "電気機器": ("manufacturing", "製造業", 5.44, 1854847, "performance_nonformula"),
    "自動車": ("manufacturing", "製造業", 5.44, 1854847, "collective_discretionary"),
    "精密機器": ("manufacturing", "製造業", 5.44, 1854847, "performance_nonformula"),
    "通信": ("information_publication", "情報・出版", 5.42, 1770611, "hybrid"),
    "銀行": ("other", "その他（金融・投資等）", 4.39, 1749584, "hybrid"),
    "その他金融": ("other", "その他（金融・投資等）", 4.39, 1749584, "hybrid"),
    "証券": ("other", "その他（金融・投資等）", 4.39, 1749584, "performance_nonformula"),
    "保険": ("other", "その他（金融・投資等）", 4.39, 1749584, "hybrid"),
    "水産": ("manufacturing", "製造業", 5.44, 1854847, "collective_discretionary"),
    "食品": ("manufacturing", "製造業", 5.44, 1854847, "collective_discretionary"),
    "小売業": ("commercial_distribution", "商業流通", 3.87, 1169622, "hybrid"),
    "サービス": ("service_hotel", "サービス・ホテル", 4.04, 890000, "performance_nonformula"),
    "鉱業": ("manufacturing", "製造業", 5.44, 1854847, "performance_nonformula"),
    "繊維": ("manufacturing", "製造業", 5.44, 1854847, "collective_discretionary"),
    "パルプ・紙": ("manufacturing", "製造業", 5.44, 1854847, "collective_discretionary"),
    "化学": ("manufacturing", "製造業", 5.44, 1854847, "hybrid"),
    "石油": ("manufacturing", "製造業", 5.44, 1854847, "performance_nonformula"),
    "ゴム": ("manufacturing", "製造業", 5.44, 1854847, "collective_discretionary"),
    "窯業": ("manufacturing", "製造業", 5.44, 1854847, "collective_discretionary"),
    "鉄鋼": ("manufacturing", "製造業", 5.44, 1854847, "performance_nonformula"),
    "非鉄・金属": ("manufacturing", "製造業", 5.44, 1854847, "performance_nonformula"),
    "商社": ("other", "その他（金融・投資等）", 4.39, 1749584, "performance_nonformula"),
    "建設": ("manufacturing", "製造業", 5.44, 1854847, "collective_discretionary"),
    "機械": ("manufacturing", "製造業", 5.44, 1854847, "performance_nonformula"),
    "造船": ("manufacturing", "製造業", 5.44, 1854847, "performance_nonformula"),
    "輸送用機器": ("manufacturing", "製造業", 5.44, 1854847, "collective_discretionary"),
    "その他製造": ("manufacturing", "製造業", 5.44, 1854847, "hybrid"),
    "不動産": ("other", "その他（金融・投資等）", 4.39, 1749584, "hybrid"),
    "鉄道・バス": ("transport", "交通運輸", 4.42, 917078, "collective_discretionary"),
    "陸運": ("transport", "交通運輸", 4.42, 917078, "collective_discretionary"),
    "海運": ("transport", "交通運輸", 4.42, 917078, "performance_nonformula"),
    "空運": ("transport", "交通運輸", 4.42, 917078, "collective_discretionary"),
    "倉庫": ("transport", "交通運輸", 4.42, 917078, "collective_discretionary"),
    "電力": ("other", "その他（金融・投資等）", 4.39, 1749584, "collective_discretionary"),
    "ガス": ("other", "その他（金融・投資等）", 4.39, 1749584, "collective_discretionary"),
}

MECHANISMS = {
    "performance_nonformula": ("算式非開示型・業績連動", "high", 0.80, 0.85),
    "hybrid": ("ハイブリッド", "medium", 0.55, 0.45),
    "collective_discretionary": ("労使妥結・総合判断", "low", 0.25, 0.15),
}


def fetch_constituents() -> tuple[str, list[dict[str, str]]]:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "bonus-model/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8", errors="replace")
    text = html.unescape(re.sub(r"<[^>]+>", "\n", raw))
    text = re.sub(r"[\t\r ]+", " ", text)
    updated = re.search(r"更新日付[：:]\s*(\d{4}\.\d{2}\.\d{2})", text)
    as_of = updated.group(1).replace(".", "-") if updated else "unknown"
    industries = sorted(SECTOR_RULES, key=len, reverse=True)
    current = None
    rows: list[dict[str, str]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if line in industries:
            current = line
            continue
        if current and re.fullmatch(r"(?:\d{4}|\d{3}[A-Z])", line):
            nearby = lines[index + 1 : index + 8]
            names = [
                value
                for value in nearby
                if value not in {"コード", "銘柄名", "社名"}
                and not re.fullmatch(r"(?:\d{4}|\d{3}[A-Z])", value)
            ]
            if len(names) >= 2:
                rows.append(
                    {
                        "stock_code": line,
                        "display_name_ja": names[0],
                        "company_name_ja": names[1],
                        "nikkei_industry": current,
                    }
                )
    unique = {row["stock_code"]: row for row in rows}
    if len(unique) != 225:
        raise RuntimeError(
            f"official Nikkei page parsed {len(unique)} constituents, expected 225"
        )
    return as_of, sorted(unique.values(), key=lambda row: row["stock_code"])


def synthetic_record(row: dict[str, str]) -> dict:
    sector_id, sector_name, months, observed_annual_amount, mechanism_id = (
        SECTOR_RULES[row["nikkei_industry"]]
    )
    label, upside, upside_score, upper_add = MECHANISMS[mechanism_id]
    low = round(max(0.5, months - 1.10), 2)
    central = round(months, 2)
    high = round(months + 1.10 + upper_add, 2)
    broad = (
        "performance_linked"
        if mechanism_id == "performance_nonformula"
        else "hybrid"
        if mechanism_id == "hybrid"
        else "discretionary"
    )
    amount_conversion = {
        "status": "unavailable",
        "amount_sample_id": f"rengo-2026-final:{sector_id}:amount",
        "months_sample_id": f"rengo-2026-final:{sector_id}:months",
        "matched_population": False,
        "aggregation": "worker_weighted_average",
        "reason": "amount and months aggregates use different respondent samples",
        "monthly_base_yen": None,
    }
    return {
        "stock_code": row["stock_code"],
        "company_name_ja": row["company_name_ja"],
        "subject": "employees",
        "employee_scope": "個社一次資料未確認。日経業種と連合2026業種実測を用いた低信頼度の初期推定。",
        "classification": None,
        "evidence_status": "unknown",
        "as_of": "2026-08-02",
        "bonus": {
            "frequency_per_year": None,
            "annual_months": None,
            "pool_basis": None,
            "allocation_logic": None,
            "base_salary_link": None,
        },
        "notes": ["日経225拡張時のセクター事前分布。個社資料取得時に置換する。"],
        "sources": [],
        "survey": {
            "stage": "queued",
            "open_questions": [
                "会社・労組一次資料で制度類型、年間月数、支給回数を確認する。"
            ],
        },
        "estimate": {
            "status": "sector_prior",
            "method": "nikkei_industry_sector_prior",
            "sector_id": sector_id,
            "sector_name_ja": sector_name,
            "nikkei_industry": row["nikkei_industry"],
            "months": {"minimum": low, "central": central, "maximum": high},
            "amount_yen": None,
            "amount_status": "unavailable",
            "amount_method": "not_estimable_from_available_samples",
            "amount_conversion": amount_conversion,
            "frequency_per_year": 2,
            "classification": broad,
            "mechanism": {
                "id": mechanism_id,
                "label_ja": label,
                "upside_profile": upside,
                "upside_score": upside_score,
                "formula_disclosure": (
                    "not_disclosed"
                    if mechanism_id == "performance_nonformula"
                    else "unknown"
                ),
                "source_url": SOURCE_URL,
                "source_note": "日経業種からの初期仮説であり、個社制度の確認事実ではない。",
            },
            "official_observations": {
                "annual_months": None,
                "latest_seasonal": None,
            },
            "confidence": {
                "score": 0.30,
                "level": "low",
                "amount_score": None,
            },
            "weights": {"company_prior": 0.0, "sector_actual": 1.0},
            "anchors": {
                "company_prior_months": {
                    "minimum": low,
                    "central": central,
                    "maximum": high,
                },
                "sector_actual_months": months,
                "sector_actual_amount_yen": observed_annual_amount,
                "sector_implied_monthly_base_yen": None,
                "sector_sample_months": {},
                "sector_sample_amount": {},
                "source_url": RENGo_SOURCE_URL,
            },
            "override_note": None,
            "basis": [
                {
                    "type": "official_index_industry",
                    "statement": f"日経平均公式構成銘柄の業種は{row['nikkei_industry']}。",
                    "reference": SOURCE_URL,
                },
                {
                    "type": "official_sector_actual",
                    "statement": (
                        f"連合2026最終集計の{sector_name}実測を月数初期アンカーに使用。"
                    ),
                    "reference": RENGo_SOURCE_URL,
                },
            ],
            "assumptions": [
                "同じ日経業種内では個社資料がない段階の中心値として業種実測月数を使用する。",
                "金額と月数の回答標本が異なるため、個社参考金額を算定しない。",
            ],
            "falsifiers": [
                "会社・労組一次資料で制度類型または年間月数が確認された場合。"
            ],
            "amount_caution": "業種の金額平均と月数平均は回答標本が異なるため、比から個社金額を算定しない。",
        },
    }


def main() -> int:
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    official_as_of, constituents = fetch_constituents()
    existing = {row["stock_code"]: row for row in payload["records"]}
    records = [
        existing.get(row["stock_code"], synthetic_record(row))
        for row in constituents
    ]
    if len(records) != 225 or len({row["stock_code"] for row in records}) != 225:
        raise RuntimeError("expanded payload must contain exactly 225 unique constituents")

    available_amounts = [
        row["estimate"]["amount_yen"]["central"]
        for row in records
        if row["estimate"].get("amount_status") == "available"
        and isinstance(row["estimate"].get("amount_yen"), dict)
    ]

    payload["records"] = records
    payload["universe"] = {
        "source_file": SOURCE_URL,
        "source_as_of": official_as_of,
        "mutation_policy": "official_snapshot_at_build",
        "tracked_companies": 225,
        "covered_companies": 225,
        "coverage_ratio": 1.0,
        "company_specific_prior_count": len(existing),
        "sector_prior_count": 225 - len(existing),
    }
    payload["summary"]["record_count"] = 225
    payload["summary"]["quantified_company_count"] = 225
    payload["summary"]["company_specific_prior_count"] = len(existing)
    payload["summary"]["sector_prior_count"] = 225 - len(existing)
    payload["summary"]["amount_available_company_count"] = len(available_amounts)
    payload["summary"]["amount_unavailable_company_count"] = 225 - len(
        available_amounts
    )
    payload["summary"]["median_estimated_amount_yen"] = (
        int(statistics.median(available_amounts)) if available_amounts else None
    )
    payload["summary"]["mechanism_counts"] = dict(
        Counter(row["estimate"]["mechanism"]["id"] for row in records)
    )
    PAYLOAD.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Expanded Pages JSON to {len(records)} Nikkei 225 constituents "
        f"(official {official_as_of}); amount available={len(available_amounts)} "
        f"unavailable={225 - len(available_amounts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
