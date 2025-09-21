#!/usr/bin/env python3
import datetime
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
COMPANY_DIR = BASE_DIR / "companies"
PHASE3_DIR = BASE_DIR / "analysis" / "phase3_estimates"
SUMMARY_DIR = BASE_DIR / "analysis" / "summary"

STAGE_PRIORITY = ("researched", "phase3_estimate")
CLASSIFICATION_ORDER = ("業績連動型", "基本給連動型", "総合判断型", "ハイブリッド型")


def load_yaml(path: Path):
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def normalise_classification(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    result = value.strip()
    for delimiter in ("（", "("):
        if delimiter in result:
            result = result.split(delimiter, 1)[0].strip()
    if not result:
        return None
    alias_map = {
        "業績連動": "業績連動型",
        "基本給連動": "基本給連動型",
        "総合判断": "総合判断型",
        "ハイブリッド": "ハイブリッド型",
    }
    return alias_map.get(result, result)


def normalise_confidence(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    return value[0].upper()


def to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def consider_record(records: Dict[str, dict], record: dict):
    code = record.get("stock_code")
    if not code:
        return
    prev = records.get(code)
    if not prev:
        records[code] = record
        return
    prev_score = prev.get("reliability_score")
    new_score = record.get("reliability_score")
    if new_score is not None and (prev_score is None or new_score > prev_score):
        records[code] = record


def parse_company_file(path: Path) -> Optional[dict]:
    data = load_yaml(path)
    if not isinstance(data, dict):
        return None
    profile = data.get("company_profile")
    if not isinstance(profile, dict):
        return None
    classification_raw = None
    confidence_raw = None
    reliability_raw = None
    bonus_block = data.get("bonus_system")
    if isinstance(bonus_block, dict):
        classification_raw = bonus_block.get("classification")
        confidence_raw = bonus_block.get("confidence_level")
        reliability_raw = bonus_block.get("reliability_score")
    stock_code = str(profile.get("stock_code") or "").strip()
    if not stock_code:
        stock_code = path.name.split("_", 1)[0]
    stock_code = stock_code.zfill(4)
    classification = normalise_classification(classification_raw)
    if not classification:
        return None
    record = {
        "stock_code": stock_code,
        "company_name": profile.get("company_name"),
        "sector": profile.get("sector"),
        "classification": classification,
        "raw_classification": classification_raw,
        "confidence_level": normalise_confidence(confidence_raw),
        "reliability_score": to_float(reliability_raw),
        "stage": "researched",
        "source_file": str(path.relative_to(BASE_DIR)),
    }
    return record


def parse_phase3_file(path: Path) -> Optional[dict]:
    data = load_yaml(path)
    if not isinstance(data, dict):
        return None
    profile = data.get("company_profile")
    if not isinstance(profile, dict):
        return None
    bonus_block = data.get("bonus_system_estimate")
    if not isinstance(bonus_block, dict):
        return None
    stock_code = str(profile.get("stock_code") or "").strip()
    if not stock_code:
        stock_code = path.name.split("_", 1)[0]
    stock_code = stock_code.zfill(4)
    classification = normalise_classification(bonus_block.get("classification"))
    if not classification:
        return None
    record = {
        "stock_code": stock_code,
        "company_name": profile.get("company_name"),
        "sector": profile.get("sector"),
        "classification": classification,
        "raw_classification": bonus_block.get("classification"),
        "confidence_level": normalise_confidence(bonus_block.get("confidence_level")),
        "reliability_score": to_float(bonus_block.get("reliability_score")),
        "stage": "phase3_estimate",
        "source_file": str(path.relative_to(BASE_DIR)),
    }
    return record


def gather_records():
    records_by_stage: Dict[str, Dict[str, dict]] = {stage: {} for stage in STAGE_PRIORITY}

    for path in COMPANY_DIR.rglob("*.yaml"):
        record = parse_company_file(path)
        if record:
            consider_record(records_by_stage["researched"], record)

    if PHASE3_DIR.exists():
        for path in PHASE3_DIR.rglob("*.yaml"):
            record = parse_phase3_file(path)
            if record:
                consider_record(records_by_stage["phase3_estimate"], record)

    overall: Dict[str, dict] = {}
    for stage in STAGE_PRIORITY:
        stage_records = records_by_stage.get(stage, {})
        for code, record in stage_records.items():
            if code in overall:
                continue
            overall[code] = record.copy()

    return overall, records_by_stage


def build_summary(overall: Dict[str, dict], records_by_stage: Dict[str, Dict[str, dict]]):
def order_counts(counter: Counter) -> Dict[str, int]:
    ordered: Dict[str, int] = {}
    for key in CLASSIFICATION_ORDER:
        if key in counter:
            ordered[key] = counter[key]
    for key in sorted(counter):
        if key not in ordered:
            ordered[key] = counter[key]
    return ordered


def summarise(records_dict: Dict[str, dict]):
        classification_counts = Counter()
        confidence_counts = Counter()
        reliability_scores = []
        sectors = Counter()
        for record in records_dict.values():
            classification_counts[record.get("classification")] += 1
            confidence_counts[record.get("confidence_level") or "Unknown"] += 1
            sectors[record.get("sector") or "Unknown"] += 1
            score = record.get("reliability_score")
            if score is not None:
                reliability_scores.append(score)
        average_reliability = None
        if reliability_scores:
            average_reliability = round(statistics.mean(reliability_scores), 2)
        return {
            "total_companies": len(records_dict),
            "classification_counts": order_counts(classification_counts),
            "confidence_counts": dict(sorted(confidence_counts.items())),
            "sector_counts": dict(sorted(sectors.items())),
            "average_reliability": average_reliability,
        }

    stage_summary = {
        stage: summarise(records)
        for stage, records in records_by_stage.items()
        if records
    }

    overall_summary = summarise(overall)

    sector_breakdown = {}
    sector_records: Dict[str, list] = defaultdict(list)
    for record in overall.values():
        sector = record.get("sector") or "Unknown"
        sector_records[sector].append(record)

    for sector, items in sorted(sector_records.items()):
        classification_counts = Counter()
        stage_counts = Counter()
        confidence_counts = Counter()
        reliability_scores = []
        for record in items:
            classification_counts[record.get("classification")] += 1
            stage_counts[record.get("stage") or "Unknown"] += 1
            confidence_counts[record.get("confidence_level") or "Unknown"] += 1
            score = record.get("reliability_score")
            if score is not None:
                reliability_scores.append(score)
        average_reliability = None
        if reliability_scores:
            average_reliability = round(statistics.mean(reliability_scores), 2)
        sector_breakdown[sector] = {
            "total_companies": len(items),
            "classification_counts": order_counts(classification_counts),
            "stage_counts": dict(sorted(stage_counts.items())),
            "confidence_counts": dict(sorted(confidence_counts.items())),
            "average_reliability": average_reliability,
        }

    return overall_summary, stage_summary, sector_breakdown


def main():
    overall, records_by_stage = gather_records()
    overall_summary, stage_summary, sector_breakdown = build_summary(overall, records_by_stage)

    summary_payload = {
        "summary_generated_on": datetime.date.today().isoformat(),
        "total_unique_companies": len(overall),
        "stage_priority": list(STAGE_PRIORITY),
        "overall": overall_summary,
        "by_stage": stage_summary,
    }

    sector_payload = {
        "summary_generated_on": datetime.date.today().isoformat(),
        "total_sectors": len(sector_breakdown),
        "sector_breakdown": sector_breakdown,
    }

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    with (SUMMARY_DIR / "bonus_overview.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(summary_payload, fh, sort_keys=False, allow_unicode=True)
    with (SUMMARY_DIR / "bonus_sector_breakdown.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(sector_payload, fh, sort_keys=False, allow_unicode=True)

    print(f"Wrote summary for {len(overall)} companies")


if __name__ == "__main__":
    main()
