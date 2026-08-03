#!/usr/bin/env python3
"""Apply issue #15 as an atomic, fail-closed source migration."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one exact match in {path}, found {count}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def insert_before_once(path: Path, marker: str, addition: str) -> None:
    replace_once(path, marker, addition + marker)


def migrate_model() -> None:
    path = ROOT / "data" / "company_estimation_model_2026-08-02.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    methodology = payload["methodology"]
    methodology["amount_formula"] = (
        "個社金額は、同一標本で金額と月数を対応確認できる場合、または会社公式の季別モデル額と月数がある場合だけ算出する。"
        "異なる標本の平均金額÷平均月数は使用しない。"
    )
    methodology["disclosure_policy"] = (
        "月数推定と金額可用性を分離し、金額を統計的に算定できない場合はnullとunavailableを公開する。"
    )
    for sector_id, sector in payload["sectors"].items():
        sector["amount_conversion"] = {
            "status": "unavailable",
            "amount_sample_id": f"rengo-2026-final:{sector_id}:amount",
            "months_sample_id": f"rengo-2026-final:{sector_id}:months",
            "matched_population": False,
            "aggregation": "worker_weighted_average",
            "reason": "amount and months aggregates use different respondent samples",
        }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )


def migrate_company_estimates() -> None:
    path = ROOT / "scripts" / "company_estimates.py"
    insert_before_once(
        path,
        "\ndef _validate_official_months(value: Any, field_name: str) -> dict[str, Any]:\n",
        '''\ndef _validate_amount_conversion(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be an object")
    amount_sample_id = _required_text(value.get("amount_sample_id"), f"{field_name}.amount_sample_id")
    months_sample_id = _required_text(value.get("months_sample_id"), f"{field_name}.months_sample_id")
    aggregation = _required_text(value.get("aggregation"), f"{field_name}.aggregation")
    reason = _required_text(value.get("reason"), f"{field_name}.reason")
    matched_population = value.get("matched_population")
    if not isinstance(matched_population, bool):
        raise ValidationError(f"{field_name}.matched_population must be boolean")
    if matched_population and amount_sample_id != months_sample_id:
        raise ValidationError(
            f"{field_name} cannot be matched when amount_sample_id and months_sample_id differ"
        )
    status = "matched_sample" if matched_population else "unavailable"
    if value.get("status") != status:
        raise ValidationError(f"{field_name}.status must be {status}")
    return {
        **value,
        "status": status,
        "amount_sample_id": amount_sample_id,
        "months_sample_id": months_sample_id,
        "matched_population": matched_population,
        "aggregation": aggregation,
        "reason": reason,
    }

''',
    )
    replace_once(
        path,
        '        sample_amount = _positive_sample(item.get("sample_amount"), f"{prefix}.sample_amount")\n        source_url = item.get("source_url")\n',
        '        sample_amount = _positive_sample(item.get("sample_amount"), f"{prefix}.sample_amount")\n'
        '        amount_conversion = _validate_amount_conversion(\n'
        '            item.get("amount_conversion"), f"{prefix}.amount_conversion"\n'
        '        )\n'
        '        source_url = item.get("source_url")\n',
    )
    replace_once(
        path,
        '            "sample_amount": sample_amount,\n            "company_codes": normalized_codes,\n',
        '            "sample_amount": sample_amount,\n            "amount_conversion": amount_conversion,\n            "company_codes": normalized_codes,\n',
    )
    old_sector = '''def _amount_from_sector(
    estimate: dict[str, float], sector: dict[str, Any]
) -> tuple[dict[str, int], str, str]:
    implied_monthly_base = sector["response_amount_yen"] / sector["response_months"]
    amount = {
        key: int(round(implied_monthly_base * estimate[key] / 1000) * 1000)
        for key in ("minimum", "central", "maximum")
    }
    return (
        amount,
        "sector_implied",
        "業種平均基本月額を仮定した参考換算であり、個社の実支給額ではない。月数と金額の標本も一致しない。",
    )
'''
    new_sector = '''def _amount_from_sector(
    estimate: dict[str, float], sector: dict[str, Any]
) -> tuple[dict[str, int] | None, str, str, dict[str, Any]]:
    conversion = dict(sector["amount_conversion"])
    if not conversion["matched_population"]:
        return (
            None,
            "not_estimable_from_available_samples",
            "業種の金額平均と月数平均は回答標本が異なるため、比から個社金額を算定しない。",
            {**conversion, "status": "unavailable", "monthly_base_yen": None},
        )
    if conversion["amount_sample_id"] != conversion["months_sample_id"]:
        raise ValidationError("matched amount conversion requires one shared sample id")
    implied_monthly_base = sector["response_amount_yen"] / sector["response_months"]
    amount = {
        key: int(round(implied_monthly_base * estimate[key] / 1000) * 1000)
        for key in ("minimum", "central", "maximum")
    }
    return (
        amount,
        "matched_sector_sample",
        "同一回答標本の年間金額と年間月数から得た基礎月額による参考換算。",
        {
            **conversion,
            "status": "matched_sample",
            "monthly_base_yen": int(round(implied_monthly_base)),
        },
    )
'''
    replace_once(path, old_sector, new_sector)
    old_official = '''def _amount_from_official_seasonal_base(
    annual_months: dict[str, float], seasonal: dict[str, Any]
) -> tuple[dict[str, int], str, str]:
    monthly_base = seasonal["amount_yen"] / seasonal["months"]
    central = monthly_base * annual_months["central"]
    uncertainty_rate = 0.05
    return (
        {
            "minimum": int(round(central * (1 - uncertainty_rate) / 1000) * 1000),
            "central": int(round(central / 1000) * 1000),
            "maximum": int(round(central * (1 + uncertainty_rate) / 1000) * 1000),
        },
        "official_company_base_projection",
        (
            f"{seasonal['period']}の会社公式モデル額÷月数で基礎月額を推定し、別期間の公式年間月数を乗じた参考値。"
            "期間差を含むため実支給額ではない。"
        ),
    )
'''
    new_official = '''def _amount_from_official_seasonal_base(
    annual_months: dict[str, float], seasonal: dict[str, Any]
) -> tuple[dict[str, int], str, str, dict[str, Any]]:
    monthly_base = seasonal["amount_yen"] / seasonal["months"]
    central = monthly_base * annual_months["central"]
    uncertainty_rate = 0.05
    sample_id = f"company-official:{seasonal['source_url']}:{seasonal['period']}"
    return (
        {
            "minimum": int(round(central * (1 - uncertainty_rate) / 1000) * 1000),
            "central": int(round(central / 1000) * 1000),
            "maximum": int(round(central * (1 + uncertainty_rate) / 1000) * 1000),
        },
        "official_company_base_projection",
        (
            f"{seasonal['period']}の会社公式モデル額÷月数で基礎月額を推定し、別期間の公式年間月数を乗じた参考値。"
            "期間差を含むため実支給額ではない。"
        ),
        {
            "status": "company_official",
            "amount_sample_id": sample_id,
            "months_sample_id": sample_id,
            "matched_population": True,
            "aggregation": "company_model_employee",
            "reason": "company published amount and months for the same seasonal model employee",
            "monthly_base_yen": int(round(monthly_base)),
        },
    )
'''
    replace_once(path, old_official, new_official)
    replace_once(
        path,
        '''        if override.get("amount_policy") == "project_from_official_seasonal_base":
            amount, amount_method, amount_caution = _amount_from_official_seasonal_base(
                estimate, override["latest_seasonal"]
            )
        else:
            amount, amount_method, amount_caution = _amount_from_sector(estimate, sector)
''',
        '''        if override.get("amount_policy") == "project_from_official_seasonal_base":
            amount, amount_method, amount_caution, amount_conversion = _amount_from_official_seasonal_base(
                estimate, override["latest_seasonal"]
            )
        else:
            amount, amount_method, amount_caution, amount_conversion = _amount_from_sector(
                estimate, sector
            )
        amount_status = "available" if amount is not None else "unavailable"
''',
    )
    replace_once(
        path,
        '''        amount_score = confidence_score - 0.15
        if amount_method == "sector_implied":
            if sector["sample_amount"]["organizations"] < 10:
                amount_score -= 0.12
            elif sector["sample_amount"]["organizations"] < 50:
                amount_score -= 0.05
        else:
            amount_score = max(amount_score, 0.65)
        amount_score = round(_clamp(amount_score, 0.15, 0.82), 2)
''',
        '''        if amount is None:
            amount_score = None
        else:
            amount_score = confidence_score - 0.15
            if amount_method == "matched_sector_sample":
                if sector["sample_amount"]["organizations"] < 10:
                    amount_score -= 0.12
                elif sector["sample_amount"]["organizations"] < 50:
                    amount_score -= 0.05
            else:
                amount_score = max(amount_score, 0.65)
            amount_score = round(_clamp(amount_score, 0.15, 0.82), 2)
''',
    )
    replace_once(
        path,
        '''        if amount_method == "sector_implied":
            assumptions.append("参考換算額では個社の基本月額を業種平均と同じと仮定する。")
        else:
            assumptions.append("会社公式の季別モデル基礎額が年間換算にも近似利用できる。")
''',
        '''        if amount_method == "not_estimable_from_available_samples":
            assumptions.append("金額標本と月数標本が異なるため、個社参考金額を算定しない。")
        elif amount_method == "matched_sector_sample":
            assumptions.append("同一標本から得た業種基礎月額を個社月数へ参考適用する。")
        else:
            assumptions.append("会社公式の季別モデル基礎額が年間換算にも近似利用できる。")
''',
    )
    replace_once(
        path,
        '            "amount_yen": amount,\n            "amount_method": amount_method,\n',
        '            "amount_yen": amount,\n            "amount_status": amount_status,\n            "amount_method": amount_method,\n            "amount_conversion": amount_conversion,\n',
    )
    replace_once(
        path,
        '''                "sector_implied_monthly_base_yen": int(
                    round(sector["response_amount_yen"] / sector["response_months"])
                ),
''',
        '''                "sector_implied_monthly_base_yen": amount_conversion.get("monthly_base_yen"),
''',
    )


def migrate_public_generator() -> None:
    path = ROOT / "scripts" / "generate_pages_data.py"
    replace_once(path, '        "schema_version": 4,\n', '        "schema_version": 5,\n')
    replace_once(
        path,
        '                "sample_amount": item["sample_amount"],\n                "company_count": len(item["company_codes"]),\n',
        '                "sample_amount": item["sample_amount"],\n                "amount_conversion": item["amount_conversion"],\n                "company_count": len(item["company_codes"]),\n',
    )
    replace_once(
        path,
        '    central_amounts = [item["estimate"]["amount_yen"]["central"] for item in public_records]\n',
        '''    central_amounts = [
        item["estimate"]["amount_yen"]["central"]
        for item in public_records
        if isinstance(item["estimate"].get("amount_yen"), dict)
        and isinstance(item["estimate"]["amount_yen"].get("central"), (int, float))
    ]
    amount_available_count = sum(
        item["estimate"].get("amount_status") == "available" for item in public_records
    )
''',
    )
    replace_once(
        path,
        '            "quantified_company_count": len(estimates_by_code),\n            "median_estimated_months": round(statistics.median(central_months), 2),\n            "median_estimated_amount_yen": int(statistics.median(central_amounts)),\n',
        '            "quantified_company_count": len(estimates_by_code),\n            "amount_available_company_count": amount_available_count,\n            "amount_unavailable_company_count": len(public_records) - amount_available_count,\n            "median_estimated_months": round(statistics.median(central_months), 2),\n            "median_estimated_amount_yen": (\n                int(statistics.median(central_amounts)) if central_amounts else None\n            ),\n',
    )


def migrate_app() -> None:
    path = ROOT / "docs" / "app.js"
    replace_once(
        path,
        '''const amountMethodLabels = {
  sector_implied: '業種平均基礎額による参考換算',
  official_company_base_projection: '会社公式の季別モデル額から年間投影'
};
''',
        '''const amountMethodLabels = {
  matched_sector_sample: '同一業種標本による参考換算',
  official_company_base_projection: '会社公式の季別モデル額から年間投影',
  not_estimable_from_available_samples: '対応標本がなく算定不可'
};
''',
    )
    replace_once(
        path,
        "function yen(value) { return `¥${number(Math.round(Number(value)))}`; }\n",
        "function yen(value) { return value == null ? '—' : `¥${number(Math.round(Number(value)))}`; }\n",
    )
    replace_once(
        path,
        "function percent(value) { return `${Math.round(Number(value) * 100)}%`; }\n",
        "function percent(value) { return value == null ? '—' : `${Math.round(Number(value) * 100)}%`; }\n",
    )
    replace_once(
        path,
        '      <dt>金額算定</dt><dd>${escapeHtml(amountMethodLabels[estimate.amount_method] || estimate.amount_method)}</dd>\n',
        '      <dt>金額算定</dt><dd>${escapeHtml(amountMethodLabels[estimate.amount_method] || estimate.amount_method)}</dd>\n'
        '      <dt>金額可用性</dt><dd>${escapeHtml(estimate.amount_status)}</dd>\n'
        '      <dt>金額標本</dt><dd>${escapeHtml(estimate.amount_conversion?.amount_sample_id || "—")}</dd>\n'
        '      <dt>月数標本</dt><dd>${escapeHtml(estimate.amount_conversion?.months_sample_id || "—")}</dd>\n',
    )
    replace_once(
        path,
        '''  const amount = estimate.amount_yen;
  const anchor = estimate.anchors;
''',
        '''  const amount = estimate.amount_yen;
  const amountCell = amount && amount.central != null
    ? `<strong>${yen(amount.central)}</strong><span>${yen(amount.minimum)}–${yen(amount.maximum)}</span><small>${escapeHtml(amountMethodLabels[estimate.amount_method] || '参考換算')}</small>`
    : `<strong>算定不可</strong><span>金額・月数の対応標本なし</span><small>${escapeHtml(amountMethodLabels[estimate.amount_method] || estimate.amount_method)}</small>`;
  const anchor = estimate.anchors;
''',
    )
    replace_once(
        path,
        '    <td class="numeric"><strong>${yen(amount.central)}</strong><span>${yen(amount.minimum)}–${yen(amount.maximum)}</span><small>${escapeHtml(amountMethodLabels[estimate.amount_method] || \'参考換算\')}</small></td>\n',
        '    <td class="numeric">${amountCell}</td>\n',
    )
    replace_once(
        path,
        '    <td><span class="confidence confidence-${escapeHtml(estimate.confidence.level)}">${escapeHtml(confidenceLabels[estimate.confidence.level])} ${percent(estimate.confidence.score)}</span><small>金額 ${percent(estimate.confidence.amount_score)}</small></td>\n',
        '    <td><span class="confidence confidence-${escapeHtml(estimate.confidence.level)}">${escapeHtml(confidenceLabels[estimate.confidence.level])} ${percent(estimate.confidence.score)}</span><small>金額 ${estimate.confidence.amount_score == null ? \'算定不可\' : percent(estimate.confidence.amount_score)}</small></td>\n',
    )
    replace_once(
        path,
        "  if (key === 'amount') return record.estimate.amount_yen.central;\n",
        "  if (key === 'amount') return record.estimate.amount_yen?.central ?? Number.NEGATIVE_INFINITY;\n",
    )


def migrate_validator() -> None:
    path = ROOT / "scripts" / "validate_pages.py"
    replace_once(path, '    assert public["schema_version"] == 4\n', '    assert public["schema_version"] == 5\n')
    replace_once(
        path,
        '    assert public["summary"]["median_estimated_amount_yen"] > 0\n',
        '    assert public["summary"]["amount_available_company_count"] > 0\n'
        '    assert public["summary"]["amount_unavailable_company_count"] > 0\n'
        '    assert public["summary"]["median_estimated_amount_yen"] > 0\n',
    )
    replace_once(
        path,
        '''        assert 0 < amount["minimum"] <= amount["central"] <= amount["maximum"]
        assert 0 <= estimate["confidence"]["score"] <= 1
        assert 0 <= estimate["confidence"]["amount_score"] <= 1
''',
        '''        if estimate["amount_status"] == "available":
            assert isinstance(amount, dict)
            assert 0 < amount["minimum"] <= amount["central"] <= amount["maximum"]
            assert 0 <= estimate["confidence"]["amount_score"] <= 1
            assert estimate["amount_conversion"]["status"] in {"matched_sample", "company_official"}
        else:
            assert estimate["amount_status"] == "unavailable"
            assert amount is None
            assert estimate["confidence"]["amount_score"] is None
            assert estimate["amount_method"] == "not_estimable_from_available_samples"
            assert estimate["amount_conversion"]["matched_population"] is False
        assert 0 <= estimate["confidence"]["score"] <= 1
''',
    )
    replace_once(
        path,
        '        "record.estimate", "estimate.months", "estimate.amount_yen", "estimate.weights",\n',
        '        "record.estimate", "estimate.months", "estimate.amount_yen", "estimate.amount_status", "estimate.weights",\n',
    )


def migrate_tests() -> None:
    path = ROOT / "tests" / "test_company_estimates.py"
    replace_once(
        path,
        '                "sample_amount": {"organizations": 1003, "workers": 739115},\n                "source_url": "https://example.com/rengo.pdf",\n',
        '                "sample_amount": {"organizations": 1003, "workers": 739115},\n'
        '                "amount_conversion": {\n'
        '                    "status": "unavailable",\n'
        '                    "amount_sample_id": "rengo:manufacturing:amount",\n'
        '                    "months_sample_id": "rengo:manufacturing:months",\n'
        '                    "matched_population": False,\n'
        '                    "aggregation": "worker_weighted_average",\n'
        '                    "reason": "different respondent samples",\n'
        '                },\n'
        '                "source_url": "https://example.com/rengo.pdf",\n',
    )
    replace_once(
        path,
        '        self.assertEqual(result["mechanism"]["upside_profile"], "medium")\n',
        '        self.assertEqual(result["mechanism"]["upside_profile"], "medium")\n'
        '        self.assertIsNone(result["amount_yen"])\n'
        '        self.assertEqual(result["amount_status"], "unavailable")\n',
    )
    replace_once(
        path,
        '''        self.assertEqual(result["amount_method"], "sector_implied")
        self.assertIn("業種平均基本月額", result["amount_caution"])
''',
        '''        self.assertEqual(result["amount_method"], "not_estimable_from_available_samples")
        self.assertEqual(result["amount_status"], "unavailable")
        self.assertIsNone(result["amount_yen"])
        self.assertIn("回答標本", result["amount_caution"])
''',
    )
    replace_once(
        path,
        '        self.assertEqual(result["amount_method"], "official_company_base_projection")\n        self.assertGreater(result["amount_yen"]["central"], 2_000_000)\n',
        '        self.assertEqual(result["amount_method"], "official_company_base_projection")\n'
        '        self.assertEqual(result["amount_status"], "available")\n'
        '        self.assertEqual(result["amount_conversion"]["status"], "company_official")\n'
        '        self.assertGreater(result["amount_yen"]["central"], 2_000_000)\n',
    )
    addition = '''
    def test_matched_sector_sample_allows_amount_conversion(self):
        payload = model()
        conversion = payload["sectors"]["manufacturing"]["amount_conversion"]
        conversion.update(
            {
                "status": "matched_sample",
                "amount_sample_id": "rengo:manufacturing:matched",
                "months_sample_id": "rengo:manufacturing:matched",
                "matched_population": True,
                "reason": "same respondents answered both fields",
            }
        )
        validated = validate_company_estimation_model(payload, {"6146"})
        result = build_company_estimates({"6146": hypothesis()}, [], validated)["6146"]
        self.assertEqual(result["amount_status"], "available")
        self.assertEqual(result["amount_method"], "matched_sector_sample")
        self.assertGreater(result["amount_yen"]["central"], 0)
        self.assertGreater(result["amount_conversion"]["monthly_base_yen"], 0)

    def test_rejects_mismatched_ids_claimed_as_matched_population(self):
        payload = model()
        conversion = payload["sectors"]["manufacturing"]["amount_conversion"]
        conversion.update(
            {
                "status": "matched_sample",
                "matched_population": True,
            }
        )
        with self.assertRaises(ValidationError):
            validate_company_estimation_model(payload, {"6146"})

'''
    insert_before_once(path, '\n\nif __name__ == "__main__":\n', addition)


def migrate_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "7. 参考換算額は、業種実測金額 ÷ 業種実測月数で得た業種平均基本月額を使う。個社の実支給額とは扱わない。\n",
        "7. 参考換算額は、金額と月数が同一標本で対応する場合、または会社公式モデル額がある場合だけ算出する。\n",
    )
    text = text.replace(
        "8. 会社・労組の明示値が得られた時点で、推定を確認事実へ置換する。\n",
        "8. 金額標本と月数標本が異なる場合は金額をnull・unavailableとし、月数推定だけを公開する。\n9. 会社・労組の明示値が得られた時点で、推定を確認事実へ置換する。\n",
    )
    text = text.replace("同じスキーマ4", "同じスキーマ5")
    section = '''
## 金額換算の可用性

業種集計の年間一時金額と年間月数について、回答組織・回答労働者が異なる場合、その二つの平均値の比を基本月額とは扱いません。対応標本を確認できない会社は`amount_status: unavailable`、`amount_yen: null`として公開し、0円とは区別します。

金額を公開できる経路は次の二つです。

- 同一標本について金額と月数が対応している`matched_sector_sample`
- 会社公式の季別モデル額と月数を使う`official_company_base_projection`

標本ID、対応可否、集計方式、算定理由は`amount_conversion`へ保存します。旧方式との差分は`audit/amount_conversion_diff.json`で監査できます。

'''
    if "## 金額換算の可用性" not in text:
        marker = "## 2026年の業種実測アンカー\n"
        if marker not in text:
            raise RuntimeError("README insertion marker not found")
        text = text.replace(marker, section + marker, 1)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    migrate_model()
    migrate_company_estimates()
    migrate_public_generator()
    migrate_app()
    migrate_validator()
    migrate_tests()
    migrate_readme()
    print("PASS: issue 15 source migration applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
