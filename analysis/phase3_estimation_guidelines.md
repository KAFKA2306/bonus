# Phase 3 Estimation Guidelines

Phase 3 relies on structured estimation instead of primary-source confirmation. Use this checklist when populating `phase3_estimation_template.yaml` for each company.

## Required Elements

- **company_profile.phase3_reason**: Capture the rationale for remaining in Phase 3 (e.g. "情報公開が限定的な中堅製造業").
- **bonus_system_estimate.classification**: Keep within the standard four-type taxonomy; note that "推定" is implied so avoid qualifiers like "可能性".
- **bonus_system_estimate.estimation_summary**: Two-three sentences summarising the logic, citing the most influential indicators, patterns, or peer analogues.
- **estimation_inputs.indicators**: Prioritize quantitative metrics available from financial statements, industry averages, or analyst reports. Include at least one profitability or payout benchmark.
- **estimation_inputs.comparable_companies**: List up to three peers with known bonus structures that anchor the estimate.
- **validation_plan.immediate_checks**: Define quick follow-up actions, such as "Phase 2でEDINET確認予定" or "次期決算説明会で確認".

## Confidence Calibration

- `confidence_level` should default to **C** unless the estimate rests on a single weak signal (then **D**).
- `reliability_score` range: 30–60. Anchor scores near 30 when relying on broad industry averages; move toward 60 when peer comparables closely match.

## Workflow Tips

1. Start with industry-wide assumptions. Reference `analysis/manufacturing_bonus_patterns_analysis.yaml` or sector notes for baseline multiples.
2. Note any outlier signals (union activity, restructuring) under `qualitative_signals` to revisit during Phase 2 escalations.
3. Update `meta.analyst` with your handle and keep `version` at 1 unless the structure itself changes.

## Output Expectations

- Store completed estimation records under `analysis/phase3_estimates/` (create if absent) with filenames `{stock_code}_{shortname}_phase3.yaml`.
- Maintain ASCII-only text; keep lines under 120 characters for readability.

Following this template keeps the Phase 3 dataset consistent while flagging which assumptions need future validation.
