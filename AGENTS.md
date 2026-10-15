# Repository Guidelines

## Project Structure & Module Organization
- `companies/` – per-company YAML dossiers grouped by sector (`tech/6758_sony.yaml`).
- `analysis/` – aggregated outputs (`summary/` YAMLs) and generated charts (`graphs/`).
- `scripts/` – Python utilities for data curation (`generate_bonus_summary.py`, `plot_bonus_graphs.py`).
- `reports/`, `sectors/` – narrative summaries and sector-level analyses.
- Root YAMLs (`nikkei225_bonus_survey_2024*.yaml`) provide master survey data; treat them as the single source of truth when syncing files below them.

## Build, Test, and Development Commands
- `python3 scripts/generate_bonus_summary.py` – refreshes aggregate YAML metrics under `analysis/summary/`.
- `python3 scripts/plot_bonus_graphs.py` – regenerates bar charts in `analysis/graphs/` (install a CJK font for clean labels).
- `python3 scripts/visualizer.py --chart stats-table` – rebuilds the large comparison table (`analysis/graphs/company_statistics_table.png`).
- `python3 scripts/data_gap_analyzer.py` – optional helper to highlight missing fields across YAML sources.

## Coding Style & Naming Conventions
- Python: keep to PEP 8, 4-space indentation, and ASCII unless a source requires Unicode (e.g., Japanese names).
- YAML files use snake_case keys, quoted strings for Japanese text, and filenames `{stock_code}_{slug}.yaml` within sector folders.
- Prefer descriptive key names (`bonus_system`, `performance_metrics`) and retain comment blocks at file tops.

## Testing Guidelines
- After editing YAML, run `python3 scripts/generate_bonus_summary.py` to surface schema errors.
- Use `python3 scripts/plot_bonus_graphs.py` to confirm new data integrates into visuals without nulls.
- For bulk updates, execute `python3 scripts/data_gap_analyzer.py --report` and address flagged omissions before committing.

## Commit & Pull Request Guidelines
- Follow imperative, scope-aware messages (e.g., “Update bonus metrics and regenerate graphs”).
- Group related data/script changes together; include regenerated artifacts in the same commit.
- PRs should summarize affected companies or sectors, note regenerated outputs, link to supporting sources, and attach chart diffs if visuals change.
- Ensure CI-equivalent steps: rerun summary generation and visualization before requesting review.

## Agent-Specific Tips
- Keep generated binaries (`analysis/graphs/*.png`) in sync with YAML changes.
- Record source citations directly inside YAML `evidence` blocks to simplify future fact checks.
- When adding estimates (Phase 3), populate `analysis/phase3_estimates/` and re-run summary scripts to maintain coverage tables.

## Outcome-first task interpretation (Altitude / Depth / Range / Speed)

- **Altitude:** Interpret a task as a verifiable user outcome, not only the named intermediate step. Infer the smallest justified acceptance criteria from the request and current repository state; do not silently expand scope or override explicit instructions.
- **Depth:** Use primary/current evidence and real domain tools and methods. Verify the actual postcondition independently of agent self-report, preferably on the exact changed revision or target runtime.
- **Range:** Combine code, data, tests, runtime/UI, and domain expertise when that materially improves the outcome, while respecting canonical owners and avoiding duplicate systems.
- **Speed:** Complete authorized read-only and reversible work through implementation, focused checks, and read-back without unnecessary approval loops. Bound retries and stop at the proven outcome.
- These principles never bypass repository-specific security, privacy, permission, publication, spending, destructive-action, or release boundaries. State verified results and any remaining `UNVERIFIED` layers.
