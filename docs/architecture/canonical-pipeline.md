# Canonical bonus pipeline

Issue #33 の ratchet として、賞与比較の正準経路を次の1本に固定する。

```text
canonical inputs
  data/verified_bonus_facts_YYYY-MM-DD.yaml
  data/bonus_hypotheses_YYYY-MM-DD.yaml
  data/company_estimation_model_YYYY-MM-DD.yaml
  data/source_survey_YYYY-MM-DD.yaml
  data/quantitative_benchmarks_YYYY-MM-DD.yaml
        ↓
canonical calculation
  scripts/company_estimates.py
        ↓
canonical public materialization
  scripts/generate_pages_data.py
        ↓
  docs/data/bonus.json
        ↓
  docs/index.html + docs/app.js
```

## Boundary

- `docs/data/bonus.json` は生成物であり、手編集しない。
- `scripts/` は実行コードだけを置く。PNGなどの生成物を保存しない。
- 確認事実とモデル推定を同じ値として扱わない。
- 金額と月数の標本対応が確認できない場合、金額を `0` や推定値で補完しない。
- 数式・モデル重みの変更は `scripts/company_estimates.py` と対応する入力データ・テストを同時に変更する。

## KPI

複雑性の監視は次の3点だけとする。

1. canonical pipeline の再生成テスト成功率
2. 手動補正が必要な公開レコード数
3. `scripts/` に混入した生成artifact件数

## CI contract

`tests/test_repository_ratchet.py` が以下を直接検証する。

- 正準入力・計算・出力の主要pathが存在する
- `scripts/` にPNG生成物を置かない
- 既知の重複artifactを再導入しない

既存の `pages.yml` はさらに unit test、決定論的生成、schema/境界条件、JavaScript syntax、生成後のclean checkoutを検証する。
