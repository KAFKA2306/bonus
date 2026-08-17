# 主要30社 賞与定量モデル

[![Verify bonus facts](https://github.com/KAFKA2306/bonus/actions/workflows/verified-bonus.yml/badge.svg)](https://github.com/KAFKA2306/bonus/actions/workflows/verified-bonus.yml)
[![Deploy Nikkei 225 quantified bonus dashboard](https://github.com/KAFKA2306/bonus/actions/workflows/pages.yml/badge.svg)](https://github.com/KAFKA2306/bonus/actions/workflows/pages.yml)
[![Audit live bonus dashboard](https://github.com/KAFKA2306/bonus/actions/workflows/live-audit.yml/badge.svg)](https://github.com/KAFKA2306/bonus/actions/workflows/live-audit.yml)

**賞与は「会社別の数字」を集めるほど、比較を間違えやすくなる。**

会社公式の明示値、労組集計、業種平均、モデル推定は同じ数字ではありません。このrepositoryは、主要30社の賞与を **「確認できた事実」と「比較のための推定」を混ぜずに並べ、どこまで信じてよいか判断できる状態**へ変える定量比較基盤です。

- 公開ページ: https://kafka2306.github.io/bonus/
- Repository: https://github.com/KAFKA2306/bonus

## Vision

賞与比較を「大きい金額順のランキング」から、**会社ごとの制度・月数・推定range・根拠・不確実性を同じ画面で読み、比較条件まで確認できる体験**へ変えます。

利用者が知りたいのは単一点ではありません。

- その値は会社公式なのかモデル推定なのか
- 年間月数と季別金額を混ぜていないか
- 金額と月数は同じ標本なのか
- 同一sampleでなければ、なぜ金額を出していないのか
- どのsector実測をanchorにしたか
- 一次資料が見つかったら、推定はどう置換されるか

## Design philosophy

- **Fact and estimate stay separate.** verified factをmodel valueで上書きしない。
- **Null is not zero.** 対応sampleを確認できない金額は`null`として残し、0円へ変換しない。
- **Range before false precision.** point estimateだけでなくlower / center / upper、weight、confidenceを公開する。
- **Matched samples before currency conversion.** amountとmonthsのsampleが一致しないなら比を基本月額として扱わない。
- **Primary evidence overrides priors.** company / unionの明示値が得られたら推定を確認事実へ置換する。
- **Universe stays fixed for comparison.** 主要30社を途中で都合よく入れ替えず、coverageと比較条件をCIで固定する。

## Why / 差別化

賞与記事やdashboardは、金額・月数の大きさだけを比較しやすいです。しかし平均金額と平均月数の回答sampleが違えば、その比から作った「基本月額」は存在しない架空の値になります。

このrepositoryの差別化は経験ベイズ式そのものではなく、**比較したいからといって対応していない数字を無理に接続しないこと**です。

その結果、値が出せないcompanyは空欄ではなく「なぜ unavailable なのか」まで説明できます。

## Reader journey

```text
companyを選ぶ
  → verified factsを見る
  → estimated months rangeを見る
  → sector anchor / company weightを見る
  → amount availabilityを確認
  → source / sample boundaryを見る
  → comparisonに使える範囲を判断
```

## Root logic

1. company / union / EDINET / TDnetの制度・回数・月数をverified factとして保存
2. 個社年間月数が不明なら旧調査rangeをcompany priorとして保持
3. 連合2026最終集計のsector annual bonusをobserved anchorとして使用
4. evidence strengthに応じてcompany priorをsector observationへshrink
5. lower / center / upper、company / sector weight、confidenceを公開
6. primary sourceのexplicit constraintをmodelより優先
7. amountはmatched sampleまたはofficial company projectionだけで算出
8. sample mismatchなら`amount_status: unavailable`, `amount_yen: null`
9. verified company fact取得時にestimateを置換

## Estimation model

```text
estimated annual months
  = company_weight × company_prior_center
  + sector_weight × sector_observed_months

sector_weight = 1 - company_weight
```

company weightはprior confidenceと一次資料の有無を反映し、現行modelでは0.55〜0.90に制限します。

amount conversionを許可する経路:

- `matched_sector_sample`
- `official_company_base_projection`

sample ID / match status / aggregation method / reasonは`amount_conversion`へ保存します。

## 2026 sector anchors

`data/company_estimation_model_YYYY-MM-DD.yaml` がsector observationとcompany assignmentを保持します。

代表例:

| sector | annual months | annual amount | example allocation |
|---|---:|---:|---|
| 製造業 | 5.44 | 1,854,847円 | トヨタ、ソニー、東京エレクトロン、ディスコ等 |
| 商業流通 | 3.87 | 1,169,622円 | ファーストリテイリング |
| 交通運輸 | 4.42 | 917,078円 | JR東日本、JR東海、ANA、JAL |
| サービス・ホテル | 4.04 | 890,000円 | リクルートHD |
| 情報・出版 | 5.42 | 1,770,611円 | KDDI、NTT、野村総合研究所 |
| その他 | 4.39 | 1,749,584円 | ソフトバンクグループ、MUFG、証券2社 |

最新の正準値はdata fileを優先し、README固定値を正本とはしません。

## Official benchmark snapshots

`data/quantitative_benchmarks_YYYY-MM-DD.yaml` には比較用の公式集計を保存します。

- 連合 2026春季生活闘争 最終集計
- 経団連 2026年夏季賞与・一時金 第1回集計
- 厚生労働省 民間主要企業 2025年夏季・年末一時金

amount / months、request / settlement、annual / seasonal、weighted / company averageを別seriesとして保持します。

## Public dashboard

表示するもの:

- company-level estimated months range
- reference amount range when available
- sector observed anchor
- method / frequency
- confidence
- source / equation / assumption / falsification condition
- official quantitative benchmark

主表を「未調査queue」にはせず、比較可能な30社universeとして表示します。

## Canonical data

- `nikkei225_bonus_survey_2024_en.yaml` — fixed universe
- `data/verified_bonus_facts_YYYY-MM-DD.yaml` — verified facts
- `data/bonus_hypotheses_YYYY-MM-DD.yaml` — company priors
- `data/company_estimation_model_YYYY-MM-DD.yaml` — sector anchors / model config
- `data/source_survey_YYYY-MM-DD.yaml` — source ledger
- `data/quantitative_benchmarks_YYYY-MM-DD.yaml` — official benchmark snapshots
- `scripts/company_estimates.py` — estimator
- `scripts/generate_pages_data.py` — public JSON generation
- `scripts/validate_pages.py` — Pages audit
- `scripts/audit_live_pages.py` — production audit

## Quality gate

CI verifies at least:

- fixed 30-company universe
- priors cover exactly 30 companies
- sector assignment covers exactly 30 without duplicate
- sector months / amount / sample / official URL completeness
- ordered positive estimate ranges
- company + sector weights = 1
- explicit verified constraints override model
- verified facts and estimates remain separate fields
- unmatched amount samples stay unavailable/null
- public JSON / HTML / CSS / JS schema consistency
- deployed artifact byte parity

## Local verification

```bash
python -m pip install PyYAML==6.0.3
python -m unittest discover -s tests -v
python scripts/generate_bonus_summary.py --check
python scripts/generate_pages_data.py
python scripts/validate_pages.py
node --check docs/app.js
```

`docs/data/bonus.json` と `gh-pages` branchは直接編集せず、canonical inputsから生成します。

## Done

成功指標は30社すべてに金額を埋めることではありません。

**比較できる値と比較できない値を分け、利用者が「これは事実・これは推定・これはsample不一致で算出不可」と判断できること**をDoneとします。
