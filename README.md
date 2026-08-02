# 主要30社 賞与制度・推定比較データベース

**比較表:** https://kafka2306.github.io/bonus/  
**Repository:** https://github.com/KAFKA2306/bonus

主要30社の従業員賞与について、一次情報で確認した事実と、未確認部分に対する仮説推定を分離して管理するプロジェクトです。GitHub Pagesでは、30社を1つの比較表で検索・並べ替えでき、各行から根拠、前提、反証条件、一次資料を展開できます。

## 基本方針

- `nikkei225_bonus_survey_2024_en.yaml`の既存30社を固定し、銘柄追加・削除を行いません。
- 従業員賞与だけを対象とし、役員報酬は除外します。
- 会社公式、労働組合公式、法定開示資料だけを確認事実として扱います。
- 一次情報で月数を確定できない場合は、別レイヤーに検証可能な仮説レンジを置きます。
- 仮説値を確認済み月数の平均へ混入させません。

## 公開比較表

ページ名は**「主要30社 賞与制度・推定比較表」**です。

表示列:

- 会社名・証券コード
- 一次情報の確認状態
- 制度分類
- 一次情報で確認した年換算月数
- 仮説の下限・中心値・上限
- 支給回数
- 仮説確度
- 根拠・前提・反証条件・一次資料

列見出しによる並べ替え、全文検索、確認状態フィルター、固定ヘッダー、先頭列固定、モバイル横スクロールに対応します。

## 二層データモデル

### 確認事実

`data/verified_bonus_facts_YYYY-MM-DD.yaml`

- 一次情報URL、対象範囲、確認日を必須化
- 月数・回数は一次資料に明記された値だけを格納
- 確認できない項目は`null`または`unknown`

### 仮説推定

`data/bonus_hypotheses_YYYY-MM-DD.yaml`

- 固定Universe 30社すべてを収載
- 年換算月数を下限・中心値・上限で提示
- 制度分類と支給回数の仮説を保持
- 確度レベルと0〜1のスコアを表示
- 推定根拠、成立前提、反証条件を必須化
- `not_for_verified_aggregate: true`を必須化

推定は正解値ではありません。新しい公式資料が出たときに棄却・更新できる状態にすることが目的です。

## 現在のカバレッジ

- 固定Universe: 30社
- 公開収載: 30社
- 仮説推定: 30社
- カバレッジ: 100%
- 一次情報による制度監査済み: 5社

## 正規データと生成物

- 固定調査銘柄: `nikkei225_bonus_survey_2024_en.yaml`
- 確認事実: `data/verified_bonus_facts_YYYY-MM-DD.yaml`
- 仮説推定: `data/bonus_hypotheses_YYYY-MM-DD.yaml`
- 確認事実の自動集計: `analysis/summary/verified_bonus_overview.yaml`
- 公開ビュー: `docs/`
- 公開JSON: `docs/data/bonus.json`
- 公開ブランチ: `gh-pages`（workflow生成物のみ。直接編集しない）
- 事実検証: `scripts/generate_verified_bonus_summary.py`
- 仮説検証: `scripts/bonus_hypotheses.py`
- Pages JSON生成: `scripts/generate_pages_data.py`
- Pages監査: `scripts/validate_pages.py`
- 本番監査: `scripts/audit_live_pages.py`

`docs/data/bonus.json`と`gh-pages`ブランチは直接編集しません。`main`の正規データから生成し、全テスト通過後に公開します。

## 実行方法

```bash
python -m pip install PyYAML==6.0.3
python -m unittest discover -s tests -v
python scripts/generate_bonus_summary.py --check
python scripts/generate_pages_data.py
python scripts/validate_pages.py
node --check docs/app.js
```

検証処理は次を確認します。

- 仮説の証券コード集合が固定Universe 30社と完全一致すること
- 確認事実と仮説の双方で重複コードがないこと
- 確認事実にHTTPSの一次情報URLがあること
- 仮説レンジが`minimum <= central <= maximum`を満たすこと
- 仮説に確度、根拠、前提、反証条件があること
- 仮説がverified集計から除外されること
- 公開JSONが最新の事実・仮説スナップショットと一致すること
- Pagesがカードではなく比較テーブルであること
- 日本語タイトル、検索、列ソート、状態フィルター、アクセシビリティ、文字コントラストが監査契約を満たすこと

## CI / Pages

- `.github/workflows/verified-bonus.yml`: 確認事実、仮説、固定Universeの単体テスト
- `.github/workflows/pages.yml`: JSON生成、テーブルUI監査、JavaScript構文確認、`gh-pages`同期、本番バイト一致監査
- `.github/workflows/live-audit.yml`: 公開中のHTML・CSS・JavaScript・JSONを独立監査

**最終UI更新:** 2026-08-02
