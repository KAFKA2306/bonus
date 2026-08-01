# bonus — 従業員賞与制度の一次情報データベース

**Dashboard:** https://kafka2306.github.io/bonus/  
**Repository:** https://github.com/KAFKA2306/bonus

既存の調査銘柄を対象に、従業員賞与の決定方式を一次情報から記録するプロジェクトです。GitHub Pagesでは、確認状態、制度分類、支給回数、明示月数、根拠URLを企業別に閲覧できます。

## 更新方針

- `nikkei225_bonus_survey_2024_en.yaml`の既存30社を固定し、銘柄追加・削除を行いません。
- `nikkei225_companies.yaml`は過去の225銘柄スナップショットとして変更しません。
- 従業員賞与だけを対象とし、役員報酬は集計しません。
- 会社公式、労働組合公式、法定開示資料だけを検証済み根拠として扱います。
- 月数・回数は一次資料に明記された値だけを保存し、文章や口コミから推定しません。
- 一次情報で確認できない項目は`null`または`unknown`として残します。

## 正規データと生成物

- 固定調査銘柄: `nikkei225_bonus_survey_2024_en.yaml`
- 過去の225銘柄スナップショット: `nikkei225_companies.yaml`
- 検証済み事実: `data/verified_bonus_facts_YYYY-MM-DD.yaml`
- 自動集計: `analysis/summary/verified_bonus_overview.yaml`
- 公開ビュー: `docs/`
- 公開JSON: `docs/data/bonus.json`（デプロイ時に最新スナップショットから生成）
- 事実検証: `scripts/generate_verified_bonus_summary.py`
- Pages JSON生成: `scripts/generate_pages_data.py`
- Pages監査: `scripts/validate_pages.py`

`docs/data/bonus.json`は編集対象ではありません。`data/`配下の最新スナップショットを正として、Pages workflowとローカルコマンドが決定論的に生成します。

従来の`companies/`配下は履歴資料です。出典URLのない月数や推定値を含むため、現在の検証済み集計には使用しません。

## データ状態

2026年8月2日時点の一次情報を反映しています。

| 証券コード | 会社 | 状態 | 確認内容 |
|---|---|---|---|
| 6146 | ディスコ | confirmed | 年4回、年4.0か月を下限とする業績連動テーブル、Will賞与の二段階配分 |
| 9433 | KDDI | confirmed | 会社業績賞与と個人業績連動賞与、サステナビリティKPI連動 |
| 6503 | 三菱電機 | confirmed | 前年度の成果・行動評価による総合評価を賃金・賞与へ直接反映 |
| 6758 | ソニーグループ | partially_confirmed | 2027年4月入社向け募集要項で、2026年度予定の業績給を年1回と確認 |
| 6861 | キーエンス | unknown | 最新法定開示は確認済みだが、賞与原資・月数・配分式は一次情報で未確認 |

各事実のURL、対象範囲、確認日は検証済みスナップショット内に保存しています。

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

- データ内の証券コードが既存30社の固定集合に含まれること
- 重複コードがないこと
- 検証済みレコードにHTTPSの一次情報URLがあること
- `unknown`に分類や数値を混入させていないこと
- 月数が文章から推定された値ではないこと
- 役員報酬を従業員賞与として扱っていないこと
- 公開JSONが最新スナップショットの全公開項目と完全一致すること
- 日本語検索、フィルター状態、結果件数、キーボードフォーカス、文字コントラストが監査契約を満たすこと

## 集計ルール

`explicit_point_months_average`は、`confirmed`かつ一次資料に単一の年換算月数が明記されたレコードだけで計算します。下限、上限、範囲、推定値、口コミ値は平均へ入れません。該当値がなければ`null`です。

## CI / Pages

- `.github/workflows/verified-bonus.yml`: 単体テスト、固定銘柄監査、スナップショット検証、集計再生成
- `.github/workflows/pages.yml`: 最新JSON生成、UI・アクセシビリティ監査、JavaScript構文確認、Pagesデプロイ
- デプロイ後に公開HTML・CSS・JavaScript・JSONを再取得し、リポジトリから生成した成果物とのバイト一致を確認します。

**最終監査:** 2026-08-02
