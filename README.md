# bonus — 従業員賞与制度の一次情報データベース

**Repository:** https://github.com/KAFKA2306/bonus

日経平均採用企業のうち、このリポジトリで既に管理している銘柄を対象に、従業員賞与の決定方式を一次情報から記録する調査プロジェクトです。

## 更新方針

- `nikkei225_companies.yaml`の銘柄集合は固定し、今回の更新では変更しません。
- 従業員賞与だけを対象とし、役員報酬は集計しません。
- 会社公式、労働組合公式、法定開示資料だけを検証済み根拠として扱います。
- 月数・回数は資料に明記された値だけを保存し、文章や口コミから推定しません。
- 一次情報で確認できない項目は`null`または`unknown`として残します。

## 現在の正規データ

- 固定銘柄スナップショット: `nikkei225_companies.yaml`
- 検証済み事実: `data/verified_bonus_facts_2026-08-02.yaml`
- 自動集計: `analysis/summary/verified_bonus_overview.yaml`
- 検証ロジック: `scripts/generate_verified_bonus_summary.py`
- 互換エントリーポイント: `scripts/generate_bonus_summary.py`

`nikkei225_bonus_survey_2024_en.yaml`と従来の`companies/`配下は履歴資料です。出典URLのない月数や推定値を含むため、現在の検証済み集計には使用しません。

## データ状態

2026年8月2日時点の一次情報を反映しています。

| 証券コード | 会社 | 状態 | 確認内容 |
|---|---|---|---|
| 6146 | ディスコ | confirmed | 年4回、年4.0か月を下限とする業績連動テーブル、Will賞与の二段階配分 |
| 9433 | KDDI | confirmed | 会社業績賞与と個人業績連動賞与、サステナビリティKPI連動 |
| 6503 | 三菱電機 | confirmed | 前年度の成果・行動評価による総合評価を賃金・賞与へ直接反映 |
| 6758 | ソニーグループ | partially_confirmed | 2026年度新卒募集要項の対象会社では業績給を年1回 |
| 6861 | キーエンス | unknown | 最新法定開示は確認済みだが、賞与原資・月数・配分式は一次情報で未確認 |

各事実のURL、対象範囲、確認日は検証済みスナップショット内に保存しています。

## 実行方法

```bash
python -m pip install PyYAML==6.0.3
python -m unittest discover -s tests -v
python scripts/generate_bonus_summary.py --check
python scripts/generate_bonus_summary.py
```

`--check`は次を検証します。

- データ内の証券コードが固定銘柄集合に含まれること
- 重複コードがないこと
- 検証済みレコードにHTTPSの一次情報URLがあること
- `unknown`に分類や数値を混入させていないこと
- 月数が文章から推定された値ではないこと
- 役員報酬を従業員賞与として扱っていないこと

## 集計ルール

`explicit_point_months_average`は、`confirmed`かつ一次資料に単一の年換算月数が明記されたレコードだけで計算します。下限、上限、範囲、推定値、口コミ値は平均へ入れません。該当値がなければ`null`です。

## CI

`.github/workflows/verified-bonus.yml`が、テスト、固定銘柄監査、スナップショット検証、集計再生成、生成物差分の有無を確認します。

**最終監査:** 2026-08-02
