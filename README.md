# bonus — 従業員賞与制度の事実・仮説データベース

**Dashboard:** https://kafka2306.github.io/bonus/  
**Repository:** https://github.com/KAFKA2306/bonus

既存の調査銘柄を対象に、従業員賞与の決定方式を「確認事実」と「仮説推定」に分けて記録するプロジェクトです。GitHub Pagesでは、一次情報で確認した制度と、月数レンジ・確度・前提・反証条件を持つ推定を企業別に閲覧できます。

## 基本方針

- `nikkei225_bonus_survey_2024_en.yaml`の既存30社を固定し、銘柄追加・削除を行いません。
- `nikkei225_companies.yaml`は過去の225銘柄スナップショットとして変更しません。
- 従業員賞与だけを対象とし、役員報酬は集計しません。
- 会社公式、労働組合公式、法定開示資料だけを確認事実として扱います。
- 一次情報で月数を確定できない場合も`unknown`だけで止めず、別レイヤーに検証可能な仮説レンジを置きます。
- 仮説は確認済み月数の平均へ混入させません。

## 二層データモデル

### 1. 確認事実

`data/verified_bonus_facts_YYYY-MM-DD.yaml`

- 一次情報URL、対象範囲、確認日を必須化
- 月数・回数は一次資料に明記された値だけを格納
- 確認できない項目は`null`または`unknown`

### 2. 仮説推定

`data/bonus_hypotheses_YYYY-MM-DD.yaml`

- 年換算月数を下限・中心値・上限で提示
- 制度分類と支給回数の仮説を保持
- 確度レベルと0〜1のスコアを表示
- 旧調査値、確認事実、業界構造などの推定根拠を明示
- 前提と反証条件を必須化
- `not_for_verified_aggregate: true`を必須化

推定は正解値ではありません。新しい公式資料が出たときに棄却・更新できる状態へすることが目的です。

## 現在の仮説レンジ

| 証券コード | 会社 | 仮説レンジ | 中心値 | 確度 |
|---|---|---:|---:|---|
| 6146 | ディスコ | 9.0〜12.0か月 | 10.5か月 | 中 0.58 |
| 9433 | KDDI | 4.5〜5.5か月 | 5.0か月 | 中 0.55 |
| 6503 | 三菱電機 | 5.0〜6.0か月 | 5.5か月 | 中 0.54 |
| 6758 | ソニーグループ | 0.5〜1.5か月 | 1.0か月 | 低 0.42 |
| 6861 | キーエンス | 6.0〜8.0か月 | 7.0か月 | 低 0.40 |

各レンジの根拠、前提、反証条件は仮説スナップショットと公開カードに保存しています。

## 正規データと生成物

- 固定調査銘柄: `nikkei225_bonus_survey_2024_en.yaml`
- 確認事実: `data/verified_bonus_facts_YYYY-MM-DD.yaml`
- 仮説推定: `data/bonus_hypotheses_YYYY-MM-DD.yaml`
- 確認事実の自動集計: `analysis/summary/verified_bonus_overview.yaml`
- 公開ビュー: `docs/`
- 公開JSON: `docs/data/bonus.json`（最新の事実・仮説から生成）
- 公開ブランチ: `gh-pages`（workflow生成物のみ。直接編集しない）
- 事実検証: `scripts/generate_verified_bonus_summary.py`
- 仮説検証: `scripts/bonus_hypotheses.py`
- Pages JSON生成: `scripts/generate_pages_data.py`
- Pages監査: `scripts/validate_pages.py`
- 本番監査: `scripts/audit_live_pages.py`

`docs/data/bonus.json`と`gh-pages`ブランチは編集対象ではありません。`main`の正規データから生成し、全テスト通過後に`gh-pages`へ同期します。

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

- 証券コードが既存30社の固定集合に含まれること
- 確認事実と仮説の双方で重複コードがないこと
- 確認事実にHTTPSの一次情報URLがあること
- 仮説レンジが`minimum <= central <= maximum`を満たすこと
- 仮説に確度、根拠、前提、反証条件があること
- 仮説がverified集計から明示的に除外されること
- 公開JSONが最新の事実・仮説スナップショットと一致すること
- 日本語検索、推定フィルター、アクセシビリティ、文字コントラストが監査契約を満たすこと

## CI / Pages

- `.github/workflows/verified-bonus.yml`: 確認事実の単体テスト、固定銘柄監査、集計再生成
- `.github/workflows/pages.yml`: 事実・仮説JSON生成、UI監査、JavaScript構文確認、`gh-pages`自動同期、本番バイト一致監査
- `.github/workflows/live-audit.yml`: 公開中のHTML・CSS・JavaScript・JSONを独立監査

**最終モデル更新:** 2026-08-02
