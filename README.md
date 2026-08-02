# 主要30社 賞与ソース・メタサーベイ

**公開ページ:** https://kafka2306.github.io/bonus/  
**Repository:** https://github.com/KAFKA2306/bonus

主要30社の賞与額を先に推定するのではなく、会社ごとに「何を確認するために、どの公式サイト・一次資料を、どの順序で参照するか」を定義し、資料に明示された事実だけを公開データへ昇格する調査基盤です。

## 根本ロジック

1. 会社公式、労組公式、EDINET、TDnetを個社調査の必須チャネルとする。
2. 連合、経団連、厚生労働省、e-Statは交渉・業界・全国水準の比較に使う。
3. 報道、転職、口コミ、給与集計サイトは一次資料を発見するためだけに使う。
4. 個社の月数、支給回数、算定方式は、当該会社または労組の一次資料に明記された場合だけ確認済みとする。
5. 集計平均、平均給与、賞与引当金、口コミ値から個社の賞与月数を逆算しない。
6. 資料の対象範囲を、全社員・新卒・組合員・単体・グループ会社のいずれかとして記録する。

## 参照ソースの階層

| 階層 | 主な参照先 | 用途 | 個社値への使用 |
|---|---|---|---|
| 会社一次 | 会社IR、人事、採用、制度説明 | 制度、回数、月数、算定方式、対象範囲 | 明示値のみ可 |
| 労使一次 | 会社・産業別労組、連合の個別表 | 春闘要求・妥結、一時金、対象組合員 | 個別企業・組合を識別できる場合のみ可 |
| 公式開示 | EDINET、TDnet、上場会社情報 | 法定開示、制度改定、賃上げ・報酬方針 | 明示された制度事実のみ可 |
| 公的ベンチマーク | 経団連、連合集計、厚生労働省、e-Stat | 業界・規模・全国水準との比較 | 不可 |
| 探索専用 | 報道、転職、口コミ、給与集計 | 一次資料名・労組名・検索語の発見 | 不可 |

各ソースの用途、確認可能項目、限界、公式URLは `data/source_survey_YYYY-MM-DD.yaml` で管理します。

## 公開ページ

GitHub Pagesでは、次の2表を表示します。

- **参照サイトの役割:** 優先順位、区分、確認できる項目、限界、公式リンク
- **企業別の調査キュー:** 調査段階、必須チャネル確認率、次に見る一次資料、確認済み事実、未解決の問い

旧UIにあった30社一律の推定レンジと確度は公開対象から除外しました。未確認企業は推定値を置かず、次に確認すべき一次資料を表示します。

## データモデル

- 固定調査銘柄: `nikkei225_bonus_survey_2024_en.yaml`
- 確認事実: `data/verified_bonus_facts_YYYY-MM-DD.yaml`
- ソース・メタサーベイ: `data/source_survey_YYYY-MM-DD.yaml`
- 旧仮説スナップショット: `data/bonus_hypotheses_YYYY-MM-DD.yaml`（履歴として保持し、公開JSONには使用しない）
- 公開ビュー: `docs/`
- 公開JSON: `docs/data/bonus.json`（CIで再生成）
- ソース台帳検証: `scripts/source_survey.py`
- Pages JSON生成: `scripts/generate_pages_data.py`
- Pages監査: `scripts/validate_pages.py`
- 本番監査: `scripts/audit_live_pages.py`

## 検証契約

CIは次を確認します。

- 固定Universe 30社を変更していないこと
- 確認事実のURLがHTTPSの適格な一次資料であること
- ソース台帳のID、階層、用途、限界、参照関係が妥当であること
- 必須チャネルとベンチマークチャネルが分離されていること
- 公開JSONに旧仮説データが含まれないこと
- 全30社に調査段階、確認済みチャネル、次の参照先、未解決項目があること
- PagesのHTML、CSS、JavaScript、JSONが生成結果と一致すること
- デプロイ後の公開ファイルがローカル生成物とバイト一致すること

## 実行方法

```bash
python -m pip install PyYAML==6.0.3
python -m unittest discover -s tests -v
python scripts/generate_bonus_summary.py --check
python scripts/generate_pages_data.py
python scripts/validate_pages.py
node --check docs/app.js
```

`docs/data/bonus.json` と `gh-pages` ブランチは直接編集せず、`main` の確認事実とソース台帳から生成します。

**最終ロジック更新:** 2026-08-02
