# 調査実行スクリプト・ツール群

## 📁 ファイル構成

```
scripts/
├── README.md                   # このファイル
├── survey_methodology.md       # 調査方法論の詳細マニュアル
├── phase2_execution_guide.md   # Phase 2実行のための具体的ガイド
├── data_validation_tools.md    # データ検証ツールの使用方法
└── automation_scripts/         # 自動化スクリプト（将来実装予定）
```

## 🚀 即座に実行可能なコマンド

### Phase 2調査の開始
```bash
# 作業ディレクトリに移動
cd /home/kafka/projects/bonus

# Phase 2対象企業リストの確認
cat nikkei225_companies.yaml | grep -E "(stock_code|company_name)" | head -140

# 製造業企業の抽出
grep -A 3 -B 1 "製造業\|機械\|化学\|鉄鋼" nikkei225_companies.yaml
```

### 新規企業データ作成
```bash
# テンプレートファイルを基に新規YAML作成
cp companies/tech/6861_keyence.yaml companies/manufacturing/6301_komatsu.yaml

# 企業情報を置換
sed -i 's/キーエンス/コマツ/g' companies/manufacturing/6301_komatsu.yaml
sed -i 's/6861/6301/g' companies/manufacturing/6301_komatsu.yaml
```

### データ品質チェック
```bash
# 作成済みYAMLファイルの一覧表示
find companies/ -name "*.yaml" -exec basename {} \; | sort

# 信頼度A級企業の確認
grep -r "confidence_level: \"A\"" companies/

# 業績連動型企業の一覧
grep -r "業績連動" companies/ | cut -d: -f1 | sort | uniq
```

## 📊 調査進捗の可視化

### 現在の完了状況
```bash
# 調査完了企業数
find companies/ -name "*.yaml" | wc -l

# 業界別完了状況
for sector in tech finance automotive retail pharma trading utilities manufacturing other; do
  count=$(find companies/$sector/ -name "*.yaml" 2>/dev/null | wc -l)
  echo "$sector: $count社"
done
```

### 信頼度分布の確認
```bash
# 信頼度別企業数
echo "A級:" $(grep -r "confidence_level: \"A\"" companies/ | wc -l)
echo "B級:" $(grep -r "confidence_level: \"B\"" companies/ | wc -l)
echo "C級:" $(grep -r "confidence_level: \"C\"" companies/ | wc -l)
echo "D級:" $(grep -r "confidence_level: \"D\"" companies/ | wc -l)
```

## 🔍 効率的な情報収集コマンド

### 企業の基本情報収集
```bash
# 特定企業の有価証券報告書検索（例：トヨタ）
echo "EDINET検索: https://disclosure.edinet-fsa.go.jp/E01EW/BLMainController.jsp"

# 企業IR情報の確認
curl -s "https://www.toyota.co.jp/jpn/company/profile/" | grep -i "賞与\|ボーナス\|一時金"
```

### 業界情報の一括収集
```bash
# 春闘関連ニュースの検索
echo "春闘 2024年 賞与" | xargs -I {} echo "検索キーワード: {}"

# 業界レポートの検索
echo "業界別賞与水準 2024年" | xargs -I {} echo "検索キーワード: {}"
```

## 📝 データ入力の効率化

### YAMLテンプレートの活用
```bash
# 業界別テンプレートの作成
mkdir -p templates/
cp companies/tech/6861_keyence.yaml templates/tech_template.yaml
cp companies/automotive/7203_toyota.yaml templates/automotive_template.yaml
cp companies/trading/8058_mitsubishi_corp.yaml templates/trading_template.yaml
```

### 一括データ処理
```bash
# 企業名の一括置換
for file in companies/*/*.yaml; do
  # 会社名の更新例
  echo "Processing: $file"
done

# 日付の一括更新
find companies/ -name "*.yaml" -exec sed -i 's/2025-01-21/2025-01-22/g' {} \;
```

## 🎯 Phase 2実行の具体的手順

### 1. 対象企業の選定（10分）
```bash
# 製造業企業の抽出
grep -E "機械|化学|鉄鋼|非鉄金属|金属製品" nikkei225_companies.yaml > phase2_target_list.txt

# 企業数の確認
wc -l phase2_target_list.txt
```

### 2. 調査実行（1社30-45分）
```bash
# 企業情報収集の開始
company_code="6113"  # アマダの例
company_name="アマダ"

# 基本情報の記録
echo "調査開始: $company_name ($company_code) - $(date)"

# YAMLファイルの作成
cp templates/manufacturing_template.yaml companies/manufacturing/${company_code}_${company_name,,}.yaml
```

### 3. データ品質チェック（10分）
```bash
# 必須項目の確認
yaml_file="companies/manufacturing/6113_amada.yaml"
grep -q "confidence_level" $yaml_file && echo "OK: 信頼度記載済み" || echo "NG: 信頼度未記載"
grep -q "evidence" $yaml_file && echo "OK: 証拠記載済み" || echo "NG: 証拠未記載"
```

## 🔧 トラブルシューティング

### よくある問題と解決策
```bash
# YAMLファイルの構文チェック
python3 -c "
import yaml
with open('companies/tech/6861_keyence.yaml', 'r', encoding='utf-8') as f:
    try:
        yaml.safe_load(f)
        print('OK: Valid YAML')
    except yaml.YAMLError as e:
        print(f'Error: {e}')
"

# 文字化け対策
find companies/ -name "*.yaml" -exec file {} \; | grep -v UTF-8
```

### データバックアップ
```bash
# 定期バックアップの作成
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz companies/ sectors/ analysis/

# Gitでのバージョン管理（推奨）
git init
git add .
git commit -m "Phase 1完了時点のデータ"
```

## 📈 進捗レポートの生成

### 簡易レポート作成
```bash
# 調査状況サマリー
echo "=== 調査進捗レポート ===" > progress_report.txt
echo "作成日時: $(date)" >> progress_report.txt
echo "完了企業数: $(find companies/ -name "*.yaml" | wc -l)/225" >> progress_report.txt
echo "" >> progress_report.txt

# 業界別進捗
echo "=== 業界別進捗 ===" >> progress_report.txt
for sector in tech finance automotive retail pharma trading utilities manufacturing other; do
  count=$(find companies/$sector/ -name "*.yaml" 2>/dev/null | wc -l)
  echo "$sector: $count社" >> progress_report.txt
done
```

---

**このスクリプト集を活用することで、**
**Phase 2以降の調査を効率的かつ一貫性を持って実行できます。**