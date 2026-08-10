# 賞与比較ウィジェット / Media Data Widget

転職・キャリア系メディアの記事内で、KAFKA Bonusの正準公開データから2〜5社の賞与情報を比較表示するための技術デモです。

## 無料デモ

GitHub Pages上の埋め込みページをiframeで利用できます。

```html
<iframe
  src="https://kafka2306.github.io/bonus/embed/compare/?companies=7203,6758&partner=demo-media&campaign=pilot-01"
  title="賞与比較"
  loading="lazy"
  width="100%"
  height="520">
</iframe>
```

`companies` は証券コードを2〜5件、`partner` と `campaign` は英数字・`.`・`_`・`-`だけを最大64文字で受け付けます。ウィジェット自身はcookie、メールアドレス、氏名、IPアドレス等を収集しません。

## データ契約

表示は `/data/media-widget-v1.json` の `bonus.media-widget.v1` だけを読みます。このcontractは正準 `/data/bonus.json` から `scripts/build_media_widget_contract.py` が決定論的に生成します。別の手入力賞与表を持ちません。

各recordは最低限、会社ID、会社名、基準日、`verified / estimated / unavailable`、月数range、金額availability、confidence、公開可能なsource URLを持ちます。

- `verified`: 当該recordの一次資料側に年間月数があり、evidence statusが確認済み系の場合
- `estimated`: 正準モデルの月数推定が存在する場合
- `unavailable`: 月数を安全に表示できない場合
- 金額は正準recordの `amount_status == available` の場合だけ表示し、それ以外を0円や推定額へ変換しません

## attribution / measurement contract

外部埋め込みでは、親windowへ `postMessage` で次の匿名イベントだけを通知します。サーバー送信は行いません。

- `embed_loaded`
- `source_opened`
- `full_comparison_opened`
- `business_inquiry_started`

payloadは `source`, `schema_version`, `event`, `partner`, `campaign`, `company_id` だけです。媒体側で計測する場合も、このcontractへ個人識別情報を追加しないでください。

## 法人PoC候補

有償PoCを実施する場合の対象は、媒体側の掲載レイアウト調整、対象会社セット、更新確認、出典表示、埋め込み導入支援です。契約前の導入実績、提携先、売上、効果を実績として表示しません。

このデモは賞与額・転職結果・将来の支給を保証せず、投資・転職判断の助言でもありません。元データの定義・対象範囲・推定状態と一次資料を確認してください。

## 問い合わせ

- [媒体への導入を相談する](https://github.com/KAFKA2306/bonus/issues/new)
- [正準の比較サイトを見る](https://kafka2306.github.io/bonus/)

## 45日KPI

実測値は `metrics/media-widget-kpi.json` に、確認できた事実だけを記録します。未実施の提案・埋め込み・相談・有償PoCを実績として補完しません。
