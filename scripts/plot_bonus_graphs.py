from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from matplotlib import font_manager
import yaml
BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARY_PATH = BASE_DIR / "analysis" / "summary" / "bonus_graph_data.yaml"
GRAPH_DIR = BASE_DIR / "analysis" / "graphs"
CLASSIFICATION_ORDER = ("業績連動型", "基本給連動型", "総合判断型", "ハイブリッド型")
def configure_font():
    preferred = [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAPGothic",
        "Yu Gothic",
        "MS Gothic",
        "TakaoPGothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams['font.family'] = name
            return name
    plt.rcParams['font.family'] = 'DejaVu Sans'
    return 'DejaVu Sans'
def load_records() -> List[Dict]:
    with SUMMARY_PATH.open(encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    records = payload.get("records", [])
    return [rec for rec in records if rec.get("bonus_months_estimate") is not None]
def aggregate_averages(records: List[Dict], key: str, order=None, label_field=None):
    buckets: Dict[str, List[float]] = defaultdict(list)
    labels: Dict[str, str] = {}
    for record in records:
        value = record.get(key) or "Unknown"
        buckets[value].append(record["bonus_months_estimate"])
        if label_field:
            label_value = record.get(label_field)
            if label_value:
                labels[value] = label_value
    averages = []
    def label_for(value: str) -> str:
        if label_field:
            if value in labels:
                return labels[value]
            if (value is None or value == "" or value == "Unknown") and "Unknown" in labels:
                return labels["Unknown"]
        return value
    if order:
        for item in order:
            if item in buckets:
                vals = buckets.pop(item)
                averages.append((label_for(item), sum(vals) / len(vals)))
    for item, vals in sorted(buckets.items(), key=lambda x: x[1], reverse=True):
        averages.append((label_for(item), sum(vals) / len(vals)))
    return averages
def render_bar_chart(pairs, title, filename, ylabel="Estimated Annual Bonus (months)"):
    if not pairs:
        return None
    labels = [label for label, _ in pairs]
    values = [val for _, val in pairs]
    x_pos = range(len(pairs))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x_pos, values, color="
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            f"{value:.2f}",
            ha='center',
            va='bottom',
            fontsize=9,
        )
    fig.tight_layout()
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GRAPH_DIR / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
def main():
    configure_font()
    records = load_records()
    if not records:
        print("No records with bonus estimates found; graphs not generated.")
        return
    sector_pairs = aggregate_averages(records, "sector", label_field="sector_name_eb")
    classification_pairs = aggregate_averages(
        records,
        "classification",
        order=CLASSIFICATION_ORDER,
        label_field="classification_name_eb",
    )
    outputs = []
    path = render_bar_chart(
        sector_pairs,
        "Estimated Annual Bonus by Industry (months)",
        "industry_bonus_months.png",
    )
    if path:
        outputs.append(path)
    path = render_bar_chart(
        classification_pairs,
        "Estimated Annual Bonus by Bonus Method (months)",
        "classification_bonus_months.png",
    )
    if path:
        outputs.append(path)
    if outputs:
        print("Generated:")
        for path in outputs:
            print(f" - {path.relative_to(BASE_DIR)}")
    else:
        print("No graphs generated.")
if __name__ == "__main__":
    main()
