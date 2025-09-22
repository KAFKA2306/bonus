#!/usr/bin/env python3
"""
Bonus Data Visualizer for Nikkei 225 Companies
Visualizes bonus methods, company sectors, and bonus distribution data.
Enhanced to pull detailed bonus statistics from company YAML files and
extract Japanese/English evidence for income, bonus months, and frequency.
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # noqa: F401 (kept for compatibility)
import numpy as np  # noqa: F401 (kept for compatibility)
import yaml


class BonusVisualizer:
    def __init__(self, data_dir="../"):
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = self.data_dir / "analysis" / "graphs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.companies_data = None
        self.bonus_data = None
        self.company_index = defaultdict(list)

        self.load_data()

    # ------------------------------------------------------------------
    # Data loading and indexing
    # ------------------------------------------------------------------

    def load_data(self):
        """Load company list and bonus survey data"""
        try:
            companies_path = self.data_dir / "nikkei225_companies.yaml"
            if companies_path.exists():
                with companies_path.open('r', encoding='utf-8') as f:
                    content = f.read()
                fixed_content = self._fix_company_yaml_structure(content)
                self.companies_data = yaml.safe_load(fixed_content)
                self._index_company_files()

            english_file = self.data_dir / "nikkei225_bonus_survey_2024_en.yaml"
            japanese_file = self.data_dir / "nikkei225_bonus_survey_2024.yaml"

            if english_file.exists():
                with english_file.open('r', encoding='utf-8') as f:
                    self.bonus_data = yaml.safe_load(f)
                print("Using English bonus survey data")
            elif japanese_file.exists():
                with japanese_file.open('r', encoding='utf-8') as f:
                    self.bonus_data = yaml.safe_load(f)
                print("Using Japanese bonus survey data")
            else:
                raise FileNotFoundError("Bonus survey YAML not found")

            if self.companies_data:
                companies = self.companies_data['nikkei225']['companies']
                actual_companies = [c for c in companies if isinstance(c, dict) and 'sector' in c]
                print(f"Loaded {len(actual_companies)} companies")
            print(f"Loaded {len(self.bonus_data['companies'])} bonus survey entries")

        except (FileNotFoundError, yaml.YAMLError) as exc:
            raise RuntimeError(f"Failed to load data: {exc}")

    def _fix_company_yaml_structure(self, raw_content: str) -> str:
        """Adjust the company YAML to ensure notes are outside the list."""
        lines = raw_content.split('\n')
        fixed_lines = []
        in_companies = False
        notes_started = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('companies:'):
                in_companies = True
                fixed_lines.append(line)
                continue
            if stripped.startswith('# 注記') or stripped.startswith('notes:'):
                if in_companies:
                    fixed_lines.append('\nnotes:')
                    in_companies = False
                    notes_started = True
                if stripped.startswith('notes:'):
                    continue
            if notes_started and line.startswith('      -'):
                fixed_lines.append('  ' + stripped)
            else:
                fixed_lines.append(line)
        return '\n'.join(fixed_lines)

    def _index_company_files(self):
        companies_dir = self.data_dir / "companies"
        if not companies_dir.exists():
            return
        for path in companies_dir.rglob('*.yaml'):
            stem = path.stem
            code = stem.split('_', 1)[0]
            if code.isdigit():
                self.company_index[code].append(path)

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def plot_sector_distribution(self):
        if not self.companies_data:
            print("Companies data not available - skipping sector distribution chart")
            return {}

        companies = self.companies_data['nikkei225']['companies']
        companies = [c for c in companies if isinstance(c, dict) and 'sector' in c]
        sectors = [c.get('sector_en', c['sector']) for c in companies]
        sector_counts = Counter(sectors)

        plt.figure(figsize=(12, 8))
        plt.pie(sector_counts.values(), labels=sector_counts.keys(), autopct='%1.1f%%', startangle=90)
        plt.title('Nikkei 225 Companies by Sector Distribution', fontsize=16, pad=20)
        plt.axis('equal')
        plt.tight_layout()
        output_path = self.output_dir / 'sector_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return sector_counts

    def plot_bonus_methods(self):
        bonus_companies = self.bonus_data['companies']
        methods = [company.get('bonus_method') for company in bonus_companies]

        translation = {
            '業績連動': 'Performance-linked',
            '総合判断': 'Comprehensive judgment',
            'ハイブリッド': 'Hybrid',
            '基本給連動': 'Base salary-linked'
        }
        translated = [translation.get(method, method) for method in methods]
        method_counts = Counter(translated)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(method_counts.keys(), method_counts.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Bonus Method Distribution in Survey Companies', fontsize=14)
        plt.xlabel('Bonus Method')
        plt.ylabel('Number of Companies')
        plt.xticks(rotation=45)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{int(height)}', ha='center', va='bottom')
        plt.tight_layout()
        output_path = self.output_dir / 'bonus_methods.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return method_counts

    def plot_confidence_levels(self):
        bonus_companies = self.bonus_data['companies']
        confidence_levels = [company.get('confidence_level', 'Unknown') for company in bonus_companies]
        confidence_counts = Counter(confidence_levels)

        plt.figure(figsize=(8, 6))
        colors = {'A': '#2ECC71', 'B': '#F39C12', 'C': '#E74C3C', 'D': '#9B59B6', 'Unknown': '#95A5A6'}
        bar_colors = [colors.get(level, '#95A5A6') for level in confidence_counts.keys()]
        bars = plt.bar(confidence_counts.keys(), confidence_counts.values(), color=bar_colors)
        plt.title('Survey Data Confidence Level Distribution', fontsize=14)
        plt.xlabel('Confidence Level')
        plt.ylabel('Number of Companies')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{int(height)}', ha='center', va='bottom')
        plt.tight_layout()
        output_path = self.output_dir / 'confidence_levels.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return confidence_counts

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def create_summary_table(self):
        bonus_companies = self.bonus_data['companies']
        translation = {
            '業績連動': 'Performance-linked',
            '総合判断': 'Comprehensive judgment',
            'ハイブリッド': 'Hybrid',
            '基本給連動': 'Base salary-linked'
        }
        summary = []
        for company in bonus_companies:
            name = company.get('company_name_en', company['company_name'])
            method = translation.get(company.get('bonus_method'), company.get('bonus_method'))
            summary.append({
                'Company': name,
                'Stock Code': company['stock_code'],
                'Bonus Method': method,
                'Confidence': company.get('confidence_level', 'Unknown')
            })
        df = pd.DataFrame(summary)
        plt.figure(figsize=(14, 8))
        plt.axis('tight')
        plt.axis('off')
        table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        for i, row in df.iterrows():
            color = self._confidence_color(row['Confidence'])
            for j in range(len(df.columns)):
                table[(i + 1, j)].set_facecolor(color)
        plt.title('Bonus Survey Summary Table', fontsize=16, pad=20)
        output_path = self.output_dir / 'summary_table.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return df

    def create_detailed_summary_table(self):
        bonus_companies = self.bonus_data['companies']
        translation = {
            '業績連動': 'Performance-linked',
            '総合判断': 'Comprehensive judgment',
            'ハイブリッド': 'Hybrid',
            '基本給連動': 'Base salary-linked'
        }
        rows = []
        for company in bonus_companies:
            name = company.get('company_name_en', company['company_name'])
            method = translation.get(company.get('bonus_method'), company.get('bonus_method'))
            calc = company.get('calculation_method', company.get('calculation_method_ja', 'N/A'))
            notes = company.get('notes', company.get('notes_ja', 'N/A'))
            rows.append({
                'Company': name,
                'Code': company['stock_code'],
                'Method': method,
                'Conf': company.get('confidence_level', 'U'),
                'Calculation': self._truncate(calc, 60),
                'Notes': self._truncate(notes, 50)
            })
        df = pd.DataFrame(rows)
        plt.figure(figsize=(20, 12))
        plt.axis('tight')
        plt.axis('off')
        table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.8)
        cell_dict = table.get_celld()
        for i in range(len(df) + 1):
            cell_dict[(i, 0)].set_width(0.15)
            cell_dict[(i, 1)].set_width(0.08)
            cell_dict[(i, 2)].set_width(0.15)
            cell_dict[(i, 3)].set_width(0.06)
            cell_dict[(i, 4)].set_width(0.35)
            cell_dict[(i, 5)].set_width(0.21)
        for i, row in df.iterrows():
            color = self._confidence_color(row['Conf'])
            for j in range(len(df.columns)):
                table[(i + 1, j)].set_facecolor(color)
        plt.title('Detailed Bonus Survey Summary Table', fontsize=16, pad=20)
        output_path = self.output_dir / 'detailed_summary_table.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return df

    def create_company_statistics_table(self):
        bonus_companies = self.bonus_data['companies']
        translation = {
            '業績連動': 'Performance-linked',
            '総合判断': 'Comprehensive judgment',
            'ハイブリッド': 'Hybrid',
            '基本給連動': 'Base salary-linked'
        }
        rows = []
        for company in bonus_companies:
            stock_code = company['stock_code']
            name = company.get('company_name_en', company['company_name'])
            metrics = self._extract_company_metrics(stock_code, company)
            bonus_method = translation.get(company.get('bonus_method'), company.get('bonus_method'))
            rows.append({
                'Company': name,
                'Code': stock_code,
                'Avg Income': metrics.get('avg_income', 'N/A'),
                'Bonus': metrics.get('bonus', 'N/A'),
                'Frequency': metrics.get('frequency', 'N/A'),
                'Method': bonus_method,
                'Confidence': company.get('confidence_level', 'U')
            })
        df = pd.DataFrame(rows)
        plt.figure(figsize=(16, 10))
        plt.axis('tight')
        plt.axis('off')
        table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.6)
        for i, row in df.iterrows():
            color = self._confidence_color(row['Confidence'])
            for j in range(len(df.columns)):
                table[(i + 1, j)].set_facecolor(color)
        plt.title('Company Statistics & Bonus Metrics Table', fontsize=16, pad=20)
        output_path = self.output_dir / 'company_statistics_table.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return df

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self):
        print("Generating Bonus Data Visualization Report...")
        print("=" * 50)
        sector_dist = self.plot_sector_distribution()
        bonus_methods = self.plot_bonus_methods()
        confidence_levels = self.plot_confidence_levels()
        self.create_summary_table()
        self.create_detailed_summary_table()
        self.create_company_statistics_table()

        print("\nSummary Statistics:")
        if self.companies_data:
            companies = self.companies_data['nikkei225']['companies']
            actual_companies = [c for c in companies if isinstance(c, dict) and 'sector' in c]
            coverage = len(self.bonus_data['companies']) / len(actual_companies) * 100
            print(f"Total Nikkei 225 companies: {len(actual_companies)}")
            print(f"Companies in bonus survey: {len(self.bonus_data['companies'])}")
            print(f"Survey coverage: {coverage:.1f}%")
        else:
            print(f"Companies in bonus survey: {len(self.bonus_data['companies'])}")
            print("Full company list not available")

        print("\nSector Distribution (Top 5):")
        for sector, count in sorted(sector_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {sector}: {count} companies")

        print("\nBonus Method Distribution:")
        for method, count in bonus_methods.items():
            print(f"  {method}: {count} companies")

        print("\nConfidence Level Distribution:")
        for level, count in confidence_levels.items():
            print(f"  Level {level}: {count} companies")

        print("\nVisualization files saved under analysis/graphs:")
        for filename in [
            'sector_distribution.png',
            'bonus_methods.png',
            'confidence_levels.png',
            'summary_table.png',
            'detailed_summary_table.png',
            'company_statistics_table.png',
        ]:
            print(f"  - {filename}")

    # ------------------------------------------------------------------
    # Metric extraction helpers
    # ------------------------------------------------------------------

    def _extract_company_metrics(self, stock_code: str, survey_entry: dict) -> dict:
        metrics = {
            'avg_income': 'N/A',
            'bonus': 'N/A',
            'frequency': 'N/A',
        }
        company_path = self._select_company_file(stock_code)
        if company_path:
            try:
                with company_path.open('r', encoding='utf-8') as fh:
                    data = yaml.safe_load(fh)
                self._update_metrics_from_company_yaml(metrics, data)
            except yaml.YAMLError:
                pass

        for field in ('calculation_method', 'calculation_method_ja', 'notes', 'notes_ja'):
            value = survey_entry.get(field)
            if isinstance(value, str):
                self._update_metrics_from_text(metrics, value)

        for evidence in survey_entry.get('evidence', []):
            for key in ('detail', 'detail_ja'):
                value = evidence.get(key)
                if isinstance(value, str):
                    self._update_metrics_from_text(metrics, value)
        return metrics

    def _select_company_file(self, stock_code: str):
        paths = self.company_index.get(stock_code)
        if not paths:
            return None

        def rank(path: Path):
            name = path.name
            return (name.count('-'), len(name))

        return sorted(paths, key=rank)[0]

    def _update_metrics_from_company_yaml(self, metrics: dict, data: dict):
        bonus_system = data.get('bonus_system') or {}
        if not isinstance(bonus_system, dict):
            bonus_system = {}
        methodology = bonus_system.get('methodology') or {}
        if not isinstance(methodology, dict):
            methodology = {}
        performance_metrics = bonus_system.get('performance_metrics') or {}
        if not isinstance(performance_metrics, dict):
            performance_metrics = {}
        financial_data = data.get('financial_data') or {}
        if not isinstance(financial_data, dict):
            financial_data = {}

        if isinstance(methodology.get('payment_frequency'), str):
            self._apply_frequency(metrics, methodology['payment_frequency'])

        for key in ('base_salary_months', 'bonus_range', 'annual_bonus_amount'):
            value = performance_metrics.get(key)
            if isinstance(value, str):
                self._apply_bonus_months(metrics, value)

        profile = data.get('company_profile') or {}
        if not isinstance(profile, dict):
            profile = {}
        avg_income_candidates = [
            performance_metrics.get('average_annual_income'),
            financial_data.get('average_annual_income'),
            profile.get('average_annual_income'),
        ]
        for candidate in avg_income_candidates:
            if isinstance(candidate, str):
                self._apply_income(metrics, candidate)

        notes = data.get('notes')
        if isinstance(notes, str):
            self._update_metrics_from_text(metrics, notes)

    def _update_metrics_from_text(self, metrics: dict, text: str):
        self._apply_income(metrics, text)
        self._apply_bonus_months(metrics, text)
        self._apply_frequency(metrics, text)

    def _apply_income(self, metrics: dict, text: str):
        if metrics['avg_income'] != 'N/A':
            return
        if not any(keyword in text for keyword in ('平均年収', 'average annual income', 'average income', '平均給与')):
            return
        amount = self._parse_money_to_million_yen(text)
        if amount is not None:
            metrics['avg_income'] = f"¥{amount:.2f}M"

    def _apply_bonus_months(self, metrics: dict, text: str):
        if metrics['bonus'] != 'N/A':
            return
        parsed = self._parse_bonus_months(text)
        if parsed:
            metrics['bonus'] = parsed

    def _apply_frequency(self, metrics: dict, text: str):
        if metrics['frequency'] != 'N/A':
            return
        parsed = self._parse_frequency(text)
        if parsed:
            metrics['frequency'] = parsed

    # ------------------------------------------------------------------
    # Parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_money_to_million_yen(text: str):
        clean = text.replace(',', '')
        # 兆円 -> x * 10000
        match = re.search(r'約?([0-9]+(?:\.[0-9]+)?)\s*兆円', clean)
        if match:
            return float(match.group(1)) * 10000
        match = re.search(r'約?([0-9]+(?:\.[0-9]+)?)\s*億円', clean)
        if match:
            return float(match.group(1)) * 100
        match = re.search(r'約?([0-9]+(?:\.[0-9]+)?)\s*万円', clean)
        if match:
            return float(match.group(1)) / 100
        match = re.search(r'¥\s*([0-9]+(?:\.[0-9]+)?)M', clean)
        if match:
            return float(match.group(1))
        match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*million\s*yen', clean, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*百万円', clean)
        if match:
            return float(match.group(1))
        return None

    @staticmethod
    def _parse_bonus_months(text: str) -> str:
        clean = text.replace('～', '~')
        range_pattern = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*[~\-]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:カ月|か月|ヶ月|月分|months)', clean)
        if range_pattern:
            start, end = range_pattern.groups()
            return f"{start}-{end} months"
        single_pattern = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(?:カ月|か月|ヶ月|月分|months)', clean)
        if single_pattern:
            value = single_pattern.group(1)
            return f"{value} months"
        return ''

    @staticmethod
    def _parse_frequency(text: str) -> str:
        clean = text.lower()
        match = re.search(r'年\s*(\d+)\s*回', text)
        if match:
            return f"{match.group(1)}x/year"
        match = re.search(r'(\d+)\s*x\s*/\s*year', clean)
        if match:
            return f"{match.group(1)}x/year"
        if 'quarterly' in clean or '四半期' in text:
            return '4x/year'
        if any(keyword in clean for keyword in ('twice-yearly', 'twice yearly', 'biannual')) or '年2回' in text:
            return '2x/year'
        if '年1回' in text or 'once a year' in clean:
            return '1x/year'
        if '年4回' in text:
            return '4x/year'
        return ''

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if text is None:
            return 'N/A'
        text = str(text)
        return text if len(text) <= limit else text[:limit - 3] + '...'

    @staticmethod
    def _confidence_color(level: str) -> str:
        if level == 'A':
            return '#D5F4E6'
        if level == 'B':
            return '#FFF3CD'
        if level == 'C':
            return '#F8D7DA'
        return '#F0F0F0'


def main():
    parser = argparse.ArgumentParser(description='Visualize Nikkei 225 bonus data')
    parser.add_argument('--data-dir', default='../', help='Directory containing YAML data files')
    parser.add_argument('--chart', choices=['sectors', 'methods', 'confidence', 'table', 'detailed-table', 'stats-table', 'all'],
                        default='all', help='Specific chart to generate')
    args = parser.parse_args()

    visualizer = BonusVisualizer(args.data_dir)

    if args.chart == 'sectors':
        visualizer.plot_sector_distribution()
    elif args.chart == 'methods':
        visualizer.plot_bonus_methods()
    elif args.chart == 'confidence':
        visualizer.plot_confidence_levels()
    elif args.chart == 'table':
        visualizer.create_summary_table()
    elif args.chart == 'detailed-table':
        visualizer.create_detailed_summary_table()
    elif args.chart == 'stats-table':
        visualizer.create_company_statistics_table()
    else:
        visualizer.generate_report()


if __name__ == "__main__":
    main()
