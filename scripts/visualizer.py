#!/usr/bin/env python3
"""
Bonus Data Visualizer for Nikkei 225 Companies
Visualizes bonus methods, company sectors, and bonus distribution data
"""

import yaml
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import Counter
import argparse
import re


class BonusVisualizer:
    def __init__(self, data_dir="../"):
        self.data_dir = Path(data_dir)
        self.companies_data = None
        self.bonus_data = None
        self.load_data()

    def load_data(self):
        """Load YAML data files"""
        try:
            # Try to load the companies file, handling potential YAML structure issues
            with open(self.data_dir / "nikkei225_companies.yaml", 'r', encoding='utf-8') as f:
                content = f.read()
                # Fix the YAML structure by moving notes outside companies array
                lines = content.split('\n')
                fixed_lines = []
                in_companies = False
                notes_started = False

                for line in lines:
                    if '  companies:' in line:
                        in_companies = True
                        fixed_lines.append(line)
                    elif line.strip().startswith('# 注記') or line.strip().startswith('notes:'):
                        if in_companies:
                            # End companies array and start notes at root level
                            fixed_lines.append('\n  notes:')
                            in_companies = False
                            notes_started = True
                        if line.strip().startswith('notes:'):
                            continue  # Skip the original notes line
                    elif notes_started and line.startswith('      -'):
                        # Adjust indentation for notes items
                        fixed_lines.append('    ' + line.strip())
                    else:
                        fixed_lines.append(line)

                fixed_content = '\n'.join(fixed_lines)
                self.companies_data = yaml.safe_load(fixed_content)

            # Try to load English version first, fallback to Japanese version
            english_file = self.data_dir / "nikkei225_bonus_survey_2024_en.yaml"
            japanese_file = self.data_dir / "nikkei225_bonus_survey_2024.yaml"

            if english_file.exists():
                with open(english_file, 'r', encoding='utf-8') as f:
                    self.bonus_data = yaml.safe_load(f)
                print("Using English bonus survey data")
            else:
                with open(japanese_file, 'r', encoding='utf-8') as f:
                    self.bonus_data = yaml.safe_load(f)
                print("Using Japanese bonus survey data")

            # Count actual companies
            companies = self.companies_data['nikkei225']['companies']
            actual_companies = [c for c in companies if isinstance(c, dict) and 'sector' in c]

            print(f"Loaded {len(actual_companies)} companies")
            print(f"Loaded {len(self.bonus_data['companies'])} bonus survey entries")

        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            print("Attempting to load with basic parsing...")
            # Fallback: load only the bonus survey data
            english_file = self.data_dir / "nikkei225_bonus_survey_2024_en.yaml"
            japanese_file = self.data_dir / "nikkei225_bonus_survey_2024.yaml"

            if english_file.exists():
                with open(english_file, 'r', encoding='utf-8') as f:
                    self.bonus_data = yaml.safe_load(f)
                print("Using English bonus survey data (fallback)")
            else:
                with open(japanese_file, 'r', encoding='utf-8') as f:
                    self.bonus_data = yaml.safe_load(f)
                print("Using Japanese bonus survey data (fallback)")
            self.companies_data = None

    def plot_sector_distribution(self):
        """Plot distribution of companies by sector"""
        if not self.companies_data:
            print("Companies data not available - skipping sector distribution chart")
            return {}

        companies = self.companies_data['nikkei225']['companies']
        # Filter out non-company entries (like notes)
        companies = [company for company in companies if isinstance(company, dict) and 'sector' in company]
        # Use English sector names
        sectors = [company.get('sector_en', company['sector']) for company in companies]
        sector_counts = Counter(sectors)

        plt.figure(figsize=(12, 8))
        sectors_list = list(sector_counts.keys())
        counts = list(sector_counts.values())

        plt.pie(counts, labels=sectors_list, autopct='%1.1f%%', startangle=90)
        plt.title('Nikkei 225 Companies by Sector Distribution', fontsize=16, pad=20)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('sector_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        return sector_counts

    def plot_bonus_methods(self):
        """Plot distribution of bonus methods"""
        bonus_companies = self.bonus_data['companies']
        bonus_methods = [company['bonus_method'] for company in bonus_companies]

        # Translate Japanese bonus methods to English
        method_translation = {
            '業績連動': 'Performance-linked',
            '総合判断': 'Comprehensive judgment',
            'ハイブリッド': 'Hybrid',
            '基本給連動': 'Base salary-linked'
        }

        translated_methods = [method_translation.get(method, method) for method in bonus_methods]
        method_counts = Counter(translated_methods)

        plt.figure(figsize=(10, 6))
        methods = list(method_counts.keys())
        counts = list(method_counts.values())

        bars = plt.bar(methods, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Bonus Method Distribution in Survey Companies', fontsize=14)
        plt.xlabel('Bonus Method')
        plt.ylabel('Number of Companies')
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('bonus_methods.png', dpi=300, bbox_inches='tight')
        plt.show()

        return method_counts

    def plot_confidence_levels(self):
        """Plot confidence level distribution"""
        bonus_companies = self.bonus_data['companies']
        confidence_levels = [company.get('confidence_level', 'Unknown') for company in bonus_companies]
        confidence_counts = Counter(confidence_levels)

        plt.figure(figsize=(8, 6))
        levels = list(confidence_counts.keys())
        counts = list(confidence_counts.values())

        colors = {'A': '#2ECC71', 'B': '#F39C12', 'C': '#E74C3C', 'D': '#9B59B6', 'Unknown': '#95A5A6'}
        bar_colors = [colors.get(level, '#95A5A6') for level in levels]

        bars = plt.bar(levels, counts, color=bar_colors)
        plt.title('Survey Data Confidence Level Distribution', fontsize=14)
        plt.xlabel('Confidence Level')
        plt.ylabel('Number of Companies')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('confidence_levels.png', dpi=300, bbox_inches='tight')
        plt.show()

        return confidence_counts

    def create_summary_table(self):
        """Create a summary table of bonus survey data"""
        bonus_companies = self.bonus_data['companies']

        summary_data = []

        # Bonus method translation
        method_translation = {
            '業績連動': 'Performance-linked',
            '総合判断': 'Comprehensive judgment',
            'ハイブリッド': 'Hybrid',
            '基本給連動': 'Base salary-linked'
        }

        for company in bonus_companies:
            # Use English company name if available, otherwise use Japanese name
            company_name = company.get('company_name_en', company['company_name'])
            bonus_method = method_translation.get(company['bonus_method'], company['bonus_method'])

            summary_data.append({
                'Company': company_name,
                'Stock Code': company['stock_code'],
                'Bonus Method': bonus_method,
                'Confidence': company.get('confidence_level', 'Unknown')
            })

        df = pd.DataFrame(summary_data)

        plt.figure(figsize=(14, 8))
        plt.axis('tight')
        plt.axis('off')

        table = plt.table(cellText=df.values, colLabels=df.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Color code by confidence level
        for i, row in df.iterrows():
            confidence = row['Confidence']
            if confidence == 'A':
                color = '#D5F4E6'  # Light green
            elif confidence == 'B':
                color = '#FFF3CD'  # Light yellow
            elif confidence == 'C':
                color = '#F8D7DA'  # Light red
            else:
                color = '#F0F0F0'  # Light gray

            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor(color)

        plt.title('Bonus Survey Summary Table', fontsize=16, pad=20)
        plt.savefig('summary_table.png', dpi=300, bbox_inches='tight')
        plt.show()

        return df

    def create_detailed_summary_table(self):
        """Create a detailed summary table with calculation methods and notes"""
        bonus_companies = self.bonus_data['companies']

        summary_data = []

        # Bonus method translation
        method_translation = {
            '業績連動': 'Performance-linked',
            '総合判断': 'Comprehensive judgment',
            'ハイブリッド': 'Hybrid',
            '基本給連動': 'Base salary-linked'
        }

        for company in bonus_companies:
            # Use English company name if available, otherwise use Japanese name
            company_name = company.get('company_name_en', company['company_name'])
            bonus_method = method_translation.get(company['bonus_method'], company['bonus_method'])

            # Get calculation method (prefer English)
            calc_method = company.get('calculation_method', company.get('calculation_method_ja', 'N/A'))
            if len(calc_method) > 60:  # Truncate long descriptions
                calc_method = calc_method[:57] + '...'

            # Get notes (prefer English)
            notes = company.get('notes', company.get('notes_ja', 'N/A'))
            if len(notes) > 50:  # Truncate long notes
                notes = notes[:47] + '...'

            summary_data.append({
                'Company': company_name,
                'Code': company['stock_code'],
                'Method': bonus_method,
                'Conf': company.get('confidence_level', 'U'),
                'Calculation': calc_method,
                'Notes': notes
            })

        df = pd.DataFrame(summary_data)

        plt.figure(figsize=(20, 12))
        plt.axis('tight')
        plt.axis('off')

        table = plt.table(cellText=df.values, colLabels=df.columns,
                         cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.8)

        # Color code by confidence level and adjust column widths
        for i, row in df.iterrows():
            confidence = row['Conf']
            if confidence == 'A':
                color = '#D5F4E6'  # Light green
            elif confidence == 'B':
                color = '#FFF3CD'  # Light yellow
            elif confidence == 'C':
                color = '#F8D7DA'  # Light red
            else:
                color = '#F0F0F0'  # Light gray

            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor(color)

        # Adjust column widths
        cellDict = table.get_celld()
        for i in range(len(df) + 1):
            cellDict[(i, 0)].set_width(0.15)  # Company
            cellDict[(i, 1)].set_width(0.08)  # Code
            cellDict[(i, 2)].set_width(0.15)  # Method
            cellDict[(i, 3)].set_width(0.06)  # Conf
            cellDict[(i, 4)].set_width(0.35)  # Calculation
            cellDict[(i, 5)].set_width(0.21)  # Notes

        plt.title('Detailed Bonus Survey Summary Table', fontsize=16, pad=20)
        plt.savefig('detailed_summary_table.png', dpi=300, bbox_inches='tight')
        plt.show()

        return df

    def create_company_statistics_table(self):
        """Create a statistics table showing company metrics and evidence"""
        bonus_companies = self.bonus_data['companies']

        stats_data = []

        for company in bonus_companies:
            company_name = company.get('company_name_en', company['company_name'])

            # Extract key statistics from evidence if available
            avg_income = 'N/A'
            bonus_months = 'N/A'
            frequency = 'N/A'

            if 'evidence' in company and company['evidence']:
                for evidence in company['evidence']:
                    detail = evidence.get('detail', '')
                    # Look for annual income mentions
                    if 'million yen' in detail or '万円' in detail:
                        if 'average annual income' in detail.lower():
                            # Extract the number before 'million yen'
                            match = re.search(r'(\d+(?:,\d+)*)\s*million yen', detail)
                            if match:
                                avg_income = f"¥{match.group(1)}M"

                    # Look for bonus months
                    if 'months' in detail:
                        match = re.search(r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*months', detail)
                        if match:
                            bonus_months = f"{match.group(1)} months"

                    # Look for frequency
                    if 'twice-yearly' in detail.lower() or 'twice yearly' in detail.lower():
                        frequency = '2x/year'
                    elif 'quarterly' in detail.lower():
                        frequency = '4x/year'

            stats_data.append({
                'Company': company_name,
                'Code': company['stock_code'],
                'Avg Income': avg_income,
                'Bonus': bonus_months,
                'Frequency': frequency,
                'Method': company['bonus_method'],
                'Confidence': company.get('confidence_level', 'U')
            })

        df = pd.DataFrame(stats_data)

        plt.figure(figsize=(16, 10))
        plt.axis('tight')
        plt.axis('off')

        table = plt.table(cellText=df.values, colLabels=df.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.6)

        # Color code by confidence level
        for i, row in df.iterrows():
            confidence = row['Confidence']
            if confidence == 'A':
                color = '#D5F4E6'  # Light green
            elif confidence == 'B':
                color = '#FFF3CD'  # Light yellow
            elif confidence == 'C':
                color = '#F8D7DA'  # Light red
            else:
                color = '#F0F0F0'  # Light gray

            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor(color)

        plt.title('Company Statistics & Bonus Metrics Table', fontsize=16, pad=20)
        plt.savefig('company_statistics_table.png', dpi=300, bbox_inches='tight')
        plt.show()

        return df

    def generate_report(self):
        """Generate a comprehensive visual report"""
        print("Generating Bonus Data Visualization Report...")
        print("=" * 50)

        # Generate all visualizations
        sector_dist = self.plot_sector_distribution()
        bonus_methods = self.plot_bonus_methods()
        confidence_levels = self.plot_confidence_levels()
        summary_df = self.create_summary_table()
        detailed_df = self.create_detailed_summary_table()
        stats_df = self.create_company_statistics_table()

        # Print summary statistics
        print("\nSummary Statistics:")
        if self.companies_data:
            # Count actual companies (excluding notes)
            companies = self.companies_data['nikkei225']['companies']
            actual_companies = [c for c in companies if isinstance(c, dict) and 'sector' in c]
            print(f"Total Nikkei 225 companies: {len(actual_companies)}")
            print(f"Companies in bonus survey: {len(self.bonus_data['companies'])}")
            print(f"Survey coverage: {len(self.bonus_data['companies'])/len(actual_companies)*100:.1f}%")
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

        print("\nVisualization files saved:")
        print("  - sector_distribution.png")
        print("  - bonus_methods.png")
        print("  - confidence_levels.png")
        print("  - summary_table.png")
        print("  - detailed_summary_table.png")
        print("  - company_statistics_table.png")


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