#!/usr/bin/env python3
"""
Bonus Method Analysis - Which method is better for employees?
Analyzes employee benefits across different bonus calculation methods
"""

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional


class BonusMethodAnalyzer:
    def __init__(self, data_dir="../"):
        self.data_dir = Path(data_dir)
        self.bonus_data = None
        self.load_data()

    def load_data(self):
        """Load the English bonus survey data"""
        english_file = self.data_dir / "nikkei225_bonus_survey_2024_en.yaml"

        with open(english_file, 'r', encoding='utf-8') as f:
            self.bonus_data = yaml.safe_load(f)

        print(f"Loaded {len(self.bonus_data['companies'])} companies for analysis")

    def extract_salary_data(self, detail_text: str) -> Optional[float]:
        """Extract average salary from evidence text"""
        # Look for patterns like "average annual income ¥XX.XXM" or "平均年収XXX万円"
        patterns = [
            r'average annual income ¥(\d+(?:,\d+)*(?:\.\d+)?)\s*[Mm]',
            r'平均年収(\d+(?:,\d+)*)\s*万円',
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*million yen'
        ]

        for pattern in patterns:
            match = re.search(pattern, detail_text)
            if match:
                value_str = match.group(1).replace(',', '')
                try:
                    if 'million' in pattern or 'M' in pattern:
                        return float(value_str)  # Already in millions
                    else:
                        return float(value_str) / 100  # Convert 万円 to millions
                except ValueError:
                    continue
        return None

    def extract_bonus_months(self, detail_text: str) -> Optional[float]:
        """Extract bonus months from evidence text"""
        patterns = [
            r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*months',
            r'(\d+(?:\.\d+)?(?:〜\d+(?:\.\d+)?)?)\s*ヶ?月',
            r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*ヶ?月'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, detail_text)
            if matches:
                try:
                    # Handle ranges by taking the average
                    for match in matches:
                        if '-' in match or '〜' in match:
                            parts = re.split(r'[-〜]', match)
                            if len(parts) == 2:
                                start, end = float(parts[0]), float(parts[1])
                                return (start + end) / 2
                        else:
                            value = float(match)
                            if 2 <= value <= 20:  # Reasonable bonus range
                                return value
                except ValueError:
                    continue
        return None

    def extract_bonus_volatility(self, evidence_list: List[Dict], notes: str) -> str:
        """Determine bonus volatility level"""
        volatility_indicators = {
            'high': ['volatile', 'high volatility', 'cycles', 'variable', 'fluctuate', 'up to', 'minimum'],
            'medium': ['comprehensive', 'judgment', 'performance', 'results'],
            'low': ['stable', 'fixed', 'consistent', 'regular', 'standard']
        }

        all_text = notes.lower()
        for evidence in evidence_list:
            all_text += " " + evidence.get('detail', '').lower()

        scores = {}
        for level, indicators in volatility_indicators.items():
            scores[level] = sum(1 for indicator in indicators if indicator in all_text)

        return max(scores.items(), key=lambda x: x[1])[0] if max(scores.values()) > 0 else 'medium'

    def analyze_by_method(self) -> Dict:
        """Analyze employee benefits by bonus method"""
        method_data = defaultdict(list)

        for company in self.bonus_data['companies']:
            method = company['bonus_method']

            # Extract salary data
            salary = None
            bonus_months = None

            for evidence in company.get('evidence', []):
                detail = evidence.get('detail', '')

                if not salary:
                    salary = self.extract_salary_data(detail)

                if not bonus_months:
                    bonus_months = self.extract_bonus_months(detail)

            # Extract volatility
            volatility = self.extract_bonus_volatility(
                company.get('evidence', []),
                company.get('notes', '')
            )

            method_data[method].append({
                'company': company['company_name'],
                'stock_code': company['stock_code'],
                'salary_millions': salary,
                'bonus_months': bonus_months,
                'volatility': volatility,
                'confidence': company.get('confidence_level', 'Unknown')
            })

        return dict(method_data)

    def calculate_method_statistics(self, method_data: Dict) -> Dict:
        """Calculate statistics for each bonus method"""
        stats = {}

        for method, companies in method_data.items():
            salaries = [c['salary_millions'] for c in companies if c['salary_millions']]
            bonus_months_list = [c['bonus_months'] for c in companies if c['bonus_months']]
            volatilities = [c['volatility'] for c in companies]
            confidences = [c['confidence'] for c in companies]

            # Calculate bonus as percentage of total compensation
            bonus_percentages = []
            for company in companies:
                if company['salary_millions'] and company['bonus_months']:
                    # Estimate bonus percentage (assuming bonus months apply to monthly base)
                    annual_months = 12
                    bonus_ratio = company['bonus_months'] / annual_months
                    bonus_pct = bonus_ratio / (1 + bonus_ratio) * 100
                    bonus_percentages.append(bonus_pct)

            stats[method] = {
                'company_count': len(companies),
                'avg_salary_millions': np.mean(salaries) if salaries else None,
                'median_salary_millions': np.median(salaries) if salaries else None,
                'avg_bonus_months': np.mean(bonus_months_list) if bonus_months_list else None,
                'median_bonus_months': np.median(bonus_months_list) if bonus_months_list else None,
                'avg_bonus_percentage': np.mean(bonus_percentages) if bonus_percentages else None,
                'volatility_distribution': {v: volatilities.count(v) for v in set(volatilities)},
                'confidence_distribution': {c: confidences.count(c) for c in set(confidences)},
                'high_confidence_count': confidences.count('A') + confidences.count('B'),
                'companies': companies
            }

        return stats

    def score_employee_benefit(self, stats: Dict) -> Dict:
        """Score each method from employee perspective"""
        scores = {}

        # Weights for different factors (from employee perspective)
        weights = {
            'salary_level': 0.25,      # Higher salary is better
            'bonus_amount': 0.25,      # More bonus months is better
            'stability': 0.30,         # Lower volatility is better for security
            'data_reliability': 0.20   # Higher confidence in data
        }

        # Normalize data for scoring (0-100 scale)
        all_salaries = [s['avg_salary_millions'] for s in stats.values() if s['avg_salary_millions']]
        all_bonus_months = [s['avg_bonus_months'] for s in stats.values() if s['avg_bonus_months']]

        max_salary = max(all_salaries) if all_salaries else 1
        max_bonus = max(all_bonus_months) if all_bonus_months else 1

        for method, data in stats.items():
            score_components = {}

            # Salary score (0-100)
            if data['avg_salary_millions']:
                score_components['salary_level'] = (data['avg_salary_millions'] / max_salary) * 100
            else:
                score_components['salary_level'] = 50  # Neutral if no data

            # Bonus amount score (0-100)
            if data['avg_bonus_months']:
                score_components['bonus_amount'] = (data['avg_bonus_months'] / max_bonus) * 100
            else:
                score_components['bonus_amount'] = 50  # Neutral if no data

            # Stability score (0-100, higher is better)
            volatility_scores = {'low': 100, 'medium': 60, 'high': 20}
            vol_dist = data['volatility_distribution']
            total_companies = sum(vol_dist.values())
            if total_companies > 0:
                weighted_stability = sum(
                    volatility_scores[vol] * count / total_companies
                    for vol, count in vol_dist.items()
                )
                score_components['stability'] = weighted_stability
            else:
                score_components['stability'] = 60  # Neutral

            # Data reliability score (0-100)
            conf_scores = {'A': 100, 'B': 80, 'C': 60, 'D': 40, 'Unknown': 30}
            conf_dist = data['confidence_distribution']
            total_conf = sum(conf_dist.values())
            if total_conf > 0:
                weighted_reliability = sum(
                    conf_scores[conf] * count / total_conf
                    for conf, count in conf_dist.items()
                )
                score_components['data_reliability'] = weighted_reliability
            else:
                score_components['data_reliability'] = 50  # Neutral

            # Calculate weighted total score
            total_score = sum(
                score_components[factor] * weight
                for factor, weight in weights.items()
            )

            scores[method] = {
                'total_score': total_score,
                'components': score_components,
                'rank': 0  # Will be set after ranking
            }

        # Rank methods by total score
        ranked_methods = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        for rank, (method, score_data) in enumerate(ranked_methods, 1):
            scores[method]['rank'] = rank

        return scores

    def create_comparison_visualization(self, stats: Dict, scores: Dict):
        """Create comprehensive comparison visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        methods = list(stats.keys())
        colors = ['#2E8B57', '#4169E1', '#FF6347', '#DAA520']  # Professional colors

        # 1. Average Salary Comparison
        salaries = [stats[m]['avg_salary_millions'] or 0 for m in methods]
        bars1 = ax1.bar(methods, salaries, color=colors)
        ax1.set_title('Average Annual Salary by Bonus Method')
        ax1.set_ylabel('Annual Salary (Million Yen)')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars1, salaries):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'¥{value:.1f}M', ha='center', va='bottom', fontweight='bold')

        # 2. Average Bonus Months
        bonus_months = [stats[m]['avg_bonus_months'] or 0 for m in methods]
        bars2 = ax2.bar(methods, bonus_months, color=colors)
        ax2.set_title('Average Bonus Months by Method')
        ax2.set_ylabel('Bonus Months per Year')
        ax2.tick_params(axis='x', rotation=45)

        for bar, value in zip(bars2, bonus_months):
            if value > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

        # 3. Employee Benefit Score
        total_scores = [scores[m]['total_score'] for m in methods]
        bars3 = ax3.bar(methods, total_scores, color=colors)
        ax3.set_title('Employee Benefit Score (0-100)')
        ax3.set_ylabel('Benefit Score')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 100)

        for bar, value, rank in zip(bars3, total_scores, [scores[m]['rank'] for m in methods]):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}\n(#{rank})', ha='center', va='bottom', fontweight='bold')

        # 4. Volatility Distribution
        volatility_data = {}
        for method in methods:
            vol_dist = stats[method]['volatility_distribution']
            total = sum(vol_dist.values())
            volatility_data[method] = {
                'high': vol_dist.get('high', 0) / total * 100 if total > 0 else 0,
                'medium': vol_dist.get('medium', 0) / total * 100 if total > 0 else 0,
                'low': vol_dist.get('low', 0) / total * 100 if total > 0 else 0
            }

        vol_categories = ['high', 'medium', 'low']
        vol_colors = ['#FF6B6B', '#FFE66D', '#4ECDC4']

        bottom = np.zeros(len(methods))
        for i, vol_cat in enumerate(vol_categories):
            values = [volatility_data[m][vol_cat] for m in methods]
            ax4.bar(methods, values, bottom=bottom, label=f'{vol_cat.title()} Volatility', color=vol_colors[i])
            bottom += values

        ax4.set_title('Bonus Volatility Distribution (%)')
        ax4.set_ylabel('Percentage of Companies')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        ax4.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig('bonus_method_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_detailed_report(self, stats: Dict, scores: Dict):
        """Generate detailed analysis report"""
        print("\n" + "="*80)
        print("BONUS METHOD ANALYSIS: WHICH IS BETTER FOR EMPLOYEES?")
        print("="*80)

        # Overall ranking
        ranked_methods = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)

        print(f"\n🏆 EMPLOYEE BENEFIT RANKING:")
        for rank, (method, score_data) in enumerate(ranked_methods, 1):
            print(f"{rank}. {method:<20} Score: {score_data['total_score']:.1f}/100")

        # Detailed analysis by method
        for method, score_data in ranked_methods:
            stat_data = stats[method]
            print(f"\n{'='*60}")
            print(f"#{score_data['rank']} {method.upper()}")
            print(f"{'='*60}")

            print(f"📊 OVERALL SCORE: {score_data['total_score']:.1f}/100")
            print(f"📈 Companies surveyed: {stat_data['company_count']}")

            if stat_data['avg_salary_millions']:
                print(f"💰 Average salary: ¥{stat_data['avg_salary_millions']:.1f}M")
            else:
                print(f"💰 Average salary: No data available")

            if stat_data['avg_bonus_months']:
                print(f"🎁 Average bonus: {stat_data['avg_bonus_months']:.1f} months/year")
            else:
                print(f"🎁 Average bonus: No data available")

            if stat_data['avg_bonus_percentage']:
                print(f"📊 Bonus percentage: {stat_data['avg_bonus_percentage']:.1f}% of total compensation")

            # Component scores
            components = score_data['components']
            print(f"\n📋 COMPONENT SCORES:")
            print(f"   Salary Level:     {components['salary_level']:.1f}/100")
            print(f"   Bonus Amount:     {components['bonus_amount']:.1f}/100")
            print(f"   Stability:        {components['stability']:.1f}/100")
            print(f"   Data Reliability: {components['data_reliability']:.1f}/100")

            # Volatility breakdown
            vol_dist = stat_data['volatility_distribution']
            total_vol = sum(vol_dist.values())
            if total_vol > 0:
                print(f"\n🎲 VOLATILITY BREAKDOWN:")
                for vol_level, count in vol_dist.items():
                    pct = count / total_vol * 100
                    print(f"   {vol_level.title()} volatility: {count} companies ({pct:.1f}%)")

            # Top companies example
            high_conf_companies = [c for c in stat_data['companies']
                                 if c['confidence'] in ['A', 'B'] and c['salary_millions']]
            if high_conf_companies:
                print(f"\n🏢 EXAMPLE COMPANIES (High confidence data):")
                for company in high_conf_companies[:3]:
                    salary_str = f"¥{company['salary_millions']:.1f}M" if company['salary_millions'] else "N/A"
                    bonus_str = f"{company['bonus_months']:.1f}mo" if company['bonus_months'] else "N/A"
                    print(f"   {company['company']}: {salary_str}, {bonus_str} bonus")

        # Summary recommendations
        print(f"\n{'='*80}")
        print("💡 EMPLOYEE RECOMMENDATIONS")
        print(f"{'='*80}")

        best_method = ranked_methods[0][0]
        worst_method = ranked_methods[-1][0]

        print(f"\n🥇 BEST FOR EMPLOYEES: {best_method}")
        best_stats = stats[best_method]
        best_scores = scores[best_method]

        reasons = []
        if best_scores['components']['salary_level'] > 80:
            reasons.append("High average salary")
        if best_scores['components']['bonus_amount'] > 80:
            reasons.append("Generous bonus amounts")
        if best_scores['components']['stability'] > 80:
            reasons.append("High stability/predictability")
        if best_scores['components']['data_reliability'] > 80:
            reasons.append("Reliable data quality")

        if reasons:
            print(f"   Strengths: {', '.join(reasons)}")

        print(f"\n🥉 LEAST FAVORABLE: {worst_method}")
        worst_stats = stats[worst_method]
        worst_scores = scores[worst_method]

        concerns = []
        if worst_scores['components']['stability'] < 50:
            concerns.append("High volatility/uncertainty")
        if worst_scores['components']['salary_level'] < 50:
            concerns.append("Lower average salaries")
        if worst_scores['components']['bonus_amount'] < 50:
            concerns.append("Smaller bonus amounts")
        if worst_scores['components']['data_reliability'] < 60:
            concerns.append("Limited reliable data")

        if concerns:
            print(f"   Concerns: {', '.join(concerns)}")

        print(f"\n📝 KEY INSIGHTS:")
        print(f"• Salary range across methods: ¥{min(s['avg_salary_millions'] or 0 for s in stats.values()):.1f}M - ¥{max(s['avg_salary_millions'] or 0 for s in stats.values()):.1f}M")
        print(f"• Bonus range: {min(s['avg_bonus_months'] or 0 for s in stats.values()):.1f} - {max(s['avg_bonus_months'] or 0 for s in stats.values()):.1f} months")
        print(f"• Most stable method: {max(stats.items(), key=lambda x: scores[x[0]]['components']['stability'])[0]}")
        print(f"• Highest paying method: {max(stats.items(), key=lambda x: x[1]['avg_salary_millions'] or 0)[0]}")

    def run_analysis(self):
        """Run complete analysis"""
        print("Starting bonus method analysis...")

        # Extract and analyze data
        method_data = self.analyze_by_method()
        stats = self.calculate_method_statistics(method_data)
        scores = self.score_employee_benefit(stats)

        # Generate outputs
        self.create_comparison_visualization(stats, scores)
        self.generate_detailed_report(stats, scores)

        print(f"\n✅ Analysis complete! Visualization saved as 'bonus_method_comparison.png'")

        return stats, scores


def main():
    analyzer = BonusMethodAnalyzer()
    stats, scores = analyzer.run_analysis()

    return stats, scores


if __name__ == "__main__":
    main()