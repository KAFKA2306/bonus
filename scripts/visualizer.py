"""
Bonus Data Visualizer for Nikkei 225 Companies
Visualizes bonus methods, company sectors, and bonus distribution data.
Also performs deep analysis of bonus methods.
"""
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
import datetime
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
class BonusMethodAnalyzer:
    def __init__(self, data_dir=None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.resolve()
        else:
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
                        return float(value_str)
                    else:
                        return float(value_str) / 100
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
                    for match in matches:
                        if '-' in match or '〜' in match:
                            parts = re.split(r'[-〜]', match)
                            if len(parts) == 2:
                                start, end = float(parts[0]), float(parts[1])
                                return (start + end) / 2
                        else:
                            value = float(match)
                            if 2 <= value <= 20:
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
            salary = None
            bonus_months = None
            for evidence in company.get('evidence', []):
                detail = evidence.get('detail', '')
                if not salary:
                    salary = self.extract_salary_data(detail)
                if not bonus_months:
                    bonus_months = self.extract_bonus_months(detail)
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
            bonus_percentages = []
            for company in companies:
                if company['salary_millions'] and company['bonus_months']:
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
        weights = {
            'salary_level': 0.25,
            'bonus_amount': 0.25,
            'stability': 0.30,
            'data_reliability': 0.20
        }
        all_salaries = [s['avg_salary_millions'] for s in stats.values() if s['avg_salary_millions']]
        all_bonus_months = [s['avg_bonus_months'] for s in stats.values() if s['avg_bonus_months']]
        max_salary = max(all_salaries) if all_salaries else 1
        max_bonus = max(all_bonus_months) if all_bonus_months else 1
        for method, data in stats.items():
            score_components = {}
            if data['avg_salary_millions']:
                score_components['salary_level'] = (data['avg_salary_millions'] / max_salary) * 100
            else:
                score_components['salary_level'] = 50
            if data['avg_bonus_months']:
                score_components['bonus_amount'] = (data['avg_bonus_months'] / max_bonus) * 100
            else:
                score_components['bonus_amount'] = 50
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
                score_components['stability'] = 60
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
                score_components['data_reliability'] = 50
            total_score = sum(
                score_components[factor] * weight
                for factor, weight in weights.items()
            )
            scores[method] = {
                'total_score': total_score,
                'components': score_components,
                'rank': 0
            }
        ranked_methods = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        for rank, (method, score_data) in enumerate(ranked_methods, 1):
            scores[method]['rank'] = rank
        return scores
class BonusAnalysisVisualizer:
    def __init__(self, data_dir=None):
        self.analyzer = BonusMethodAnalyzer(data_dir)
        self.method_data = None
        self.stats = None
        self.scores = None
        self.results_yaml = None
    def run_analysis_and_export(self):
        """Run analysis and export results to YAML"""
        print("Running comprehensive bonus method analysis...")
        self.method_data = self.analyzer.analyze_by_method()
        self.stats = self.analyzer.calculate_method_statistics(self.method_data)
        self.scores = self.analyzer.score_employee_benefit(self.stats)
        self.create_results_yaml()
        self.create_comprehensive_dashboard()
        self.create_employee_decision_matrix()
        self.create_risk_reward_scatter()
        self.create_company_comparison_heatmap()
        self.save_yaml_results()
        print("✅ Analysis complete with YAML export and visualizations!")
    def create_results_yaml(self):
        """Create structured YAML output of analysis results"""
        ranked_methods = sorted(self.scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        self.results_yaml = {
            'analysis_metadata': {
                'generated_on': datetime.datetime.now().isoformat(),
                'total_companies_analyzed': sum(s['company_count'] for s in self.stats.values()),
                'analysis_methodology': 'Multi-factor employee benefit scoring',
                'scoring_weights': {
                    'salary_level': 0.25,
                    'bonus_amount': 0.25,
                    'stability': 0.30,
                    'data_reliability': 0.20
                }
            },
            'ranking_summary': {
                'best_for_employees': ranked_methods[0][0],
                'worst_for_employees': ranked_methods[-1][0],
                'highest_paying': max(self.stats.items(), key=lambda x: x[1]['avg_salary_millions'] or 0)[0],
                'most_stable': max(self.stats.items(), key=lambda x: self.scores[x[0]]['components']['stability'])[0],
                'highest_bonus': max(self.stats.items(), key=lambda x: x[1]['avg_bonus_months'] or 0)[0]
            },
            'method_analysis': {},
            'company_details': {},
            'insights': {
                'salary_range': {
                    'min_millions': min(s['avg_salary_millions'] or 0 for s in self.stats.values()),
                    'max_millions': max(s['avg_salary_millions'] or 0 for s in self.stats.values())
                },
                'bonus_range': {
                    'min_months': min(s['avg_bonus_months'] or 0 for s in self.stats.values()),
                    'max_months': max(s['avg_bonus_months'] or 0 for s in self.stats.values())
                },
                'volatility_patterns': {},
                'employee_recommendations': {}
            }
        }
        for rank, (method, score_data) in enumerate(ranked_methods, 1):
            stat_data = self.stats[method]
            self.results_yaml['method_analysis'][method] = {
                'rank': rank,
                'overall_score': round(score_data['total_score'], 1),
                'score_components': {
                    'salary_level': round(score_data['components']['salary_level'], 1),
                    'bonus_amount': round(score_data['components']['bonus_amount'], 1),
                    'stability': round(score_data['components']['stability'], 1),
                    'data_reliability': round(score_data['components']['data_reliability'], 1)
                },
                'statistics': {
                    'company_count': stat_data['company_count'],
                    'avg_salary_millions': stat_data['avg_salary_millions'],
                    'avg_bonus_months': stat_data['avg_bonus_months'],
                    'avg_bonus_percentage': stat_data['avg_bonus_percentage']
                },
                'volatility_distribution': stat_data['volatility_distribution'],
                'confidence_distribution': stat_data['confidence_distribution'],
                'strengths': self._identify_strengths(score_data['components']),
                'concerns': self._identify_concerns(score_data['components'])
            }
        for method, companies in self.method_data.items():
            self.results_yaml['company_details'][method] = []
            for company in companies:
                self.results_yaml['company_details'][method].append({
                    'company_name': company['company'],
                    'stock_code': company['stock_code'],
                    'salary_millions': company['salary_millions'],
                    'bonus_months': company['bonus_months'],
                    'volatility': company['volatility'],
                    'confidence_level': company['confidence']
                })
        self._add_insights()
    def _identify_strengths(self, components):
        strengths = []
        if components['salary_level'] > 80: strengths.append('High average salary')
        if components['bonus_amount'] > 80: strengths.append('Generous bonus amounts')
        if components['stability'] > 80: strengths.append('High stability and predictability')
        if components['data_reliability'] > 80: strengths.append('Reliable data quality')
        return strengths
    def _identify_concerns(self, components):
        concerns = []
        if components['stability'] < 50: concerns.append('High volatility and uncertainty')
        if components['salary_level'] < 50: concerns.append('Lower average salaries')
        if components['bonus_amount'] < 50: concerns.append('Smaller bonus amounts')
        if components['data_reliability'] < 60: concerns.append('Limited reliable data')
        return concerns
    def _add_insights(self):
        for method, stats in self.stats.items():
            vol_dist = stats['volatility_distribution']
            total = sum(vol_dist.values())
            if total > 0:
                dominant_volatility = max(vol_dist.items(), key=lambda x: x[1])[0]
                self.results_yaml['insights']['volatility_patterns'][method] = {
                    'dominant_pattern': dominant_volatility,
                    'low_volatility_percentage': vol_dist.get('low', 0) / total * 100,
                    'high_volatility_percentage': vol_dist.get('high', 0) / total * 100
                }
        ranked_methods = sorted(self.scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        best_method = ranked_methods[0][0]
        self.results_yaml['insights']['employee_recommendations'] = {
            'risk_averse_employees': {
                'recommended_method': max(self.stats.items(), key=lambda x: self.scores[x[0]]['components']['stability'])[0],
                'reason': 'Highest stability and predictability'
            },
            'growth_oriented_employees': {
                'recommended_method': max(self.stats.items(), key=lambda x: x[1]['avg_salary_millions'] or 0)[0],
                'reason': 'Highest earning potential'
            },
            'balanced_approach': {
                'recommended_method': best_method,
                'reason': 'Best overall employee benefit score'
            },
            'career_stage_considerations': {
                'early_career': 'Base salary-linked for stability',
                'mid_career': 'Performance-linked for growth',
                'late_career': 'Comprehensive judgment for balance'
            }
        }
    def create_comprehensive_dashboard(self):
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        methods = list(self.stats.keys())
        colors = ['
        ax1 = fig.add_subplot(gs[0, :2])
        scores = [self.scores[m]['total_score'] for m in methods]
        ranks = [self.scores[m]['rank'] for m in methods]
        bars = ax1.bar(methods, scores, color=colors)
        ax1.set_title('Employee Benefit Score by Method', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Benefit Score (0-100)')
        ax1.set_ylim(0, 100)
        for bar, score, rank in zip(bars, scores, ranks):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{score:.1f}\n
        ax2 = fig.add_subplot(gs[0, 2:], projection='polar')
        categories = ['Salary\nLevel', 'Bonus\nAmount', 'Stability', 'Data\nReliability']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        for i, method in enumerate(methods):
            values = [
                self.scores[method]['components']['salary_level'],
                self.scores[method]['components']['bonus_amount'],
                self.scores[method]['components']['stability'],
                self.scores[method]['components']['data_reliability']
            ]
            values += values[:1]
            ax2.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax2.fill(angles, values, alpha=0.1, color=colors[i])
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 100)
        ax2.set_title('Component Score Breakdown', fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax3 = fig.add_subplot(gs[1, :2])
        for i, method in enumerate(methods):
            stat = self.stats[method]
            if stat['avg_salary_millions'] and stat['avg_bonus_months']:
                ax3.scatter(stat['avg_salary_millions'], stat['avg_bonus_months'],
                          s=200, color=colors[i], label=method, alpha=0.7)
                ax3.annotate(method, (stat['avg_salary_millions'], stat['avg_bonus_months']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax3.set_xlabel('Average Salary (Million Yen)')
        ax3.set_ylabel('Average Bonus (Months)')
        ax3.set_title('Salary vs Bonus Relationship', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax4 = fig.add_subplot(gs[1, 2:])
        volatility_data = np.zeros((len(methods), 3))
        vol_labels = ['High', 'Medium', 'Low']
        for i, method in enumerate(methods):
            vol_dist = self.stats[method]['volatility_distribution']
            total = sum(vol_dist.values())
            if total > 0:
                volatility_data[i] = [
                    vol_dist.get('high', 0) / total * 100,
                    vol_dist.get('medium', 0) / total * 100,
                    vol_dist.get('low', 0) / total * 100
                ]
        bottom = np.zeros(len(methods))
        vol_colors = ['
        for i, vol_level in enumerate(vol_labels):
            ax4.bar(methods, volatility_data[:, i], bottom=bottom,
                   label=f'{vol_level} Volatility', color=vol_colors[i])
            bottom += volatility_data[:, i]
        ax4.set_title('Volatility Distribution by Method', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Percentage of Companies (%)')
        ax4.legend()
        ax4.set_ylim(0, 100)
        ax5 = fig.add_subplot(gs[2, :2])
        company_counts = [self.stats[m]['company_count'] for m in methods]
        high_conf_counts = [self.stats[m]['high_confidence_count'] for m in methods]
        x = np.arange(len(methods))
        width = 0.35
        bars1 = ax5.bar(x - width/2, company_counts, width, label='Total Companies', color='lightblue')
        bars2 = ax5.bar(x + width/2, high_conf_counts, width, label='High Confidence (A+B)', color='darkblue')
        ax5.set_xlabel('Bonus Methods')
        ax5.set_ylabel('Number of Companies')
        ax5.set_title('Data Coverage and Confidence', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(methods, rotation=45)
        ax5.legend()
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1, 
                        f'{int(height)}', ha='center', va='bottom')
        ax6 = fig.add_subplot(gs[2, 2:])
        for i, method in enumerate(methods):
            risk = 100 - self.scores[method]['components']['stability']
            reward = self.scores[method]['components']['salary_level']
            ax6.scatter(risk, reward, s=300, color=colors[i], alpha=0.8,
                       marker='s', edgecolor='black', linewidth=2)
            ax6.annotate(method, (risk, reward), xytext=(5, 5),
                        textcoords='offset points', fontsize=10, fontweight='bold')
        ax6.set_xlabel('Risk Level (100 - Stability Score)')
        ax6.set_ylabel('Reward Level (Salary Score)')
        ax6.set_title('Risk vs Reward Matrix', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax6.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        ax6.text(25, 75, 'Low Risk\nHigh Reward', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        ax6.text(75, 75, 'High Risk\nHigh Reward', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.5))
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('tight')
        ax7.axis('off')
        table_data = []
        for method in methods:
            stat = self.stats[method]
            score = self.scores[method]
            table_data.append([
                method,
                f"
                f"{score['total_score']:.1f}",
                f"¥{stat['avg_salary_millions']:.1f}M" if stat['avg_salary_millions'] else "N/A",
                f"{stat['avg_bonus_months']:.1f}" if stat['avg_bonus_months'] else "N/A",
                f"{stat['company_count']}",
                f"{score['components']['stability']:.1f}"
            ])
        table = ax7.table(cellText=table_data,
                         colLabels=['Method', 'Rank', 'Score', 'Avg Salary', 'Avg Bonus', 'Companies', 'Stability'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        rank_colors = ['
        for i, row in enumerate(table_data):
            rank = int(row[1][1]) - 1
            for j in range(len(row)):
                table[(i+1, j)].set_facecolor(rank_colors[rank] if rank < len(rank_colors) else '
        ax7.set_title('Summary Statistics by Method', fontsize=14, fontweight='bold', pad=20)
        plt.suptitle('Comprehensive Bonus Method Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
        plt.savefig('bonus_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    def create_employee_decision_matrix(self):
        fig = plt.figure(figsize=(16, 12))
        grid = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.25)
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[1, 0], polar=True)
        ax4 = fig.add_subplot(grid[1, 1])
        methods = list(self.stats.keys())
        colors = ['
        career_stages = ['Early Career\n(Stability)', 'Mid Career\n(Growth)', 'Late Career\n(Balance)']
        recommendations = ['Base salary-linked', 'Performance-linked', 'Comprehensive judgment']
        y_pos = np.arange(len(career_stages))
        ax1.barh(y_pos, [90, 85, 75], color=['
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(career_stages)
        ax1.set_xlabel('Recommendation Strength (%)')
        ax1.set_title('Recommendations by Career Stage')
        for i, (stage, rec) in enumerate(zip(career_stages, recommendations)):
            ax1.text(45, i, rec, ha='center', va='center', fontweight='bold', color='white')
        risk_levels = ['Risk Averse', 'Moderate Risk', 'High Risk']
        method_scores_by_risk = {
            'Base salary-linked': [95, 70, 40],
            'Performance-linked': [30, 75, 95],
            'Comprehensive judgment': [60, 85, 60],
            'Hybrid': [50, 60, 50]
        }
        x = np.arange(len(risk_levels))
        width = 0.2
        for i, (method, scores) in enumerate(method_scores_by_risk.items()):
            ax2.bar(x + i*width, scores, width, label=method, color=colors[i])
        ax2.set_xlabel('Risk Tolerance Level')
        ax2.set_ylabel('Suitability Score')
        ax2.set_title('Method Suitability by Risk Tolerance')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(risk_levels)
        ax2.legend()
        priorities = ['Stability', 'Maximum Earnings', 'Work-Life Balance']
        priority_scores = {
            'Base salary-linked': [95, 60, 85],
            'Performance-linked': [40, 95, 50],
            'Comprehensive judgment': [70, 75, 80],
            'Hybrid': [60, 65, 70]
        }
        angles = np.linspace(0, 2 * np.pi, len(priorities), endpoint=False).tolist()
        angles += angles[:1]
        ax3.set_theta_offset(np.pi / 2)
        ax3.set_theta_direction(-1)
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(priorities)
        ax3.set_ylim(0, 100)
        for i, (method, scores) in enumerate(priority_scores.items()):
            values = scores + scores[:1]
            ax3.plot(angles, values, linewidth=2, label=method, color=colors[i])
            ax3.fill(angles, values, alpha=0.15, color=colors[i])
        ax3.set_title('Method Alignment with Employee Priorities', pad=20)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05))
        ax4.text(0.5, 0.9, 'BONUS METHOD DECISION TREE', ha='center', va='center',
                fontsize=14, fontweight='bold', transform=ax4.transAxes)
        tree_text = '''
        START: What's most important to you?
        🔹 STABILITY & PREDICTABILITY
           → Base Salary-Linked
           ✓ Consistent bonuses
           ✓ Lower stress
           ✓ Easy financial planning
        🔹 MAXIMUM EARNING POTENTIAL
           → Performance-Linked
           ✓ Highest salaries
           ✓ Large bonuses possible
           ⚠ High volatility
        🔹 BALANCED APPROACH
           → Comprehensive Judgment
           ✓ Moderate risk/reward
           ✓ Management flexibility
           ~ Medium predictability
        🔹 COMPANY-SPECIFIC BENEFITS
           → Hybrid Systems
           ✓ Customized approach
           ⚠ Limited data available
        '''
        ax4.text(0.05, 0.8, tree_text, ha='left', va='top', fontsize=10,
                transform=ax4.transAxes, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        plt.tight_layout()
        plt.savefig('employee_decision_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    def create_risk_reward_scatter(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        methods = list(self.stats.keys())
        colors = ['
        for i, method in enumerate(methods):
            companies = self.method_data[method]
            for company in companies:
                if company['salary_millions'] and company['volatility']:
                    risk_score = {'low': 20, 'medium': 60, 'high': 90}[company['volatility']]
                    reward_score = company['salary_millions'] / 10
                    ax1.scatter(risk_score, reward_score, s=100, color=colors[i], alpha=0.6)
                    if company['salary_millions'] > 1000 or company['company'] in ['KEYENCE', 'DISCO', 'Fanuc']:
                        ax1.annotate(company['company'], (risk_score, reward_score),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
        for i, method in enumerate(methods):
            risk = 100 - self.scores[method]['components']['stability']
            reward = self.scores[method]['components']['salary_level']
            ax1.scatter(risk, reward, s=400, color=colors[i], alpha=0.8,
                       marker='s', edgecolor='black', linewidth=2, label=method)
        ax1.set_xlabel('Risk Level (Volatility & Uncertainty)')
        ax1.set_ylabel('Reward Level (Salary Potential)')
        ax1.set_title('Risk vs Reward: Individual Companies & Method Averages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        for i, method in enumerate(methods):
            stability = self.scores[method]['components']['stability']
            total_score = self.scores[method]['total_score']
            company_count = self.stats[method]['company_count']
            ax2.scatter(stability, total_score, s=company_count*50, color=colors[i], 
                       alpha=0.7, label=method, edgecolor='black', linewidth=1)
            ax2.annotate(f"{method}\n({company_count} companies)",
                        (stability, total_score), xytext=(5, 5),
                        textcoords='offset points', fontsize=9, fontweight='bold')
        ax2.set_xlabel('Stability Score')
        ax2.set_ylabel('Overall Employee Benefit Score')
        ax2.set_title('Stability vs Overall Benefit (Bubble size = Company count)')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('risk_reward_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    def create_company_comparison_heatmap(self):
        company_data = []
        company_names = []
        for method, companies in self.method_data.items():
            for company in companies:
                if company['salary_millions'] and company['bonus_months']:
                    company_data.append([
                        company['salary_millions'] / 100,
                        company['bonus_months'],
                        {'low': 1, 'medium': 2, 'high': 3}[company['volatility']],
                        {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'Unknown': 0}[company['confidence']]
                    ])
                    company_names.append(f"{company['company']}\n({method})")
        if company_data:
            df = pd.DataFrame(company_data,
                            columns=['Salary (×100M)', 'Bonus Months', 'Volatility', 'Confidence'],
                            index=company_names)
            plt.figure(figsize=(12, max(8, len(company_names) * 0.4)))
            sns.heatmap(df.T, annot=True, cmap='RdYlBu_r', cbar_kws={'label': 'Score'})
            plt.title('Company Comparison Heatmap\n(Salary, Bonus, Volatility, Confidence)')
            plt.tight_layout()
            plt.savefig('company_comparison_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
    def _convert_numpy_types(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list): return [self._convert_numpy_types(item) for item in obj]
        else: return obj
    def save_yaml_results(self):
        output_file = Path('../analysis/bonus_method_analysis_results.yaml')
        clean_results = self._convert_numpy_types(self.results_yaml)
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(clean_results, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        print(f"✅ Analysis results saved to: {output_file}")
    def print_yaml_summary(self):
        print("\n" + "="*80)
        print("YAML ANALYSIS RESULTS SUMMARY")
        print("="*80)
        print(f"\n📊 ANALYSIS METADATA:")
        metadata = self.results_yaml['analysis_metadata']
        print(f"   Generated: {metadata['generated_on']}")
        print(f"   Total companies: {metadata['total_companies_analyzed']}")
        print(f"   Methodology: {metadata['analysis_methodology']}")
        print(f"\n🏆 KEY RANKINGS:")
        rankings = self.results_yaml['ranking_summary']
        for key, value in rankings.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        print(f"\n📈 INSIGHTS:")
        insights = self.results_yaml['insights']
        salary_range = insights['salary_range']
        bonus_range = insights['bonus_range']
        print(f"   Salary range: ¥{salary_range['min_millions']:.1f}M - ¥{salary_range['max_millions']:.1f}M")
        print(f"   Bonus range: {bonus_range['min_months']:.1f} - {bonus_range['max_months']:.1f} months")
class BonusVisualizer:
    def __init__(self, data_dir=None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.resolve()
        else:
            self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "analysis" / "graphs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.companies_data = None
        self.bonus_data = None
        self.company_index = defaultdict(list)
        self.load_data()
    def load_data(self):
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
            if stripped.startswith('
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
        for base in [self.data_dir / "companies", self.data_dir / "analysis" / "phase3_estimates"]:
            if not base.exists():
                continue
            for path in base.rglob('*.yaml'):
                stem = path.stem
                code = stem.split('_', 1)[0]
                if code.isdigit():
                    self.company_index[code].append(path)
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
        bars = plt.bar(method_counts.keys(), method_counts.values(), color=['
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
        colors = {'A': '
        bar_colors = [colors.get(level, '
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
            'sector_distribution.png', 'bonus_methods.png', 'confidence_levels.png',
            'summary_table.png', 'detailed_summary_table.png', 'company_statistics_table.png',
        ]:
            print(f"  - {filename}")
    def _extract_company_metrics(self, stock_code: str, survey_entry: dict) -> dict:
        metrics = {'avg_income': 'N/A', 'bonus': 'N/A', 'frequency': 'N/A'}
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
        if not paths: return None
        def rank(path: Path):
            name = path.name
            return (name.count('-'), len(name))
        return sorted(paths, key=rank)[0]
    def _update_metrics_from_company_yaml(self, metrics: dict, data: dict):
        bonus_system = data.get('bonus_system') or {}
        bonus_system_estimate = data.get('bonus_system_estimate') or {}
        if not isinstance(bonus_system, dict): bonus_system = {}
        if not isinstance(bonus_system_estimate, dict): bonus_system_estimate = {}
        methodology = bonus_system.get('methodology') or {}
        if not isinstance(methodology, dict): methodology = {}
        performance_metrics = bonus_system.get('performance_metrics') or {}
        if not isinstance(performance_metrics, dict): performance_metrics = {}
        estimate_metrics = bonus_system_estimate
        financial_data = data.get('financial_data') or {}
        if not isinstance(financial_data, dict): financial_data = {}
        if isinstance(methodology.get('payment_frequency'), str):
            self._apply_frequency(metrics, methodology['payment_frequency'])
        if isinstance(estimate_metrics.get('estimated_bonus_multiple'), str):
            self._apply_bonus_months(metrics, estimate_metrics['estimated_bonus_multiple'])
        for key in ('base_salary_months', 'bonus_range', 'annual_bonus_amount'):
            value = performance_metrics.get(key)
            if isinstance(value, str):
                self._apply_bonus_months(metrics, value)
        profile = data.get('company_profile') or {}
        if not isinstance(profile, dict): profile = {}
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
        if metrics['avg_income'] != 'N/A': return
        if not any(keyword in text for keyword in ('平均年収', 'average annual income', 'average income', '平均給与')): return
        amount = self._parse_money_to_million_yen(text)
        if amount is not None:
            metrics['avg_income'] = f"¥{amount:.2f}M"
    def _apply_bonus_months(self, metrics: dict, text: str):
        if metrics['bonus'] != 'N/A': return
        parsed = self._parse_bonus_months(text)
        if parsed: metrics['bonus'] = parsed
    def _apply_frequency(self, metrics: dict, text: str):
        if metrics['frequency'] != 'N/A': return
        parsed = self._parse_frequency(text)
        if parsed: metrics['frequency'] = parsed
    @staticmethod
    def _parse_money_to_million_yen(text: str):
        clean = text.replace(',', '')
        match = re.search(r'約?([0-9]+(?:\.[0-9]+)?)\s*兆円', clean)
        if match: return float(match.group(1)) * 10000
        match = re.search(r'約?([0-9]+(?:\.[0-9]+)?)\s*億円', clean)
        if match: return float(match.group(1)) * 100
        match = re.search(r'約?([0-9]+(?:\.[0-9]+)?)\s*万円', clean)
        if match: return float(match.group(1)) / 100
        match = re.search(r'¥\s*([0-9]+(?:\.[0-9]+)?)M', clean)
        if match: return float(match.group(1))
        match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*million\s*yen', clean, re.IGNORECASE)
        if match: return float(match.group(1))
        match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*百万円', clean)
        if match: return float(match.group(1))
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
        if match: return f"{match.group(1)}x/year"
        match = re.search(r'(\d+)\s*x\s*/\s*year', clean)
        if match: return f"{match.group(1)}x/year"
        if 'quarterly' in clean or '四半期' in text: return '4x/year'
        if any(keyword in clean for keyword in ('twice-yearly', 'twice yearly', 'biannual')) or '年2回' in text: return '2x/year'
        if '年1回' in text or 'once a year' in clean: return '1x/year'
        if '年4回' in text: return '4x/year'
        return ''
    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if text is None: return 'N/A'
        text = str(text)
        return text if len(text) <= limit else text[:limit - 3] + '...'
    @staticmethod
    def _confidence_color(level: str) -> str:
        if level == 'A': return '
        if level == 'B': return '
        if level == 'C': return '
        return '
def main():
    parser = argparse.ArgumentParser(description='Visualize and analyze Nikkei 225 bonus data')
    parser.add_argument('--data-dir', help='Directory containing YAML data files')
    parser.add_argument('--task', choices=['report', 'analysis', 'all'], default='all',
                        help='Specify the task to run: "report" for basic visualization, "analysis" for deep analysis, "all" for both.')
    args = parser.parse_args()
    if args.task in ['report', 'all']:
        visualizer = BonusVisualizer(args.data_dir)
        visualizer.generate_report()
    if args.task in ['analysis', 'all']:
        analysis_visualizer = BonusAnalysisVisualizer(args.data_dir)
        analysis_visualizer.run_analysis_and_export()
        analysis_visualizer.print_yaml_summary()
if __name__ == "__main__":
    main()
