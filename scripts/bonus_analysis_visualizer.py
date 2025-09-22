#!/usr/bin/env python3
"""
Bonus Analysis Results Visualizer
Creates comprehensive visualizations and YAML output from bonus method analysis
"""

import yaml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import datetime
from bonus_method_analysis import BonusMethodAnalyzer


class BonusAnalysisVisualizer:
    def __init__(self):
        self.analyzer = BonusMethodAnalyzer()
        self.method_data = None
        self.stats = None
        self.scores = None
        self.results_yaml = None

    def run_analysis_and_export(self):
        """Run analysis and export results to YAML"""
        print("Running comprehensive bonus method analysis...")

        # Get analysis results
        self.method_data = self.analyzer.analyze_by_method()
        self.stats = self.analyzer.calculate_method_statistics(self.method_data)
        self.scores = self.analyzer.score_employee_benefit(self.stats)

        # Create YAML structure
        self.create_results_yaml()

        # Generate all visualizations
        self.create_comprehensive_dashboard()
        self.create_employee_decision_matrix()
        self.create_risk_reward_scatter()
        self.create_company_comparison_heatmap()

        # Save YAML results
        self.save_yaml_results()

        print("✅ Analysis complete with YAML export and visualizations!")

    def create_results_yaml(self):
        """Create structured YAML output of analysis results"""
        # Rank methods by score
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

        # Fill in method analysis
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

        # Fill in company details
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

        # Add insights
        self._add_insights()

    def _identify_strengths(self, components):
        """Identify strengths based on component scores"""
        strengths = []
        if components['salary_level'] > 80:
            strengths.append('High average salary')
        if components['bonus_amount'] > 80:
            strengths.append('Generous bonus amounts')
        if components['stability'] > 80:
            strengths.append('High stability and predictability')
        if components['data_reliability'] > 80:
            strengths.append('Reliable data quality')
        return strengths

    def _identify_concerns(self, components):
        """Identify concerns based on component scores"""
        concerns = []
        if components['stability'] < 50:
            concerns.append('High volatility and uncertainty')
        if components['salary_level'] < 50:
            concerns.append('Lower average salaries')
        if components['bonus_amount'] < 50:
            concerns.append('Smaller bonus amounts')
        if components['data_reliability'] < 60:
            concerns.append('Limited reliable data')
        return concerns

    def _add_insights(self):
        """Add analytical insights to YAML"""
        # Volatility patterns
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

        # Employee recommendations
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
        """Create a comprehensive dashboard visualization"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        methods = list(self.stats.keys())
        colors = ['#2E8B57', '#4169E1', '#FF6347', '#DAA520']

        # 1. Overall Score Comparison (Top Left)
        ax1 = fig.add_subplot(gs[0, :2])
        scores = [self.scores[m]['total_score'] for m in methods]
        ranks = [self.scores[m]['rank'] for m in methods]
        bars = ax1.bar(methods, scores, color=colors)
        ax1.set_title('Employee Benefit Score by Method', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Benefit Score (0-100)')
        ax1.set_ylim(0, 100)

        for bar, score, rank in zip(bars, scores, ranks):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{score:.1f}\n#{rank}', ha='center', va='bottom', fontweight='bold')

        # 2. Component Breakdown Radar Chart (Top Right)
        ax2 = fig.add_subplot(gs[0, 2:], projection='polar')

        categories = ['Salary\nLevel', 'Bonus\nAmount', 'Stability', 'Data\nReliability']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for i, method in enumerate(methods):
            values = [
                self.scores[method]['components']['salary_level'],
                self.scores[method]['components']['bonus_amount'],
                self.scores[method]['components']['stability'],
                self.scores[method]['components']['data_reliability']
            ]
            values += values[:1]  # Complete the circle

            ax2.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax2.fill(angles, values, alpha=0.1, color=colors[i])

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 100)
        ax2.set_title('Component Score Breakdown', fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # 3. Salary vs Bonus Scatter (Middle Left)
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

        # 4. Volatility Distribution (Middle Right)
        ax4 = fig.add_subplot(gs[1, 2:])

        volatility_data = np.zeros((len(methods), 3))  # high, medium, low
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
        vol_colors = ['#FF6B6B', '#FFE66D', '#4ECDC4']

        for i, vol_level in enumerate(vol_labels):
            ax4.bar(methods, volatility_data[:, i], bottom=bottom,
                   label=f'{vol_level} Volatility', color=vol_colors[i])
            bottom += volatility_data[:, i]

        ax4.set_title('Volatility Distribution by Method', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Percentage of Companies (%)')
        ax4.legend()
        ax4.set_ylim(0, 100)

        # 5. Company Count and Confidence (Bottom Left)
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

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')

        # 6. Risk-Reward Matrix (Bottom Right)
        ax6 = fig.add_subplot(gs[2, 2:])

        for i, method in enumerate(methods):
            risk = 100 - self.scores[method]['components']['stability']  # Inverse of stability
            reward = self.scores[method]['components']['salary_level']

            ax6.scatter(risk, reward, s=300, color=colors[i], alpha=0.7, label=method)
            ax6.annotate(method, (risk, reward), xytext=(5, 5),
                        textcoords='offset points', fontsize=10, fontweight='bold')

        ax6.set_xlabel('Risk Level (100 - Stability Score)')
        ax6.set_ylabel('Reward Level (Salary Score)')
        ax6.set_title('Risk vs Reward Matrix', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # Add quadrant labels
        ax6.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax6.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        ax6.text(25, 75, 'Low Risk\nHigh Reward', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        ax6.text(75, 75, 'High Risk\nHigh Reward', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.5))

        # 7. Summary Statistics Table (Bottom Full Width)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('tight')
        ax7.axis('off')

        # Create summary table
        table_data = []
        for method in methods:
            stat = self.stats[method]
            score = self.scores[method]
            table_data.append([
                method,
                f"#{score['rank']}",
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

        # Color code by rank
        rank_colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#D3D3D3']  # Gold, Silver, Bronze, Gray
        for i, row in enumerate(table_data):
            rank = int(row[1][1]) - 1  # Extract rank number
            for j in range(len(row)):
                table[(i+1, j)].set_facecolor(rank_colors[rank] if rank < len(rank_colors) else '#F0F0F0')

        ax7.set_title('Summary Statistics by Method', fontsize=14, fontweight='bold', pad=20)

        plt.suptitle('Comprehensive Bonus Method Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
        plt.savefig('bonus_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_employee_decision_matrix(self):
        """Create employee decision matrix visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        methods = list(self.stats.keys())
        colors = ['#2E8B57', '#4169E1', '#FF6347', '#DAA520']

        # 1. Career Stage Recommendations
        career_stages = ['Early Career\n(Stability)', 'Mid Career\n(Growth)', 'Late Career\n(Balance)']
        recommendations = [
            'Base salary-linked',
            'Performance-linked',
            'Comprehensive judgment'
        ]

        y_pos = np.arange(len(career_stages))
        ax1.barh(y_pos, [90, 85, 75], color=['#4ECDC4', '#FF6B6B', '#FFE66D'])
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(career_stages)
        ax1.set_xlabel('Recommendation Strength (%)')
        ax1.set_title('Recommendations by Career Stage')

        for i, (stage, rec) in enumerate(zip(career_stages, recommendations)):
            ax1.text(45, i, rec, ha='center', va='center', fontweight='bold', color='white')

        # 2. Risk Tolerance Matrix
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

        # 3. Income Priority Matrix
        priorities = ['Stability', 'Maximum Earnings', 'Work-Life Balance']
        priority_scores = {
            'Base salary-linked': [95, 60, 85],
            'Performance-linked': [40, 95, 50],
            'Comprehensive judgment': [70, 75, 80],
            'Hybrid': [60, 65, 70]
        }

        angles = np.linspace(0, 2 * np.pi, len(priorities), endpoint=False).tolist()
        angles += angles[:1]

        for i, (method, scores) in enumerate(priority_scores.items()):
            values = scores + scores[:1]
            ax3.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax3.fill(angles, values, alpha=0.1, color=colors[i])

        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(priorities)
        ax3.set_ylim(0, 100)
        ax3.set_title('Method Alignment with Employee Priorities')
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # 4. Decision Tree Visualization
        ax4.text(0.5, 0.9, 'BONUS METHOD DECISION TREE', ha='center', va='center',
                fontsize=14, fontweight='bold', transform=ax4.transAxes)

        # Decision tree structure
        tree_text = """
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
        """

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
        """Create detailed risk-reward analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        methods = list(self.stats.keys())
        colors = ['#2E8B57', '#4169E1', '#FF6347', '#DAA520']

        # 1. Risk vs Reward with Company Examples
        for i, method in enumerate(methods):
            companies = self.method_data[method]

            for company in companies:
                if company['salary_millions'] and company['volatility']:
                    risk_score = {'low': 20, 'medium': 60, 'high': 90}[company['volatility']]
                    reward_score = company['salary_millions'] / 10  # Scale for visibility

                    ax1.scatter(risk_score, reward_score, s=100, color=colors[i], alpha=0.6)

                    # Label high-profile companies
                    if company['salary_millions'] > 1000 or company['company'] in ['KEYENCE', 'DISCO', 'Fanuc']:
                        ax1.annotate(company['company'], (risk_score, reward_score),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Add method centroids
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

        # 2. Stability vs Total Compensation
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
        """Create company comparison heatmap"""
        # Collect company data for heatmap
        company_data = []
        company_names = []

        for method, companies in self.method_data.items():
            for company in companies:
                if company['salary_millions'] and company['bonus_months']:
                    company_data.append([
                        company['salary_millions'] / 100,  # Scale for heatmap
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
        """Convert numpy types to native Python types for YAML serialization"""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def save_yaml_results(self):
        """Save analysis results to YAML file"""
        output_file = Path('../analysis/bonus_method_analysis_results.yaml')

        # Convert numpy types to native Python types
        clean_results = self._convert_numpy_types(self.results_yaml)

        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(clean_results, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

        print(f"✅ Analysis results saved to: {output_file}")

    def print_yaml_summary(self):
        """Print key findings from YAML"""
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


def main():
    visualizer = BonusAnalysisVisualizer()
    visualizer.run_analysis_and_export()
    visualizer.print_yaml_summary()


if __name__ == "__main__":
    main()