#!/usr/bin/env python3
"""
Data Gap Analyzer - Identifies missing data and suggests research targets
"""

import yaml
import re
from pathlib import Path
from collections import defaultdict


class DataGapAnalyzer:
    def __init__(self, data_dir="../"):
        self.data_dir = Path(data_dir)
        self.bonus_data = None
        self.load_data()

    def load_data(self):
        """Load the English bonus survey data"""
        english_file = self.data_dir / "nikkei225_bonus_survey_2024_en.yaml"
        with open(english_file, 'r', encoding='utf-8') as f:
            self.bonus_data = yaml.safe_load(f)

    def extract_data_status(self):
        """Extract current data status for each company"""
        results = []

        for company in self.bonus_data['companies']:
            company_data = {
                'company_name': company['company_name'],
                'stock_code': company['stock_code'],
                'bonus_method': company['bonus_method'],
                'confidence_level': company.get('confidence_level', 'Unknown'),
                'has_salary_data': False,
                'has_bonus_months': False,
                'has_frequency_data': False,
                'salary_value': None,
                'bonus_value': None,
                'frequency_value': None,
                'evidence_count': len(company.get('evidence', [])),
                'needs_research': False
            }

            # Check evidence for data availability
            if 'evidence' in company:
                for evidence in company['evidence']:
                    detail = evidence.get('detail', '')

                    # Check for salary data
                    if not company_data['has_salary_data']:
                        salary_patterns = [
                            r'average annual income.*?(\d+(?:,\d+)*(?:\.\d+)?)\s*million yen',
                            r'平均年収.*?(\d+(?:,\d+)*(?:\.\d+)?)\s*万円',
                            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*million yen'
                        ]
                        for pattern in salary_patterns:
                            match = re.search(pattern, detail, re.IGNORECASE)
                            if match:
                                company_data['has_salary_data'] = True
                                company_data['salary_value'] = match.group(1)
                                break

                    # Check for bonus months
                    if not company_data['has_bonus_months']:
                        bonus_patterns = [
                            r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*months',
                            r'(\d+(?:\.\d+)?(?:〜\d+(?:\.\d+)?)?)\s*ヶ?月'
                        ]
                        for pattern in bonus_patterns:
                            match = re.search(pattern, detail)
                            if match:
                                company_data['has_bonus_months'] = True
                                company_data['bonus_value'] = match.group(1)
                                break

                    # Check for frequency
                    if not company_data['has_frequency_data']:
                        frequency_indicators = [
                            'twice-yearly', 'twice yearly', 'quarterly', 'four times',
                            '年2回', '年4回', '4回', 'semi-annual'
                        ]
                        for indicator in frequency_indicators:
                            if indicator in detail.lower():
                                company_data['has_frequency_data'] = True
                                company_data['frequency_value'] = indicator
                                break

            # Determine if needs research
            missing_critical_data = (
                not company_data['has_salary_data'] or
                not company_data['has_bonus_months']
            )
            low_confidence = company_data['confidence_level'] in ['C', 'D', 'Unknown']
            limited_evidence = company_data['evidence_count'] < 2

            company_data['needs_research'] = missing_critical_data or (low_confidence and limited_evidence)

            results.append(company_data)

        return results

    def prioritize_research_targets(self, data_status):
        """Prioritize companies for research based on gaps and importance"""
        priority_scores = []

        for company in data_status:
            score = 0
            reasons = []

            # Missing salary data (high priority)
            if not company['has_salary_data']:
                score += 20
                reasons.append("Missing salary data")

            # Missing bonus months (high priority)
            if not company['has_bonus_months']:
                score += 20
                reasons.append("Missing bonus months")

            # Missing frequency (medium priority)
            if not company['has_frequency_data']:
                score += 10
                reasons.append("Missing frequency data")

            # Low confidence level
            if company['confidence_level'] in ['D']:
                score += 15
                reasons.append("Low confidence (D)")
            elif company['confidence_level'] in ['C']:
                score += 10
                reasons.append("Medium-low confidence (C)")

            # Limited evidence
            if company['evidence_count'] < 2:
                score += 8
                reasons.append("Limited evidence sources")

            # Performance-linked methods get higher priority (more volatile, need better data)
            if company['bonus_method'] == 'Performance-linked':
                score += 5
                reasons.append("Performance-linked method")

            priority_scores.append({
                'company': company['company_name'],
                'stock_code': company['stock_code'],
                'method': company['bonus_method'],
                'confidence': company['confidence_level'],
                'priority_score': score,
                'reasons': reasons,
                'missing_data': {
                    'salary': not company['has_salary_data'],
                    'bonus_months': not company['has_bonus_months'],
                    'frequency': not company['has_frequency_data']
                }
            })

        # Sort by priority score (highest first)
        return sorted(priority_scores, key=lambda x: x['priority_score'], reverse=True)

    def generate_research_report(self):
        """Generate comprehensive research gap report"""
        data_status = self.extract_data_status()
        research_priorities = self.prioritize_research_targets(data_status)

        print("="*80)
        print("DATA GAP ANALYSIS & RESEARCH PRIORITIZATION")
        print("="*80)

        # Overall statistics
        total_companies = len(data_status)
        missing_salary = sum(1 for c in data_status if not c['has_salary_data'])
        missing_bonus = sum(1 for c in data_status if not c['has_bonus_months'])
        missing_frequency = sum(1 for c in data_status if not c['has_frequency_data'])
        low_confidence = sum(1 for c in data_status if c['confidence_level'] in ['C', 'D'])

        print(f"\n📊 OVERALL DATA STATUS:")
        print(f"   Total companies: {total_companies}")
        print(f"   Missing salary data: {missing_salary} ({missing_salary/total_companies*100:.1f}%)")
        print(f"   Missing bonus months: {missing_bonus} ({missing_bonus/total_companies*100:.1f}%)")
        print(f"   Missing frequency: {missing_frequency} ({missing_frequency/total_companies*100:.1f}%)")
        print(f"   Low confidence (C/D): {low_confidence} ({low_confidence/total_companies*100:.1f}%)")

        # Top research priorities
        print(f"\n🎯 TOP 10 RESEARCH PRIORITIES:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Company':<25} {'Code':<6} {'Score':<6} {'Missing Data':<20} {'Confidence'}")
        print("-" * 80)

        for i, priority in enumerate(research_priorities[:10], 1):
            missing_items = []
            if priority['missing_data']['salary']:
                missing_items.append('Salary')
            if priority['missing_data']['bonus_months']:
                missing_items.append('Bonus')
            if priority['missing_data']['frequency']:
                missing_items.append('Freq')

            missing_str = ', '.join(missing_items) if missing_items else 'None'

            print(f"{i:<4} {priority['company']:<25} {priority['stock_code']:<6} "
                  f"{priority['priority_score']:<6} {missing_str:<20} {priority['confidence']}")

        # Method-specific gaps
        print(f"\n📈 GAPS BY BONUS METHOD:")
        method_gaps = defaultdict(lambda: {'salary': 0, 'bonus': 0, 'frequency': 0, 'total': 0})

        for company in data_status:
            method = company['bonus_method']
            method_gaps[method]['total'] += 1
            if not company['has_salary_data']:
                method_gaps[method]['salary'] += 1
            if not company['has_bonus_months']:
                method_gaps[method]['bonus'] += 1
            if not company['has_frequency_data']:
                method_gaps[method]['frequency'] += 1

        for method, gaps in method_gaps.items():
            print(f"\n   {method}:")
            print(f"     Total companies: {gaps['total']}")
            print(f"     Missing salary: {gaps['salary']} ({gaps['salary']/gaps['total']*100:.1f}%)")
            print(f"     Missing bonus: {gaps['bonus']} ({gaps['bonus']/gaps['total']*100:.1f}%)")
            print(f"     Missing frequency: {gaps['frequency']} ({gaps['frequency']/gaps['total']*100:.1f}%)")

        # Research suggestions
        print(f"\n💡 RESEARCH SUGGESTIONS:")
        critical_companies = [p for p in research_priorities if p['priority_score'] >= 30]

        if critical_companies:
            print(f"\n🔴 CRITICAL (Score ≥30): {len(critical_companies)} companies")
            for company in critical_companies[:5]:
                print(f"   • {company['company']} ({company['stock_code']}): {', '.join(company['reasons'])}")

        medium_companies = [p for p in research_priorities if 15 <= p['priority_score'] < 30]
        if medium_companies:
            print(f"\n🟡 MEDIUM PRIORITY (Score 15-29): {len(medium_companies)} companies")
            for company in medium_companies[:3]:
                print(f"   • {company['company']} ({company['stock_code']}): {', '.join(company['reasons'])}")

        low_companies = [p for p in research_priorities if p['priority_score'] < 15]
        if low_companies:
            print(f"\n🟢 LOW PRIORITY (Score <15): {len(low_companies)} companies")

        return research_priorities

    def generate_research_queries(self, research_priorities):
        """Generate specific search queries for top priority companies"""
        print(f"\n🔍 SUGGESTED RESEARCH QUERIES:")
        print("="*60)

        top_priorities = research_priorities[:5]

        for i, company in enumerate(top_priorities, 1):
            print(f"\n{i}. {company['company']} ({company['stock_code']})")
            print(f"   Priority Score: {company['priority_score']}")

            # Generate Japanese search queries
            company_ja = company['company']  # We'd need to add Japanese names to our data

            if company['missing_data']['salary']:
                print(f"   Salary Research:")
                print(f"     • \"{company['company']} 平均年収 2024年\"")
                print(f"     • \"{company['company']} salary average annual income\"")
                print(f"     • \"{company['company']} 有価証券報告書 平均年収\"")

            if company['missing_data']['bonus_months']:
                print(f"   Bonus Research:")
                print(f"     • \"{company['company']} 賞与 ボーナス 何ヶ月分 2024\"")
                print(f"     • \"{company['company']} bonus months calculation method\"")
                print(f"     • \"{company['company']} 年間賞与 支給\"")

            if company['missing_data']['frequency']:
                print(f"   Frequency Research:")
                print(f"     • \"{company['company']} 賞与 支給回数 年何回\"")
                print(f"     • \"{company['company']} bonus payment frequency\"")

            print(f"   Sources to check:")
            print(f"     • Company investor relations page")
            print(f"     • Securities reports (有価証券報告書)")
            print(f"     • Employee review sites (OpenWork, 転職会議)")
            print(f"     • Industry salary surveys")


def main():
    analyzer = DataGapAnalyzer()
    research_priorities = analyzer.generate_research_report()
    analyzer.generate_research_queries(research_priorities)

    return research_priorities


if __name__ == "__main__":
    main()