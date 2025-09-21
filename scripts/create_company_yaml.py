
import os
import yaml
import argparse

def create_company_yaml(stock_code, sector):
    """
    Generates a YAML template for a given company.
    """
    company_data = {
        "company_profile": {
            "company_name": "",
            "stock_code": stock_code,
            "sector": sector,
            "market_cap": "",
            "employees": ""
        },
        "bonus_system": {
            "classification": "",
            "confidence_level": "",
            "reliability_score": 0
        },
        "evidence": [],
        "calculation_method": "",
        "notes": ""
    }

    # Create sector directory if it doesn't exist
    sector_dir = os.path.join("companies", sector)
    if not os.path.exists(sector_dir):
        os.makedirs(sector_dir)

    # Write the YAML file
    file_path = os.path.join(sector_dir, f"{stock_code}_.yaml")
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(company_data, f, allow_unicode=True, sort_keys=False)

    print(f"Created YAML template for stock code {stock_code} in {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a YAML template for a company's bonus survey.")
    parser.add_argument("--stock-code", required=True, help="Stock code of the company.")
    parser.add_argument("--sector", required=True, help="Sector of the company.")
    args = parser.parse_args()

    create_company_yaml(args.stock_code, args.sector)
