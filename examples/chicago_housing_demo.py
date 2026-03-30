"""Small demo for pandas_ai using a Chicago housing style dataset."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pandas_ai import ask_ai, setup_ai
from pandas_ai.interactive import enable_ai_result_displayhook


def build_demo_dataframe():
    records = [
        {"community_area": "Lincoln Park", "zip_code": "60614", "housing_type": "Condo", "bedrooms": 2, "bathrooms": 2.0, "sqft": 1250, "year_built": 2008, "list_price": 525000, "days_on_market": 18},
        {"community_area": "Lincoln Park", "zip_code": "60614", "housing_type": "Single Family", "bedrooms": 4, "bathrooms": 3.5, "sqft": 2850, "year_built": 1998, "list_price": 1295000, "days_on_market": 34},
        {"community_area": "Lakeview", "zip_code": "60657", "housing_type": "Condo", "bedrooms": 1, "bathrooms": 1.0, "sqft": 780, "year_built": 2005, "list_price": 289000, "days_on_market": 11},
        {"community_area": "Lakeview", "zip_code": "60657", "housing_type": "Townhome", "bedrooms": 3, "bathrooms": 2.5, "sqft": 1880, "year_built": 2014, "list_price": 699000, "days_on_market": 27},
        {"community_area": "Near North Side", "zip_code": "60610", "housing_type": "Condo", "bedrooms": 2, "bathrooms": 2.0, "sqft": 1325, "year_built": 2016, "list_price": 615000, "days_on_market": 22},
        {"community_area": "Near North Side", "zip_code": "60610", "housing_type": "Condo", "bedrooms": 3, "bathrooms": 2.5, "sqft": 1760, "year_built": 2019, "list_price": 879000, "days_on_market": 40},
        {"community_area": "Hyde Park", "zip_code": "60615", "housing_type": "Condo", "bedrooms": 2, "bathrooms": 2.0, "sqft": 1180, "year_built": 1978, "list_price": 319000, "days_on_market": 15},
        {"community_area": "Hyde Park", "zip_code": "60615", "housing_type": "Single Family", "bedrooms": 5, "bathrooms": 3.0, "sqft": 3120, "year_built": 1915, "list_price": 845000, "days_on_market": 51},
        {"community_area": "Logan Square", "zip_code": "60647", "housing_type": "Two Flat", "bedrooms": 4, "bathrooms": 3.0, "sqft": 2410, "year_built": 1922, "list_price": 735000, "days_on_market": 29},
        {"community_area": "Logan Square", "zip_code": "60647", "housing_type": "Condo", "bedrooms": 2, "bathrooms": 2.0, "sqft": 1090, "year_built": 2001, "list_price": 399000, "days_on_market": 13},
    ]
    return pd.DataFrame.from_records(records)

 

if __name__ == "__main__":
    df = build_demo_dataframe()

    use_stream = True
    enable_ai_result_displayhook()
    if use_stream:
        
        setup_ai(   backend="lmstudio",
                    base_url="http://192.168.40.1:1234/v1",
                    model="google/gemma-3-4b",
                    stream=True, 
                    stream_output=True,
                    stream_parse_code=True,
                    stream_delay=0.1, 
                    color="\033[32m",
                    )
    else:        
        setup_ai(   stream=False,
                    color="\033[32m",)

    print("""try -> ask_ai("average list_price by community_area" )""")
  
