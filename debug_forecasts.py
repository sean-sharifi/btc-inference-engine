
import sys
import os
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from btc_engine.database.client import db_client
    
    print("--- Database Diagnostics ---")
    
    # Check model states
    states_df = db_client.query_to_dataframe("SELECT count(*) as count, max(timestamp) as latest FROM model_states")
    print("\nModel States:")
    print(states_df)
    
    # Check forecasts
    forecasts_df = db_client.query_to_dataframe("SELECT * FROM forecasts ORDER BY forecast_timestamp DESC LIMIT 5")
    print("\nForecasts (Top 5):")
    if forecasts_df.empty:
        print("NO FORECASTS FOUND")
    else:
        print(forecasts_df[['forecast_timestamp', 'target_timestamp', 'horizon', 'quantile_50']])
        
    # Check distinct horizons
    horizons_df = db_client.query_to_dataframe("SELECT DISTINCT horizon FROM forecasts")
    print("\nDistinct Horizons:")
    print(horizons_df)

except Exception as e:
    print(f"Error: {e}")
