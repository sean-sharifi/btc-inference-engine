
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Mock logger to print to stdout
import logging
logging.basicConfig(level=logging.INFO)

try:
    from btc_engine.models.forecasting import DistributionalForecaster
    from btc_engine.database.client import db_client
    
    print("--- Forecasting Diagnostics ---")
    
    forecaster = DistributionalForecaster()
    print(f"Horizons config: {forecaster.horizons}")
    
    # 1. Fetch Training Data
    print("\n1. Fetching training data...")
    try:
        df = forecaster.fetch_training_data()
        print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")
        print(f"Head:\n{df.head(2)}")
        print(f"Tail:\n{df.tail(2)}")
    except Exception as e:
        print(f"FETCH FAILED: {e}")
        sys.exit(1)
        
    # 2. Check Data Sufficiency for Horizons
    print("\n2. Checking horizons...")
    for horizon, periods in forecaster.horizons.items():
        print(f"Checking horizon '{horizon}' (periods={periods})...")
        X, y = forecaster.create_features(df.copy(), periods)
        print(f"  Features shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        
        if len(X) < 50:
            print(f"  WARNING: Insufficient data (<50 samples). This horizon will specify SKIPPED.")
        else:
            print(f"  Data sufficient. Training...")
            try:
                forecaster.train_quantile_models(X, y, horizon)
                print(f"  Training SUCCESS.")
            except Exception as e:
                print(f"  Training FAILED: {e}")

    forecaster.is_fitted = True # Force true to test generation
    
    # 3. Generate Forecast
    print("\n3. Testing Generation...")
    latest_ts = db_client.get_latest_timestamp("model_states")
    if latest_ts:
        print(f"Latest state timestamp: {latest_ts}")
        try:
            forecasts = forecaster.generate_forecasts(latest_ts, "test_version")
            print(f"Generated {len(forecasts)} forecasts.")
            if len(forecasts) > 0:
                print(forecasts[0])
            else:
                print("Generated 0 forecasts.")
                # Check why
                print("Debugging generation logic...")
                # ... (Additional logic could go here)
        except Exception as e:
            print(f"Generation FAILED: {e}")
    else:
        print("No model states found, cannot test generation.")
        
except ImportError as e:
    print(f"ImportError: {e}")
    print("Ensure you are running from the project root and requirements are installed.")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
