
import duckdb
import pandas as pd
import numpy as np

DB_PATH = "data/btc_engine.duckdb"

def debug_full_pipeline():
    print(f"Connecting to {DB_PATH}...")
    conn = duckdb.connect(DB_PATH)
    
    # 1. Fetch Model States with ALL columns used
    print("Fetching model_states with feature columns...")
    state_query = """
        SELECT timestamp, 
               regime_1_prob, regime_2_prob, regime_3_prob,
               state_risk_appetite, state_leverage_stress, 
               state_dealer_stabilization, state_tail_demand
        FROM model_states
        ORDER BY timestamp
    """
    df = conn.execute(state_query).df()
    print(f"Model States loaded: {len(df)} rows")
    
    # Check for NaNs in states
    state_nans = df.isna().sum()
    print("NaNs in state columns:")
    print(state_nans[state_nans > 0])

    # 2. Fetch Prices
    print("\nFetching prices...")
    price_query = """
        SELECT timestamp, AVG(underlying_price) as btc_price
        FROM raw_deribit_ticker_snapshots
        GROUP BY timestamp
        ORDER BY timestamp
    """
    df_price = conn.execute(price_query).df()
    print(f"Prices loaded: {len(df_price)} rows")
    
    # 3. Merge
    print("\nMerging...")
    df = df.merge(df_price, on='timestamp', how='left')
    print(f"Merged size: {len(df)}")
    
    print("Filling Price NaNs (ffill)...")
    df['btc_price'] = df['btc_price'].fillna(method='ffill')
    
    print("Calculating Returns...")
    df['returns'] = df['btc_price'].pct_change()
    
    # 4. Feature Simulation (Horizon = 24h -> 1 period)
    print("\nSimulating 24h Feature Creation...")
    horizon_periods = 1
    
    # Lags
    for lag in [1, 2, 3, 5, 10]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
    # Target (rolling sum shift)
    df[f'target_{horizon_periods}'] = df['returns'].rolling(horizon_periods).sum().shift(-horizon_periods)
    
    # 5. Drop NaNs
    before = len(df)
    df_clean = df.dropna()
    after = len(df_clean)
    
    print(f"\n--- DROP REPORT ---")
    print(f"Rows before dropna: {before}")
    print(f"Rows after dropna:  {after}")
    print(f"Rows dropped:       {before - after}")
    
    if after < 50:
        print("\nCRITICAL: Final row count is LESS THAN 50! Training will be skipped.")
        
        # Analyze why
        print("\nNull count analysis:")
        print(df.isna().sum())
    else:
        print(f"\nSUCCESS: {after} rows available for training. This should pass the >50 check.")

    conn.close()

if __name__ == "__main__":
    debug_full_pipeline()
