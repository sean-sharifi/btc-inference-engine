
import duckdb
import pandas as pd
import numpy as np

DB_PATH = "data/btc_engine.duckdb"

def check_returns():
    print(f"Connecting to {DB_PATH}...")
    conn = duckdb.connect(DB_PATH)
    
    # 1. Fetch Prices aggregated by timestamp (matches forecasting.py)
    query = """
        SELECT timestamp, AVG(underlying_price) as btc_price
        FROM raw_deribit_ticker_snapshots
        GROUP BY timestamp
        ORDER BY timestamp
    """
    df = conn.execute(query).df()
    print(f"Loaded {len(df)} price points.")
    
    # 2. Calculate Returns
    df['returns'] = df['btc_price'].pct_change()
    df = df.dropna()
    
    print("\nReturns Statistics:")
    print(df['returns'].describe())
    
    print("\nSample Returns (head 10):")
    print(df['returns'].head(10))
    
    # Check for exact zeros
    zeros = (df['returns'] == 0).sum()
    print(f"\nExact zeros: {zeros} / {len(df)}")
    
    # Check magnitude
    small = (df['returns'].abs() < 1e-6).sum()
    print(f"Valid but < 1e-6: {small}")
    
    conn.close()

if __name__ == "__main__":
    check_returns()
