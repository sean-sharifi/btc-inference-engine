
import duckdb
import pandas as pd
import numpy as np

DB_PATH = "data/btc_engine.duckdb"

def debug_data():
    print(f"Connecting to {DB_PATH}...")
    conn = duckdb.connect(DB_PATH)
    
    # 1. Fetch Model States
    print("Fetching model_states...")
    df_states = conn.execute("SELECT timestamp FROM model_states ORDER BY timestamp").df()
    print(f"Model States: {len(df_states)} rows")
    if len(df_states) > 0:
        print(f"Sample state timestamps: {df_states['timestamp'].head(3).tolist()}")
        print(f"First timestamp type: {type(df_states['timestamp'].iloc[0])}")

    # 2. Fetch Prices
    print("\nFetching prices (raw_deribit_ticker_snapshots)...")
    price_query = """
        SELECT timestamp, AVG(underlying_price) as btc_price
        FROM raw_deribit_ticker_snapshots
        GROUP BY timestamp
        ORDER BY timestamp
    """
    df_prices = conn.execute(price_query).df()
    print(f"Prices: {len(df_prices)} rows")
    if len(df_prices) > 0:
        print(f"Sample price timestamps: {df_prices['timestamp'].head(3).tolist()}")

    # 3. Try Merge
    print("\nAttempting Merge...")
    if len(df_states) > 0 and len(df_prices) > 0:
        merged = df_states.merge(df_prices, on='timestamp', how='left')
        print(f"Merged size: {len(merged)} rows")
        
        # Check NaNs
        nans = merged['btc_price'].isna().sum()
        print(f"Rows with NaN price after merge: {nans}")
        
        if nans > 0:
            print("WARNING: Price data not matching state timestamps!")
            print("First 5 missing match timestamps:")
            print(merged[merged['btc_price'].isna()]['timestamp'].head(5))
            
            # Check precision
            t_state = df_states['timestamp'].iloc[0]
            # Find closest in price
            # Assuming sorted
            print("\nPrecision Check:")
            print(f"State TS: {t_state} ({t_state.value})")
            # matches?
            matches = df_prices[df_prices['timestamp'] == t_state]
            print(f"Exact matches in price df: {len(matches)}")
            
    else:
        print("Cannot merge, one dataframe is empty.")

    conn.close()

if __name__ == "__main__":
    debug_data()
