
import duckdb
import os

DB_PATH = "data/btc_engine.duckdb"

def inspect_db():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database file not found at {DB_PATH}")
        return

    print(f"Connecting to {DB_PATH}...")
    conn = duckdb.connect(DB_PATH)
    
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
    print(f"Tables found: {tables}")
    
    if 'model_states' in tables:
        count = conn.execute("SELECT COUNT(*) FROM model_states").fetchone()[0]
        print(f"Row count for 'model_states': {count}")
        if count > 0:
            latest = conn.execute("SELECT MAX(timestamp) FROM model_states").fetchone()[0]
            print(f"Latest model state timestamp: {latest}")
    else:
        print("Table 'model_states' DOES NOT EXIST.")
        
    if 'forecasts' in tables:
        count = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        print(f"Row count for 'forecasts': {count}")
        if count > 0:
            print("Sample forecasts (latest 5):")
            rows = conn.execute("SELECT * FROM forecasts ORDER BY forecast_timestamp DESC LIMIT 5").fetchall()
            for r in rows:
                print(r)
        else:
            print("Table 'forecasts' is empty.")
    else:
        print("Table 'forecasts' DOES NOT EXIST.")

    if 'features_options_surface' in tables:
        count = conn.execute("SELECT COUNT(*) FROM features_options_surface").fetchone()[0]
        print(f"Row count for 'features_options_surface': {count}")
        if count > 0:
            latest = conn.execute("SELECT MAX(timestamp) FROM features_options_surface").fetchone()[0]
            print(f"Latest features timestamp: {latest}")
    else:
        print("Table 'features_options_surface' DOES NOT EXIST.")

if __name__ == "__main__":
    inspect_db()
