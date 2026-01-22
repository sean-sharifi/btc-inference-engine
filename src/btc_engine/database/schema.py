"""DuckDB schema definitions and setup"""

from typing import List
import duckdb
from pathlib import Path

from btc_engine.utils.logging_config import logger


# Schema definitions
SCHEMA_DEFINITIONS = {
    # Raw data tables
    "raw_deribit_instruments": """
        CREATE TABLE IF NOT EXISTS raw_deribit_instruments (
            timestamp TIMESTAMP,
            instrument_name VARCHAR,
            strike DECIMAL(18, 2),
            expiration_timestamp TIMESTAMP,
            option_type VARCHAR,  -- 'call' or 'put'
            creation_timestamp TIMESTAMP,
            is_active BOOLEAN,
            min_trade_amount DECIMAL(18, 8),
            tick_size DECIMAL(18, 8),
            PRIMARY KEY (timestamp, instrument_name)
        )
    """,
    
    "raw_deribit_ticker_snapshots": """
        CREATE TABLE IF NOT EXISTS raw_deribit_ticker_snapshots (
            timestamp TIMESTAMP,
            instrument_name VARCHAR,
            mark_price DECIMAL(18, 8),
            mark_iv DECIMAL(10, 6),
            bid_price DECIMAL(18, 8),
            bid_iv DECIMAL(10, 6),
            ask_price DECIMAL(18, 8),
            ask_iv DECIMAL(10, 6),
            underlying_price DECIMAL(18, 2),
            open_interest DECIMAL(18, 8),
            volume_24h DECIMAL(18, 8),
            last_price DECIMAL(18, 8),
            greeks_delta DECIMAL(10, 6),
            greeks_gamma DECIMAL(10, 8),
            greeks_vega DECIMAL(10, 6),
            greeks_theta DECIMAL(10, 6),
            PRIMARY KEY (timestamp, instrument_name)
        )
    """,
    
    "raw_deribit_funding": """
        CREATE TABLE IF NOT EXISTS raw_deribit_funding (
            timestamp TIMESTAMP,
            instrument_name VARCHAR,
            funding_rate DECIMAL(10, 8),
            funding_8h DECIMAL(10, 8),
            index_price DECIMAL(18, 2),
            mark_price DECIMAL(18, 2),
            open_interest DECIMAL(18, 8),
            PRIMARY KEY (timestamp, instrument_name)
        )
    """,
    
    "raw_glassnode_metrics": """
        CREATE TABLE IF NOT EXISTS raw_glassnode_metrics (
            timestamp TIMESTAMP,
            metric_name VARCHAR,
            value DECIMAL(28, 8),
            resolution VARCHAR,
            PRIMARY KEY (timestamp, metric_name)
        )
    """,
    
    # Feature tables
    "features_options_surface": """
        CREATE TABLE IF NOT EXISTS features_options_surface (
            timestamp TIMESTAMP PRIMARY KEY,
            level DECIMAL(10, 6),
            skew DECIMAL(10, 6),
            curvature DECIMAL(10, 6),
            term_structure DECIMAL(10, 6),
            wing_asymmetry DECIMAL(10, 6),
            surface_shock DECIMAL(10, 6),
            pca_factor_1 DECIMAL(10, 6),
            pca_factor_2 DECIMAL(10, 6),
            pca_factor_3 DECIMAL(10, 6),
            pca_factor_4 DECIMAL(10, 6),
            pca_factor_5 DECIMAL(10, 6)
        )
    """,
    
    "features_risk_neutral": """
        CREATE TABLE IF NOT EXISTS features_risk_neutral (
            timestamp TIMESTAMP,
            expiry_days INTEGER,
            left_tail_mass DECIMAL(10, 6),
            right_tail_mass DECIMAL(10, 6),
            jump_risk_proxy DECIMAL(10, 6),
            skew_elasticity DECIMAL(10, 6),
            PRIMARY KEY (timestamp, expiry_days)
        )
    """,
    
    "features_hedging_pressure": """
        CREATE TABLE IF NOT EXISTS features_hedging_pressure (
            timestamp TIMESTAMP PRIMARY KEY,
            stabilization_index DECIMAL(10, 6),
            acceleration_index DECIMAL(10, 6),
            max_gamma_strike DECIMAL(18, 2),
            zero_gamma_level DECIMAL(18, 2),
            total_gamma_exposure DECIMAL(18, 8),
            gamma_skew DECIMAL(10, 6)
        )
    """,
    
    "features_hedging_pressure_grid": """
        CREATE TABLE IF NOT EXISTS features_hedging_pressure_grid (
            timestamp TIMESTAMP,
            spot_level DECIMAL(18, 2),
            gamma_exposure DECIMAL(18, 8),
            hedge_flow_1pct DECIMAL(18, 8),
            PRIMARY KEY (timestamp, spot_level)
        )
    """,
    
    "features_onchain_indices": """
        CREATE TABLE IF NOT EXISTS features_onchain_indices (
            timestamp TIMESTAMP PRIMARY KEY,
            supply_elasticity DECIMAL(10, 6),
            forced_flow_index DECIMAL(10, 6),
            liquidity_impulse DECIMAL(10, 6),
            elasticity_zscore DECIMAL(10, 6),
            forced_flow_zscore DECIMAL(10, 6),
            liquidity_zscore DECIMAL(10, 6)
        )
    """,
    
    "features_divergence": """
        CREATE TABLE IF NOT EXISTS features_divergence (
            timestamp TIMESTAMP PRIMARY KEY,
            divergence_score DECIMAL(10, 6),
            classification VARCHAR,
            options_tail_signal DECIMAL(10, 6),
            onchain_pressure_signal DECIMAL(10, 6),
            top_signal_1 VARCHAR,
            top_signal_1_value DECIMAL(10, 6),
            top_signal_2 VARCHAR,
            top_signal_2_value DECIMAL(10, 6),
            top_signal_3 VARCHAR,
            top_signal_3_value DECIMAL(10, 6),
            top_signal_4 VARCHAR,
            top_signal_4_value DECIMAL(10, 6),
            top_signal_5 VARCHAR,
            top_signal_5_value DECIMAL(10, 6)
        )
    """,
    
    # Model tables
    "model_states": """
        CREATE TABLE IF NOT EXISTS model_states (
            timestamp TIMESTAMP PRIMARY KEY,
            regime_1_prob DECIMAL(10, 6),
            regime_2_prob DECIMAL(10, 6),
            regime_3_prob DECIMAL(10, 6),
            state_risk_appetite DECIMAL(10, 6),
            state_leverage_stress DECIMAL(10, 6),
            state_dealer_stabilization DECIMAL(10, 6),
            state_tail_demand DECIMAL(10, 6),
            state_inventory_imbalance DECIMAL(10, 6),
            state_liquidity_regime DECIMAL(10, 6),
            model_version VARCHAR
        )
    """,
    
    "forecasts": """
        CREATE TABLE IF NOT EXISTS forecasts (
            forecast_timestamp TIMESTAMP,
            target_timestamp TIMESTAMP,
            horizon VARCHAR,
            quantile_05 DECIMAL(18, 8),
            quantile_25 DECIMAL(18, 8),
            quantile_50 DECIMAL(18, 8),
            quantile_75 DECIMAL(18, 8),
            quantile_95 DECIMAL(18, 8),
            expected_shortfall_5pct DECIMAL(18, 8),
            left_tail_mass DECIMAL(10, 6),
            right_tail_mass DECIMAL(10, 6),
            vol_of_vol DECIMAL(10, 6),
            model_version VARCHAR,
            PRIMARY KEY (forecast_timestamp, target_timestamp, horizon)
        )
    """,
    
    "evaluations": """
        CREATE TABLE IF NOT EXISTS evaluations (
            evaluation_timestamp TIMESTAMP,
            backtest_window_start TIMESTAMP,
            backtest_window_end TIMESTAMP,
            metric_name VARCHAR,
            metric_value DECIMAL(18, 8),
            horizon VARCHAR,
            model_version VARCHAR,
            PRIMARY KEY (evaluation_timestamp, metric_name, horizon)
        )
    """,
    
    # Pipeline metadata
    "pipeline_checkpoints": """
        CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
            task_name VARCHAR PRIMARY KEY,
            last_run_timestamp TIMESTAMP,
            last_success_timestamp TIMESTAMP,
            status VARCHAR,
            records_processed INTEGER,
            error_message VARCHAR
        )
    """,
}


def create_tables(db_path: str) -> None:
    """Create all database tables
    
    Args:
        db_path: Path to DuckDB database file
    """
    logger.info(f"Creating database tables at {db_path}")
    
    # Ensure parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = duckdb.connect(db_path)
    
    try:
        for table_name, create_sql in SCHEMA_DEFINITIONS.items():
            logger.debug(f"Creating table: {table_name}")
            conn.execute(create_sql)
        
        conn.commit()
        logger.info(f"Successfully created {len(SCHEMA_DEFINITIONS)} tables")
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise
    finally:
        conn.close()


def get_table_info(db_path: str) -> dict:
    """Get information about all tables in database
    
    Args:
        db_path: Path to DuckDB database file
        
    Returns:
        Dictionary with table names and row counts
    """
    conn = duckdb.connect(db_path, read_only=True)
    
    try:
        tables = {}
        for table_name in SCHEMA_DEFINITIONS.keys():
            try:
                result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                tables[table_name] = result[0] if result else 0
            except Exception as e:
                logger.debug(f"Could not query {table_name}: {e}")
                tables[table_name] = -1
        
        return tables
        
    finally:
        conn.close()
