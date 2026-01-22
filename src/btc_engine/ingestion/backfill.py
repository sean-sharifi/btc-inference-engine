"""Historical data backfill using synthetic data generation"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
from btc_engine.database.client import db_client
from btc_engine.utils.logging_config import logger


def generate_historical_volatility_series(days: int = 180) -> pd.Series:
    """Generate realistic historical volatility series
    
    Uses GBM with mean reversion to simulate realistic vol dynamics
    
    Args:
        days: Number of days to simulate
        
    Returns:
        Series of daily volatility levels (annualized)
    """
    # Start at current typical BTC vol level
    vol_0 = 0.65  # 65% annualized
    vol_mean = 0.60  # Long-term mean
    vol_std = 0.15  # Volatility of volatility
    mean_reversion = 0.1  # Mean reversion speed
    
    vols = [vol_0]
    for _ in range(days - 1):
        shock = np.random.normal(0, vol_std / np.sqrt(252))
        mean_rev = mean_reversion * (vol_mean - vols[-1]) / 252
        new_vol = vols[-1] + mean_rev + shock
        new_vol = max(0.3, min(1.2, new_vol))  # Bound between 30% and 120%
        vols.append(new_vol)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    return pd.Series(vols, index=dates)


def scale_snapshot_by_vol(snapshot_df: pd.DataFrame, vol_ratio: float) -> pd.DataFrame:
    """Scale a Deribit snapshot by volatility ratio
    
    Args:
        snapshot_df: DataFrame with current snapshot
        vol_ratio: Ratio to scale IVs by (new_vol / current_vol)
        
    Returns:
        Scaled DataFrame
    """
    df = snapshot_df.copy()
    
    # Scale implied volatilities
    if 'iv' in df.columns:
        df['iv'] = df['iv'] * vol_ratio
    if 'mark_iv' in df.columns:
        df['mark_iv'] = df['mark_iv'] * vol_ratio
    if 'bid_iv' in df.columns:
        df['bid_iv'] = df['bid_iv'] * vol_ratio
    if 'ask_iv' in df.columns:
        df['ask_iv'] = df['ask_iv'] * vol_ratio
        
    # Scale option prices proportionally (simple approximation)
    price_scale = np.sqrt(vol_ratio)  # Approximate vega scaling
    if 'mark_price' in df.columns:
        df['mark_price'] = df['mark_price'] * price_scale
    if 'bid_price' in df.columns:
        df['bid_price'] = df['bid_price'] * price_scale
    if 'ask_price' in df.columns:
        df['ask_price'] = df['ask_price'] * price_scale
        
    # Add some noise to make it realistic
    if 'iv' in df.columns:
        noise = np.random.normal(0, 0.02, len(df))  # 2% noise
        df['iv'] = df['iv'] * (1 + noise)
        df['iv'] = df['iv'].clip(lower=0.1)  # Min 10% IV
    
    return df


def backfill_deribit_history(days: int = 180) -> dict:
    """Backfill Deribit historical data using synthetic generation
    
    Args:
        days: Number of days to backfill
        
    Returns:
        Dictionary with backfill statistics
    """
    logger.info(f"Starting Deribit historical backfill for {days} days")
    
    # Get current snapshot as template
    current_instruments = db_client.query_to_dataframe(
        "SELECT * FROM raw_deribit_instruments ORDER BY timestamp DESC LIMIT 1000"
    )
    current_tickers = db_client.query_to_dataframe(
        "SELECT * FROM raw_deribit_ticker_snapshots ORDER BY timestamp DESC LIMIT 1000"
    )
    
    if current_instruments.empty or current_tickers.empty:
        raise ValueError("No current Deribit data found. Run 'btc-engine ingest-deribit' first.")
    
    logger.info(f"Using {len(current_instruments)} instruments and {len(current_tickers)} tickers as template")
    
    # Generate historical volatility series
    vol_series = generate_historical_volatility_series(days)
    current_vol = vol_series.iloc[-1]
    
    total_instruments = 0
    total_tickers = 0
    
    # Generate daily snapshots going backwards
    for date, hist_vol in vol_series.iloc[:-1].items():  # Skip last day (current)
        vol_ratio = hist_vol / current_vol
        timestamp = pd.Timestamp(date).replace(hour=12)  # Noon UTC
        
        # Scale instruments
        hist_instruments = current_instruments.copy()
        hist_instruments['timestamp'] = timestamp
        if 'creation_timestamp' in hist_instruments.columns:
            hist_instruments['creation_timestamp'] = timestamp
        if 'expiration_timestamp' in hist_instruments.columns:
            # Keep expiration dates relative to the historical timestamp
            exp_diff = hist_instruments['expiration_timestamp'] - current_instruments['timestamp'].iloc[0]
            hist_instruments['expiration_timestamp'] = timestamp + exp_diff
        
        # Scale tickers
        hist_tickers = scale_snapshot_by_vol(current_tickers.copy(), vol_ratio)
        hist_tickers['timestamp'] = timestamp
        
        # Insert into database (using INSERT OR REPLACE for idempotency)
        db_client.insert_dataframe('raw_deribit_instruments', hist_instruments, if_exists='append')
        db_client.insert_dataframe('raw_deribit_ticker_snapshots', hist_tickers, if_exists='append')
        
        total_instruments += len(hist_instruments)
        total_tickers += len(hist_tickers)
        
        if date.day % 30 == 0:  # Progress logging every 30 days
            logger.info(f"Backfilled up to {date.strftime('%Y-%m-%d')}")
    
    logger.info(f"Backfill complete: {total_instruments} instruments, {total_tickers} tickers")
    
    return {
        'days_backfilled': days - 1,
        'instruments_created': total_instruments,
        'tickers_created': total_tickers,
        'vol_range': (vol_series.min(), vol_series.max())
    }


def backfill_glassnode_history(days: int = 180) -> dict:
    """Backfill Glassnode data if needed
    
    Glassnode ingestion already handles historical data,
    so this just ensures we have enough coverage
    
    Args:
        days: Number of days to ensure coverage for
        
    Returns:
        Dictionary with status
    """
    from btc_engine.ingestion.glassnode_ingest import run_glassnode_ingestion
    
    logger.info(f"Ensuring Glassnode coverage for {days} days")
    
    # Check existing coverage
    earliest = db_client.execute_query(
        "SELECT MIN(timestamp) FROM raw_glassnode_metrics",
        read_only=True
    )
    
    current_days = 0
    if earliest and earliest[0] and earliest[0][0]:
        earliest_date = pd.Timestamp(earliest[0][0])
        current_days = (datetime.now() - earliest_date).days
    
    if current_days >= days:
        logger.info(f"Glassnode already has {current_days} days of data")
        return {'status': 'sufficient', 'days': current_days}
    
    # Re-ingest with sufficient history
    logger.info(f"Re-ingesting Glassnode with {days} days")
    result = run_glassnode_ingestion(days=days, incremental=False)
    
    return {'status': 'backfilled', 'records': result.get('records_inserted', 0)}
