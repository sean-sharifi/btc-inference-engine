"""Onchain mechanical indices: supply elasticity, forced flow, liquidity impulse"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from btc_engine.database.client import db_client
from btc_engine.utils.config_loader import load_yaml_config
from btc_engine.utils.logging_config import logger
from btc_engine.features.utils import calculate_z_score, exponential_smoothing, detect_spikes


class OnchainIndicesCalculator:
    """Calculate mechanical onchain indices from Glassnode data"""
    
    def __init__(self):
        """Initialize onchain indices calculator"""
        config = load_yaml_config("features")
        self.config = config["onchain_indices"]
        
        self.lookback_days = self.config["lookback_days"]
        self.smoothing_window = self.config["smoothing_window"]
        self.zscore_window = self.config["zscore_window"]
        self.min_data_points = self.config["min_data_points"]
        
        self.elasticity_config = self.config["elasticity"]
        self.forced_flow_config = self.config["forced_flow"]
        self.liquidity_config = self.config["liquidity_impulse"]
    
    def fetch_glassnode_metrics(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch Glassnode metrics from database
        
        Args:
            since: Start date
            until: End date
            
        Returns:
            DataFrame with metrics
        """
        if until is None:
            until = datetime.now()
        if since is None:
            since = until - timedelta(days=self.lookback_days * 2)  # Extra buffer for rolling calculations
        
        query = """
            SELECT 
                timestamp,
                metric_name,
                value
            FROM raw_glassnode_metrics
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        
        df = db_client.query_to_dataframe(query, (since, until))
        
        if len(df) == 0:
            logger.warning(f"No Glassnode data found between {since} and {until}")
        
        return df
    
    def pivot_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot metrics DataFrame to wide format
        
        Args:
            df: Long-format DataFrame
            
        Returns:
            Wide-format DataFrame with one column per metric
        """
        pivoted = df.pivot(index='timestamp', columns='metric_name', values='value')
        pivoted = pivoted.reset_index()
        
        return pivoted
    
    def calculate_supply_elasticity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Supply Elasticity Index
        
        Combines:
        - LTH supply (low elasticity if rising - holders not selling)
        - Exchange reserve (high elasticity if rising - more supply available)
        - Dormancy (low elasticity if high - old coins staying dormant)
        
        Args:
            df: DataFrame with required metrics
            
        Returns:
            Supply elasticity index series
        """
        weights = self.elasticity_config["weights"]
        
        # LTH supply: inverse z-score (rising LTH → lower elasticity)
        if 'supply_lth' in df.columns:
            lth_zscore = calculate_z_score(df['supply_lth'], window=self.zscore_window)
            lth_component = -lth_zscore * weights["lth_supply"]
        else:
            lth_component = 0
        
        # Exchange reserve: positive z-score (rising reserves → higher elasticity)
        if 'exchange_reserve' in df.columns:
            reserve_zscore = calculate_z_score(df['exchange_reserve'], window=self.zscore_window)
            reserve_component = reserve_zscore * weights["exchange_reserve"]
        else:
            reserve_component = 0
        
        # Dormancy: inverse z-score (high dormancy → lower elasticity)
        if 'dormancy' in df.columns:
            dormancy_zscore = calculate_z_score(df['dormancy'], window=self.zscore_window)
            dormancy_component = -dormancy_zscore * weights["dormancy"]
        else:
            dormancy_component = 0
        
        # Combine
        elasticity = lth_component + reserve_component + dormancy_component
        
        # Smooth
        elasticity = exponential_smoothing(elasticity, alpha=0.3)
        
        return elasticity
    
    def calculate_forced_flow(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Forced Flow Index
        
        Detects non-discretionary selling pressure:
        - Exchange inflow spikes
        - Young coin movement (STH selling)
        - Netflow acceleration
        
        Args:
            df: DataFrame with required metrics
            
        Returns:
            Forced flow index series
        """
        weights = self.forced_flow_config["weights"]
        spike_threshold = self.forced_flow_config["spike_threshold"]
        
        components = []
        
        # Exchange inflow spikes
        if 'exchange_flows_inflow' in df.columns:
            inflow_spikes = detect_spikes(
                df['exchange_flows_inflow'],
                threshold=spike_threshold,
                window=self.zscore_window
            ).astype(float)
            components.append(inflow_spikes * weights["exchange_inflow_spike"])
        
        # Young coin movement (STH supply decreases → selling)
        if 'supply_sth' in df.columns:
            sth_change = df['supply_sth'].pct_change()
            sth_zscore = calculate_z_score(sth_change, window=self.zscore_window)
            # Negative change (selling) → positive signal
            components.append(-sth_zscore * weights["young_coin_movement"])
        
        # Netflow acceleration (increasing outflows from exchanges → potential selling)
        if 'exchange_netflow' in df.columns:
            netflow_accel = df['exchange_netflow'].diff()
            netflow_zscore = calculate_z_score(netflow_accel, window=self.zscore_window)
            components.append(netflow_zscore * weights["netflow_acceleration"])
        
        # Combine
        if len(components) > 0:
            forced_flow = sum(components)
            forced_flow = exponential_smoothing(forced_flow, alpha=0.3)
        else:
            forced_flow = pd.Series(0, index=df.index)
        
        return forced_flow
    
    def calculate_liquidity_impulse(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Liquidity Impulse Index
        
        Combines:
        - Stablecoin supply changes
        - Stablecoin exchange inflows
        
        Args:
            df: DataFrame with required metrics
            
        Returns:
            Liquidity impulse index series
        """
        weights = self.liquidity_config["weights"]
        alpha = self.liquidity_config["smoothing_alpha"]
        
        components = []
        
        # Stablecoin supply change
        if 'stablecoin_supply_ratio' in df.columns:
            stablecoin_change = df['stablecoin_supply_ratio'].pct_change()
            stablecoin_zscore = calculate_z_score(stablecoin_change, window=self.zscore_window)
            components.append(stablecoin_zscore * weights["stablecoin_supply_change"])
        
        # Stablecoin exchange netflow
        if 'stablecoin_exchange_netflow' in df.columns:
            netflow_zscore = calculate_z_score(
                df['stablecoin_exchange_netflow'],
                window=self.zscore_window
            )
            components.append(netflow_zscore * weights["stablecoin_exchange_inflow"])
        
        # Combine and smooth
        if len(components) > 0:
            liquidity = sum(components)
            liquidity = exponential_smoothing(liquidity, alpha=alpha)
        else:
            liquidity = pd.Series(0, index=df.index)
        
        return liquidity
    
    def calculate_all_indices(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Calculate all onchain indices
        
        Args:
            since: Start date
            until: End date
            
        Returns:
            DataFrame with all indices
        """
        logger.info(f"Calculating onchain indices from {since} to {until}")
        
        # Fetch raw metrics
        df = self.fetch_glassnode_metrics(since, until)
        
        if len(df) < self.min_data_points:
            logger.warning(f"Insufficient data for onchain indices ({len(df)} points)")
            return pd.DataFrame()
        
        # Pivot to wide format
        df_wide = self.pivot_metrics(df)
        
        # Calculate indices
        df_wide['supply_elasticity'] = self.calculate_supply_elasticity(df_wide)
        df_wide['forced_flow_index'] = self.calculate_forced_flow(df_wide)
        df_wide['liquidity_impulse'] = self.calculate_liquidity_impulse(df_wide)
        
        # Calculate z-scores for storage
        df_wide['elasticity_zscore'] = calculate_z_score(df_wide['supply_elasticity'], window=self.zscore_window)
        df_wide['forced_flow_zscore'] = calculate_z_score(df_wide['forced_flow_index'], window=self.zscore_window)
        df_wide['liquidity_zscore'] = calculate_z_score(df_wide['liquidity_impulse'], window=self.zscore_window)
        
        # Select output columns
        output = df_wide[[
            'timestamp',
            'supply_elasticity',
            'forced_flow_index',
            'liquidity_impulse',
            'elasticity_zscore',
            'forced_flow_zscore',
            'liquidity_zscore'
        ]].copy()
        
        # Remove rows with all NaN
        output = output.dropna(subset=['supply_elasticity', 'forced_flow_index', 'liquidity_impulse'], how='all')
        
        logger.info(f"Calculated {len(output)} onchain index records")
        
        return output


def calculate_and_store_onchain_indices(
    since: Optional[datetime] = None,
    until: Optional[datetime] = None
) -> pd.DataFrame:
    """Calculate and store onchain indices
    
    Args:
        since: Start date
        until: End date
        
    Returns:
        DataFrame with indices
    """
    calculator = OnchainIndicesCalculator()
    indices = calculator.calculate_all_indices(since, until)
    
    if len(indices) > 0:
        db_client.insert_dataframe("features_onchain_indices", indices, if_exists="append")
        logger.info(f"Stored {len(indices)} onchain index records")
    
    return indices
