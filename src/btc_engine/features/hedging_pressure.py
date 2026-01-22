"""Hedging pressure field calculation (gamma/vanna/charm aggregation)"""

from typing import Dict, Optional, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm

from btc_engine.database.client import db_client
from btc_engine.utils.config_loader import load_yaml_config
from btc_engine.utils.logging_config import logger


class HedgingPressureCalculator:
    """Calculate dealer hedging pressure from options Greeks and OI"""
    
    def __init__(self):
        """Initialize hedging pressure calculator"""
        config = load_yaml_config("features")
        self.config = config["hedging_pressure"]
        
        self.spot_grid_points = self.config["spot_grid_points"]
        self.spot_range_pct = self.config["spot_range_pct"]
        self.gamma_scaling = float(self.config["gamma_scaling"])
        self.aggregation_method = self.config["aggregation_method"]
        self.min_oi_threshold = self.config["min_oi_threshold"]
    
    def fetch_ticker_with_greeks(self, timestamp: datetime) -> pd.DataFrame:
        """Fetch ticker data with Greeks"""
        query = """
            SELECT 
                timestamp,
                instrument_name,
                underlying_price,
                open_interest,
                greeks_delta,
                greeks_gamma,
                greeks_vega,
                greeks_theta
            FROM raw_deribit_ticker_snapshots
            WHERE timestamp = ?
            AND open_interest > ?
            AND greeks_gamma IS NOT NULL
        """
        return db_client.query_to_dataframe(query, (timestamp, self.min_oi_threshold))
    
    def build_pressure_grid(
        self,
        df: pd.DataFrame,
        spot: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build hedging pressure grid across spot levels
        
        Args:
            df: DataFrame with options data and Greeks
            spot: Current spot price
            
        Returns:
            Tuple of (spot_levels, gamma_exposure)
        """
        # Create spot grid
        spot_min = spot * (1 + self.spot_range_pct[0] / 100.0)
        spot_max = spot * (1 + self.spot_range_pct[1] / 100.0)
        spot_levels = np.linspace(spot_min, spot_max, self.spot_grid_points)
        
        # Initialize exposure arrays
        gamma_exposure = np.zeros(self.spot_grid_points)
        
        # Aggregate gamma exposure
        for _, row in df.iterrows():
            # Parse instrument name for strike
            try:
                parts = row['instrument_name'].split('-')
                if len(parts) == 4:
                    strike = float(parts[2])
                    opt_type = parts[3]  # 'C' or 'P'
                    
                    gamma = row['greeks_gamma']
                    oi = row['open_interest']
                    
                    # Dealers are short options, so their gamma is negative of client's gamma
                    # Client long call/put → positive gamma
                    # Dealer short call/put → negative gamma
                    dealer_gamma = -gamma * oi
                    
                    # Distribute gamma exposure around strike using Gaussian kernel
                    # Gamma peaks at strike
                    for i, spot_level in enumerate(spot_levels):
                        distance = abs(spot_level - strike) / strike
                        weight = np.exp(-0.5 * (distance / 0.1)**2)  # Gaussian with std=10%
                        gamma_exposure[i] += dealer_gamma * weight
                        
            except Exception as e:
                logger.debug(f"Failed to process {row['instrument_name']}: {e}")
                continue
        
        # Scale gamma exposure
        gamma_exposure = gamma_exposure * self.gamma_scaling
        
        return spot_levels, gamma_exposure
    
    def calculate_hedge_flow(
        self,
        spot_levels: np.ndarray,
        gamma_exposure: np.ndarray,
        spot: float
    ) -> np.ndarray:
        """Calculate expected hedge flow per 1% spot move
        
        Args:
            spot_levels: Array of spot levels
            gamma_exposure: Array of gamma exposure at each level
            spot: Current spot
            
        Returns:
            Array of hedge flow per 1% move
        """
        # Hedge flow ≈ -gamma × spot × move size
        # For 1% move: hedge_flow = -gamma × spot × 0.01
        hedge_flow = -gamma_exposure * spot_levels * 0.01
        
        return hedge_flow
    
    def calculate_pressure_indices(
        self,
        spot_levels: np.ndarray,
        gamma_exposure: np.ndarray,
        spot: float
    ) -> Dict[str, float]:
        """Calculate aggregate pressure indices
        
        Args:
            spot_levels: Spot level grid
            gamma_exposure: Gamma exposure grid
            spot: Current spot
            
        Returns:
            Dictionary with pressure metrics
        """
        # Find current spot index
        spot_idx = np.argmin(np.abs(spot_levels - spot))
        
        # Stabilization index: gamma concentration above spot (mean-reverting pressure)
        # Positive gamma above spot → dealers buy dips, sell rallies → stabilizing
        above_spot_gamma = gamma_exposure[spot_idx:]
        stabilization_index = np.mean(above_spot_gamma)
        
        # Acceleration index: negative gamma concentration (trend-amplifying pressure)
        # Negative gamma → dealers sell dips, buy rallies → destabilizing
        negative_gamma_mass = np.sum(gamma_exposure < 0)
        acceleration_index = -np.mean(gamma_exposure[gamma_exposure < 0]) if negative_gamma_mass > 0 else 0.0
        
        # Find max gamma strike (point of maximum exposure)
        max_gamma_idx = np.argmax(np.abs(gamma_exposure))
        max_gamma_strike = spot_levels[max_gamma_idx]
        
        # Find zero-crossing (gamma flip level)
        zero_crossings = np.where(np.diff(np.sign(gamma_exposure)))[0]
        zero_gamma_level = spot_levels[zero_crossings[0]] if len(zero_crossings) > 0 else spot
        
        # Total gamma exposure (absolute)
        total_gamma_exposure = np.sum(np.abs(gamma_exposure))
        
        # Gamma skew (asymmetry)
        gamma_skew = np.mean(gamma_exposure[spot_idx:]) - np.mean(gamma_exposure[:spot_idx])
        
        return {
            'stabilization_index': stabilization_index,
            'acceleration_index': acceleration_index,
            'max_gamma_strike': max_gamma_strike,
            'zero_gamma_level': zero_gamma_level,
            'total_gamma_exposure': total_gamma_exposure,
            'gamma_skew': gamma_skew,
        }
    
    def calculate_hedging_pressure(self, timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """Calculate hedging pressure metrics
        
        Args:
            timestamp: Timestamp to analyze
            
        Returns:
            Dictionary with pressure metrics
        """
        logger.info(f"Calculating hedging pressure for {timestamp}")
        
        # Fetch data
        df = self.fetch_ticker_with_greeks(timestamp)
        
        if len(df) == 0:
            logger.warning("No options data with Greeks found")
            return {
                'timestamp': timestamp,
                'stabilization_index': np.nan,
                'acceleration_index': np.nan,
                'max_gamma_strike': np.nan,
                'zero_gamma_level': np.nan,
                'total_gamma_exposure': np.nan,
                'gamma_skew': np.nan,
            }
        
        spot = df['underlying_price'].iloc[0]
        
        # Build pressure grid
        spot_levels, gamma_exposure = self.build_pressure_grid(df, spot)
        
        # Calculate pressure indices
        indices = self.calculate_pressure_indices(spot_levels, gamma_exposure, spot)
        indices['timestamp'] = timestamp
        
        # Store pressure grid for visualization
        grid_df = pd.DataFrame({
            'timestamp': timestamp,
            'spot_level': spot_levels,
            'gamma_exposure': gamma_exposure,
            'hedge_flow_1pct': self.calculate_hedge_flow(spot_levels, gamma_exposure, spot)
        })
        
        db_client.insert_dataframe("features_hedging_pressure_grid", grid_df, if_exists="append")
        
        logger.info(f"Calculated hedging pressure: stab={indices['stabilization_index']:.4f}, accel={indices['acceleration_index']:.4f}")
        
        return indices


def calculate_and_store_hedging_pressure(timestamp: Optional[datetime] = None) -> Dict[str, float]:
    """Calculate and store hedging pressure metrics
    
    Args:
        timestamp: Timestamp to process
        
    Returns:
        Dictionary with metrics
    """
    calculator = HedgingPressureCalculator()
    metrics = calculator.calculate_hedging_pressure(timestamp)
    
    # Only store if we have valid data
    if metrics.get('timestamp') is not None:
        df = pd.DataFrame([metrics])
        db_client.insert_dataframe("features_hedging_pressure", df, if_exists="append")
        logger.info("Stored hedging pressure metrics")
    else:
        logger.warning("No hedging pressure metrics to store (no options data)")
    
    return metrics
