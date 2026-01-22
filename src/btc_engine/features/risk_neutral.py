"""Risk-neutral distribution proxies and tail measures"""

from typing import Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm

from btc_engine.database.client import db_client
from btc_engine.utils.config_loader import load_yaml_config
from btc_engine.utils.logging_config import logger


class RiskNeutralAnalyzer:
    """Compute risk-neutral distribution proxies from IV surface"""
    
    def __init__(self):
        """Initialize risk-neutral analyzer"""
        config = load_yaml_config("features")
        self.config = config["risk_neutral"]
        
        self.density_method = self.config["density_method"]
        self.strike_spacing = self.config["strike_spacing"]
        self.tail_threshold = self.config["tail_threshold"]
        self.min_strikes = self.config["min_strikes_per_expiry"]
    
    def fetch_ticker_snapshot(self, timestamp: datetime) -> pd.DataFrame:
        """Fetch ticker snapshot"""
        query = """
            SELECT * FROM raw_deribit_ticker_snapshots
            WHERE timestamp = ? AND mark_iv IS NOT NULL
        """
        return db_client.query_to_dataframe(query, (timestamp,))
    
    def breeden_litzenberger_proxy(
        self,
        strikes: np.ndarray,
        call_prices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate risk-neutral density using Breeden-Litzenberger
        
        Args:
            strikes: Array of strikes
            call_prices: Array of call prices
            
        Returns:
            Tuple of (strikes, density values)
        """
        # Sort by strike
        sorted_idx = np.argsort(strikes)
        strikes = strikes[sorted_idx]
        call_prices = call_prices[sorted_idx]
        
        # Compute second derivative via finite differences
        # d²C/dK² ≈ (C(K+h) - 2C(K) + C(K-h)) / h²
        
        if len(strikes) < 3:
            return strikes, np.zeros_like(strikes)
        
        # Use central differences where possible
        density = np.zeros_like(call_prices)
        
        for i in range(1, len(strikes) - 1):
            h_left = strikes[i] - strikes[i-1]
            h_right = strikes[i+1] - strikes[i]
            h = (h_left + h_right) / 2.0
            
            density[i] = (call_prices[i+1] - 2*call_prices[i] + call_prices[i-1]) / (h**2)
        
        # Handle boundaries with forward/backward differences
        if len(strikes) >= 2:
            density[0] = density[1]
            density[-1] = density[-2]
        
        # Ensure non-negative (RN density must be >= 0)
        density = np.maximum(density, 0)
        
        # Normalize
        if np.sum(density) > 0:
            density = density / np.sum(density)
        
        return strikes, density
    
    def compute_tail_masses(
        self,
        strikes: np.ndarray,
        density: np.ndarray,
        spot: float
    ) -> Dict[str, float]:
        """Compute left and right tail masses
        
        Args:
            strikes: Strike array
            density: Density array
            spot: Current spot price
            
        Returns:
            Dictionary with tail mass metrics
        """
        # Define tail thresholds (e.g., 10% OTM)
        left_threshold = spot * (1 - self.tail_threshold)
        right_threshold = spot * (1 + self.tail_threshold)
        
        # Integrate density in tails
        left_tail_mass = np.sum(density[strikes < left_threshold])
        right_tail_mass = np.sum(density[strikes > right_threshold])
        
        return {
            'left_tail_mass': left_tail_mass,
            'right_tail_mass': right_tail_mass,
        }
    
    def compute_jump_risk_proxy(
        self,
        strikes: np.ndarray,
        iv_values: np.ndarray,
        spot: float
    ) -> float:
        """Compute jump risk proxy from far-wing steepening
        
        Args:
            strikes: Strike array
            iv_values: IV array
            spot: Spot price
            
        Returns:
            Jump risk proxy value
        """
        # Focus on far OTM puts (large downside jumps)
        far_otm_threshold = 0.8 * spot
        far_otm_mask = strikes < far_otm_threshold
        
        if np.sum(far_otm_mask) < 2:
            return 0.0
        
        far_strikes = strikes[far_otm_mask]
        far_ivs = iv_values[far_otm_mask]
        
        # Compute rate of IV increase as we go further OTM
        # Use linear regression slope
        if len(far_strikes) >= 2:
            slope = np.polyfit(far_strikes, far_ivs, 1)[0]
            return abs(slope)
        
        return 0.0
    
    def analyze_expiry(
        self,
        df_expiry: pd.DataFrame,
        spot: float,
        expiry_days: int
    ) -> Dict[str, float]:
        """Analyze single expiry
        
        Args:
            df_expiry: DataFrame for single expiry
            spot: Spot price
            expiry_days: Days to expiry
            
        Returns:
            Dictionary with metrics
        """
        if len(df_expiry) < self.min_strikes:
            return {
                'expiry_days': expiry_days,
                'left_tail_mass': np.nan,
                'right_tail_mass': np.nan,
                'jump_risk_proxy': np.nan,
                'skew_elasticity': np.nan,
            }
        
        # Extract strikes and prices (using mark prices)
        strikes = []
        ivs = []
        call_prices = []
        
        for _, row in df_expiry.iterrows():
            # Parse instrument name to get strike and type
            parts = row['instrument_name'].split('-')
            if len(parts) == 4:
                strike = float(parts[2])
                opt_type = parts[3]
                
                strikes.append(strike)
                ivs.append(row['mark_iv'] / 100.0)
                
                # Use mark price as call price proxy
                if opt_type == 'C':
                    call_prices.append(row['mark_price'])
                else:
                    # For puts, use put-call parity approximation if needed
                    call_prices.append(row['mark_price'])
        
        strikes = np.array(strikes)
        ivs = np.array(ivs)
        call_prices = np.array(call_prices)
        
        # Compute RN density
        _, density = self.breeden_litzenberger_proxy(strikes, call_prices)
        
        # Tail masses
        tail_metrics = self.compute_tail_masses(strikes, density, spot)
        
        # Jump risk
        jump_risk = self.compute_jump_risk_proxy(strikes, ivs, spot)
        
        # Skew elasticity (placeholder - needs historical data for proper calculation)
        skew_elasticity = 0.0
        
        return {
            'expiry_days': expiry_days,
            'left_tail_mass': tail_metrics['left_tail_mass'],
            'right_tail_mass': tail_metrics['right_tail_mass'],
            'jump_risk_proxy': jump_risk,
            'skew_elasticity': skew_elasticity,
        }
    
    def calculate_risk_neutral_metrics(self, timestamp: Optional[datetime] = None) -> pd.DataFrame:
        """Calculate risk-neutral metrics for all expiries
        
        Args:
            timestamp: Timestamp to analyze
            
        Returns:
            DataFrame with metrics per expiry
        """
        logger.info(f"Calculating risk-neutral metrics for {timestamp}")
        
        # Fetch ticker data
        df = self.fetch_ticker_snapshot(timestamp)
        
        if len(df) == 0:
            return pd.DataFrame()
        
        spot = df['underlying_price'].iloc[0]
        
        # Group by expiry
        results = []
        
        # Extract expiry from instrument names and group
        df['expiry_str'] = df['instrument_name'].apply(lambda x: x.split('-')[1] if len(x.split('-')) == 4 else None)
        
        for expiry_str, group in df.groupby('expiry_str'):
            if expiry_str is None:
                continue
            
            try:
                expiry_date = datetime.strptime(expiry_str, '%d%b%y')
                expiry_days = (expiry_date - timestamp).days
                
                if expiry_days < 0:
                    continue
                
                metrics = self.analyze_expiry(group, spot, expiry_days)
                metrics['timestamp'] = timestamp
                results.append(metrics)
                
            except Exception as e:
                logger.warning(f"Failed to analyze expiry {expiry_str}: {e}")
                continue
        
        return pd.DataFrame(results)


def calculate_and_store_risk_neutral(timestamp: Optional[datetime] = None) -> pd.DataFrame:
    """Calculate and store risk-neutral metrics
    
    Args:
        timestamp: Timestamp to process
        
    Returns:
        DataFrame with metrics
    """
    analyzer = RiskNeutralAnalyzer()
    metrics = analyzer.calculate_risk_neutral_metrics(timestamp)
    
    if len(metrics) > 0:
        db_client.insert_dataframe("features_risk_neutral", metrics, if_exists="append")
        logger.info(f"Stored {len(metrics)} risk-neutral metric records")
    
    return metrics
