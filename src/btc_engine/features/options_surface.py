"""Options surface factorization and PCA analysis"""

from typing import Optional, Dict, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from btc_engine.database.client import db_client
from btc_engine.utils.config_loader import load_yaml_config
from btc_engine.utils.logging_config import logger
from btc_engine.features.utils import (
    calculate_log_moneyness,
    calculate_time_to_expiry,
    interpolate_surface_rbf,
    robust_standardize
)


class OptionsSurfaceFactorizer:
    """Extract factors from options IV surface"""
    
    def __init__(self):
        """Initialize surface factorizer"""
        config = load_yaml_config("features")
        self.config = config["options_surface"]
        
        self.moneyness_range = self.config["moneyness_range"]
        self.moneyness_grid_points = self.config["moneyness_grid_points"]
        self.expiry_range_days = self.config["expiry_range_days"]
        self.smoothing_method = self.config["smoothing"]["method"]
        self.epsilon = self.config["smoothing"]["epsilon"]
        self.n_pca_components = self.config["pca"]["n_components"]
        self.pca_standardize = self.config["pca"]["standardize"]
        
        self.pca_model = None
        self.scaler = None
    
    def fetch_ticker_snapshot(self, timestamp: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch ticker snapshot from database
        
        Args:
            timestamp: Specific timestamp to fetch. If None, uses latest.
            
        Returns:
            DataFrame with ticker data
        """
        if timestamp is None:
            # Get latest timestamp
            latest_ts = db_client.get_latest_timestamp("raw_deribit_ticker_snapshots")
            if latest_ts is None:
                raise ValueError("No ticker data found in database")
            timestamp = latest_ts
        
        query = """
            SELECT 
                timestamp,
                instrument_name,
                mark_iv,
                underlying_price,
                open_interest
            FROM raw_deribit_ticker_snapshots
            WHERE timestamp = ?
            AND mark_iv IS NOT NULL
            AND mark_iv > 0
        """
        
        df = db_client.query_to_dataframe(query, (timestamp,))
        
        if len(df) == 0:
            raise ValueError(f"No ticker data found for timestamp {timestamp}")
        
        logger.debug(f"Fetched {len(df)} ticker records for {timestamp}")
        return df
    
    def extract_strike_expiry_from_name(self, instrument_name: str) -> Tuple[Optional[float], Optional[datetime], Optional[str]]:
        """Parse instrument name to extract strike, expiry, and option type
        
        Args:
            instrument_name: e.g., 'BTC-31MAR23-25000-C'
            
        Returns:
            Tuple of (strike, expiry_date, option_type)
        """
        try:
            parts = instrument_name.split('-')
            if len(parts) != 4:
                return None, None, None
            
            # Parse expiry date (e.g., '31MAR23')
            expiry_str = parts[1]
            expiry_date = datetime.strptime(expiry_str, '%d%b%y')
            
            # Parse strike
            strike = float(parts[2])
            
            # Parse option type
            option_type = parts[3]  # 'C' or 'P'
            
            return strike, expiry_date, option_type
            
        except Exception as e:
            logger.warning(f"Failed to parse instrument name {instrument_name}: {e}")
            return None, None, None
    
    def build_surface_grid(self, df: pd.DataFrame, current_time: datetime) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build interpolated IV surface grid
        
        Args:
            df: DataFrame with ticker data
            current_time: Current timestamp
            
        Returns:
            Tuple of (moneyness_grid, expiry_grid, iv_grid)
        """
        # Extract strike and expiry from instrument names
        strikes = []
        expiries_years = []
        iv_values = []
        
        spot = df['underlying_price'].iloc[0] if len(df) > 0 else None
        if spot is None:
            raise ValueError("No spot price available")
        
        for _, row in df.iterrows():
            strike, expiry_date, option_type = self.extract_strike_expiry_from_name(row['instrument_name'])
            
            if strike is None or expiry_date is None:
                continue
            
            # Calculate log-moneyness
            log_m = calculate_log_moneyness(strike, spot)
            
            # Calculate time to expiry
            tte = calculate_time_to_expiry(expiry_date, current_time)
            
            # Filter by configured ranges
            if log_m < self.moneyness_range[0] or log_m > self.moneyness_range[1]:
                continue
            if tte * 365 < self.expiry_range_days[0] or tte * 365 > self.expiry_range_days[1]:
                continue
            
            strikes.append(log_m)
            expiries_years.append(tte)
            iv_values.append(row['mark_iv'] / 100.0)  # Convert from percentage
        
        if len(iv_values) < 10:
            logger.warning(f"Insufficient data points for surface grid ({len(iv_values)})")
            return None, None, None
        
        # Create interpolation grid
        moneyness_grid = np.linspace(self.moneyness_range[0], self.moneyness_range[1], self.moneyness_grid_points)
        
        # Use actual expiry range from data
        expiry_min = max(min(expiries_years), self.expiry_range_days[0] / 365.0)
        expiry_max = min(max(expiries_years), self.expiry_range_days[1] / 365.0)
        expiry_grid = np.linspace(expiry_min, expiry_max, 10)
        
        # Create meshgrid for interpolation
        M, E = np.meshgrid(moneyness_grid, expiry_grid)
        
        # Interpolate IV surface
        iv_grid = interpolate_surface_rbf(
            np.array(strikes),
            np.array(expiries_years),
            np.array(iv_values),
            M.flatten(),
            E.flatten(),
            epsilon=self.epsilon
        )
        
        iv_grid = iv_grid.reshape(M.shape)
        
        return moneyness_grid, expiry_grid, iv_grid
    
    def extract_handcrafted_factors(self, moneyness_grid: np.ndarray, expiry_grid: np.ndarray, iv_grid: np.ndarray) -> Dict[str, float]:
        """Extract interpretable handcrafted factors from IV surface
        
        Args:
            moneyness_grid: Log-moneyness grid
            expiry_grid: Expiry grid (years)
            iv_grid: Interpolated IV grid (expiries × moneyness)
            
        Returns:
            Dictionary of factor values
        """
        factors = {}
        
        # Level: ATM vol (average around moneyness = 0)
        atm_idx = np.argmin(np.abs(moneyness_grid))
        atm_vols = iv_grid[:, max(0, atm_idx-2):min(len(moneyness_grid), atm_idx+3)]
        factors['level'] = np.nanmean(atm_vols)
        
        # Skew: slope around ATM (put wing - call wing)
        # Use -25 delta and +25 delta proxies
        left_idx = np.argmin(np.abs(moneyness_grid - (-0.1)))
        right_idx = np.argmin(np.abs(moneyness_grid - 0.1))
        
        left_iv = np.nanmean(iv_grid[:, left_idx])
        right_iv = np.nanmean(iv_grid[:, right_idx])
        factors['skew'] = (left_iv - right_iv)
        
        # Curvature: smile intensity (butterfly)
        # (left + right) / 2 - atm
        factors['curvature'] = ((left_iv + right_iv) / 2.0 - factors['level'])
        
        # Term structure: front vs back vol spread
        if len(expiry_grid) >= 2:
            front_vol = np.nanmean(iv_grid[0, :])  # First expiry
            back_vol = np.nanmean(iv_grid[-1, :])  # Last expiry
            factors['term_structure'] = back_vol - front_vol
        else:
            factors['term_structure'] = 0.0
        
        # Wing asymmetry: call wing slope - put wing slope
        far_left_idx = 0
        far_right_idx = -1
        
        far_left_iv = np.nanmean(iv_grid[:, far_left_idx])
        far_right_iv = np.nanmean(iv_grid[:, far_right_idx])
        
        left_slope = far_left_iv - left_iv
        right_slope = far_right_iv - right_iv
        factors['wing_asymmetry'] = right_slope - left_slope
        
        # Surface shock: will be calculated as time derivative
        factors['surface_shock'] = 0.0  # Placeholder, needs historical comparison
        
        return factors
    
    def extract_pca_factors(self, iv_grid: np.ndarray, fit: bool = False) -> Dict[str, float]:
        """Extract PCA factors from IV surface
        
        Args:
            iv_grid: Interpolated IV grid
            fit: If True, fit new PCA model. Otherwise use existing.
            
        Returns:
            Dictionary of PCA factor values
        """
        # Flatten and standardize surface
        surface_flat = iv_grid.flatten()
        
        # Remove NaNs
        if np.any(np.isnan(surface_flat)):
            surface_flat = np.nan_to_num(surface_flat, nan=np.nanmean(surface_flat))
        
        surface_flat = surface_flat.reshape(1, -1)
        
        # Check if we have enough features for PCA
        n_features = surface_flat.shape[1]
        if n_features < self.n_pca_components:
            logger.warning(f"Insufficient features ({n_features}) for PCA with {self.n_pca_components} components. Returning zeros.")
            return {f'pca_factor_{i+1}': 0.0 for i in range(self.n_pca_components)}
        
        if fit or self.pca_model is None:
            # Fit new PCA model
            if self.pca_standardize:
                self.scaler = StandardScaler()
                surface_std = self.scaler.fit_transform(surface_flat)
            else:
                surface_std = surface_flat
            
            self.pca_model = PCA(n_components=self.n_pca_components)
            pca_factors = self.pca_model.fit_transform(surface_std)
            
            logger.debug(f"PCA explained variance: {self.pca_model.explained_variance_ratio_}")
        else:
            # Transform using existing model
            if self.scaler is not None:
                surface_std = self.scaler.transform(surface_flat)
            else:
                surface_std = surface_flat
            
            pca_factors = self.pca_model.transform(surface_std)
        
        # Return as dictionary
        factors = {}
        for i in range(min(self.n_pca_components, pca_factors.shape[1])):
            factors[f'pca_factor_{i+1}'] = pca_factors[0, i]
        
        return factors
    
    def calculate_surface_factors(self, timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """Calculate all surface factors for a timestamp
        
        Args:
            timestamp: Timestamp to calculate factors for. If None, uses latest.
            
        Returns:
            Dictionary with all factor values
        """
        logger.info(f"Calculating surface factors for {timestamp}")
        
        # Fetch ticker data
        df = self.fetch_ticker_snapshot(timestamp)
        
        if timestamp is None:
            timestamp = df['timestamp'].iloc[0]
        
        # Build surface grid
        moneyness_grid, expiry_grid, iv_grid = self.build_surface_grid(df, timestamp)
        
        if iv_grid is None:
            logger.warning("Surface grid construction failed, returning NaN factors")
            return {
                'timestamp': timestamp,
                'level': np.nan,
                'skew': np.nan,
                'curvature': np.nan,
                'term_structure': np.nan,
                'wing_asymmetry': np.nan,
                'surface_shock': np.nan,
            }
        
        # Extract handcrafted factors
        handcrafted = self.extract_handcrafted_factors(moneyness_grid, expiry_grid, iv_grid)
        
        # Extract PCA factors (fit on first call)
        pca_factors = self.extract_pca_factors(iv_grid, fit=False)
        
        # Combine all factors
        all_factors = {'timestamp': timestamp}
        all_factors.update(handcrafted)
        all_factors.update(pca_factors)
        
        logger.info(f"Extracted {len(all_factors)-1} surface factors")
        
        return all_factors


def calculate_and_store_surface_factors(timestamp: Optional[datetime] = None) -> Dict[str, float]:
    """Calculate surface factors and store to database
    
    Args:
        timestamp: Timestamp to process
        
    Returns:
        Dictionary with factors
    """
    factorizer = OptionsSurfaceFactorizer()
    factors = factorizer.calculate_surface_factors(timestamp)
    
    # Store to database
    df = pd.DataFrame([factors])
    db_client.insert_dataframe("features_options_surface", df, if_exists="append")
    
    logger.info(f"Stored surface factors to database")
    
    return factors
