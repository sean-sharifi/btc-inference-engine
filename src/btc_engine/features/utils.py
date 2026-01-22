"""Feature engineering utilities for interpolation, smoothing, and transformations"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf, griddata
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

from btc_engine.utils.logging_config import logger


def calculate_log_moneyness(strike: float, spot: float) -> float:
    """Calculate log-moneyness
    
    Args:
        strike: Strike price
        spot: Spot price
        
    Returns:
        Log-moneyness: log(strike / spot)
    """
    return np.log(strike / spot)


def calculate_time_to_expiry(expiry_date, current_date) -> float:
    """Calculate time to expiry in years
    
    Args:
        expiry_date: Expiration datetime
        current_date: Current datetime
        
    Returns:
        Time to expiry in years
    """
    delta = expiry_date - current_date
    return delta.total_seconds() / (365.25 * 24 * 3600)


def interpolate_surface_rbf(
    strikes: np.ndarray,
    expiries: np.ndarray,
    iv_values: np.ndarray,
    grid_strikes: np.ndarray,
    grid_expiries: np.ndarray,
    epsilon: float = 0.1
) -> np.ndarray:
    """Interpolate IV surface using Radial Basis Functions
    
    Args:
        strikes: Array of strikes
        expiries: Array of expiries (in years)
        iv_values: Array of IV values
        grid_strikes: Grid of strikes for interpolation
        grid_expiries: Grid of expiries for interpolation
        epsilon: RBF smoothing parameter
        
    Returns:
        Interpolated IV values on grid
    """
    # Remove NaN values
    mask = ~np.isnan(iv_values)
    strikes = strikes[mask]
    expiries = expiries[mask]
    iv_values = iv_values[mask]
    
    if len(iv_values) < 4:
        logger.warning(f"Insufficient data for RBF interpolation ({len(iv_values)} points)")
        return np.full_like(grid_strikes, np.nan, dtype=float)
    
    try:
        # Create RBF interpolator
        rbf = Rbf(strikes, expiries, iv_values, function='multiquadric', epsilon=epsilon, smooth=0.001)
        
        # Interpolate on grid
        iv_grid = rbf(grid_strikes, grid_expiries)
        
        return iv_grid
        
    except Exception as e:
        logger.error(f"RBF interpolation failed: {e}")
        return np.full_like(grid_strikes, np.nan, dtype=float)


def calculate_z_score(
    series: pd.Series,
    window: int = 30,
    min_periods: Optional[int] = None
) -> pd.Series:
    """Calculate rolling z-score
    
    Args:
        series: Time series
        window: Rolling window size
        min_periods: Minimum periods for calculation
        
    Returns:
        Z-score series
    """
    if min_periods is None:
        min_periods = max(1, window // 2)
    
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    
    z_score = (series - rolling_mean) / (rolling_std + 1e-10)
    
    return z_score


def exponential_smoothing(
    series: pd.Series,
    alpha: float = 0.3
) -> pd.Series:
    """Apply exponential smoothing
    
    Args:
        series: Time series
        alpha: Smoothing factor (0-1)
        
    Returns:
        Smoothed series
    """
    return series.ewm(alpha=alpha, adjust=False).mean()


def robust_standardize(
    data: np.ndarray,
    clip_std: float = 3.0
) -> np.ndarray:
    """Robustly standardize data with outlier clipping
    
    Args:
        data: Input data array
        clip_std: Number of std deviations for clipping
        
    Returns:
        Standardized data
    """
    # Calculate median and MAD (robust statistics)
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))
    
    # Standardize
    standardized = (data - median) / (1.4826 * mad + 1e-10)
    
    # Clip outliers
    standardized = np.clip(standardized, -clip_std, clip_std)
    
    return standardized


def smooth_time_series(
    series: pd.Series,
    window: int = 7,
    method: str = 'gaussian'
) -> pd.Series:
    """Smooth time series
    
    Args:
        series: Input time series
        window: Smoothing window size
        method: Smoothing method ('gaussian', 'rolling', 'ewm')
        
    Returns:
        Smoothed series
    """
    if method == 'gaussian':
        # Gaussian filter
        smoothed_values = gaussian_filter1d(
            series.fillna(method='ffill').fillna(method='bfill').values,
            sigma=window / 2
        )
        return pd.Series(smoothed_values, index=series.index)
    
    elif method == 'rolling':
        # Simple rolling mean
        return series.rolling(window=window, center=True, min_periods=1).mean()
    
    elif method == 'ewm':
        # Exponential weighted moving average
        return series.ewm(span=window, adjust=False).mean()
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def detect_spikes(
    series: pd.Series,
    threshold: float = 2.0,
    window: int = 30
) -> pd.Series:
    """Detect spikes in time series
    
    Args:
        series: Input time series
        threshold: Z-score threshold for spike detection
        window: Rolling window for z-score calculation
        
    Returns:
        Boolean series indicating spikes
    """
    z_scores = calculate_z_score(series, window=window)
    spikes = np.abs(z_scores) > threshold
    
    return spikes


def compute_returns(
    prices: pd.Series,
    method: str = 'log'
) -> pd.Series:
    """Compute returns
    
    Args:
        prices: Price series
        method: 'log' or 'simple'
        
    Returns:
        Returns series
    """
    if method == 'log':
        return np.log(prices / prices.shift(1))
    elif method == 'simple':
        return prices.pct_change()
    else:
        raise ValueError(f"Unknown return method: {method}")
