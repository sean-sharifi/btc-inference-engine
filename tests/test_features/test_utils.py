"""Test feature utilities"""

import numpy as np
import pandas as pd
from btc_engine.features.utils import (
    calculate_log_moneyness,
    calculate_z_score,
    exponential_smoothing,
    detect_spikes
)


def test_calculate_log_moneyness():
    """Test log-moneyness calculation"""
    strike = 50000
    spot = 48000
    
    log_m = calculate_log_moneyness(strike, spot)
    
    assert isinstance(log_m, float)
    assert log_m > 0  # Strike > spot


def test_calculate_z_score():
    """Test z-score calculation"""
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    z_scores = calculate_z_score(series, window=5)
    
    assert len(z_scores) == len(series)
    assert not np.isnan(z_scores.iloc[-1])


def test_exponential_smoothing():
    """Test exponential smoothing"""
    series = pd.Series([1, 5, 2, 8, 3, 9, 4, 10])
    
    smoothed = exponential_smoothing(series, alpha=0.3)
    
    assert len(smoothed) == len(series)
    assert smoothed.iloc[-1] != series.iloc[-1]  # Should be smoothed


def test_detect_spikes():
    """Test spike detection"""
    # Create series with a spike
    series = pd.Series([1, 1, 1, 10, 1, 1, 1])
    
    spikes = detect_spikes(series, threshold=2.0, window=5)
    
    assert spikes.iloc[3] == True  # Spike at index 3
