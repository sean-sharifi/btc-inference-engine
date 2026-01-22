"""Pytest configuration and fixtures"""

import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_ticker_data():
    """Sample ticker data for testing"""
    return {
        'timestamp': '2024-01-01 12:00:00',
        'instrument_name': 'BTC-31MAR24-50000-C',
        'mark_iv': 65.5,
        'mark_price': 1500.0,
        'underlying_price': 48000.0,
        'open_interest': 100.0,
        'greeks_delta': 0.5,
        'greeks_gamma': 0.00001,
        'greeks_vega': 25.0,
        'greeks_theta': -15.0
    }
