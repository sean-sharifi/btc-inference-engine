"""Deribit API connector for options data"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from btc_engine.connectors.base import BaseConnector
from btc_engine.utils.config_loader import settings, load_yaml_config
from btc_engine.utils.logging_config import logger


class DeribitConnector(BaseConnector):
    """Connector for Deribit exchange API"""
    
    def __init__(self):
        """Initialize Deribit connector"""
        config = load_yaml_config("data_sources")
        deribit_config = config["deribit"]
        
        super().__init__(
            base_url=deribit_config["base_url"],
            rate_limit_rps=deribit_config["rate_limit"]["requests_per_second"],
            rate_limit_burst=deribit_config["rate_limit"]["burst"],
            cache_ttl=deribit_config["cache_ttl_seconds"],
            max_retries=deribit_config["retry"]["max_attempts"],
            backoff_factor=deribit_config["retry"]["backoff_factor"]
        )
        
        self.api_key = settings.deribit_api_key
        self.api_secret = settings.deribit_api_secret
        self.currency = deribit_config["instruments"]["currency"]
        self.kind = deribit_config["instruments"]["kind"]
        
        logger.info(f"DeribitConnector initialized for {self.currency} {self.kind}s")
    
    def test_connection(self) -> bool:
        """Test Deribit API connection
        
        Returns:
            True if connection successful
        """
        try:
            response = self._make_request_with_retry(
                "GET",
                f"{self.base_url}/public/test"
            )
            logger.info("Deribit connection test successful")
            return True
        except Exception as e:
            logger.error(f"Deribit connection test failed: {e}")
            return False
    
    def get_instruments(self, expired: bool = False) -> List[Dict[str, Any]]:
        """Get list of available instruments
        
        Args:
            expired: Include expired instruments
            
        Returns:
            List of instrument dictionaries
        """
        cache_key = self._get_cache_key("instruments", expired)
        
        # Check memory cache
        if cache_key in self.cache:
            logger.debug("Loaded instruments from memory cache")
            return self.cache[cache_key]
        
        # Check file cache
        cached = self._load_from_file_cache(cache_key, ttl=3600)
        if cached:
            self.cache[cache_key] = cached
            return cached
        
        # Fetch from API
        logger.info(f"Fetching {self.currency} {self.kind} instruments from Deribit")
        
        try:
            response = self._make_request_with_retry(
                "GET",
                f"{self.base_url}/public/get_instruments",
                params={
                    "currency": self.currency,
                    "kind": self.kind,
                    "expired": str(expired).lower()
                }
            )
            
            data = response.json()
            
            if "result" not in data:
                raise ValueError(f"Unexpected response format: {data}")
            
            instruments = data["result"]
            logger.info(f"Fetched {len(instruments)} instruments")
            
            # Cache results
            self.cache[cache_key] = instruments
            self._save_to_file_cache(cache_key, instruments)
            
            return instruments
            
        except Exception as e:
            logger.error(f"Failed to fetch instruments: {e}")
            raise
    
    def get_ticker(self, instrument_name: str) -> Dict[str, Any]:
        """Get ticker data for instrument
        
        Args:
            instrument_name: Name of instrument (e.g., 'BTC-31MAR23-25000-C')
            
        Returns:
            Ticker data dictionary
        """
        try:
            response = self._make_request_with_retry(
                "GET",
                f"{self.base_url}/public/ticker",
                params={"instrument_name": instrument_name}
            )
            
            data = response.json()
            
            if "result" not in data:
                raise ValueError(f"Unexpected response format: {data}")
            
            return data["result"]
            
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {instrument_name}: {e}")
            raise
    
    def get_tickers_batch(self, instrument_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get ticker data for multiple instruments
        
        Args:
            instrument_names: List of instrument names. If None, gets all active options.
            
        Returns:
            List of ticker dictionaries
        """
        if instrument_names is None:
            # Get all active instruments
            instruments = self.get_instruments(expired=False)
            instrument_names = [inst["instrument_name"] for inst in instruments]
        
        logger.info(f"Fetching tickers for {len(instrument_names)} instruments")
        
        tickers = []
        for instrument_name in instrument_names:
            try:
                ticker = self.get_ticker(instrument_name)
                tickers.append(ticker)
            except Exception as e:
                logger.warning(f"Failed to fetch ticker for {instrument_name}: {e}")
                continue
        
        logger.info(f"Successfully fetched {len(tickers)} tickers")
        return tickers
    
    def get_order_book(self, instrument_name: str, depth: int = 5) -> Dict[str, Any]:
        """Get order book for instrument
        
        Args:
            instrument_name: Name of instrument
            depth: Order book depth
            
        Returns:
            Order book dictionary
        """
        try:
            response = self._make_request_with_retry(
                "GET",
                f"{self.base_url}/public/get_order_book",
                params={
                    "instrument_name": instrument_name,
                    "depth": depth
                }
            )
            
            data = response.json()
            return data["result"] if "result" in data else {}
            
        except Exception as e:
            logger.error(f"Failed to fetch order book for {instrument_name}: {e}")
            raise
    
    def get_funding_rate(self) -> Dict[str, Any]:
        """Get current funding rate for BTC perpetual
        
        Returns:
            Funding rate data
        """
        instrument_name = f"{self.currency}-PERPETUAL"
        
        try:
            ticker = self.get_ticker(instrument_name)
            
            return {
                "instrument_name": instrument_name,
                "funding_8h": ticker.get("funding_8h"),
                "current_funding": ticker.get("current_funding"),
                "index_price": ticker.get("index_price"),
                "mark_price": ticker.get("mark_price"),
                "open_interest": ticker.get("open_interest"),
                "timestamp": ticker.get("timestamp")
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch funding rate: {e}")
            raise
    
    def get_volatility_index(self) -> Dict[str, Any]:
        """Get DVOL (Deribit Volatility Index)
        
        Returns:
            Volatility index data
        """
        try:
            response = self._make_request_with_retry(
                "GET",
                f"{self.base_url}/public/get_index_price",
                params={"index_name": f"{self.currency}_DVOL"}
            )
            
            data = response.json()
            return data["result"] if "result" in data else {}
            
        except Exception as e:
            logger.warning(f"Failed to fetch DVOL: {e}")
            return {}
    
    def snapshot_iv_surface(self) -> pd.DataFrame:
        """Snapshot current IV surface across all active options
        
        Returns:
            DataFrame with IV surface data
        """
        logger.info("Snapshotting IV surface")
        
        tickers = self.get_tickers_batch()
        timestamp = datetime.now()
        
        records = []
        for ticker in tickers:
            record = {
                "timestamp": timestamp,
                "instrument_name": ticker.get("instrument_name"),
                "mark_price": ticker.get("mark_price"),
                "mark_iv": ticker.get("mark_iv"),
                "bid_price": ticker.get("best_bid_price"),
                "bid_iv": ticker.get("bid_iv"),
                "ask_price": ticker.get("best_ask_price"),
                "ask_iv": ticker.get("ask_iv"),
                "underlying_price": ticker.get("underlying_price"),
                "open_interest": ticker.get("open_interest"),
                "volume_24h": ticker.get("stats", {}).get("volume") if ticker.get("stats") else None,
                "last_price": ticker.get("last_price"),
                "greeks_delta": ticker.get("greeks", {}).get("delta") if ticker.get("greeks") else None,
                "greeks_gamma": ticker.get("greeks", {}).get("gamma") if ticker.get("greeks") else None,
                "greeks_vega": ticker.get("greeks", {}).get("vega") if ticker.get("greeks") else None,
                "greeks_theta": ticker.get("greeks", {}).get("theta") if ticker.get("greeks") else None,
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Snapshotted {len(df)} option tickers")
        
        return df


def create_deribit_connector() -> DeribitConnector:
    """Factory function to create Deribit connector
    
    Returns:
        Configured DeribitConnector instance
    """
    return DeribitConnector()
