"""Glassnode API connector for onchain data"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from btc_engine.connectors.base import BaseConnector
from btc_engine.utils.config_loader import settings, load_yaml_config
from btc_engine.utils.logging_config import logger


class GlassnodeConnector(BaseConnector):
    """Connector for Glassnode API"""
    
    def __init__(self):
        """Initialize Glassnode connector"""
        config = load_yaml_config("data_sources")
        glassnode_config = config["glassnode"]
        
        super().__init__(
            base_url=glassnode_config["base_url"],
            rate_limit_rps=glassnode_config["rate_limit"]["requests_per_second"],
            rate_limit_burst=glassnode_config["rate_limit"]["burst"],
            cache_ttl=glassnode_config["cache_ttl_seconds"],
            max_retries=glassnode_config["retry"]["max_attempts"],
            backoff_factor=glassnode_config["retry"]["backoff_factor"]
        )
        
        self.api_key = settings.glassnode_api_key
        self.endpoints = glassnode_config["endpoints"]
        self.resolution = glassnode_config.get("resolution", "24h")
        
        logger.info("GlassnodeConnector initialized")
    
    def test_connection(self) -> bool:
        """Test Glassnode API connection
        
        Returns:
            True if connection successful
        """
        try:
            # Test with a simple endpoint
            _ = self.get_metric(
                "supply_lth",
                since=int((datetime.now() - timedelta(days=7)).timestamp()),
                until=int(datetime.now().timestamp())
            )
            logger.info("Glassnode connection test successful")
            return True
        except Exception as e:
            logger.error(f"Glassnode connection test failed: {e}")
            return False
    
    def get_metric(
        self,
        metric_key: str,
        since: Optional[int] = None,
        until: Optional[int] = None,
        resolution: Optional[str] = None
    ) -> pd.DataFrame:
        """Get time series data for a metric
        
        Args:
            metric_key: Key in endpoints config (e.g., 'supply_lth')
            since: Start timestamp (Unix time)
            until: End timestamp (Unix time)
            resolution: Data resolution ('24h', '1h', etc.)
            
        Returns:
            DataFrame with timestamp and value columns
        """
        if metric_key not in self.endpoints:
            raise ValueError(f"Unknown metric: {metric_key}. Available: {list(self.endpoints.keys())}")
        
        endpoint = self.endpoints[metric_key]
        resolution = resolution or self.resolution
        
        # Default time range if not specified
        if since is None:
            since = int((datetime.now() - timedelta(days=90)).timestamp())
        if until is None:
            until = int(datetime.now().timestamp())
        
        # Create cache key
        cache_key = self._get_cache_key(metric_key, since, until, resolution)
        
        # Check memory cache
        if cache_key in self.cache:
            logger.debug(f"Loaded {metric_key} from memory cache")
            return self.cache[cache_key]
        
        # Check file cache
        cached = self._load_from_file_cache(cache_key, ttl=7200)  # 2 hour file cache
        if cached is not None:
            df = pd.DataFrame(cached)
            self.cache[cache_key] = df
            return df
        
        # Fetch from API
        logger.info(f"Fetching {metric_key} from Glassnode (since={since}, until={until})")
        
        try:
            response = self._make_request_with_retry(
                "GET",
                f"{self.base_url}{endpoint}",
                params={
                    "a": "BTC",
                    "s": since,
                    "u": until,
                    "i": resolution,
                    "api_key": self.api_key
                }
            )
            
            data = response.json()
            
            if not isinstance(data, list):
                raise ValueError(f"Unexpected response format for {metric_key}: {data}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if len(df) > 0:
                # Convert timestamp to datetime
                df['t'] = pd.to_datetime(df['t'], unit='s')
                df = df.rename(columns={'t': 'timestamp', 'v': 'value'})
                df['metric_name'] = metric_key
            else:
                df = pd.DataFrame(columns=['timestamp', 'value', 'metric_name'])
            
            logger.info(f"Fetched {len(df)} records for {metric_key}")
            
            # Cache results
            self.cache[cache_key] = df
            self._save_to_file_cache(cache_key, df.to_dict('records'))
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {metric_key}: {e}")
            raise
    
    def get_multiple_metrics(
        self,
        metric_keys: List[str],
        since: Optional[int] = None,
        until: Optional[int] = None,
        resolution: Optional[str] = None
    ) -> pd.DataFrame:
        """Get multiple metrics and combine into single DataFrame
        
        Args:
            metric_keys: List of metric keys
            since: Start timestamp (Unix time)
            until: End timestamp (Unix time)
            resolution: Data resolution
            
        Returns:
            Combined DataFrame with all metrics
        """
        logger.info(f"Fetching {len(metric_keys)} metrics from Glassnode")
        
        dfs = []
        for metric_key in metric_keys:
            try:
                df = self.get_metric(metric_key, since, until, resolution)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to fetch {metric_key}: {e}")
                continue
        
        if not dfs:
            logger.warning("No metrics successfully fetched")
            return pd.DataFrame(columns=['timestamp', 'value', 'metric_name'])
        
        # Combine all DataFrames
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(combined)} total records from {len(dfs)} metrics")
        
        return combined
    
    def get_exchange_flows(
        self,
        since: Optional[int] = None,
        until: Optional[int] = None
    ) -> pd.DataFrame:
        """Get exchange flow metrics (inflow, outflow, netflow)
        
        Args:
            since: Start timestamp
            until: End timestamp
            
        Returns:
            DataFrame with exchange flow data
        """
        flow_metrics = [
            "exchange_flows_inflow",
            "exchange_flows_outflow",
            "exchange_netflow"
        ]
        
        return self.get_multiple_metrics(flow_metrics, since, until)
    
    def get_supply_cohorts(
        self,
        since: Optional[int] = None,
        until: Optional[int] = None
    ) -> pd.DataFrame:
        """Get supply cohort metrics (LTH, STH, etc.)
        
        Args:
            since: Start timestamp
            until: End timestamp
            
        Returns:
            DataFrame with supply cohort data
        """
        supply_metrics = [
            "supply_lth",
            "supply_sth",
            "supply_active_1y",
            "supply_illiquid"
        ]
        
        return self.get_multiple_metrics(supply_metrics, since, until)
    
    def get_dormancy_metrics(
        self,
        since: Optional[int] = None,
        until: Optional[int] = None
    ) -> pd.DataFrame:
        """Get dormancy and age-related metrics
        
        Args:
            since: Start timestamp
            until: End timestamp
            
        Returns:
            DataFrame with dormancy metrics
        """
        dormancy_metrics = [
            "dormancy",
            "cdd",
            "soab"
        ]
        
        return self.get_multiple_metrics(dormancy_metrics, since, until)
    
    def get_stablecoin_metrics(
        self,
        since: Optional[int] = None,
        until: Optional[int] = None
    ) -> pd.DataFrame:
        """Get stablecoin supply and flow metrics
        
        Args:
            since: Start timestamp
            until: End timestamp
            
        Returns:
            DataFrame with stablecoin metrics
        """
        stablecoin_metrics = [
            "stablecoin_supply_ratio",
            "stablecoin_exchange_netflow"
        ]
        
        return self.get_multiple_metrics(stablecoin_metrics, since, until)
    
    def get_all_onchain_metrics(
        self,
        since: Optional[int] = None,
        until: Optional[int] = None
    ) -> pd.DataFrame:
        """Get all configured onchain metrics
        
        Args:
            since: Start timestamp
            until: End timestamp
            
        Returns:
            DataFrame with all metrics
        """
        all_metric_keys = list(self.endpoints.keys())
        logger.info(f"Fetching all {len(all_metric_keys)} onchain metrics")
        
        return self.get_multiple_metrics(all_metric_keys, since, until)


def create_glassnode_connector() -> GlassnodeConnector:
    """Factory function to create Glassnode connector
    
    Returns:
        Configured GlassnodeConnector instance
    """
    return GlassnodeConnector()
