"""Base connector with retry, caching, and rate limiting logic"""

from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
import time
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from cachetools import TTLCache

from btc_engine.utils.logging_config import logger
from btc_engine.utils.constants import PROJECT_ROOT


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, requests_per_second: float, burst: int = 1):
        """Initialize rate limiter
        
        Args:
            requests_per_second: Maximum requests per second
            burst: Maximum burst size
        """
        self.requests_per_second = requests_per_second
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
    
    def acquire(self) -> None:
        """Acquire a token, blocking if necessary"""
        while True:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.burst,
                self.tokens + elapsed * self.requests_per_second
            )
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                break
            else:
                sleep_time = (1 - self.tokens) / self.requests_per_second
                time.sleep(sleep_time)


class BaseConnector(ABC):
    """Base class for API connectors with retry, caching, and rate limiting"""
    
    def __init__(
        self,
        base_url: str,
        rate_limit_rps: float = 1.0,
        rate_limit_burst: int = 3,
        cache_ttl: int = 300,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ):
        """Initialize base connector
        
        Args:
            base_url: Base URL for API
            rate_limit_rps: Rate limit in requests per second
            rate_limit_burst: Maximum burst size
            cache_ttl: Cache TTL in seconds
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff factor
        """
        self.base_url = base_url.rstrip('/')
        self.rate_limiter = RateLimiter(rate_limit_rps, rate_limit_burst)
        self.cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # HTTP client with timeout
        self.client = httpx.Client(timeout=30.0)
        
        # Cache directory for persistent caching
        self.cache_dir = PROJECT_ROOT / ".cache" / self.__class__.__name__
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments
        
        Returns:
            MD5 hash of arguments
        """
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_file_cache_path(self, cache_key: str) -> Path:
        """Get file path for persistent cache
        
        Args:
            cache_key: Cache key
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_file_cache(self, cache_key: str, ttl: int) -> Optional[Any]:
        """Load data from file cache if valid
        
        Args:
            cache_key: Cache key
            ttl: Time-to-live in seconds
            
        Returns:
            Cached data or None if not found or expired
        """
        cache_file = self._get_file_cache_path(cache_key)
        
        if not cache_file.exists():
            return None
        
        # Check if cache is expired
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age > ttl:
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            logger.debug(f"Loaded from file cache: {cache_key}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load from file cache: {e}")
            return None
    
    def _save_to_file_cache(self, cache_key: str, data: Any) -> None:
        """Save data to file cache
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        cache_file = self._get_file_cache_path(cache_key)
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            logger.debug(f"Saved to file cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save to file cache: {e}")
    
    def _make_request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request with retry logic
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional arguments for httpx request
            
        Returns:
            Response object
        """
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.backoff_factor, min=1, max=60),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPError)),
            reraise=True
        )
        def _request():
            self.rate_limiter.acquire()
            response = self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        
        return _request()
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test API connection
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    def close(self) -> None:
        """Close HTTP client"""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
