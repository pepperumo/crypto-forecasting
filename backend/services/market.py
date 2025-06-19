"""
Market data service for fetching cryptocurrency prices and market information.
Supports multiple data providers with fallback capabilities.
"""

import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import os

from models import PriceData, OHLCV

logger = logging.getLogger(__name__)


class MarketDataService:
    """Service for fetching cryptocurrency market data."""
    
    def __init__(self):
        self.provider = os.getenv("MARKET_DATA_PROVIDER", "coingecko")
        self.cache_ttl = int(os.getenv("MARKET_DATA_CACHE_TTL", "300"))
        self.base_urls = {
            "coingecko": "https://api.coingecko.com/api/v3",
        }
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting properties
        self._last_request_time = datetime.now() - timedelta(seconds=10)
        self._request_cooldown = float(os.getenv("API_REQUEST_COOLDOWN", "1.2"))  # Wait at least this many seconds between requests
        self._retry_count = int(os.getenv("API_RETRY_COUNT", "3"))
        self._retry_delay = float(os.getenv("API_RETRY_DELAY", "2.0"))
        
    async def get_price_history(self, symbol: str, days: int) -> List[PriceData]:
        """
        Fetch historical price data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'bitcoin', 'ethereum')
            days: Number of days of historical data
            
        Returns:
            List of PriceData objects
        """
        try:
            cache_key = f"price_history_{symbol}_{days}"
            
            # Check cache first
            if self._is_cache_valid(cache_key):
                logger.debug(f"Returning cached price history for {symbol}")
                return self._cache[cache_key]["data"]
            
            # Fetch from API
            if self.provider == "coingecko":
                data = await self._fetch_coingecko_price_history(symbol, days)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Cache the result
            self._cache[cache_key] = {
                "data": data,
                "timestamp": datetime.now()
            }
            
            logger.info(f"Fetched {len(data)} price points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching price history for {symbol}: {e}")
            
            # Try to return cached data if available (even if expired)
            if cache_key in self._cache:
                logger.warning(f"Returning expired cached data for {symbol}")
                return self._cache[cache_key]["data"]
            
            return []
    
    async def get_ohlcv_history(self, symbol: str, days: int, interval: str = "daily") -> List[OHLCV]:
        """
        Fetch OHLCV historical data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days of historical data
            interval: Data interval ('daily', 'hourly')
            
        Returns:
            List of OHLCV objects
        """
        try:
            cache_key = f"ohlcv_history_{symbol}_{days}_{interval}"
            
            # Check cache first
            if self._is_cache_valid(cache_key):
                logger.debug(f"Returning cached OHLCV history for {symbol}")
                return self._cache[cache_key]["data"]
            
            # Fetch from API
            if self.provider == "coingecko":
                data = await self._fetch_coingecko_ohlcv_history(symbol, days, interval)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Cache the result
            self._cache[cache_key] = {
                "data": data,
                "timestamp": datetime.now()
            }
            
            logger.info(f"Fetched {len(data)} OHLCV points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV history for {symbol}: {e}")
            
            # Try to return cached data if available
            if cache_key in self._cache:
                logger.warning(f"Returning expired cached OHLCV data for {symbol}")
                return self._cache[cache_key]["data"]
            
            return []
    
    async def get_current_price(self, symbol: str) -> Optional[PriceData]:
        """
        Get the current price for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            PriceData object with current price information
        """
        try:
            cache_key = f"current_price_{symbol}"
            
            # Check cache (shorter TTL for current price)
            if self._is_cache_valid(cache_key, ttl=60):  # 1-minute cache for current price
                logger.debug(f"Returning cached current price for {symbol}")
                return self._cache[cache_key]["data"]
            
            # Fetch from API
            if self.provider == "coingecko":
                data = await self._fetch_coingecko_current_price(symbol)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Cache the result
            self._cache[cache_key] = {
                "data": data,
                "timestamp": datetime.now()
            }
            
            logger.info(f"Fetched current price for {symbol}: ${data.price}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
              # Try to return cached data if available
            if cache_key in self._cache:
                logger.warning(f"Returning expired cached current price for {symbol}")
                return self._cache[cache_key]["data"]
            
            return None
            
    async def _fetch_coingecko_price_history(self, symbol: str, days: int) -> List[PriceData]:
        """Fetch price history from CoinGecko API."""
        url = f"{self.base_urls['coingecko']}/coins/{symbol}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": str(days),
            "interval": "daily" if days > 1 else "hourly"
        }
        
        # Get data using our rate-limited request method
        data = await self._rate_limited_request(url, params)
        
        # Process the data
        price_data = []
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        market_caps = data.get("market_caps", [])
        
        for i, price_point in enumerate(prices):
            timestamp = datetime.fromtimestamp(price_point[0] / 1000)
            price = price_point[1]
            volume = volumes[i][1] if i < len(volumes) else None
            market_cap = market_caps[i][1] if i < len(market_caps) else None
            
            price_data.append(PriceData(
                timestamp=timestamp,
                price=price,
                volume=volume,
                market_cap=market_cap
            ))
        return price_data
    
    async def _fetch_coingecko_ohlcv_history(self, symbol: str, days: int, interval: str) -> List[OHLCV]:
        """Fetch OHLCV history from CoinGecko API."""
        url = f"{self.base_urls['coingecko']}/coins/{symbol}/ohlc"
        params = {
            "vs_currency": "usd",
            "days": str(days)
        }
        
        # Get data using our rate-limited request method
        data = await self._rate_limited_request(url, params)
        
        ohlcv_data = []
        for ohlc_point in data:
            timestamp = datetime.fromtimestamp(ohlc_point[0] / 1000)
            open_price = ohlc_point[1]
            high_price = ohlc_point[2]
            low_price = ohlc_point[3]
            close_price = ohlc_point[4]
            
            # Note: CoinGecko OHLC endpoint doesn't include volume
            # We'll set a placeholder or fetch separately if needed
            volume = 0.0
            
            ohlcv_data.append(OHLCV(
                timestamp=timestamp,
                open=open_price,
                high=high_price,                low=low_price,
                close=close_price,
                volume=volume
            ))
        
        return ohlcv_data
    
    async def _fetch_coingecko_current_price(self, symbol: str) -> PriceData:
        """Fetch current price from CoinGecko API."""
        url = f"{self.base_urls['coingecko']}/simple/price"
        params = {
            "ids": symbol,
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true"
        }
        
        # Use rate-limited request method
        data = await self._rate_limited_request(url, params)
        
        if symbol not in data:
            raise Exception(f"Symbol {symbol} not found in CoinGecko response")
        
        symbol_data = data[symbol]
        
        return PriceData(
            timestamp=datetime.now(),
            price=symbol_data["usd"],
            volume=symbol_data.get("usd_24h_vol"),
            market_cap=symbol_data.get("usd_market_cap")
        )
    
    async def _rate_limited_request(self, url: str, params: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Make a rate-limited HTTP request with retries.
        
        Args:
            url: The URL to request
            params: Optional query parameters
            
        Returns:
            JSON response data
        """
        for attempt in range(self._retry_count):
            # Respect rate limits by waiting between requests
            time_since_last_request = (datetime.now() - self._last_request_time).total_seconds()
            if time_since_last_request < self._request_cooldown:
                wait_time = self._request_cooldown - time_since_last_request
                logger.debug(f"Rate limiting: Waiting {wait_time:.2f} seconds before request")
                await asyncio.sleep(wait_time)
            
            try:
                self._last_request_time = datetime.now()
                
                async with aiohttp.ClientSession() as session:
                    logger.debug(f"Requesting URL: {url} with params: {params}")
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:  # Rate limit exceeded
                            # Exponential backoff
                            retry_delay = self._retry_delay * (2 ** attempt)
                            logger.warning(f"Rate limit exceeded. Retrying after {retry_delay} seconds (attempt {attempt+1}/{self._retry_count})")
                            await asyncio.sleep(retry_delay)
                        else:
                            raise Exception(f"API error: {response.status}")
            except Exception as e:
                logger.error(f"Request error (attempt {attempt+1}/{self._retry_count}): {e}")
                if attempt < self._retry_count - 1:
                    retry_delay = self._retry_delay * (2 ** attempt)
                    logger.info(f"Retrying after {retry_delay} seconds")
                    await asyncio.sleep(retry_delay)
                else:
                    raise
                    
        raise Exception(f"Failed after {self._retry_count} attempts")
    
    def _is_cache_valid(self, cache_key: str, ttl: Optional[int] = None) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        
        cache_age = (datetime.now() - self._cache[cache_key]["timestamp"]).total_seconds()
        ttl = ttl or self.cache_ttl
        
        return cache_age < ttl
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Market data cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the market data service."""
        try:
            # Try to fetch a simple price to test API connectivity
            test_price = await self.get_current_price("bitcoin")
            
            return {
                "status": "healthy" if test_price else "degraded",
                "provider": self.provider,
                "cache_entries": len(self._cache),
                "last_check": datetime.now()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider,
                "error": str(e),
                "last_check": datetime.now()
            }
