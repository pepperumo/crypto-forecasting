"""
Tests for the market data service.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from services.market import MarketDataService
from models import PriceData


class TestMarketDataService:
    """Test the market data service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.market_service = MarketDataService()
        
    @pytest.mark.asyncio
    async def test_get_price_history_success(self):
        """Test successful price data fetch."""
        # Mock the HTTP response
        mock_response_data = {
            "prices": [[1640995200000, 50000.0], [1641081600000, 51000.0]],
            "total_volumes": [[1640995200000, 1000000.0], [1641081600000, 1100000.0]],
            "market_caps": [[1640995200000, 900000000.0], [1641081600000, 950000000.0]]
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await self.market_service.get_price_history("bitcoin", 2)
            
            assert len(result) == 2
            assert isinstance(result[0], PriceData)
            assert result[0].price == 50000.0
            assert result[1].price == 51000.0
    
    @pytest.mark.asyncio
    async def test_get_price_history_api_error(self):
        """Test price data fetch with API error."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await self.market_service.get_price_history("invalid", 2)
            
            # Should return empty list on error
            assert result == []
    
    def test_cache_functionality(self):
        """Test cache validity checking."""
        cache_key = "test_key"
        
        # Cache should be invalid initially
        assert not self.market_service._is_cache_valid(cache_key)
        
        # Add item to cache
        self.market_service._cache[cache_key] = {
            "data": ["test_data"],
            "timestamp": self.market_service._cache.get("timestamp", 
                                                       __import__("datetime").datetime.now())
        }
        
        # Cache should be valid now
        assert self.market_service._is_cache_valid(cache_key)
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test service health check."""
        # Mock a successful price fetch
        with patch.object(self.market_service, 'get_current_price') as mock_get_price:
            mock_get_price.return_value = PriceData(
                timestamp=__import__("datetime").datetime.now(),
                price=50000.0
            )
            
            health = await self.market_service.health_check()
            
            assert health["status"] == "healthy"
            assert health["provider"] == "coingecko"


if __name__ == "__main__":
    pytest.main([__file__])
