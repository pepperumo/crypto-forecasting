"""
Price-related API endpoints.
Handles fetching current prices and historical data.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from models import PriceResponse, OHLCVResponse, SymbolEnum
from services.market import MarketDataService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize market data service
market_service = MarketDataService()


@router.get("/prices/{symbol}", response_model=PriceResponse)
async def get_prices(
    symbol: SymbolEnum,
    days: int = Query(default=30, ge=1, le=365, description="Number of days of historical data"),
    background_tasks: BackgroundTasks = None
):
    """
    Get historical price data for a cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., bitcoin, ethereum)
        days: Number of days of historical data to fetch (1-365)
        
    Returns:
        PriceResponse: Historical price data with metadata
    """
    try:
        logger.info(f"Fetching {days} days of price data for {symbol}")
        
        # Fetch price data from market service
        price_data = await market_service.get_price_history(symbol.value, days)
        
        if not price_data:
            raise HTTPException(
                status_code=404,
                detail=f"No price data found for {symbol}"
            )
        
        # Schedule background update of WebSocket connections
        if background_tasks:
            background_tasks.add_task(
                _notify_price_update,
                symbol.value,
                price_data[-1] if price_data else None
            )
        
        return PriceResponse(
            symbol=symbol.value,
            data=price_data,
            count=len(price_data),
            last_updated=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching price data for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch price data: {str(e)}"
        )


@router.get("/ohlcv/{symbol}", response_model=OHLCVResponse)
async def get_ohlcv(
    symbol: SymbolEnum,
    days: int = Query(default=30, ge=1, le=365, description="Number of days of OHLCV data"),
    interval: str = Query(default="daily", description="Data interval (daily, hourly)")
):
    """
    Get OHLCV (Open, High, Low, Close, Volume) data for a cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol
        days: Number of days of historical data
        interval: Data interval (daily, hourly)
        
    Returns:
        OHLCVResponse: OHLCV data with metadata
    """
    try:
        logger.info(f"Fetching {days} days of OHLCV data for {symbol} with {interval} interval")
        
        # Fetch OHLCV data from market service
        ohlcv_data = await market_service.get_ohlcv_history(symbol.value, days, interval)
        
        if not ohlcv_data:
            raise HTTPException(
                status_code=404,
                detail=f"No OHLCV data found for {symbol}"
            )
        
        return OHLCVResponse(
            symbol=symbol.value,
            data=ohlcv_data,
            count=len(ohlcv_data),
            last_updated=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch OHLCV data: {str(e)}"
        )


@router.get("/prices/{symbol}/current")
async def get_current_price(symbol: SymbolEnum):
    """
    Get the current price for a cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol
        
    Returns:
        Current price data
    """
    try:
        logger.info(f"Fetching current price for {symbol}")
        
        # Fetch current price from market service
        current_price = await market_service.get_current_price(symbol.value)
        
        if current_price is None:
            raise HTTPException(
                status_code=404,
                detail=f"Current price not found for {symbol}"
            )
        
        return {
            "symbol": symbol.value,
            "price": current_price.price,
            "timestamp": current_price.timestamp,
            "volume": current_price.volume,
            "market_cap": current_price.market_cap
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching current price for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch current price: {str(e)}"
        )


@router.get("/prices/{symbol}/stats")
async def get_price_stats(
    symbol: SymbolEnum,
    days: int = Query(default=30, ge=1, le=365, description="Period for statistics calculation")
):
    """
    Get price statistics for a cryptocurrency over a specified period.
    
    Args:
        symbol: Cryptocurrency symbol
        days: Period for statistics calculation
        
    Returns:
        Price statistics (min, max, average, volatility, etc.)
    """
    try:
        logger.info(f"Calculating price statistics for {symbol} over {days} days")
        
        # Fetch price data
        price_data = await market_service.get_price_history(symbol.value, days)
        
        if not price_data:
            raise HTTPException(
                status_code=404,
                detail=f"No price data found for {symbol}"
            )
        
        # Calculate statistics
        prices = [p.price for p in price_data]
        
        stats = {
            "symbol": symbol.value,
            "period_days": days,
            "data_points": len(prices),
            "current_price": prices[-1] if prices else None,
            "min_price": min(prices) if prices else None,
            "max_price": max(prices) if prices else None,
            "avg_price": sum(prices) / len(prices) if prices else None,
            "price_change": prices[-1] - prices[0] if len(prices) > 1 else 0,
            "price_change_percent": ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) > 1 and prices[0] != 0 else 0,
            "volatility": _calculate_volatility(prices) if len(prices) > 1 else 0,
            "calculated_at": datetime.now()
        }
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating price statistics for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate price statistics: {str(e)}"
        )


async def _notify_price_update(symbol: str, latest_price_data):
    """
    Background task to notify WebSocket connections of price updates.
    """
    try:
        # Import here to avoid circular imports
        from main import broadcast_price_update
        
        if latest_price_data:
            await broadcast_price_update(symbol, {
                "price": latest_price_data.price,
                "timestamp": latest_price_data.timestamp.isoformat(),
                "volume": latest_price_data.volume,
                "market_cap": latest_price_data.market_cap
            })
    except Exception as e:
        logger.error(f"Error notifying price update: {e}")


def _calculate_volatility(prices: list) -> float:
    """
    Calculate price volatility (standard deviation of returns).
    
    Args:
        prices: List of price values
        
    Returns:
        Volatility as a percentage
    """
    if len(prices) < 2:
        return 0.0
    
    # Calculate daily returns
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            return_val = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(return_val)
    
    if not returns:
        return 0.0
    
    # Calculate standard deviation
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    volatility = (variance ** 0.5) * 100  # Convert to percentage
    
    return round(volatility, 4)
