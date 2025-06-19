"""
Forecasting API endpoints.
Handles model predictions and forecast-related operations.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from models import ForecastResponse, SymbolEnum
from services.forecast import ForecastService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize forecast service
forecast_service = ForecastService()


@router.get("/forecast/{symbol}", response_model=ForecastResponse)
async def get_forecast(
    symbol: SymbolEnum,
    horizon: int = Query(default=7, ge=1, le=30, description="Forecast horizon in days"),
    background_tasks: BackgroundTasks = None
):
    """
    Get price forecast for a cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol
        horizon: Forecast horizon in days (1-30)
        
    Returns:
        ForecastResponse: Forecast data with confidence intervals and metrics
    """
    try:
        logger.info(f"Generating {horizon}-day forecast for {symbol}")
        
        # Generate forecast using the forecast service
        forecast_data = await forecast_service.generate_forecast(symbol.value, horizon)
        
        if not forecast_data:
            raise HTTPException(
                status_code=404,
                detail=f"Unable to generate forecast for {symbol}. Model may not be trained."
            )
        
        # Schedule background update of WebSocket connections
        if background_tasks:
            background_tasks.add_task(
                _notify_forecast_update,
                symbol.value,
                forecast_data
            )
        
        return forecast_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate forecast: {str(e)}"
        )


@router.get("/forecast/{symbol}/metrics")
async def get_forecast_metrics(symbol: SymbolEnum):
    """
    Get model performance metrics for a cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol
        
    Returns:
        Model performance metrics
    """
    try:
        logger.info(f"Fetching forecast metrics for {symbol}")
        
        # Get model metrics from forecast service
        metrics = await forecast_service.get_model_metrics(symbol.value)
        
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail=f"No model metrics found for {symbol}. Model may not be trained."
            )
        
        return {
            "symbol": symbol.value,
            "metrics": metrics,
            "retrieved_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching forecast metrics for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch forecast metrics: {str(e)}"
        )


@router.get("/forecast/{symbol}/model-info")
async def get_model_info(symbol: SymbolEnum):
    """
    Get information about the trained model for a cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol
        
    Returns:
        Model information and metadata
    """
    try:
        logger.info(f"Fetching model info for {symbol}")
        
        # Get model information from forecast service
        model_info = await forecast_service.get_model_info(symbol.value)
        
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"No model found for {symbol}. Train a model first."
            )
        
        return {
            "symbol": symbol.value,
            "model_info": model_info,
            "retrieved_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model info for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch model info: {str(e)}"
        )


@router.post("/forecast/{symbol}/refresh")
async def refresh_forecast(
    symbol: SymbolEnum,
    horizon: int = Query(default=7, ge=1, le=30, description="Forecast horizon in days"),
    background_tasks: BackgroundTasks = None
):
    """
    Refresh the forecast for a cryptocurrency using the latest data.
    
    Args:
        symbol: Cryptocurrency symbol
        horizon: Forecast horizon in days
        
    Returns:
        Updated forecast data
    """
    try:
        logger.info(f"Refreshing {horizon}-day forecast for {symbol}")
        
        # Force refresh the forecast with latest data
        forecast_data = await forecast_service.refresh_forecast(symbol.value, horizon)
        
        if not forecast_data:
            raise HTTPException(
                status_code=404,
                detail=f"Unable to refresh forecast for {symbol}. Model may not be trained."
            )
        
        # Schedule background update of WebSocket connections
        if background_tasks:
            background_tasks.add_task(
                _notify_forecast_update,
                symbol.value,
                forecast_data
            )
        
        return {
            "message": f"Forecast refreshed for {symbol}",
            "forecast": forecast_data,
            "refreshed_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing forecast for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh forecast: {str(e)}"
        )


@router.get("/forecast/{symbol}/validate")
async def validate_forecast(
    symbol: SymbolEnum,
    days_back: int = Query(default=30, ge=7, le=90, description="Days to look back for validation")
):
    """
    Validate forecast accuracy by comparing predictions with actual prices.
    
    Args:
        symbol: Cryptocurrency symbol
        days_back: Number of days to look back for validation
        
    Returns:
        Validation results and accuracy metrics
    """
    try:
        logger.info(f"Validating forecast accuracy for {symbol} over {days_back} days")
        
        # Perform forecast validation
        validation_results = await forecast_service.validate_forecast(symbol.value, days_back)
        
        if not validation_results:
            raise HTTPException(
                status_code=404,
                detail=f"Unable to validate forecast for {symbol}. Insufficient data or model not trained."
            )
        
        return {
            "symbol": symbol.value,
            "validation_period_days": days_back,
            "results": validation_results,
            "validated_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating forecast for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate forecast: {str(e)}"
        )


async def _notify_forecast_update(symbol: str, forecast_data):
    """
    Background task to notify WebSocket connections of forecast updates.
    """
    try:
        # Import here to avoid circular imports
        from main import broadcast_forecast_update
        
        # Prepare forecast data for WebSocket broadcast
        forecast_summary = {
            "horizon": forecast_data.horizon,
            "forecast_points": len(forecast_data.forecast),
            "generated_at": forecast_data.generated_at.isoformat(),
            "model_version": forecast_data.model_version,
            "latest_prediction": forecast_data.forecast[-1].dict() if forecast_data.forecast else None
        }
        
        await broadcast_forecast_update(symbol, forecast_summary)
        
    except Exception as e:
        logger.error(f"Error notifying forecast update: {e}")
