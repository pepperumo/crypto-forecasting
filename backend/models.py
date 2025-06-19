"""
Pydantic models for API request/response schemas.
Defines the data structures used throughout the Crypto Foresight application.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


class SymbolEnum(str, Enum):
    """Supported cryptocurrency symbols."""
    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"
    SOLANA = "solana"
    CARDANO = "cardano"
    POLKADOT = "polkadot"
    CHAINLINK = "chainlink"


class SeasonalityEnum(str, Enum):
    """Supported seasonality modes for Prophet model."""
    DAILY = "daily"
    WEEKLY = "weekly"
    YEARLY = "yearly"


class ChartTypeEnum(str, Enum):
    """Supported chart visualization types."""
    LINE = "line"
    CANDLESTICK = "candlestick"
    AREA = "area"


class PriceData(BaseModel):
    """Individual price data point."""
    timestamp: datetime
    price: float
    volume: Optional[float] = None
    market_cap: Optional[float] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OHLCV(BaseModel):
    """OHLCV (Open, High, Low, Close, Volume) data point."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ForecastPoint(BaseModel):
    """Individual forecast data point."""
    timestamp: datetime
    predicted_price: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    mae: Optional[float] = Field(None, description="Mean Absolute Error")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error")
    rmse: Optional[float] = Field(None, description="Root Mean Square Error")
    r2_score: Optional[float] = Field(None, description="R-squared Score")
    training_time: Optional[float] = Field(None, description="Training time in seconds")
    last_trained: Optional[datetime] = Field(None, description="Last training timestamp")
    data_points: Optional[int] = Field(None, description="Number of training data points")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PriceResponse(BaseModel):
    """Response model for price data."""
    symbol: str
    data: List[PriceData]
    count: int
    last_updated: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OHLCVResponse(BaseModel):
    """Response model for OHLCV data."""
    symbol: str
    data: List[OHLCV]
    count: int
    last_updated: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ForecastResponse(BaseModel):
    """Response model for forecast data."""
    symbol: str
    horizon: int
    forecast: List[ForecastPoint]
    metrics: Optional[ModelMetrics] = None
    generated_at: datetime
    model_version: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingRequest(BaseModel):
    """Request model for training a model."""
    symbol: SymbolEnum
    horizon: int = Field(default=7, ge=1, le=30, description="Forecast horizon in days")
    seasonality: SeasonalityEnum = Field(default=SeasonalityEnum.WEEKLY)
    changepoint_prior_scale: float = Field(default=0.05, ge=0.001, le=0.5)
    n_iter: int = Field(default=1000, ge=100, le=5000)
    include_features: bool = Field(default=True, description="Include technical indicators")
    
    @validator('changepoint_prior_scale')
    def validate_changepoint_prior(cls, v):
        if v <= 0 or v > 0.5:
            raise ValueError('changepoint_prior_scale must be between 0.001 and 0.5')
        return v


class TrainingStatus(BaseModel):
    """Model training status."""
    task_id: str
    status: Literal["pending", "running", "completed", "failed"]
    progress: Optional[float] = Field(None, ge=0, le=100)
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingResponse(BaseModel):
    """Response model for training initiation."""
    task_id: str
    message: str
    status_url: str


class WebSocketMessage(BaseModel):
    """WebSocket message structure."""
    type: Literal["price_update", "forecast_update", "error", "heartbeat"]
    symbol: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "unhealthy"]
    timestamp: datetime
    version: str
    uptime: float
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
