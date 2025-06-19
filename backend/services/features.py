"""
Feature engineering service for cryptocurrency data.
Generates technical indicators and derived features for model training.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from models import PriceData, OHLCV

logger = logging.getLogger(__name__)


class FeatureService:
    """Service for generating features from cryptocurrency price data."""
    
    def __init__(self):
        self.default_config = {
            "sma_windows": [7, 14, 30],
            "ema_windows": [12, 26],
            "rsi_period": 14,
            "return_periods": [1, 3, 7],
            "volatility_window": 30
        }
    
    def generate_features(self, price_data: List[PriceData], config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate features from price data.
        
        Args:
            price_data: List of PriceData objects
            config: Feature configuration (optional)
            
        Returns:
            DataFrame with engineered features
        """
        if not price_data:
            return pd.DataFrame()
        
        config = config or self.default_config
        
        # Convert to DataFrame
        df = self._price_data_to_dataframe(price_data)
        
        if df.empty:
            return df
        
        logger.info(f"Generating features from {len(df)} price points")
        
        # Sort by timestamp to ensure proper order
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Generate base features
        df = self._add_price_features(df, config)
        
        # Generate technical indicators
        df = self._add_technical_indicators(df, config)
        
        # Generate time-based features
        df = self._add_time_features(df)
        
        # Generate lag features
        df = self._add_lag_features(df, config)
        
        # Drop rows with NaN values (from rolling windows)
        df = df.dropna().reset_index(drop=True)
        
        logger.info(f"Generated {len(df.columns)} features for {len(df)} data points")
        
        return df
    
    def generate_features_from_ohlcv(self, ohlcv_data: List[OHLCV], config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate features from OHLCV data.
        
        Args:
            ohlcv_data: List of OHLCV objects
            config: Feature configuration (optional)
            
        Returns:
            DataFrame with engineered features
        """
        if not ohlcv_data:
            return pd.DataFrame()
        
        config = config or self.default_config
        
        # Convert to DataFrame
        df = self._ohlcv_data_to_dataframe(ohlcv_data)
        
        if df.empty:
            return df
        
        logger.info(f"Generating features from {len(df)} OHLCV points")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Generate OHLCV-specific features
        df = self._add_ohlcv_features(df, config)
        
        # Generate technical indicators (using close price)
        df = self._add_technical_indicators(df, config, price_col='close')
        
        # Generate time-based features
        df = self._add_time_features(df)
        
        # Generate lag features
        df = self._add_lag_features(df, config, price_col='close')
        
        # Drop rows with NaN values
        df = df.dropna().reset_index(drop=True)
        
        logger.info(f"Generated {len(df.columns)} features for {len(df)} data points")
        
        return df
    
    def _price_data_to_dataframe(self, price_data: List[PriceData]) -> pd.DataFrame:
        """Convert PriceData list to DataFrame."""
        data = []
        for price_point in price_data:
            data.append({
                'timestamp': price_point.timestamp,
                'price': price_point.price,
                'volume': price_point.volume or 0,
                'market_cap': price_point.market_cap or 0
            })
        
        return pd.DataFrame(data)
    
    def _ohlcv_data_to_dataframe(self, ohlcv_data: List[OHLCV]) -> pd.DataFrame:
        """Convert OHLCV list to DataFrame."""
        data = []
        for ohlcv_point in ohlcv_data:
            data.append({
                'timestamp': ohlcv_point.timestamp,
                'open': ohlcv_point.open,
                'high': ohlcv_point.high,
                'low': ohlcv_point.low,
                'close': ohlcv_point.close,
                'volume': ohlcv_point.volume
            })
        
        return pd.DataFrame(data)
    
    def _add_price_features(self, df: pd.DataFrame, config: Dict, price_col: str = 'price') -> pd.DataFrame:
        """Add price-based features."""
        if price_col not in df.columns:
            return df
        
        # Price returns
        for period in config.get('return_periods', [1, 3, 7]):
            if len(df) > period:
                df[f'return_{period}d'] = df[price_col].pct_change(period)
        
        # Log returns
        df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Price volatility (rolling standard deviation of returns)
        vol_window = config.get('volatility_window', 30)
        if len(df) > vol_window:
            df[f'volatility_{vol_window}d'] = df['log_return'].rolling(window=vol_window).std()
        
        # Price momentum features
        df['price_momentum_3d'] = df[price_col] / df[price_col].shift(3) - 1
        df['price_momentum_7d'] = df[price_col] / df[price_col].shift(7) - 1
        
        return df
    
    def _add_ohlcv_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Add OHLCV-specific features."""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return df
        
        # Price range features
        df['price_range'] = df['high'] - df['low']
        df['price_range_pct'] = (df['high'] - df['low']) / df['close']
        
        # Body and shadow features (candlestick analysis)
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_size_pct'] = df['body_size'] / df['close']
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        
        # Gap features
        df['gap_up'] = (df['open'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['low'].shift(1)).astype(int)
        
        # True Range (for ATR calculation)
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Average True Range
        df['atr_14'] = df['true_range'].rolling(window=14).mean()
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame, config: Dict, price_col: str = 'price') -> pd.DataFrame:
        """Add technical indicators."""
        if price_col not in df.columns:
            return df
        
        # Simple Moving Averages
        for window in config.get('sma_windows', [7, 14, 30]):
            if len(df) > window:
                df[f'sma_{window}'] = df[price_col].rolling(window=window).mean()
                df[f'price_to_sma_{window}'] = df[price_col] / df[f'sma_{window}']
        
        # Exponential Moving Averages
        for window in config.get('ema_windows', [12, 26]):
            if len(df) > window:
                df[f'ema_{window}'] = df[price_col].ewm(span=window).mean()
                df[f'price_to_ema_{window}'] = df[price_col] / df[f'ema_{window}']
        
        # RSI (Relative Strength Index)
        rsi_period = config.get('rsi_period', 14)
        if len(df) > rsi_period:
            df['rsi'] = self._calculate_rsi(df[price_col], rsi_period)
        
        # MACD
        if len(df) > 26:
            ema_12 = df[price_col].ewm(span=12).mean()
            ema_26 = df[price_col].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        if len(df) > 20:
            sma_20 = df[price_col].rolling(window=20).mean()
            std_20 = df[price_col].rolling(window=20).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_lower'] = sma_20 - (std_20 * 2)
            df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'timestamp' not in df.columns:
            return df
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time components
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['year'] = df['timestamp'].dt.year
        
        # Cyclical encoding for temporal features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, config: Dict, price_col: str = 'price') -> pd.DataFrame:
        """Add lagged features."""
        if price_col not in df.columns:
            return df
        
        # Price lags
        for lag in [1, 2, 3, 7, 14]:
            if len(df) > lag:
                df[f'{price_col}_lag_{lag}'] = df[price_col].shift(lag)
        
        # Return lags
        if 'log_return' in df.columns:
            for lag in [1, 2, 3]:
                if len(df) > lag:
                    df[f'log_return_lag_{lag}'] = df['log_return'].shift(lag)
        
        # Volume lags (if available)
        if 'volume' in df.columns:
            for lag in [1, 3, 7]:
                if len(df) > lag:
                    df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'price') -> Dict[str, float]:
        """
        Calculate feature importance using correlation with target.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        if target_col not in df.columns:
            return {}
        
        # Calculate correlations
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        
        # Remove the target itself and NaN values
        correlations = correlations.drop(target_col, errors='ignore').dropna()
        
        return correlations.to_dict()
    
    def select_features(self, df: pd.DataFrame, target_col: str = 'price', max_features: int = 20) -> List[str]:
        """
        Select top features based on correlation with target.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        feature_importance = self.get_feature_importance(df, target_col)
        
        # Select top features
        selected_features = list(feature_importance.keys())[:max_features]
        
        # Always include timestamp if available
        if 'timestamp' in df.columns and 'timestamp' not in selected_features:
            selected_features = ['timestamp'] + selected_features
        
        logger.info(f"Selected {len(selected_features)} features for modeling")
        
        return selected_features
