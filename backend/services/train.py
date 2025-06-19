"""
Training service for cryptocurrency price forecasting models.
Handles model training, validation, and persistence.
"""

import logging
import os
import pickle
import yaml
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Using sklearn models only.")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models import ModelMetrics
from services.market import MarketDataService
from services.features import FeatureService

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for training cryptocurrency forecasting models."""
    
    def __init__(self):
        self.market_service = MarketDataService()
        self.feature_service = FeatureService()
        self.artifacts_path = os.getenv("ARTIFACTS_PATH", "./artifacts")
        self.config_path = os.getenv("CONFIG_PATH", "./config")
        
        # Ensure directories exist
        os.makedirs(self.artifacts_path, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
    
    def train_model(
        self,
        symbol: str,
        horizon: int = 7,
        seasonality: str = "weekly",
        changepoint_prior_scale: float = 0.05,
        n_iter: int = 1000,
        include_features: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Train a forecasting model for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            horizon: Forecast horizon in days
            seasonality: Seasonality mode ('daily', 'weekly', 'yearly')
            changepoint_prior_scale: Prophet changepoint prior scale
            n_iter: Number of iterations for Prophet
            include_features: Whether to include technical indicators
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            if progress_callback:
                progress_callback(10.0, "Starting model training...")
            
            logger.info(f"Training model for {symbol} with horizon {horizon}")
            
            # Fetch training data
            if progress_callback:
                progress_callback(20.0, "Fetching training data...")
            
            training_data = self._fetch_training_data(symbol)
            if not training_data:
                raise Exception(f"Failed to fetch training data for {symbol}")
            
            # Prepare data
            if progress_callback:
                progress_callback(40.0, "Preparing features...")
            
            features_df = self._prepare_training_data(training_data, include_features)
            if features_df.empty:
                raise Exception("Failed to prepare training data")
            
            # Choose model type based on configuration and availability
            use_prophet = PROPHET_AVAILABLE and self.config.get("model", {}).get("use_prophet", True)
            
            if use_prophet:
                result = self._train_prophet_model(
                    features_df, symbol, horizon, seasonality,
                    changepoint_prior_scale, n_iter, progress_callback
                )
            else:
                result = self._train_sklearn_model(
                    features_df, symbol, horizon, include_features, progress_callback
                )
            
            # Save model
            if progress_callback:
                progress_callback(90.0, "Saving model...")
            
            self._save_model(symbol, result)
            
            if progress_callback:
                progress_callback(100.0, "Training completed successfully")
            
            logger.info(f"Model training completed for {symbol}")
            
            return {
                "symbol": symbol,
                "status": "completed",
                "model_type": result["type"],
                "metrics": result["metrics"],
                "trained_at": datetime.now(),
                "horizon": horizon
            }
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            if progress_callback:
                progress_callback(0.0, f"Training failed: {str(e)}")
            raise
    
    async def list_trained_models(self) -> List[Dict[str, Any]]:
        """List all trained models."""
        try:
            models = []
            
            if not os.path.exists(self.artifacts_path):
                return models
            
            for symbol_dir in os.listdir(self.artifacts_path):
                symbol_path = os.path.join(self.artifacts_path, symbol_dir)
                if os.path.isdir(symbol_path):
                    model_path = os.path.join(symbol_path, "model.pkl")
                    if os.path.exists(model_path):
                        try:
                            # Load model metadata
                            with open(model_path, 'rb') as f:
                                model_data = pickle.load(f)
                            
                            models.append({
                                "symbol": symbol_dir,
                                "type": model_data.get("type", "unknown"),
                                "version": model_data.get("version", "1.0"),
                                "trained_at": model_data.get("trained_at"),
                                "horizon": model_data.get("horizon"),
                                "metrics": model_data.get("metrics"),
                                "file_size": os.path.getsize(model_path)
                            })
                        except Exception as e:
                            logger.error(f"Error loading model metadata for {symbol_dir}: {e}")
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing trained models: {e}")
            return []
    
    async def delete_model(self, symbol: str) -> bool:
        """Delete a trained model."""
        try:
            symbol_path = os.path.join(self.artifacts_path, symbol)
            model_path = os.path.join(symbol_path, "model.pkl")
            
            if os.path.exists(model_path):
                os.remove(model_path)
                
                # Remove directory if empty
                if os.path.exists(symbol_path) and not os.listdir(symbol_path):
                    os.rmdir(symbol_path)
                
                logger.info(f"Deleted model for {symbol}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting model for {symbol}: {e}")
            return False
    
    def _fetch_training_data(self, symbol: str, days: int = 365) -> List:
        """Fetch training data for a symbol."""
        try:
            # This is a synchronous wrapper for the async market service
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                data = loop.run_until_complete(
                    self.market_service.get_price_history(symbol, days)
                )
                return data
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error fetching training data for {symbol}: {e}")
            return []
    
    def _prepare_training_data(self, price_data: List, include_features: bool = True) -> pd.DataFrame:
        """Prepare training data with features."""
        try:
            if include_features:
                # Generate features using the feature service
                features_df = self.feature_service.generate_features(price_data)
            else:
                # Simple price data only
                features_df = pd.DataFrame([
                    {
                        'timestamp': point.timestamp,
                        'price': point.price
                    }
                    for point in price_data
                ])
            
            # Ensure we have enough data
            if len(features_df) < 30:
                raise Exception("Insufficient data for training (minimum 30 days required)")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame()
    
    def _train_prophet_model(
        self,
        features_df: pd.DataFrame,
        symbol: str,
        horizon: int,
        seasonality: str,
        changepoint_prior_scale: float,
        n_iter: int,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Train a Prophet model."""
        try:
            if progress_callback:
                progress_callback(50.0, "Training Prophet model...")
            
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': features_df['timestamp'],
                'y': features_df['price']
            })
            
            # Configure Prophet model
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_mode='multiplicative',
                yearly_seasonality=(seasonality == 'yearly'),
                weekly_seasonality=(seasonality in ['weekly', 'yearly']),
                daily_seasonality=(seasonality == 'daily'),
                n_iter=n_iter
            )
            
            # Fit the model
            model.fit(prophet_df)
            
            if progress_callback:
                progress_callback(70.0, "Evaluating model performance...")
            
            # Calculate metrics using cross-validation
            metrics = self._calculate_prophet_metrics(model, prophet_df)
            
            return {
                "type": "prophet",
                "model": model,
                "metrics": metrics,
                "horizon": horizon,
                "version": "1.0",
                "trained_at": datetime.now(),
                "config": {
                    "changepoint_prior_scale": changepoint_prior_scale,
                    "seasonality": seasonality,
                    "n_iter": n_iter
                }
            }
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            raise
    
    def _train_sklearn_model(
        self,
        features_df: pd.DataFrame,
        symbol: str,
        horizon: int,
        include_features: bool,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Train a scikit-learn model."""
        try:
            if progress_callback:
                progress_callback(50.0, "Training scikit-learn model...")
              # Prepare features and target
            target_col = 'price'
            feature_cols = [col for col in features_df.columns if col not in ['timestamp', target_col]]
            if not feature_cols:
                raise Exception("No features available for training")
            
            X = features_df[feature_cols].fillna(0)
            y = features_df[target_col].fillna(method='ffill')
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model (using RandomForest as it handles features well)
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            if progress_callback:
                progress_callback(70.0, "Evaluating model performance...")
            
            # Calculate metrics
            y_pred = model.predict(X_test_scaled)
            metrics = self._calculate_sklearn_metrics(y_test, y_pred)
            
            return {
                "type": "sklearn",
                "model": model,
                "scaler": scaler,
                "features": feature_cols,
                "metrics": metrics,
                "horizon": horizon,
                "version": "1.0",
                "trained_at": datetime.now(),
                "config": {
                    "model_type": "RandomForestRegressor",
                    "include_features": include_features
                }
            }
            
        except Exception as e:
            logger.error(f"Error training sklearn model: {e}")
            raise
    
    def _calculate_prophet_metrics(self, model, df: pd.DataFrame) -> ModelMetrics:
        """Calculate metrics for Prophet model."""
        try:
            # Simple holdout validation
            train_size = int(len(df) * 0.8)
            train_df = df[:train_size]
            test_df = df[train_size:]
            
            # Retrain on training data
            temp_model = Prophet()
            temp_model.fit(train_df)
            
            # Predict on test data
            forecast = temp_model.predict(test_df[['ds']])
            
            # Calculate metrics
            y_true = test_df['y'].values
            y_pred = forecast['yhat'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            return ModelMetrics(
                mae=mae,
                mape=mape,
                rmse=rmse,
                r2_score=r2,
                training_time=0.0,  # Would need to track actual training time
                last_trained=datetime.now(),
                data_points=len(df)
            )
            
        except Exception as e:
            logger.error(f"Error calculating Prophet metrics: {e}")
            return ModelMetrics(last_trained=datetime.now(), data_points=len(df))
    
    def _calculate_sklearn_metrics(self, y_true, y_pred) -> ModelMetrics:
        """Calculate metrics for sklearn model."""
        try:
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            return ModelMetrics(
                mae=mae,
                mape=mape,
                rmse=rmse,
                r2_score=r2,
                training_time=0.0,
                last_trained=datetime.now(),
                data_points=len(y_true)
            )
            
        except Exception as e:
            logger.error(f"Error calculating sklearn metrics: {e}")
            return ModelMetrics(last_trained=datetime.now(), data_points=len(y_true))
    
    def _save_model(self, symbol: str, model_data: Dict[str, Any]):
        """Save trained model to disk."""
        try:
            symbol_path = os.path.join(self.artifacts_path, symbol)
            os.makedirs(symbol_path, exist_ok=True)
            
            model_path = os.path.join(symbol_path, "model.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved for {symbol} at {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model for {symbol}: {e}")
            raise
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files."""
        try:
            config = {}
            
            # Load default config
            default_config_path = os.path.join(self.config_path, "default.yaml")
            if os.path.exists(default_config_path):
                with open(default_config_path, 'r') as f:
                    config = yaml.safe_load(f)
            
            # Load dev config if exists (overwrites default)
            dev_config_path = os.path.join(self.config_path, "dev.yaml")
            if os.path.exists(dev_config_path):
                with open(dev_config_path, 'r') as f:
                    dev_config = yaml.safe_load(f)
                    config.update(dev_config)
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}


def main():
    """CLI entry point for training models."""
    parser = argparse.ArgumentParser(description="Train cryptocurrency forecasting models")
    parser.add_argument("--symbol", required=True, help="Cryptocurrency symbol")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon in days")
    parser.add_argument("--seasonality", default="weekly", choices=["daily", "weekly", "yearly"])
    parser.add_argument("--changepoint-prior", type=float, default=0.05, help="Prophet changepoint prior scale")
    parser.add_argument("--n-iter", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--no-features", action="store_true", help="Disable feature engineering")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize training service
    training_service = TrainingService()
    
    def progress_callback(progress: float, message: str):
        print(f"[{progress:5.1f}%] {message}")
    
    try:
        # Train the model
        result = training_service.train_model(
            symbol=args.symbol,
            horizon=args.horizon,
            seasonality=args.seasonality,
            changepoint_prior_scale=args.changepoint_prior,
            n_iter=args.n_iter,
            include_features=not args.no_features,
            progress_callback=progress_callback
        )
        
        print(f"\nTraining completed successfully!")
        print(f"Symbol: {result['symbol']}")
        print(f"Model Type: {result['model_type']}")
        print(f"Horizon: {result['horizon']} days")
        
        if result.get('metrics'):
            metrics = result['metrics']
            print(f"\nModel Metrics:")
            if metrics.mae:
                print(f"  MAE: {metrics.mae:.4f}")
            if metrics.mape:
                print(f"  MAPE: {metrics.mape:.2f}%")
            if metrics.rmse:
                print(f"  RMSE: {metrics.rmse:.4f}")
            if metrics.r2_score:
                print(f"  R²: {metrics.r2_score:.4f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
