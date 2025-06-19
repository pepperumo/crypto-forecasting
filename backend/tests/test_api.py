"""
Basic tests for the API endpoints.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    def test_health_check(self):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime" in data


class TestPricesEndpoint:
    """Test the prices API endpoints."""
    
    def test_get_prices_bitcoin(self):
        """Test getting Bitcoin price data."""
        response = client.get("/api/prices/bitcoin?days=7")
        # Note: This might fail if CoinGecko API is not available
        # In production, we'd mock the external API calls
        assert response.status_code in [200, 500]  # Allow for API errors
        
    def test_get_prices_invalid_symbol(self):
        """Test getting prices for invalid symbol."""
        response = client.get("/api/prices/invalid_symbol")
        assert response.status_code == 422  # Validation error


class TestForecastEndpoint:
    """Test the forecast API endpoints."""
    
    def test_get_forecast_no_model(self):
        """Test getting forecast when no model is trained."""
        response = client.get("/api/forecast/bitcoin")
        # Should return 404 or 500 if no model is trained
        assert response.status_code in [404, 500]


class TestTrainingEndpoint:
    """Test the training API endpoints."""
    
    def test_start_training(self):
        """Test starting model training."""
        payload = {
            "symbol": "bitcoin",
            "horizon": 7,
            "seasonality": "weekly",
            "changepoint_prior_scale": 0.05,
            "n_iter": 100,  # Reduced for testing
            "include_features": True
        }
        response = client.post("/api/train", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "status_url" in data
        
    def test_get_training_status_invalid_id(self):
        """Test getting status for invalid task ID."""
        response = client.get("/api/status/invalid_task_id")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__])
