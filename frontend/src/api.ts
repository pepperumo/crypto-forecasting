/**
 * API service for communicating with the Crypto Foresight backend
 * Handles HTTP requests and response formatting
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';

// Types
interface PriceData {
  timestamp: string;
  price: number;
  volume?: number;
  market_cap?: number;
}

interface PriceResponse {
  symbol: string;
  data: PriceData[];
  count: number;
  last_updated: string;
}

interface ForecastPoint {
  timestamp: string;
  predicted_price: number;
  lower_bound?: number;
  upper_bound?: number;
  confidence?: number;
}

interface ModelMetrics {
  mae?: number;
  mape?: number;
  rmse?: number;
  r2_score?: number;
  training_time?: number;
  last_trained?: string;
  data_points?: number;
}

interface ForecastResponse {
  symbol: string;
  horizon: number;
  forecast: ForecastPoint[];
  metrics?: ModelMetrics;
  generated_at: string;
  model_version?: string;
}

interface TrainingResponse {
  task_id: string;
  message: string;
  status_url: string;
}

interface TrainingStatus {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  message?: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
  result?: any;
}

class ApiService {
  private client: AxiosInstance;
  private baseURL: string;

  constructor() {
    this.baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('[API] Request error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        console.log(`[API] Response: ${response.status} ${response.config.url}`);
        return response;
      },      (error) => {
        console.error('[API] Response error:', error.response?.data || error.message);
        
        // Handle common errors
        if (error.response?.status === 404) {
          // Check if this is the forecast endpoint and handle it specially
          if (error.config?.url?.includes('/api/forecast/')) {
            const errorData = error.response?.data;
            if (errorData?.error?.includes('Model may not be trained')) {
              throw new Error('No forecast model trained yet. Please train a model first.');
            }
          }
          throw new Error('Resource not found');
        } else if (error.response?.status === 500) {
          throw new Error('Server error occurred');
        } else if (error.code === 'ECONNABORTED') {
          throw new Error('Request timeout');
        } else if (!error.response) {
          throw new Error('Network error - please check your connection');
        }
        
        throw new Error(error.response?.data?.detail || error.message || 'Unknown error');
      }
    );
  }

  /**
   * Get historical price data for a cryptocurrency
   */
  async getPrices(symbol: string, days: number = 30): Promise<PriceResponse> {
    const response = await this.client.get<PriceResponse>(`/api/prices/${symbol}`, {
      params: { days }
    });
    return response.data;
  }

  /**
   * Get OHLCV data for a cryptocurrency
   */
  async getOHLCV(symbol: string, days: number = 30, interval: string = 'daily') {
    const response = await this.client.get(`/api/ohlcv/${symbol}`, {
      params: { days, interval }
    });
    return response.data;
  }

  /**
   * Get current price for a cryptocurrency
   */
  async getCurrentPrice(symbol: string) {
    const response = await this.client.get(`/api/prices/${symbol}/current`);
    return response.data;
  }

  /**
   * Get price statistics for a cryptocurrency
   */
  async getPriceStats(symbol: string, days: number = 30) {
    const response = await this.client.get(`/api/prices/${symbol}/stats`, {
      params: { days }
    });
    return response.data;
  }

  /**
   * Get forecast for a cryptocurrency
   */
  async getForecast(symbol: string, horizon: number = 7): Promise<ForecastResponse> {
    const response = await this.client.get<ForecastResponse>(`/api/forecast/${symbol}`, {
      params: { horizon }
    });
    return response.data;
  }

  /**
   * Refresh forecast with latest data
   */
  async refreshForecast(symbol: string, horizon: number = 7): Promise<ForecastResponse> {
    const response = await this.client.post<ForecastResponse>(`/api/forecast/${symbol}/refresh`, null, {
      params: { horizon }
    });
    return response.data;
  }

  /**
   * Get model metrics for a cryptocurrency
   */
  async getModelMetrics(symbol: string) {
    const response = await this.client.get(`/api/forecast/${symbol}/metrics`);
    return response.data;
  }

  /**
   * Get model information
   */
  async getModelInfo(symbol: string) {
    const response = await this.client.get(`/api/forecast/${symbol}/model-info`);
    return response.data;
  }

  /**
   * Validate forecast accuracy
   */
  async validateForecast(symbol: string, daysBack: number = 30) {
    const response = await this.client.get(`/api/forecast/${symbol}/validate`, {
      params: { days_back: daysBack }
    });
    return response.data;
  }

  /**
   * Start model training
   */
  async trainModel(
    symbol: string,
    horizon: number = 7,
    seasonality: string = 'weekly',
    changepointPrior: number = 0.05,
    nIter: number = 1000,
    includeFeatures: boolean = true
  ): Promise<TrainingResponse> {
    const response = await this.client.post<TrainingResponse>('/api/train', {
      symbol,
      horizon,
      seasonality,
      changepoint_prior_scale: changepointPrior,
      n_iter: nIter,
      include_features: includeFeatures
    });
    return response.data;
  }

  /**
   * Get training status
   */
  async getTrainingStatus(taskId: string): Promise<TrainingStatus> {
    const response = await this.client.get<TrainingStatus>(`/api/status/${taskId}`);
    return response.data;
  }

  /**
   * Get all training status
   */
  async getAllTrainingStatus() {
    const response = await this.client.get('/api/status');
    return response.data;
  }

  /**
   * Cancel training task
   */
  async cancelTraining(taskId: string) {
    const response = await this.client.delete(`/api/status/${taskId}`);
    return response.data;
  }

  /**
   * Retrain model with default parameters
   */
  async retrainModel(symbol: string, horizon: number = 7) {
    const response = await this.client.post(`/api/retrain/${symbol}`, null, {
      params: { horizon }
    });
    return response.data;
  }

  /**
   * List all trained models
   */
  async listModels() {
    const response = await this.client.get('/api/models');
    return response.data;
  }

  /**
   * Delete a trained model
   */
  async deleteModel(symbol: string) {
    const response = await this.client.delete(`/api/models/${symbol}`);
    return response.data;
  }

  /**
   * Get system information
   */
  async getSystemInfo() {
    const response = await this.client.get('/api/system/info');
    return response.data;
  }

  /**
   * Health check
   */
  async healthCheck() {
    const response = await this.client.get('/health');
    return response.data;
  }

  /**
   * Get WebSocket URL for real-time updates
   */
  getWebSocketUrl(symbol: string): string {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
    return `${wsUrl}/ws/prices/${symbol}`;
  }
}

// Export singleton instance
export const api = new ApiService();

// Export types for use in components
export type {
  PriceData,
  PriceResponse,
  ForecastPoint,
  ForecastResponse,
  ModelMetrics,
  TrainingResponse,
  TrainingStatus,
};
