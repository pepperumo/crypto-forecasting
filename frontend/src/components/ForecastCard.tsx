/**
 * Forecast card component showing model metrics and predictions
 */

import React from 'react';

interface ForecastData {
  symbol: string;
  horizon: number;
  forecast: Array<{
    timestamp: string;
    predicted_price: number;
    lower_bound?: number;
    upper_bound?: number;
  }>;
  metrics?: {
    mae?: number;
    mape?: number;
    rmse?: number;
    r2_score?: number;
    last_trained?: string;
  };
  generated_at: string;
}

interface ForecastCardProps {
  forecastData: ForecastData | null;
  symbol: string;
  horizon: number;
  onRefresh: () => void;
  onTrain: () => void;
  isLoading: boolean;
}

export const ForecastCard: React.FC<ForecastCardProps> = ({
  forecastData,
  symbol,
  horizon,
  onRefresh,
  onTrain,
  isLoading
}) => {
  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(price);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-slate-200">
          {horizon}-Day Forecast
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="btn-ghost p-2"
            title="Refresh forecast"
          >
            <svg className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
          <button
            onClick={onTrain}
            disabled={isLoading}
            className="btn-secondary px-3 py-1 text-xs"
          >
            Retrain
          </button>
        </div>
      </div>

      {forecastData ? (
        <div className="space-y-6">
          {/* Latest Prediction */}
          <div>
            <h4 className="text-sm font-medium text-slate-400 mb-3">Next Price Prediction</h4>
            <div className="bg-slate-700/30 rounded-xl p-4">
              <div className="text-2xl font-bold text-slate-200 mb-1">
                {formatPrice(forecastData.forecast[0]?.predicted_price || 0)}
              </div>
              <div className="text-sm text-slate-400">
                {formatDate(forecastData.forecast[0]?.timestamp || '')}
              </div>
              {forecastData.forecast[0]?.lower_bound && forecastData.forecast[0]?.upper_bound && (
                <div className="text-xs text-slate-500 mt-2">
                  Range: {formatPrice(forecastData.forecast[0].lower_bound)} - {formatPrice(forecastData.forecast[0].upper_bound)}
                </div>
              )}
            </div>
          </div>

          {/* Model Metrics */}
          {forecastData.metrics && (
            <div>
              <h4 className="text-sm font-medium text-slate-400 mb-3">Model Performance</h4>
              <div className="grid grid-cols-2 gap-3">
                {forecastData.metrics.mae && (
                  <div className="bg-slate-700/30 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">MAE</div>
                    <div className="text-sm font-medium text-slate-200">
                      {forecastData.metrics.mae.toFixed(2)}
                    </div>
                  </div>
                )}
                {forecastData.metrics.mape && (
                  <div className="bg-slate-700/30 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">MAPE</div>
                    <div className="text-sm font-medium text-slate-200">
                      {forecastData.metrics.mape.toFixed(1)}%
                    </div>
                  </div>
                )}
                {forecastData.metrics.r2_score && (
                  <div className="bg-slate-700/30 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">R² Score</div>
                    <div className="text-sm font-medium text-slate-200">
                      {formatPercentage(forecastData.metrics.r2_score)}
                    </div>
                  </div>
                )}
                {forecastData.metrics.rmse && (
                  <div className="bg-slate-700/30 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">RMSE</div>
                    <div className="text-sm font-medium text-slate-200">
                      {forecastData.metrics.rmse.toFixed(2)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Forecast Summary */}
          <div>
            <h4 className="text-sm font-medium text-slate-400 mb-3">Forecast Summary</h4>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Horizon:</span>
                <span className="text-slate-200">{forecastData.horizon} days</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Generated:</span>
                <span className="text-slate-200">{formatDate(forecastData.generated_at)}</span>
              </div>
              {forecastData.metrics?.last_trained && (
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Last Trained:</span>
                  <span className="text-slate-200">{formatDate(forecastData.metrics.last_trained)}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-8">
          <div className="w-16 h-16 bg-slate-700/30 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h4 className="text-lg font-medium text-slate-200 mb-2">No Forecast Available</h4>
          <p className="text-slate-400 mb-4">Train a model to generate price predictions</p>
          <button
            onClick={onTrain}
            disabled={isLoading}
            className="btn-primary"
          >
            {isLoading ? 'Training...' : 'Train Model'}
          </button>
        </div>
      )}
    </div>
  );
};
