import React, { useState, useEffect } from 'react';
import { Header, TickerSearch, PriceChart, ForecastCard } from './components';
import { useWebSocket } from './hooks';
import { api } from './api';

// Types
interface CryptoSymbol {
  id: string;
  name: string;
  symbol: string;
  current_price?: number;
  price_change_24h?: number;
  price_change_percentage_24h?: number;
}

interface PriceData {
  timestamp: string;
  price: number;
  volume?: number;
}

interface ForecastData {
  symbol: string;
  horizon: number;
  forecast: Array<{
    timestamp: string;
    predicted_price: number;
    lower_bound?: number;
    upper_bound?: number;
    confidence?: number;
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

const DEFAULT_SYMBOLS = ['bitcoin', 'ethereum', 'solana'];

function App() {
  // State management
  const [selectedSymbol, setSelectedSymbol] = useState<string>('bitcoin');
  const [priceData, setPriceData] = useState<PriceData[]>([]);
  const [forecastData, setForecastData] = useState<ForecastData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chartType, setChartType] = useState<'line' | 'area'>('line');
  const [forecastHorizon, setForecastHorizon] = useState(7);

  // WebSocket connection for real-time updates
  const { 
    isConnected, 
    lastMessage, 
    connect, 
    disconnect 
  } = useWebSocket(selectedSymbol);

  // Fetch initial data
  useEffect(() => {
    fetchPriceData(selectedSymbol);
    fetchForecastData(selectedSymbol, forecastHorizon);
  }, [selectedSymbol, forecastHorizon]);

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage && lastMessage.type === 'price_update') {
      // Update price data with real-time data
      const newPricePoint: PriceData = {
        timestamp: lastMessage.timestamp,
        price: lastMessage.data.price,
        volume: lastMessage.data.volume
      };
      
      setPriceData(prev => [...prev.slice(-199), newPricePoint]); // Keep last 200 points
    }
  }, [lastMessage]);

  // Connect to WebSocket when symbol changes
  useEffect(() => {
    if (selectedSymbol) {
      connect();
    }
    
    return () => {
      disconnect();
    };
  }, [selectedSymbol, connect, disconnect]);

  const fetchPriceData = async (symbol: string, days: number = 30) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await api.getPrices(symbol, days);
      setPriceData(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch price data');
      console.error('Error fetching price data:', err);
    } finally {
      setIsLoading(false);
    }
  };
  const fetchForecastData = async (symbol: string, horizon: number) => {
    try {
      setError(null);
      
      const response = await api.getForecast(symbol, horizon);
      setForecastData(response);
    } catch (err) {
      // Check if it's a model not trained error
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch forecast data';
      
      if (errorMessage.includes('No forecast model trained yet')) {
        // Set a more friendly message but don't show it as an error
        console.log('No model trained yet, showing placeholder');
        setForecastData(null); // Clear any previous forecast data
      } else {
        // For other errors, display the error message
        setError(errorMessage);
        console.error('Error fetching forecast data:', err);
      }
    }
  };

  const handleSymbolSelect = (symbol: string) => {
    setSelectedSymbol(symbol);
  };

  const handleRefreshForecast = async () => {
    if (selectedSymbol) {
      await fetchForecastData(selectedSymbol, forecastHorizon);
    }
  };

  const handleTrainModel = async () => {
    try {
      setError(null);
      await api.trainModel(selectedSymbol, forecastHorizon);
      // Refresh forecast after training
      setTimeout(() => {
        fetchForecastData(selectedSymbol, forecastHorizon);
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start model training');
      console.error('Error training model:', err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <Header 
        isConnected={isConnected}
        onThemeToggle={() => {
          // Theme toggle logic would go here
          console.log('Theme toggle clicked');
        }}
      />

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Search and Controls */}
        <div className="mb-8">
          <TickerSearch
            onSymbolSelect={handleSymbolSelect}
            selectedSymbol={selectedSymbol}
            defaultSymbols={DEFAULT_SYMBOLS}
          />
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400">
            <p className="font-medium">Error:</p>
            <p className="text-sm">{error}</p>
          </div>
        )}

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Chart Section */}
          <div className="xl:col-span-2">
            <div className="card p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold text-slate-200">
                  {selectedSymbol.charAt(0).toUpperCase() + selectedSymbol.slice(1)} Price Chart
                </h2>
                
                <div className="flex items-center gap-4">
                  {/* Chart Type Selector */}
                  <div className="flex bg-slate-700/50 rounded-lg p-1">
                    <button
                      onClick={() => setChartType('line')}
                      className={`px-3 py-1 rounded text-sm transition-all ${
                        chartType === 'line' 
                          ? 'bg-purple-600 text-white' 
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Line
                    </button>
                    <button
                      onClick={() => setChartType('area')}
                      className={`px-3 py-1 rounded text-sm transition-all ${
                        chartType === 'area' 
                          ? 'bg-purple-600 text-white' 
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Area
                    </button>                  </div>

                  {/* Forecast Horizon Selector */}
                  <select
                    value={forecastHorizon}
                    onChange={(e) => setForecastHorizon(Number(e.target.value))}
                    className="input py-2 px-3 text-sm"
                    title="Select forecast horizon"
                    aria-label="Forecast horizon in days"
                  >
                    <option value={3}>3 days</option>
                    <option value={7}>7 days</option>
                    <option value={14}>14 days</option>
                    <option value={30}>30 days</option>
                  </select>
                </div>
              </div>

              <PriceChart
                priceData={priceData}
                forecastData={forecastData}
                chartType={chartType}
                isLoading={isLoading}
                symbol={selectedSymbol}
              />
            </div>
          </div>

          {/* Forecast Card */}
          <div className="xl:col-span-1">
            <ForecastCard
              forecastData={forecastData}
              symbol={selectedSymbol}
              horizon={forecastHorizon}
              onRefresh={handleRefreshForecast}
              onTrain={handleTrainModel}
              isLoading={isLoading}
            />
          </div>
        </div>

        {/* Additional Stats or Features */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Current Price */}
          <div className="card p-6 text-center">
            <h3 className="text-sm font-medium text-slate-400 mb-2">Current Price</h3>
            <p className="text-2xl font-bold text-slate-200">
              ${priceData.length > 0 ? priceData[priceData.length - 1].price.toLocaleString() : '---'}
            </p>
          </div>

          {/* 24h Change */}
          <div className="card p-6 text-center">
            <h3 className="text-sm font-medium text-slate-400 mb-2">24h Change</h3>
            <p className="text-2xl font-bold text-emerald-400">
              {/* This would be calculated from price data */}
              +2.45%
            </p>
          </div>

          {/* Model Accuracy */}
          <div className="card p-6 text-center">
            <h3 className="text-sm font-medium text-slate-400 mb-2">Model Accuracy</h3>
            <p className="text-2xl font-bold text-blue-400">
              {forecastData?.metrics?.r2_score 
                ? `${(forecastData.metrics.r2_score * 100).toFixed(1)}%`
                : '---'
              }
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
