/**
 * Price chart component using Chart.js
 */

import React, { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartOptions
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { format } from 'date-fns';

// Define a more flexible dataset type for Chart.js
type ChartDataset = {
  label: string;
  data: (number | null)[];
  borderColor: string;
  backgroundColor: string;
  pointBackgroundColor?: string;
  pointBorderColor?: string;
  pointRadius: number;
  borderWidth: number;
  fill: boolean | string | number;
  tension?: number;
  borderDash?: number[];
};

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
  }>;
}

interface PriceChartProps {
  priceData: PriceData[];
  forecastData: ForecastData | null;
  chartType: 'line' | 'area';
  isLoading: boolean;
  symbol: string;
}

// Register required Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export const PriceChart: React.FC<PriceChartProps> = ({
  priceData,
  forecastData,
  chartType,
  isLoading,
  symbol
}) => {
  if (isLoading) {
    return (
      <div className="chart-container">
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin w-8 h-8 border-2 border-purple-600 border-t-transparent rounded-full mx-auto mb-4"></div>
            <p className="text-slate-400">Loading chart data...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!priceData || priceData.length === 0) {
    return (
      <div className="chart-container">
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <p className="text-slate-400 mb-2">No price data available</p>
            <p className="text-sm text-slate-500">Try selecting a different cryptocurrency</p>
          </div>
        </div>
      </div>
    );
  }
  // Create the chart using Chart.js
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return format(date, 'MMM d');
  };

  // Prepare data for Chart.js
  const labels = priceData.map(d => formatDate(d.timestamp));
  
  // If we have forecast data, add those labels too
  const allLabels = [...labels];
  if (forecastData && forecastData.forecast.length > 0) {
    forecastData.forecast.forEach(point => {
      const formattedDate = formatDate(point.timestamp);
      if (!allLabels.includes(formattedDate)) {
        allLabels.push(formattedDate);
      }
    });
  }
  
  // Create dataset for historical prices
  const historicalData = priceData.map(d => d.price);
  
  // Create prediction data array with nulls for historical dates
  let predictionData: (number | null)[] = Array(labels.length).fill(null);
  
  if (forecastData && forecastData.forecast.length > 0) {
    // Append prediction values
    forecastData.forecast.forEach(point => {
      const formattedDate = formatDate(point.timestamp);
      const index = allLabels.indexOf(formattedDate);
      if (index !== -1) {
        if (!predictionData[index]) {
          predictionData[index] = point.predicted_price;
        }
      }
    });
  }

  // Create upper and lower bounds arrays if available
  let upperBoundData: (number | null)[] = Array(allLabels.length).fill(null);
  let lowerBoundData: (number | null)[] = Array(allLabels.length).fill(null);
  
  if (forecastData && forecastData.forecast.some(p => p.upper_bound && p.lower_bound)) {
    forecastData.forecast.forEach(point => {
      if (point.upper_bound && point.lower_bound) {
        const formattedDate = formatDate(point.timestamp);
        const index = allLabels.indexOf(formattedDate);
        if (index !== -1) {
          upperBoundData[index] = point.upper_bound;
          lowerBoundData[index] = point.lower_bound;
        }
      }
    });
  }
  const chartData: {
    labels: string[],
    datasets: ChartDataset[]
  } = {
    labels: allLabels,
    datasets: [
      {
        label: 'Historical Price',
        data: historicalData,
        borderColor: 'rgba(59, 130, 246, 1)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        pointBackgroundColor: 'rgba(59, 130, 246, 1)',
        pointBorderColor: '#fff',
        pointRadius: 3,
        borderWidth: 2,
        fill: false,
        tension: 0.1,
      },
      {
        label: 'Price Forecast',
        data: predictionData,
        borderColor: 'rgba(168, 85, 247, 1)',
        backgroundColor: 'rgba(168, 85, 247, 0.1)',
        pointBackgroundColor: 'rgba(168, 85, 247, 1)',
        pointBorderColor: '#fff',
        pointRadius: 4,
        borderWidth: 2,
        borderDash: forecastData ? [] : [5, 5],
        fill: false,
        tension: 0.1,
      }
    ]
  };
  
  // Add upper and lower bounds if available
  if (forecastData && forecastData.forecast.some(p => p.upper_bound && p.lower_bound)) {
    chartData.datasets.push({
      label: 'Upper Bound',
      data: upperBoundData,
      borderColor: 'rgba(168, 85, 247, 0.5)',
      backgroundColor: 'rgba(168, 85, 247, 0.1)',
      pointBackgroundColor: 'rgba(168, 85, 247, 0.5)',
      pointBorderColor: '#fff',
      pointRadius: 0,
      borderWidth: 1,
      fill: false,
      borderDash: [3, 3],
      tension: 0.1
    });
    
    chartData.datasets.push({
      label: 'Lower Bound',
      data: lowerBoundData,
      borderColor: 'rgba(168, 85, 247, 0.5)',
      backgroundColor: 'rgba(168, 85, 247, 0.1)',
      pointBackgroundColor: 'rgba(168, 85, 247, 0.5)',
      pointBorderColor: '#fff',
      pointRadius: 0,
      borderWidth: 1,
      fill: 3, // This will fill between datasets
      borderDash: [3, 3],
      tension: 0.1
    });
  }

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          color: 'rgba(148, 163, 184, 1)',
          usePointStyle: true,
          pointStyle: 'circle',
          padding: 20,
          font: {
            family: "'Inter', sans-serif",
            size: 12
          }
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(15, 23, 42, 0.9)',
        titleColor: 'rgba(255, 255, 255, 0.9)',
        bodyColor: 'rgba(148, 163, 184, 1)',
        borderColor: 'rgba(71, 85, 105, 0.2)',
        borderWidth: 1,
        padding: 12,
        boxPadding: 6,
        usePointStyle: true,
        bodyFont: {
          family: "'Inter', sans-serif",
        },        titleFont: {
          family: "'Inter', sans-serif",
          weight: 'bold'
        }
      }
    },
    scales: {
      x: {
        ticks: {
          color: 'rgba(148, 163, 184, 0.8)',
          maxRotation: 45,
          minRotation: 45,
          font: {
            family: "'Inter', sans-serif",
            size: 11
          },
          autoSkip: true,
          maxTicksLimit: 8
        },        grid: {
          display: false
        },
        border: {
          display: false
        }
      },
      y: {
        ticks: {
          color: 'rgba(148, 163, 184, 0.8)',
          font: {
            family: "'Inter', sans-serif",
            size: 11
          },
          callback: function(value) {
            return '$' + value.toLocaleString();
          }
        },        grid: {
          color: 'rgba(71, 85, 105, 0.2)'
        },
        border: {
          display: false
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    },
    elements: {
      line: {
        tension: 0.1
      },
      point: {
        radius: 2,
        hitRadius: 6,
        hoverRadius: 4
      }
    }
  };
  // Apply area chart style if chartType is 'area'
  if (chartType === 'area') {
    // Use 'start' for Chart.js v4 (previously was 'origin')
    chartData.datasets[0].fill = true;
    if (forecastData) {
      chartData.datasets[1].fill = true;
    }
  }

  return (
    <div className="chart-container">
      <div className="h-96 bg-slate-800/30 rounded-xl border border-slate-700/30 p-4">
        <Line data={chartData} options={chartOptions} />
      </div>
      <div className="flex items-center justify-center gap-4 text-sm mt-3">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-blue-400 rounded"></div>
          <span className="text-slate-400">Historical Prices</span>
        </div>
        {forecastData && (
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-purple-400 rounded"></div>
            <span className="text-slate-400">Forecast</span>
          </div>
        )}
      </div>
    </div>
  );
};
