/**
 * Header component with navigation, theme toggle, and connection status
 */

import React from 'react';
import { BoltIcon, SignalIcon, SignalSlashIcon } from '@heroicons/react/24/outline';

interface HeaderProps {
  isConnected: boolean;
  onThemeToggle: () => void;
}

export const Header: React.FC<HeaderProps> = ({ isConnected, onThemeToggle }) => {
  return (
    <header className="sticky top-0 z-50 bg-slate-900/80 backdrop-blur-md border-b border-slate-700/50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl">
              <BoltIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold gradient-text">
                Crypto Foresight
              </h1>
              <p className="text-sm text-slate-400">
                AI-Powered Price Forecasting
              </p>
            </div>
          </div>

          {/* Right Side - Status and Controls */}
          <div className="flex items-center gap-4">
            {/* Connection Status */}
            <div className="flex items-center gap-2">
              {isConnected ? (
                <>
                  <SignalIcon className="w-5 h-5 text-emerald-400" />
                  <span className="text-sm text-emerald-400 hidden sm:inline">
                    Live Data
                  </span>
                </>
              ) : (
                <>
                  <SignalSlashIcon className="w-5 h-5 text-red-400" />
                  <span className="text-sm text-red-400 hidden sm:inline">
                    Disconnected
                  </span>
                </>
              )}
            </div>

            {/* Theme Toggle */}
            <button
              onClick={onThemeToggle}
              className="btn-ghost p-2"
              title="Toggle theme"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
                />
              </svg>
            </button>

            {/* Settings/Menu Button */}
            <button
              className="btn-ghost p-2"
              title="Settings"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
