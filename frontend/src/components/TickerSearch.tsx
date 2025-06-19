/**
 * Ticker search component for selecting cryptocurrency symbols
 */

import React, { useState, useRef, useEffect } from 'react';
import { MagnifyingGlassIcon, ChevronDownIcon } from '@heroicons/react/24/outline';

interface TickerSearchProps {
  onSymbolSelect: (symbol: string) => void;
  selectedSymbol: string;
  defaultSymbols: string[];
}

const SYMBOL_DISPLAY_NAMES: Record<string, string> = {
  'bitcoin': 'Bitcoin (BTC)',
  'ethereum': 'Ethereum (ETH)',
  'solana': 'Solana (SOL)',
  'cardano': 'Cardano (ADA)',
  'polkadot': 'Polkadot (DOT)',
  'chainlink': 'Chainlink (LINK)',
};

export const TickerSearch: React.FC<TickerSearchProps> = ({
  onSymbolSelect,
  selectedSymbol,
  defaultSymbols
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Filter symbols based on search term
  const filteredSymbols = defaultSymbols.filter(symbol => {
    const displayName = SYMBOL_DISPLAY_NAMES[symbol] || symbol;
    return displayName.toLowerCase().includes(searchTerm.toLowerCase());
  });

  const handleSymbolSelect = (symbol: string) => {
    onSymbolSelect(symbol);
    setIsOpen(false);
    setSearchTerm('');
  };

  const selectedDisplayName = SYMBOL_DISPLAY_NAMES[selectedSymbol] || selectedSymbol;

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Search Button/Trigger */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between w-full sm:w-auto min-w-[200px] px-4 py-3 bg-slate-800/50 border border-slate-600/50 rounded-xl text-slate-200 hover:border-slate-500/50 focus:border-purple-500/50 focus:ring-2 focus:ring-purple-500/20 transition-all duration-200"
      >
        <div className="flex items-center gap-3">
          <MagnifyingGlassIcon className="w-5 h-5 text-slate-400" />
          <span className="font-medium">{selectedDisplayName}</span>
        </div>
        <ChevronDownIcon 
          className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${
            isOpen ? 'rotate-180' : ''
          }`} 
        />
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute top-full left-0 right-0 sm:right-auto sm:w-80 mt-2 bg-slate-800/95 backdrop-blur-md border border-slate-600/50 rounded-xl shadow-2xl z-50 animate-fade-in">
          {/* Search Input */}
          <div className="p-4 border-b border-slate-700/50">
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search cryptocurrencies..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-slate-200 placeholder-slate-400 focus:border-purple-500/50 focus:ring-1 focus:ring-purple-500/20 transition-all duration-200"
                autoFocus
              />
            </div>
          </div>

          {/* Symbol List */}
          <div className="max-h-60 overflow-y-auto">
            {filteredSymbols.length > 0 ? (
              filteredSymbols.map((symbol) => {
                const displayName = SYMBOL_DISPLAY_NAMES[symbol] || symbol;
                const isSelected = symbol === selectedSymbol;

                return (
                  <button
                    key={symbol}
                    onClick={() => handleSymbolSelect(symbol)}
                    className={`w-full px-4 py-3 text-left flex items-center gap-3 hover:bg-slate-700/50 transition-all duration-150 ${
                      isSelected ? 'bg-purple-600/20 text-purple-300' : 'text-slate-200'
                    }`}
                  >
                    {/* Symbol Icon/Badge */}
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold crypto-badge ${symbol}`}>
                      {symbol.slice(0, 3).toUpperCase()}
                    </div>
                    
                    <div className="flex-1">
                      <div className="font-medium">{displayName}</div>
                      <div className="text-xs text-slate-400 capitalize">{symbol}</div>
                    </div>

                    {isSelected && (
                      <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                    )}
                  </button>
                );
              })
            ) : (
              <div className="px-4 py-6 text-center text-slate-400">
                <MagnifyingGlassIcon className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No cryptocurrencies found</p>
                <p className="text-xs mt-1">Try a different search term</p>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="p-3 border-t border-slate-700/50 text-xs text-slate-400 text-center">
            Select a cryptocurrency to view its forecast
          </div>
        </div>
      )}
    </div>
  );
};
