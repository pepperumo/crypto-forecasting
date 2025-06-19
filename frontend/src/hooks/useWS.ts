/**
 * WebSocket hook for real-time cryptocurrency price updates
 * Handles connection management, message parsing, and reconnection logic
 */

import { useState, useEffect, useCallback, useRef } from 'react';

interface WebSocketMessage {
  type: 'price_update' | 'forecast_update' | 'error' | 'heartbeat';
  symbol?: string;
  data?: any;
  timestamp: string;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  lastMessage: WebSocketMessage | null;
  error: string | null;
  connect: () => void;
  disconnect: () => void;
  sendMessage: (message: any) => void;
}

export const useWebSocket = (symbol: string): UseWebSocketReturn => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const ws = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = useRef(1000);
  const maxReconnectDelay = 30000;

  const getWebSocketUrl = useCallback(() => {
    const wsUrl = import.meta.env?.VITE_WS_URL || 'ws://localhost:8000';
    return `${wsUrl}/ws/prices/${symbol}`;
  }, [symbol]);

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      setError(null);
      const url = getWebSocketUrl();
      console.log(`[WebSocket] Connecting to ${url}`);
      
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        console.log(`[WebSocket] Connected to ${symbol}`);
        setIsConnected(true);
        setError(null);
        reconnectAttempts.current = 0;
        reconnectDelay.current = 1000;
      };

      ws.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          console.log(`[WebSocket] Message received:`, message);
          setLastMessage(message);
          
          if (message.type === 'error') {
            setError(message.data?.message || 'WebSocket error');
          }
        } catch (err) {
          console.error('[WebSocket] Error parsing message:', err);
          setError('Error parsing WebSocket message');
        }
      };

      ws.current.onclose = (event) => {
        console.log(`[WebSocket] Connection closed: ${event.code} ${event.reason}`);
        setIsConnected(false);
        
        // Attempt to reconnect if not manually closed
        if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          setTimeout(() => {
            console.log(`[WebSocket] Reconnecting... (attempt ${reconnectAttempts.current + 1})`);
            reconnectAttempts.current++;
            reconnectDelay.current = Math.min(reconnectDelay.current * 2, maxReconnectDelay);
            connect();
          }, reconnectDelay.current);
        }
      };

      ws.current.onerror = (event) => {
        console.error('[WebSocket] Error:', event);
        setError('WebSocket connection error');
        setIsConnected(false);
      };

    } catch (err) {
      console.error('[WebSocket] Failed to create connection:', err);
      setError('Failed to create WebSocket connection');
    }
  }, [getWebSocketUrl, symbol]);

  const disconnect = useCallback(() => {
    if (ws.current) {
      console.log('[WebSocket] Disconnecting...');
      ws.current.close(1000, 'User disconnected');
      ws.current = null;
      setIsConnected(false);
      setLastMessage(null);
      setError(null);
    }
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    } else {
      console.warn('[WebSocket] Cannot send message - not connected');
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (ws.current) {
        ws.current.close(1000, 'Component unmounted');
      }
    };
  }, []);

  // Reconnect when symbol changes
  useEffect(() => {
    if (symbol && ws.current?.readyState === WebSocket.OPEN) {
      disconnect();
      setTimeout(connect, 100);
    }
  }, [symbol, connect, disconnect]);

  return {
    isConnected,
    lastMessage,
    error,
    connect,
    disconnect,
    sendMessage
  };
};

export default useWebSocket;
