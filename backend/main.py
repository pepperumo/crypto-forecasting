"""
Main FastAPI application for Crypto Foresight.
Handles API routes, WebSocket connections, and application lifecycle.
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from models import HealthResponse, ErrorResponse, WebSocketMessage
from api.prices import router as prices_router
from api.forecast import router as forecast_router
from api.admin import router as admin_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Application metadata
APP_VERSION = "1.0.0"
APP_START_TIME = datetime.now()

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, list] = {}
    
    async def connect(self, websocket: WebSocket, symbol: str):
        """Accept a WebSocket connection and add to symbol group."""
        await websocket.accept()
        if symbol not in self.active_connections:
            self.active_connections[symbol] = []
        self.active_connections[symbol].append(websocket)
        logger.info(f"WebSocket connected for {symbol}. Total connections: {len(self.active_connections[symbol])}")
    
    def disconnect(self, websocket: WebSocket, symbol: str):
        """Remove WebSocket connection."""
        if symbol in self.active_connections:
            try:
                self.active_connections[symbol].remove(websocket)
                if not self.active_connections[symbol]:
                    del self.active_connections[symbol]
                logger.info(f"WebSocket disconnected for {symbol}")
            except ValueError:
                pass  # Connection wasn't in list
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
    
    async def broadcast_to_symbol(self, message: str, symbol: str):
        """Broadcast a message to all connections for a specific symbol."""
        if symbol in self.active_connections:
            disconnected = []
            for connection in self.active_connections[symbol]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to WebSocket: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                try:
                    self.active_connections[symbol].remove(conn)
                except ValueError:
                    pass
    
    async def broadcast_to_all(self, message: str):
        """Broadcast a message to all active connections."""
        for symbol in list(self.active_connections.keys()):
            await self.broadcast_to_symbol(message, symbol)


# Global connection manager instance
manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Crypto Foresight API...")
    logger.info(f"Version: {APP_VERSION}")
    
    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    
    # Start background tasks
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Crypto Foresight API...")
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass


# Create FastAPI application
app = FastAPI(
    title="Crypto Foresight API",
    description="A modern cryptocurrency price forecasting API with real-time updates",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:5174").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(prices_router, prefix="/api", tags=["prices"])
app.include_router(forecast_router, prefix="/api", tags=["forecasting"])
app.include_router(admin_router, prefix="/api", tags=["admin"])


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - APP_START_TIME).total_seconds()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=APP_VERSION,
        uptime=uptime
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Crypto Foresight API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# WebSocket endpoint for real-time price updates
@app.websocket("/ws/prices/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time price updates."""
    await manager.connect(websocket, symbol)
    try:
        # Send initial welcome message
        welcome_msg = WebSocketMessage(
            type="heartbeat",
            symbol=symbol,
            data={"message": f"Connected to {symbol} price feed"},
            timestamp=datetime.now()
        )
        await websocket.send_text(welcome_msg.json())
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client (like ping/pong)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                logger.debug(f"Received WebSocket message: {data}")
            except asyncio.TimeoutError:
                # Send heartbeat if no message received
                heartbeat_msg = WebSocketMessage(
                    type="heartbeat",
                    timestamp=datetime.now()
                )
                await websocket.send_text(heartbeat_msg.json())
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
        logger.info(f"WebSocket disconnected for {symbol}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, symbol)


# Background task for periodic heartbeat
async def heartbeat_loop():
    """Send periodic heartbeat to all WebSocket connections."""
    while True:
        try:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            heartbeat_msg = WebSocketMessage(
                type="heartbeat",
                timestamp=datetime.now()
            )
            await manager.broadcast_to_all(heartbeat_msg.json())
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    error_response = ErrorResponse(
        error="Internal server error",
        detail=str(exc) if app.debug else "An unexpected error occurred",
        timestamp=datetime.now()
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(mode='json')
    )


# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handler for HTTP exceptions."""
    error_response = ErrorResponse(
        error=exc.detail,
        timestamp=datetime.now()
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode='json')
    )


# Function to broadcast price updates (called by background services)
async def broadcast_price_update(symbol: str, price_data: Dict[str, Any]):
    """Broadcast price update to WebSocket connections."""
    message = WebSocketMessage(
        type="price_update",
        symbol=symbol,
        data=price_data,
        timestamp=datetime.now()
    )
    await manager.broadcast_to_symbol(message.json(), symbol)


# Function to broadcast forecast updates
async def broadcast_forecast_update(symbol: str, forecast_data: Dict[str, Any]):
    """Broadcast forecast update to WebSocket connections."""
    message = WebSocketMessage(
        type="forecast_update",
        symbol=symbol,
        data=forecast_data,
        timestamp=datetime.now()
    )
    await manager.broadcast_to_symbol(message.json(), symbol)


if __name__ == "__main__":
    # Run the application
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("API_DEBUG", "true").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
