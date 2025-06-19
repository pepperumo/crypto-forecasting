"""
Admin API endpoints.
Handles model training, system management, and administrative tasks.
"""

import logging
from datetime import datetime
from typing import Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from models import TrainingRequest, TrainingResponse, TrainingStatus, SymbolEnum
from services.train import TrainingService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize training service
training_service = TrainingService()

# Simple in-memory task tracking (in production, use Redis or database)
training_tasks: Dict[str, TrainingStatus] = {}

# Thread pool for background training tasks
executor = ThreadPoolExecutor(max_workers=2)


@router.post("/train", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start training a model for a cryptocurrency.
    
    Args:
        request: Training configuration request
        
    Returns:
        TrainingResponse: Task ID and status URL for monitoring progress
    """
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task status
        training_tasks[task_id] = TrainingStatus(
            task_id=task_id,
            status="pending",
            progress=0.0,
            message="Training task queued",
            started_at=datetime.now()
        )
        
        logger.info(f"Starting training task {task_id} for {request.symbol} with horizon {request.horizon}")
        
        # Start background training task
        background_tasks.add_task(
            _run_training_task,
            task_id,
            request
        )
        
        return TrainingResponse(
            task_id=task_id,
            message=f"Training started for {request.symbol}",
            status_url=f"/api/status/{task_id}"
        )
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training: {str(e)}"
        )


@router.get("/status/{task_id}", response_model=TrainingStatus)
async def get_training_status(task_id: str):
    """
    Get the status of a training task.
    
    Args:
        task_id: Training task ID
        
    Returns:
        TrainingStatus: Current status of the training task
    """
    if task_id not in training_tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Training task {task_id} not found"
        )
    
    return training_tasks[task_id]


@router.get("/status")
async def get_all_training_status():
    """
    Get the status of all training tasks.
    
    Returns:
        Dictionary of all training task statuses
    """
    return {
        "tasks": list(training_tasks.values()),
        "total_tasks": len(training_tasks),
        "active_tasks": len([t for t in training_tasks.values() if t.status in ["pending", "running"]]),
        "retrieved_at": datetime.now()
    }


@router.delete("/status/{task_id}")
async def cancel_training(task_id: str):
    """
    Cancel a training task (if still pending or running).
    
    Args:
        task_id: Training task ID to cancel
        
    Returns:
        Cancellation status
    """
    if task_id not in training_tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Training task {task_id} not found"
        )
    
    task = training_tasks[task_id]
    
    if task.status in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task in {task.status} state"
        )
    
    # Update task status to failed (simple cancellation)
    task.status = "failed"
    task.error = "Cancelled by user"
    task.completed_at = datetime.now()
    
    logger.info(f"Training task {task_id} cancelled")
    
    return {
        "message": f"Training task {task_id} cancelled",
        "cancelled_at": datetime.now()
    }


@router.post("/retrain/{symbol}")
async def retrain_model(
    symbol: SymbolEnum,
    horizon: int = 7,
    background_tasks: BackgroundTasks = None
):
    """
    Quickly retrain a model with default parameters.
    
    Args:
        symbol: Cryptocurrency symbol
        horizon: Forecast horizon in days
        
    Returns:
        Training response
    """
    # Create training request with default parameters
    request = TrainingRequest(
        symbol=symbol,
        horizon=horizon
    )
    
    return await start_training(request, background_tasks)


@router.get("/models")
async def list_trained_models():
    """
    List all trained models and their information.
    
    Returns:
        List of trained models with metadata
    """
    try:
        models_info = await training_service.list_trained_models()
        
        return {
            "models": models_info,
            "total_models": len(models_info),
            "retrieved_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error listing trained models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list trained models: {str(e)}"
        )


@router.delete("/models/{symbol}")
async def delete_model(symbol: SymbolEnum):
    """
    Delete a trained model for a cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol
        
    Returns:
        Deletion status
    """
    try:
        success = await training_service.delete_model(symbol.value)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"No model found for {symbol}"
            )
        
        logger.info(f"Model deleted for {symbol}")
        
        return {
            "message": f"Model deleted for {symbol}",
            "deleted_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {str(e)}"
        )


@router.get("/system/info")
async def get_system_info():
    """
    Get system information and health status.
    
    Returns:
        System information
    """
    try:
        system_info = {
            "server_time": datetime.now(),
            "training_service_status": "active",
            "active_training_tasks": len([t for t in training_tasks.values() if t.status in ["pending", "running"]]),
            "completed_training_tasks": len([t for t in training_tasks.values() if t.status == "completed"]),
            "failed_training_tasks": len([t for t in training_tasks.values() if t.status == "failed"]),
            "supported_symbols": [symbol.value for symbol in SymbolEnum],
            "max_horizon": 30,
            "min_horizon": 1
        }
        
        return system_info
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system info: {str(e)}"
        )


async def _run_training_task(task_id: str, request: TrainingRequest):
    """
    Background task to run model training.
    
    Args:
        task_id: Training task ID
        request: Training configuration
    """
    try:
        # Update task status to running
        training_tasks[task_id].status = "running"
        training_tasks[task_id].progress = 10.0
        training_tasks[task_id].message = "Fetching training data..."
        
        logger.info(f"Starting training for task {task_id}")
        
        # Run training in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def update_progress(progress: float, message: str):
            """Progress callback for training updates."""
            if task_id in training_tasks:
                training_tasks[task_id].progress = progress
                training_tasks[task_id].message = message
        
        # Execute training
        result = await loop.run_in_executor(
            executor,
            training_service.train_model,
            request.symbol.value,
            request.horizon,
            request.seasonality.value,
            request.changepoint_prior_scale,
            request.n_iter,
            request.include_features,
            update_progress
        )
        
        # Update task status to completed
        training_tasks[task_id].status = "completed"
        training_tasks[task_id].progress = 100.0
        training_tasks[task_id].message = "Training completed successfully"
        training_tasks[task_id].completed_at = datetime.now()
        training_tasks[task_id].result = result
        
        logger.info(f"Training completed for task {task_id}")
        
    except Exception as e:
        # Update task status to failed
        logger.error(f"Training failed for task {task_id}: {e}")
        
        if task_id in training_tasks:
            training_tasks[task_id].status = "failed"
            training_tasks[task_id].error = str(e)
            training_tasks[task_id].completed_at = datetime.now()
            training_tasks[task_id].message = f"Training failed: {str(e)}"
