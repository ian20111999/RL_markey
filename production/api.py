"""
FastAPI REST API for Production RL Market Making Models

This API provides endpoints for:
- Model inference (predictions)
- Model management (list, get, deploy)
- Health checks and monitoring
- Performance metrics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uvicorn
from pathlib import Path
import sys
import numpy as np
import yaml
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from production.model_registry import ModelRegistry, ModelMetrics
from stable_baselines3 import SAC

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize API
app = FastAPI(
    title="RL Market Making API",
    description="Production-ready API for RL-based market making models",
    version="1.0.0"
)

# Initialize model registry
registry = ModelRegistry()

# Global model cache
_loaded_models = {}


class PredictionRequest(BaseModel):
    """Request model for prediction"""
    observation: List[float] = Field(..., description="Market observation vector")
    model_id: Optional[str] = Field(None, description="Specific model ID to use (defaults to best)")
    deterministic: bool = Field(True, description="Use deterministic policy")


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    action: List[float] = Field(..., description="Predicted action vector")
    model_id: str = Field(..., description="Model ID used for prediction")
    version: str = Field(..., description="Model version")


class ModelInfo(BaseModel):
    """Model information response"""
    model_id: str
    algorithm: str
    version: str
    created_at: str
    production_ready: bool
    profitability_score: float
    mean_pnl: float
    win_rate: float
    sharpe_ratio: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: int
    registry_size: int
    production_models: int


def load_model(model_id: str) -> SAC:
    """Load a model from registry with caching"""
    if model_id in _loaded_models:
        return _loaded_models[model_id]
    
    model_path = registry.get_model_path(model_id)
    if not model_path or not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    try:
        model = SAC.load(str(model_path))
        _loaded_models[model_id] = model
        logger.info(f"Loaded model: {model_id}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "RL Market Making API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns system status and basic statistics
    """
    models = registry.list_models()
    production_models = [m for m in models if m['production_ready']]
    
    return HealthResponse(
        status="healthy",
        models_loaded=len(_loaded_models),
        registry_size=len(models),
        production_models=len(production_models)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Get trading action prediction from model
    
    Args:
        request: Prediction request with observation vector
        
    Returns:
        Predicted action and model information
    """
    # Select model
    if request.model_id:
        model_id = request.model_id
    else:
        model_id = registry.get_best_model()
        if not model_id:
            raise HTTPException(status_code=404, detail="No production-ready models available")
    
    # Load model
    model = load_model(model_id)
    metadata = registry.get_model_metadata(model_id)
    
    # Make prediction
    try:
        observation = np.array(request.observation, dtype=np.float32)
        action, _ = model.predict(observation, deterministic=request.deterministic)
        
        return PredictionResponse(
            action=action.tolist(),
            model_id=model_id,
            version=metadata.version
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models", response_model=List[ModelInfo])
async def list_models(
    production_ready_only: bool = False,
    min_score: float = 0.0
):
    """
    List all available models
    
    Args:
        production_ready_only: Filter to production-ready models only
        min_score: Minimum profitability score filter
        
    Returns:
        List of model information
    """
    models = registry.list_models(
        production_ready_only=production_ready_only,
        min_profitability_score=min_score
    )
    
    result = []
    for m in models:
        result.append(ModelInfo(
            model_id=m['model_id'],
            algorithm=m['algorithm'],
            version=m['version'],
            created_at=m['created_at'],
            production_ready=m['production_ready'],
            profitability_score=m['metrics']['profitability_score'],
            mean_pnl=m['metrics']['mean_pnl'],
            win_rate=m['metrics']['win_rate'],
            sharpe_ratio=m['metrics']['sharpe_ratio']
        ))
    
    return result


@app.get("/models/{model_id}", response_model=Dict[str, Any])
async def get_model(model_id: str):
    """
    Get detailed information about a specific model
    
    Args:
        model_id: Model identifier
        
    Returns:
        Complete model metadata
    """
    metadata = registry.get_model_metadata(model_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return metadata.to_dict()


@app.get("/models/best/current", response_model=ModelInfo)
async def get_best_model():
    """
    Get information about the current best performing model
    
    Returns:
        Best model information
    """
    model_id = registry.get_best_model()
    if not model_id:
        raise HTTPException(status_code=404, detail="No production-ready models available")
    
    metadata = registry.get_model_metadata(model_id)
    
    return ModelInfo(
        model_id=metadata.model_id,
        algorithm=metadata.algorithm,
        version=metadata.version,
        created_at=metadata.created_at,
        production_ready=metadata.production_ready,
        profitability_score=metadata.metrics.profitability_score,
        mean_pnl=metadata.metrics.mean_pnl,
        win_rate=metadata.metrics.win_rate,
        sharpe_ratio=metadata.metrics.sharpe_ratio
    )


@app.post("/models/{model_id}/load")
async def load_model_endpoint(model_id: str):
    """
    Preload a model into memory for faster predictions
    
    Args:
        model_id: Model identifier
        
    Returns:
        Success message
    """
    load_model(model_id)
    return {"message": f"Model {model_id} loaded successfully"}


@app.post("/models/{model_id}/unload")
async def unload_model(model_id: str):
    """
    Unload a model from memory
    
    Args:
        model_id: Model identifier
        
    Returns:
        Success message
    """
    if model_id in _loaded_models:
        del _loaded_models[model_id]
        logger.info(f"Unloaded model: {model_id}")
        return {"message": f"Model {model_id} unloaded successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not loaded")


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    Delete a model from the registry
    
    Args:
        model_id: Model identifier
        
    Returns:
        Success message
    """
    # Unload if loaded
    if model_id in _loaded_models:
        del _loaded_models[model_id]
    
    # Delete from registry
    success = registry.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return {"message": f"Model {model_id} deleted successfully"}


@app.get("/metrics/leaderboard", response_model=List[ModelInfo])
async def get_leaderboard(limit: int = 10):
    """
    Get model performance leaderboard
    
    Args:
        limit: Maximum number of models to return
        
    Returns:
        Top performing models
    """
    models = registry.list_models()[:limit]
    
    result = []
    for m in models:
        result.append(ModelInfo(
            model_id=m['model_id'],
            algorithm=m['algorithm'],
            version=m['version'],
            created_at=m['created_at'],
            production_ready=m['production_ready'],
            profitability_score=m['metrics']['profitability_score'],
            mean_pnl=m['metrics']['mean_pnl'],
            win_rate=m['metrics']['win_rate'],
            sharpe_ratio=m['metrics']['sharpe_ratio']
        ))
    
    return result


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server"""
    uvicorn.run(
        "production.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start RL Market Making API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    start_server(host=args.host, port=args.port, reload=args.reload)
