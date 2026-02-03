
import os
import sys
import logging
from typing import List
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from serving.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelsResponse,
    ModelInfo,
    ErrorResponse
)
from serving.ray_client import RayServeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Ray Serve client
ray_client = RayServeClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting FastAPI application...")
    logger.info("Connecting to Ray Serve...")
    if not ray_client.is_connected():
        logger.warning("Ray Serve not connected. Make sure Ray Serve is running.")
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application...")


# Create FastAPI app
app = FastAPI(
    title="MLOps Inference Platform API",
    description="Production-ready REST API for ML model serving",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


# Root endpoint
@app.get("/", tags=["General"])
async def root():
    return {
        "name": "MLOps Inference Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models"
    }


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check the health status of the API and model serving"
)
async def health_check():
    try:
        if not ray_client.is_connected():
            return HealthResponse(
                status="degraded",
                model="sentiment_classifier",
                device="unknown",
                version="1.0.0"
            )
        
        health_data = await ray_client.health_check()
        return HealthResponse(
            status=health_data.get("status", "healthy"),
            model=health_data.get("model", "sentiment_classifier"),
            device=health_data.get("device", "cpu"),
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {str(e)}"
        )


# List available models
@app.get(
    "/models",
    response_model=ModelsResponse,
    tags=["Models"],
    summary="List available models",
    description="Get a list of all available models"
)
async def list_models():
    try:
        if not ray_client.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ray Serve not connected"
            )
        
        health_data = await ray_client.health_check()
        models = [
            ModelInfo(
                name=health_data.get("model", "sentiment_classifier"),
                status="available",
                device=health_data.get("device", "cpu")
            )
        ]
        
        return ModelsResponse(models=models, total=len(models))
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Single prediction endpoint
@app.post(
    "/predict/{model_name}",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Single prediction",
    description="Make a single sentiment prediction"
)
async def predict(model_name: str, request: PredictionRequest):
    if model_name != "sentiment_classifier":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found"
        )
    
    try:
        if not ray_client.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ray Serve not connected"
            )
        
        result = await ray_client.predict(request.text)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# Batch prediction endpoint
@app.post(
    "/predict/{model_name}/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Batch prediction",
    description="Make batch predictions for multiple texts"
)
async def predict_batch(model_name: str, request: BatchPredictionRequest):
    if model_name != "sentiment_classifier":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found"
        )
    
    try:
        if not ray_client.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ray Serve not connected"
            )
        
        results = await ray_client.predict_batch(request.texts)
        
        # Convert to PredictionResponse objects
        predictions = [PredictionResponse(**result) for result in results if "error" not in result]
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Metrics endpoint (placeholder for Week 2)
@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Metrics endpoint",
    description="Prometheus metrics (to be implemented in Week 2)"
)
async def metrics():
    return {
        "message": "Metrics endpoint - to be implemented in Week 2",
        "prometheus_endpoint": "/metrics (coming soon)"
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=f"Path: {request.url.path}"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)