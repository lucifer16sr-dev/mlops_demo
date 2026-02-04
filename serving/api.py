
import os
import sys
import logging
import time
from typing import List
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from monitoring.metrics import get_metrics_collector
from monitoring.logging_config import setup_logging

# Setup structured logging
setup_logging(level="INFO", use_json=True)
logger = logging.getLogger(__name__)

# Initialize metrics collector
metrics = get_metrics_collector()

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
    start_time = time.time()
    metrics.set_active_requests(metrics.active_requests._value.get() + 1)
    
    try:
        logger.info(
            f"{request.method} {request.url.path}",
            extra={"extra_fields": {
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None
            }}
        )
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        status_code = response.status_code
        
        # Record metrics
        metrics.record_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=status_code,
            duration=duration
        )
        
        # Record errors
        if status_code >= 400:
            error_type = "client_error" if status_code < 500 else "server_error"
            metrics.record_error(error_type, request.url.path)
        
        logger.info(
            f"Response status: {status_code}",
            extra={"extra_fields": {
                "status_code": status_code,
                "duration_seconds": duration
            }}
        )
        
        return response
    except Exception as e:
        duration = time.time() - start_time
        metrics.record_error("exception", request.url.path)
        metrics.record_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=500,
            duration=duration
        )
        raise
    finally:
        metrics.set_active_requests(max(0, metrics.active_requests._value.get() - 1))



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
        is_healthy = health_data.get("status", "unknown") == "healthy"
        metrics.set_model_health("sentiment_classifier", is_healthy)
        return HealthResponse(
            status=health_data.get("status", "healthy"),
            model=health_data.get("model", "sentiment_classifier"),
            device=health_data.get("device", "cpu"),
            version="1.0.0"
        )
    except Exception as e:
        metrics.set_model_health("sentiment_classifier", False)
        logger.error(f"Health check error: {e}", exc_info=True)
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
    inference_start = time.time()
    try:
        if not ray_client.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ray Serve not connected"
            )
        
        result = await ray_client.predict(request.text)

        inference_duration = time.time() - inference_start

        if "error" in result:
            metrics.record_inference(model_name, inference_duration, status="error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        metrics.record_inference(model_name, inference_duration, status="success")
        return PredictionResponse(**result)
    except HTTPException:
        inference_duration = time.time() - inference_start
        metrics.record_inference(model_name, inference_duration, status="error")
        raise
    except Exception as e:
        inference_duration = time.time() - inference_start
        metrics.record_inference(model_name, inference_duration, status="error")
        logger.error(f"Prediction error: {e}", exc_info=True)
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
    batch_size = len(request.texts)
    inference_start = time.time()
    try:
        if not ray_client.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ray Serve not connected"
            )
        # Record batch size
        metrics.record_batch(model_name, batch_size)

        results = await ray_client.predict_batch(request.texts)
        
        inference_duration = time.time() - inference_start
        # Convert to PredictionResponse objects
        predictions = [PredictionResponse(**result) for result in results if "error" not in result]
        
        errors = len(results) - len(predictions)
         # Record inference metrics
        status_val = "success" if errors == 0 else "partial_error"
        metrics.record_inference(model_name, inference_duration, status=status_val)
        
        if errors > 0:
            metrics.record_error("batch_error", f"/predict/{model_name}/batch")
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions)
        )
    except HTTPException:
        inference_duration = time.time() - inference_start
        metrics.record_inference(model_name, inference_duration, status="error")
        raise
    except Exception as e:
        inference_duration = time.time() - inference_start
        metrics.record_inference(model_name, inference_duration, status="error")
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Metrics endpoint (placeholder for Week 2)
@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Metrics endpoint",
    description="Prometheus metrics"
)
async def metrics_endpoint():
    return Response(
        content=metrics.get_metrics(),
        media_type=metrics.get_content_type()
    )


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