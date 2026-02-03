
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class PredictionRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for sentiment", min_length=1, max_length=10000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this product! It's amazing!"
            }
        }


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "I love this product!",
                    "This is terrible."
                ]
            }
        }


class PredictionResponse(BaseModel):
    text: str = Field(..., description="Input text")
    prediction: str = Field(..., description="Predicted sentiment (positive/negative)")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    probabilities: Dict[str, float] = Field(..., description="Probability scores for each class")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this product!",
                "prediction": "positive",
                "confidence": 0.99,
                "probabilities": {
                    "negative": 0.01,
                    "positive": 0.99
                }
            }
        }


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    total: int = Field(..., description="Total number of predictions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "text": "I love this product!",
                        "prediction": "positive",
                        "confidence": 0.99,
                        "probabilities": {"negative": 0.01, "positive": 0.99}
                    }
                ],
                "total": 1
            }
        }


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model: str = Field(..., description="Model name")
    device: str = Field(..., description="Device used for inference")
    version: Optional[str] = Field(None, description="API version")


class ModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    status: str = Field(..., description="Model status")
    device: str = Field(..., description="Device used")


class ModelsResponse(BaseModel):
    models: List[ModelInfo] = Field(..., description="List of available models")
    total: int = Field(..., description="Total number of models")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")