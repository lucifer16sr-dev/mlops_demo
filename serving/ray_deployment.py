
import os
import sys
import torch
from typing import List, Dict, Any
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request
from starlette.responses import JSONResponse

from models.sentiment_classifier.model import SentimentClassifier
from models.utils import load_model, get_model_path

logger = logging.getLogger(__name__)


@serve.deployment(
    name="sentiment_classifier",
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 3,
        "target_num_ongoing_requests_per_replica": 5,
    },
)
class SentimentClassifierDeployment:
    
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        logger.info(f"Initializing SentimentClassifierDeployment on device: {self.device}")
        
        # Load model
        if model_path is None:
            model_path = get_model_path("sentiment_classifier")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Reconstruct model
        model_params = checkpoint.get('model_params', checkpoint.get('metadata', {}))
        filtered_params = {k: v for k, v in model_params.items() 
                          if k not in ['tokenizer'] and v is not None}
        
        self.model = SentimentClassifier(**filtered_params)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        metadata = checkpoint.get('metadata', {})
        self.tokenizer = metadata.get('tokenizer')
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not found in checkpoint metadata")
        
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def predict_batch(self, requests: List[Dict[str, Any]]):
        texts = [req.get('text', '') if isinstance(req, dict) else req for req in requests]
        
        results = []
        for text in texts:
            try:
                result = self.model.predict(text, self.tokenizer, device=self.device)
                results.append({
                    "text": text,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"]
                })
            except Exception as e:
                logger.error(f"Error predicting for text '{text}': {e}")
                results.append({
                    "text": text,
                    "error": str(e)
                })
        
        return results
    async def predict_batch_manual(self, requests_list: List[Dict[str, Any]]):
        if not isinstance(requests_list, list):
            logger.error(f"Expected list, got {type(requests_list)}: {requests_list}")
            return [{"error": f"Invalid request format: expected list, got {type(requests_list).__name__}"}]
        
        results = []
        for i, req in enumerate(requests_list):
            try:
                # Handle different input formats
                if isinstance(req, dict):
                    text = req.get('text', '')
                elif isinstance(req, str):
                    text = req
                else:
                    text = str(req)
                
                if not text:
                    results.append({
                        "text": "",
                        "error": f"Empty text in request {i}"
                    })
                    continue
                
                result = self.model.predict(text, self.tokenizer, device=self.device)
                results.append({
                    "text": text,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"]
                })
            except Exception as e:
                logger.error(f"Error predicting for request {i} '{req}': {e}")
                results.append({
                    "text": str(req) if not isinstance(req, dict) else req.get('text', ''),
                    "error": str(e)
                })
        
        return results
    async def predict(self, request: Dict[str, Any]):
        text = request.get('text', '')
        
        if not text:
            return {"error": "Missing 'text' field in request"}
        
        try:
            result = self.model.predict(text, self.tokenizer, device=self.device)
            return {
                "text": text,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"]
            }
        except Exception as e:
            logger.error(f"Error predicting for text '{text}': {e}")
            return {"text": text, "error": str(e)}
    
    async def health_check(self):
        return {
            "status": "healthy",
            "model": "sentiment_classifier",
            "device": self.device
        }
     # HTTP ingress methods
    async def __call__(self, request: Request) -> Dict[str, Any]:
        path = request.url.path
    
        try:
            if path == "/health_check" or path == "/health":
                result = await self.health_check()
                return JSONResponse(result)
            
            elif path == "/predict":
                if request.method == "POST":
                    body = await request.json()
                    result = await self.predict(body)
                    return JSONResponse(result)
                else:
                    return JSONResponse({"error": "Method not allowed. Use POST."}, status_code=405)
            
            elif path == "/predict_batch":
                if request.method == "POST":
                    body = await request.json()
                    logger.info(f"Received batch request body: {body}")
                    requests_list = body.get("requests", [])
                    logger.info(f"Extracted requests_list type: {type(requests_list)}, length: {len(requests_list) if isinstance(requests_list, list) else 'N/A'}")
                    if not isinstance(requests_list, list):
                        return JSONResponse({
                            "error": f"Invalid format: 'requests' must be a list, got {type(requests_list).__name__}"
                        }, status_code=400)
                    result = await self.predict_batch_manual(requests_list)
                    return JSONResponse(result)
                else:
                    return JSONResponse({"error": "Method not allowed. Use POST."}, status_code=405)
            
            else:
                return JSONResponse({
                    "error": "Endpoint not found",
                    "available_endpoints": ["/health_check", "/predict", "/predict_batch"]
                }, status_code=404)
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)


def create_app(model_path: str = None, device: str = "cpu"):
    return SentimentClassifierDeployment.bind(model_path=model_path, device=device)