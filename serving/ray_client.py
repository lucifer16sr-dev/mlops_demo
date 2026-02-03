
import os
import sys
from typing import Dict, List, Any, Optional
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ray import serve
from ray.serve.handle import DeploymentHandle

logger = logging.getLogger(__name__)


class RayServeClient:
    
    def __init__(self):
        self.deployment_handle: Optional[DeploymentHandle] = None
        self._connect()
    
    def _connect(self):
        try:
            # Get the deployment handle
            self.deployment_handle = serve.get_deployment_handle(
                "sentiment_classifier",
                app_name="default"
            )
            logger.info("Connected to Ray Serve deployment: sentiment_classifier")
        except Exception as e:
            logger.warning(f"Could not connect to Ray Serve: {e}")
            logger.warning("Ray Serve may not be running. Start it with: python serving/start_ray_serve.py")
            self.deployment_handle = None
    
    async def health_check(self):
        if not self.deployment_handle:
            raise ConnectionError("Ray Serve deployment not available")
        
        try:
            result = await self.deployment_handle.health_check.remote()
            return result
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    async def predict(self, text: str):
        if not self.deployment_handle:
            raise ConnectionError("Ray Serve deployment not available")
        
        try:
            result = await self.deployment_handle.predict.remote({"text": text})
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    async def predict_batch(self, texts: List[str]):
        if not self.deployment_handle:
            raise ConnectionError("Ray Serve deployment not available")
        
        try:
            # Prepare requests list
            requests_list = [{"text": text} for text in texts]
            result = await self.deployment_handle.predict_batch_manual.remote(requests_list)
            return result
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def is_connected(self):
        return self.deployment_handle is not None