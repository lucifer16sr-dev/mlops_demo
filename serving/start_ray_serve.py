
import os
import sys
import argparse
import logging
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ray import serve
from ray.serve import Application
from serving.ray_deployment import create_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Start Ray Serve with sentiment classifier")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint (default: models/sentiment_classifier/checkpoint.pth)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: cpu)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting Ray Serve")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Device: {args.device}")
    if args.model_path:
        logger.info(f"Model path: {args.model_path}")
    
    # Create the application
    app = create_app(model_path=args.model_path, device=args.device)
    
    # Start Ray Serve
    try:
        serve.start(http_options={"port": args.port})
        serve.run(app, route_prefix="/")
        
        logger.info("Ray Serve started successfully!")
        logger.info(f"API available at http://localhost:{args.port}")
        logger.info("Press Ctrl+C to stop the server")
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
            serve.shutdown()
            
    except Exception as e:
        logger.error(f"Error starting Ray Serve: {e}")
        raise


if __name__ == "__main__":
    main()