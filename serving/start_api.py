import os
import sys
import uvicorn
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from serving.api import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Start FastAPI application")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting FastAPI Application")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Reload: {args.reload}")
    logger.info("")
    logger.info("Make sure Ray Serve is running:")
    logger.info("  python serving/start_ray_serve.py")
    logger.info("")
    logger.info("API Documentation:")
    logger.info(f"  Swagger UI: http://{args.host}:{args.port}/docs")
    logger.info(f"  ReDoc: http://{args.host}:{args.port}/redoc")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()