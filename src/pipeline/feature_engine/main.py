"""Feature Engineering Engine — FastAPI application entry point."""

import os
import sys

# Add parent directory of feature_engine to sys.path so imports work when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from fastapi import FastAPI
from feature_engine.routes import feature_routes
from core.utils.logger import setup_logger

# Initialize main package logger
logger = setup_logger("feature_engine", logging.INFO)

app = FastAPI(
    title="AI Quantum 2026 Feature Engineering Engine",
    description="Production-ready feature computation pipeline for Constrained RL Portfolio Optimization.",
    version="1.0.0",
)

# Include routes
app.include_router(feature_routes.router)


@app.get("/")
@app.get("/health")
def health_check():
    """Simple API health check endpoint."""
    return {
        "status": "healthy",
        "app": "feature_engine",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Feature Engine uvicorn server locally...")
    uvicorn.run("feature_engine.main:app", host="0.0.0.0", port=8001, reload=True)
