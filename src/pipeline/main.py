import os
import sys
import logging

# Ensure src/pipeline is in sys.path so we can import from data_engine and feature_engine
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI
import uvicorn

from data_engine.routes.ingestion import router as ingestion_router
from feature_engine.routes.feature_routes import router as feature_router
from core.utils.logger import setup_logger

# Initialize unified package logger
logger = setup_logger("unified_pipeline", logging.INFO)

app = FastAPI(
    title="AI Quantum 2026 Unified Pipeline API",
    description="Unified API Gateway exposing endpoints for Data Ingestion and Feature Engineering.",
    version="1.0.0"
)

# Mount the routers from sub-engines
# Ingestion router handles /ingestion/*
app.include_router(ingestion_router)
# Feature router handles /features/*
app.include_router(feature_router)

@app.get("/")
@app.get("/health")
def health_check():
    """Simple API health check endpoint for the unified gateway."""
    return {
        "status": "healthy",
        "app": "unified_pipeline",
        "version": "1.0.0",
        "mounted_routers": ["/ingestion", "/features"]
    }

if __name__ == "__main__":
    logger.info("Starting Unified Pipeline uvicorn server locally on port 8000...")
    # Run using the module path so reload works properly
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)