import os
import sys

# Add parent directory of data_engine to sys.path so imports work when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from fastapi import FastAPI
from data_engine.routes import ingestion
from core.utils.logger import setup_logger

# Initialize main package logger
logger = setup_logger("data_engine", logging.INFO)

app = FastAPI(
    title="AI Quantum 2026 Data Ingestion Engine",
    description="Production-ready multi-tier financial data ingestion system exposing sync endpoints.",
    version="1.0.0"
)

# Include routes
app.include_router(ingestion.router)

@app.get("/")
@app.get("/health")
def health_check():
    """Simple API health check endpoint."""
    return {
        "status": "healthy",
        "app": "data_engine",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server locally...")
    uvicorn.run("data_engine.main:app", host="0.0.0.0", port=8000, reload=True)
