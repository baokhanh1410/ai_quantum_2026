"""FastAPI Router defining ingestion API routes."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from data_engine.services.ingestion_service import IngestionService
from core.utils.exceptions import DataEngineError

router = APIRouter(prefix="/ingestion", tags=["ingestion"])
service = IngestionService()

class IngestionRequest(BaseModel):
    """Request model for triggers accepting optional start and end date filters."""
    start_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="End date (YYYY-MM-DD)")


@router.post("/stock")
def trigger_stock_ingestion(payload: Optional[IngestionRequest] = None) -> Dict[str, Any]:
    """Exposes endpoint to trigger vnstock historical ingestion."""
    start_date = payload.start_date if payload else None
    end_date = payload.end_date if payload else None
    try:
        res = service.ingest_stocks(start_date=start_date, end_date=end_date)
        return res
    except DataEngineError as de:
        raise HTTPException(status_code=500, detail=str(de))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.post("/gold")
def trigger_gold_ingestion() -> Dict[str, Any]:
    """Exposes endpoint to trigger SJC Gold historical CSV crawl and sync."""
    try:
        res = service.ingest_gold()
        if res.get("status") == "error":
            raise HTTPException(status_code=500, detail=res.get("message"))
        return res
    except DataEngineError as de:
        raise HTTPException(status_code=500, detail=str(de))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.post("/macro")
def trigger_macro_ingestion(payload: Optional[IngestionRequest] = None) -> Dict[str, Any]:
    """Exposes endpoint to trigger SBV, Investing, and VNDirect macro syncs."""
    start_date = payload.start_date if payload else None
    end_date = payload.end_date if payload else None
    try:
        res = service.ingest_macro(start_date=start_date, end_date=end_date)
        return res
    except DataEngineError as de:
        raise HTTPException(status_code=500, detail=str(de))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.post("/all")
def trigger_all_ingestion(payload: Optional[IngestionRequest] = None) -> Dict[str, Any]:
    """Exposes endpoint to sync all ingestion pipelines sequentially."""
    start_date = payload.start_date if payload else None
    end_date = payload.end_date if payload else None
    try:
        res = service.ingest_all(start_date=start_date, end_date=end_date)
        return res
    except DataEngineError as de:
        raise HTTPException(status_code=500, detail=str(de))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
