"""FastAPI Router defining feature computation API routes."""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from feature_engine.services.feature_pipeline_service import FeaturePipelineService
from core.utils.exceptions import FeatureEngineError

logger = logging.getLogger("feature_engine.routes.features")

router = APIRouter(prefix="/features", tags=["features"])
service = FeaturePipelineService()


class FeatureComputeRequest(BaseModel):
    """Request model for the feature computation endpoint."""

    start_date: Optional[str] = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Start date (YYYY-MM-DD). Defaults to config or today.",
    )
    end_date: Optional[str] = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="End date (YYYY-MM-DD). Defaults to config or today.",
    )


@router.post("/compute")
def compute_features(payload: Optional[FeatureComputeRequest] = None) -> Dict[str, Any]:
    """Triggers the full feature engineering pipeline.

    Accepts optional start_date and end_date parameters. If omitted,
    dates fall back to api.yaml system settings or today.

    Returns:
        Summary JSON with status, total records, and metadata.
    """
    start_date = payload.start_date if payload else None
    end_date = payload.end_date if payload else None

    try:
        result = service.compute_features(start_date=start_date, end_date=end_date)
        return result
    except FeatureEngineError as fe:
        logger.error(f"Feature pipeline error: {fe}")
        raise HTTPException(status_code=500, detail=str(fe)) from fe
    except Exception as e:
        logger.error(f"Unexpected error in feature pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e
