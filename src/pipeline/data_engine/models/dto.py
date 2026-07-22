"""Pydantic DTO validation schemas matching the updated 3NF database layout."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator

class TickerDTO(BaseModel):
    """Data Transfer Object for Tickers."""
    symbol: str = Field(..., min_length=1, max_length=20)
    name: Optional[str] = Field(None, max_length=255)
    exchange: Optional[str] = Field(None, max_length=50)
    asset_class_id: int = Field(..., ge=1, le=10)
    active: bool = True

    class Config:
        from_attributes = True


class OHLCVDTO(BaseModel):
    """Data Transfer Object for OHLCV records."""
    ticker_name: str = Field(..., min_length=1, max_length=20, alias="symbol")
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    adjusted_close: Optional[float] = None
    volume: float = Field(..., ge=0.0)
    timeframe: str = Field("1D", min_length=1, max_length=10)
    source: Optional[str] = Field(None, max_length=100)
    data_quality: str = Field("good", max_length=30)

    @field_validator("adjusted_close", mode="before")
    @classmethod
    def populate_adjusted_close(cls, v: Optional[float], info) -> Optional[float]:
        """Fallbacks to close price if adjusted_close is not set."""
        if v is None:
            values = info.data
            return values.get("close")
        return v

    @field_validator("high")
    @classmethod
    def high_must_be_ge_low_and_open_close(cls, v: float, info) -> float:
        """Validate that high is greater than or equal to other prices if they are in input."""
        values = info.data
        for key in ["open", "low", "close"]:
            if key in values and v < values[key]:
                raise ValueError(f"high ({v}) cannot be less than {key} ({values[key]})")
        return v

    @field_validator("low")
    @classmethod
    def low_must_be_le_high_and_open_close(cls, v: float, info) -> float:
        """Validate that low is less than or equal to other prices if they are in input."""
        values = info.data
        for key in ["open", "high", "close"]:
            if key in values and v > values[key]:
                raise ValueError(f"low ({v}) cannot be greater than {key} ({values[key]})")
        return v

    class Config:
        from_attributes = True
        populate_by_name = True
