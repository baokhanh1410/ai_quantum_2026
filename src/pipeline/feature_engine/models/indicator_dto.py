"""Data Transfer Objects for indicator values."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class IndicatorValueDTO:
    """Represents a single computed indicator value ready for DB insertion.

    Attributes:
        ticker_id: Foreign key to the tickers table.
        indicator_type_id: Foreign key to the indicator_types table.
        timestamp: The trading date of the indicator value.
        value: The computed numeric value.
    """

    ticker_id: int
    indicator_type_id: int
    timestamp: datetime
    value: Optional[float]

    def to_dict(self):
        """Converts to a dictionary for SQLAlchemy bulk insert."""
        return {
            "ticker_id": self.ticker_id,
            "indicator_type_id": self.indicator_type_id,
            "timestamp": self.timestamp,
            "value": self.value,
        }
