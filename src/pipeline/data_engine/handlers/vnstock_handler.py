"""Handler for formatting raw vnstock historical data."""

import logging
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger("data_engine.handlers.vnstock")

class VNStockHandler:
    """Formats vnstock raw dataframe into unified dict records."""

    def format_data(self, df: pd.DataFrame, ticker_name: str) -> List[Dict[str, Any]]:
        """Transforms vnstock DataFrame to a standardized records list.

        Args:
            df: Raw DataFrame from vnstock client.
            ticker_name: The expected symbol name.

        Returns:
            List of dicts representing OHLCV records.
        """
        if df.empty:
            return []

        # Standardize columns
        df = df.copy()
        
        # vnstock returns 'time' column for historical data
        if "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"])
        elif "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"])
        else:
            # Fallback to index if datetime
            if isinstance(df.index, pd.DatetimeIndex):
                df["timestamp"] = df.index.to_series()
            else:
                logger.error("No timestamp column ('time' or 'date') found in vnstock data")
                return []

        # Map and rename
        required_cols = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        }
        
        # Filter records where price is valid
        records = []
        for _, row in df.iterrows():
            try:
                records.append({
                    "timestamp": row["timestamp"].to_pydatetime(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "timeframe": "1D",
                    "ticker_name": ticker_name
                })
            except Exception as e:
                logger.debug(f"Skipping row due to formatting error: {e}")
                continue

        return records
