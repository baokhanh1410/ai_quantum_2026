"""Handler for formatting TradingView pandas DataFrame responses."""

import logging
import datetime
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger("data_engine.handlers.tradingview")

class TradingViewHandler:
    """Formats TradingView DataFrame data into unified database records."""

    def format_data(self, df: pd.DataFrame, ticker_name: str, data_format: str = "ohlcv") -> List[Dict[str, Any]]:
        """Transforms TradingView pandas DataFrame to standard database records.

        Args:
            df: pandas DataFrame from TradingViewClient.
            ticker_name: Expected target symbol in the database (e.g. VN10YT, VN3YT).
            data_format: Format type (e.g. 'ohlcv' or 'single_value').

        Returns:
            List of OHLCV records.
        """
        if df is None or df.empty:
            return []

        # Sort chronologically
        df = df.sort_index()

        records = []
        for dt_idx, row in df.iterrows():
            try:
                # dt_idx is a Timestamp or datetime object from pandas index
                if isinstance(dt_idx, (pd.Timestamp, datetime.date, datetime.datetime)):
                    dt = pd.to_datetime(dt_idx).to_pydatetime()
                else:
                    logger.debug(f"Skipping index value {dt_idx} because it is not date-like")
                    continue
                
                close_val = float(row["close"])
                open_val = float(row["open"])
                high_val = float(row["high"])
                low_val = float(row["low"])
                
                if data_format == "single_value":
                    open_val = close_val
                    high_val = close_val
                    low_val = close_val

                vol = 1.0
                if "volume" in row and row["volume"] is not None:
                    try:
                        v = float(row["volume"])
                        if v > 0:
                            vol = v
                    except ValueError:
                        pass
                        
                records.append({
                    "timestamp": dt,
                    "open": open_val,
                    "high": high_val,
                    "low": low_val,
                    "close": close_val,
                    "volume": vol,
                    "timeframe": "1D",
                    "ticker_name": ticker_name
                })
            except Exception as e:
                logger.debug(f"Skipping TradingView row {dt_idx} for {ticker_name} due to format error: {e}")
                continue

        return records
