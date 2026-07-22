"""Handler for SJC Gold Price data (splitting buy and sell price histories)."""

import logging
import io
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger("data_engine.handlers.gold")

class GoldHandler:
    """Formats SJC CSV response and produces buy/sell records."""

    def format_data(self, csv_content: str) -> List[Dict[str, Any]]:
        """Parses raw CSV content and splits into SJC_BUY and SJC_SELL records.

        Args:
            csv_content: Raw CSV string from GoldClient.

        Returns:
            List of OHLCV records for both tickers.
        """
        if not csv_content.strip():
            return []

        try:
            df = pd.read_csv(io.StringIO(csv_content))
            if df.empty:
                return []

            # Standardize timestamp to date
            df["date_parsed"] = pd.to_datetime(df["timestamp"])
            
            records = []
            for _, row in df.iterrows():
                dt = row["date_parsed"].to_pydatetime()
                # Multiply by 1,000,000 to convert to VND as in original notebook crawing logic
                buy_val = float(row["buy_1l"]) * 1000000.0
                sell_val = float(row["sell_1l"]) * 1000000.0
                
                # Add SJC_BUY record
                records.append({
                    "timestamp": dt,
                    "open": buy_val,
                    "high": buy_val,
                    "low": buy_val,
                    "close": buy_val,
                    "volume": 1.0,  # Default volume allocation for single-value assets
                    "timeframe": "1D",
                    "ticker_name": "SJC_BUY"
                })
                
                # Add SJC_SELL record
                records.append({
                    "timestamp": dt,
                    "open": sell_val,
                    "high": sell_val,
                    "low": sell_val,
                    "close": sell_val,
                    "volume": 1.0,  # Default volume
                    "timeframe": "1D",
                    "ticker_name": "SJC_SELL"
                })

            return records
        except Exception as e:
            logger.error(f"Error handling SJC Gold CSV data: {e}")
            return []
