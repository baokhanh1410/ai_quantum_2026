"""Handler for formatting raw Yahoo Finance JSON responses."""

import logging
import datetime
from typing import List, Dict, Any

logger = logging.getLogger("data_engine.handlers.yahoo_finance")

class YahooFinanceHandler:
    """Formats Yahoo Finance JSON data into unified database records."""

    def format_data(self, data: Dict[str, Any], ticker_name: str, data_format: str = "ohlcv") -> List[Dict[str, Any]]:
        """Transforms Yahoo Finance JSON payload to standard database records.

        Args:
            data: Raw JSON dict from YahooFinanceClient.
            ticker_name: Expected target symbol in the database (e.g. DXY, USDVND, XAUUSD).
            data_format: Format type (e.g. 'ohlcv' or 'single_value').

        Returns:
            List of OHLCV records.
        """
        if not data or "chart" not in data or not data["chart"].get("result"):
            return []

        try:
            result = data["chart"]["result"][0]
            timestamps = result.get("timestamp", [])
            quote = result.get("indicators", {}).get("quote", [{}])[0]
            
            o_list = quote.get("open", [])
            h_list = quote.get("high", [])
            l_list = quote.get("low", [])
            c_list = quote.get("close", [])
            v_list = quote.get("volume", [])
        except Exception as e:
            logger.error(f"Failed to parse Yahoo Finance response structure for {ticker_name}: {e}")
            return []

        records = []
        for idx in range(len(timestamps)):
            try:
                close_val = c_list[idx]
                if close_val is None:
                    continue
                
                # Yahoo Finance timestamp is in seconds, convert to datetime
                dt = datetime.datetime.fromtimestamp(timestamps[idx])
                
                close_val = float(close_val)
                open_val = float(o_list[idx]) if o_list and o_list[idx] is not None else close_val
                high_val = float(h_list[idx]) if h_list and h_list[idx] is not None else close_val
                low_val = float(l_list[idx]) if l_list and l_list[idx] is not None else close_val
                
                if data_format == "single_value":
                    open_val = close_val
                    high_val = close_val
                    low_val = close_val
                    
                vol = 1.0
                if v_list and idx < len(v_list) and v_list[idx] is not None:
                    try:
                        v = float(v_list[idx])
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
                logger.debug(f"Skipping Yahoo Finance record {idx} for {ticker_name} due to format error: {e}")
                continue

        return records
