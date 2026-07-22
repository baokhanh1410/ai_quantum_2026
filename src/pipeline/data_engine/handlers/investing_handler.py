"""Handler for formatting raw Investing.com TVC JSON responses."""

import logging
import datetime
from typing import List, Dict, Any

logger = logging.getLogger("data_engine.handlers.investing")

class InvestingHandler:
    """Formats Investing.com TVC JSON into unified database records."""

    def format_data(self, data: Dict[str, Any], ticker_name: str, data_format: str = "ohlcv") -> List[Dict[str, Any]]:
        """Transforms TVC JSON payload to standard database records.

        Args:
            data: Raw JSON dict from InvestingClient.
            ticker_name: Expected target symbol.
            data_format: Format type (e.g. 'ohlcv' or 'single_value').

        Returns:
            List of OHLCV records.
        """
        if not data or data.get("s") != "ok":
            return []

        t_list = data.get("t", [])
        o_list = data.get("o", [])
        h_list = data.get("h", [])
        l_list = data.get("l", [])
        c_list = data.get("c", [])
        # Volume might not be present or populated
        v_list = data.get("v", [])

        records = []
        for idx in range(len(t_list)):
            try:
                # TVC timestamp is in seconds, convert to datetime
                dt = datetime.datetime.fromtimestamp(t_list[idx])
                
                open_val = float(o_list[idx])
                high_val = float(h_list[idx])
                low_val = float(l_list[idx])
                close_val = float(c_list[idx])
                
                # Single value alignment: overwrite other fields with close if format is single_value
                if data_format == "single_value":
                    open_val = close_val
                    high_val = close_val
                    low_val = close_val
                
                # Default volume for vĩ mô index is 1.0 as per spec
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
                logger.debug(f"Skipping TVC index {idx} for {ticker_name} due to format error: {e}")
                continue

        return records
