"""Handler for formatting raw VNDirect sector indices JSON responses."""

import logging
import datetime
from typing import List, Dict, Any

logger = logging.getLogger("data_engine.handlers.vndirect")

class VNDirectHandler:
    """Formats VNDirect API JSON response into unified database records."""

    def format_data(self, data: Dict[str, Any], symbol: str) -> List[Dict[str, Any]]:
        """Transforms VNDirect JSON response to unified database records.

        Args:
            data: Raw JSON dict from VNDirect client.
            symbol: Ticker symbol (e.g. VNFIN, VNREAL).

        Returns:
            List of OHLCV records.
        """
        if not data or data.get("s") != "ok" or "t" not in data:
            return []

        t_list = data.get("t", [])
        o_list = data.get("o", [])
        h_list = data.get("h", [])
        l_list = data.get("l", [])
        c_list = data.get("c", [])
        v_list = data.get("v", [])

        records = []
        for idx in range(len(t_list)):
            try:
                # Convert unix seconds to datetime
                dt = datetime.datetime.fromtimestamp(t_list[idx])
                
                records.append({
                    "timestamp": dt,
                    "open": float(o_list[idx]),
                    "high": float(h_list[idx]),
                    "low": float(l_list[idx]),
                    "close": float(c_list[idx]),
                    "volume": float(v_list[idx]) if idx < len(v_list) else 1.0,
                    "timeframe": "1D",
                    "ticker_name": symbol
                })
            except Exception as e:
                logger.debug(f"Skipping VNDirect index {idx} for {symbol} due to format error: {e}")
                continue

        return records
