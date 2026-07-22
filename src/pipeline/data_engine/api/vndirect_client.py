"""API Client for VNDirect sector indices."""

import logging
from typing import Any, Dict, Optional
from data_engine.api.base_client import BaseAPIClient
from core.config.settings import settings

logger = logging.getLogger("data_engine.api.vndirect")

class VNDirectClient(BaseAPIClient):
    """Client for pulling historical VNDirect sector index candles."""

    def __init__(self):
        vndirect_config = settings.apis.vndirect_sector_indices
        headers = vndirect_config.get("headers", {})
        if hasattr(headers, "to_dict"):
            headers_dict = headers.to_dict()
        else:
            headers_dict = dict(headers)
            
        super().__init__(default_headers=headers_dict)
        self.url = vndirect_config.url
        self.method = vndirect_config.get("method", "GET")
        self.resolution = vndirect_config.get("resolution", "D")

    def fetch_history(self, symbol: str, from_ts: int, to_ts: int) -> Dict[str, Any]:
        """Fetches history for a VNDirect sector symbol.

        Args:
            symbol: Sector ticker symbol (e.g. VNFIN, VNREAL).
            from_ts: Start time Unix timestamp (seconds).
            to_ts: End time Unix timestamp (seconds).

        Returns:
            Dict containing raw JSON response of index candles.
        """
        params = {
            "symbol": symbol,
            "resolution": self.resolution,
            "from": from_ts,
            "to": to_ts
        }
        logger.info(f"Fetching VNDirect index history for {symbol} from {from_ts} to {to_ts}")
        response = self.request(self.method, self.url, params=params)
        return response.json()
