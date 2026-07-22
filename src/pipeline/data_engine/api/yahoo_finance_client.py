"""API Client for fetching historical data from Yahoo Finance."""

import logging
import requests
from typing import Any, Dict
from data_engine.utils.retry import retry
from core.utils.exceptions import APIConnectionError

logger = logging.getLogger("data_engine.api.yahoo_finance")

class YahooFinanceClient:
    """Client fetching historical pricing from Yahoo Finance public REST endpoints."""

    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"

    @retry(max_retries=3, backoff_factor=2.0, exceptions=(Exception,))
    def fetch_history(self, symbol: str, from_ts: int, to_ts: int) -> Dict[str, Any]:
        """Fetches history of a symbol from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g. DX-Y.NYB, USDVND=X, GC=F).
            from_ts: Start time Unix timestamp.
            to_ts: End time Unix timestamp.

        Returns:
            Dict containing raw Yahoo Finance JSON response.
        """
        url = f"{self.base_url}/{symbol}"
        params = {
            "interval": "1d",
            "period1": from_ts,
            "period2": to_ts
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        logger.info(f"Querying Yahoo Finance for {symbol} ({from_ts} to {to_ts})")
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} from Yahoo Finance: {e}")
            raise APIConnectionError(f"Yahoo Finance request failed for {symbol}: {e}") from e
