"""API Client for fetching data from TradingView using tvDatafeed."""

import logging
from typing import Any, Optional
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from data_engine.utils.retry import retry
from core.utils.exceptions import APIConnectionError

logger = logging.getLogger("data_engine.api.tradingview")

class TradingViewClient:
    """Client utilizing tvDatafeed to fetch historical data from TradingView."""

    def __init__(self):
        try:
            self.tv = TvDatafeed()
        except Exception as e:
            logger.error(f"Failed to initialize tvDatafeed: {e}")
            self.tv = None

    @retry(max_retries=3, backoff_factor=3.0, exceptions=(Exception,))
    def fetch_history(self, symbol: str, exchange: str, from_ts: int, to_ts: int) -> pd.DataFrame:
        """Fetches history of a symbol from TradingView.

        Args:
            symbol: Ticker symbol on TradingView (e.g. VN10Y, VN03Y).
            exchange: Exchange code (e.g. TVC).
            from_ts: Start time Unix timestamp.
            to_ts: End time Unix timestamp.

        Returns:
            pandas DataFrame containing the historical data.
        """
        if not self.tv:
            # Try to re-init
            self.tv = TvDatafeed()
            
        # Calculate dynamic n_bars
        n_days = int((to_ts - from_ts) / 86400) + 5
        n_bars = max(n_days, 5)
        
        logger.info(f"Querying TradingView for {exchange}:{symbol} (n_bars={n_bars})")
        try:
            df = self.tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.in_daily,
                n_bars=n_bars
            )
            if df is None or df.empty:
                raise APIConnectionError(f"No data returned from TradingView for {exchange}:{symbol}")
            return df
        except Exception as e:
            logger.error(f"TradingView fetch failed for {exchange}:{symbol}: {e}")
            raise APIConnectionError(f"TradingView fetch failed: {e}") from e
