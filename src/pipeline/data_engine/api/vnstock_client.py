"""Client wrapper for the legacy vnstock library."""

import logging
import pandas as pd
from typing import Optional
from vnstock import stock_historical_data
from data_engine.utils.retry import retry
from core.utils.exceptions import APIConnectionError

logger = logging.getLogger("data_engine.api.vnstock")

class VNStockClient:
    """Client wrapper for vnstock's stock_historical_data."""

    @retry(max_retries=3, backoff_factor=2.0, exceptions=(Exception,))
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        resolution: str = "1D",
        type: str = "stock"
    ) -> pd.DataFrame:
        """Fetches historical stock prices using the vnstock library.

        Args:
            symbol: Ticker symbol (e.g. FPT, FUEVFVND).
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            resolution: Resolution (e.g. 1D).
            type: Asset type (e.g. stock, index).

        Returns:
            A Pandas DataFrame containing the historical data.
        """
        logger.info(
            f"Fetching stock history for {symbol} from {start_date} to {end_date} (res={resolution}, type={type})"
        )
        try:
            df = stock_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                resolution=resolution,
                type=type
            )
            if df is None or df.empty:
                logger.warning(f"No historical data returned for {symbol}")
                return pd.DataFrame()
            return df
        except Exception as e:
            logger.error(f"Error fetching vnstock data for {symbol}: {e}")
            raise APIConnectionError(f"vnstock API error for {symbol}: {e}") from e
