"""DuckDB repository for reading raw OHLCV data from the local analytical store."""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from core.database.connection import get_duckdb_connection
from core.utils.exceptions import DatabaseReadError

logger = logging.getLogger("feature_engine.database.duckdb")


class DuckDBRepository:
    """Repository for DuckDB read operations on OHLCV and ticker data."""

    def fetch_ohlcv(
        self,
        start_date: str,
        end_date: str,
        ticker_symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetches OHLCV data from DuckDB joined with ticker metadata.

        Args:
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            ticker_symbols: Optional list of ticker symbols to filter.

        Returns:
            DataFrame with columns: symbol, timestamp, open, high, low, close,
            adjusted_close, volume, exchange, asset_class_id.
        """
        conn = get_duckdb_connection()
        try:
            query = """
                SELECT
                    t.symbol,
                    o.timestamp,
                    o.open,
                    o.high,
                    o.low,
                    o.close,
                    o.adjusted_close,
                    o.volume,
                    t.exchange,
                    t.asset_class_id
                FROM ohlcv o
                JOIN tickers t ON o.ticker_id = t.id
                WHERE o.timestamp BETWEEN ? AND ?
            """
            params: List[Any] = [start_date, end_date]

            if ticker_symbols:
                placeholders = ", ".join(["?" for _ in ticker_symbols])
                query += f" AND t.symbol IN ({placeholders})"
                params.extend(ticker_symbols)

            query += " ORDER BY t.symbol, o.timestamp"
            df = conn.execute(query, params).fetchdf()
            df.columns = [
                "symbol", "timestamp", "open", "high", "low", "close",
                "adjusted_close", "volume", "exchange", "asset_class_id",
            ]
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV from DuckDB: {e}")
            raise DatabaseReadError(f"DuckDB OHLCV read failed: {e}") from e
        finally:
            conn.close()

    def fetch_ticker_id_map(self) -> Dict[str, int]:
        """Returns a mapping of ticker symbol -> ticker_id from DuckDB.

        Returns:
            Dictionary mapping symbol strings to their integer IDs.
        """
        conn = get_duckdb_connection()
        try:
            rows = conn.execute("SELECT symbol, id FROM tickers").fetchall()
            return {row[0]: row[1] for row in rows}
        except Exception as e:
            logger.error(f"Failed to fetch ticker map from DuckDB: {e}")
            raise DatabaseReadError(f"DuckDB ticker map read failed: {e}") from e
        finally:
            conn.close()
