"""MySQL repository for reading raw OHLCV data and writing indicator results."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.database.connection import get_mysql_session
from core.utils.exceptions import DatabaseReadError, DatabaseWriteError

logger = logging.getLogger("feature_engine.database.mysql")


class MySQLRepository:
    """Repository for MySQL CRUD operations on OHLCV and technical indicator tables."""

    def __init__(self) -> None:
        self._indicator_type_cache: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # READ: Raw data
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        start_date: str,
        end_date: str,
        ticker_symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetches OHLCV data from MySQL joined with ticker metadata.

        Args:
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            ticker_symbols: Optional list of ticker symbols to filter.

        Returns:
            DataFrame with columns: symbol, timestamp, open, high, low, close,
            adjusted_close, volume, exchange, asset_class_id.
        """
        session: Session = next(get_mysql_session())
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
                WHERE o.timestamp BETWEEN :start_date AND :end_date
            """
            params: Dict[str, Any] = {"start_date": start_date, "end_date": end_date}

            if ticker_symbols:
                placeholders = ", ".join([f":sym_{i}" for i in range(len(ticker_symbols))])
                query += f" AND t.symbol IN ({placeholders})"
                for i, sym in enumerate(ticker_symbols):
                    params[f"sym_{i}"] = sym

            query += " ORDER BY t.symbol, o.timestamp"
            result = session.execute(text(query), params)
            rows = result.fetchall()
            columns = [
                "symbol", "timestamp", "open", "high", "low", "close",
                "adjusted_close", "volume", "exchange", "asset_class_id",
            ]
            df = pd.DataFrame(rows, columns=columns)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV from MySQL: {e}")
            raise DatabaseReadError(f"MySQL OHLCV read failed: {e}") from e
        finally:
            session.close()

    def fetch_ticker_id_map(self) -> Dict[str, int]:
        """Returns a mapping of ticker symbol -> ticker_id from MySQL.

        Returns:
            Dictionary mapping symbol strings to their integer IDs.
        """
        session: Session = next(get_mysql_session())
        try:
            result = session.execute(text("SELECT symbol, id FROM tickers"))
            return {row[0]: row[1] for row in result.fetchall()}
        except Exception as e:
            logger.error(f"Failed to fetch ticker map: {e}")
            raise DatabaseReadError(f"MySQL ticker map read failed: {e}") from e
        finally:
            session.close()

    # ------------------------------------------------------------------
    # WRITE: Indicator types
    # ------------------------------------------------------------------

    def upsert_indicator_type(
        self,
        name: str,
        category: str,
        window_size: int,
        description: str,
    ) -> int:
        """Upserts an indicator type and returns its ID.

        Args:
            name: Indicator name (e.g. 'RSI').
            category: Category (e.g. 'trend').
            window_size: Window size for the indicator.
            description: Human-readable description.

        Returns:
            The integer ID of the indicator type.
        """
        if name in self._indicator_type_cache:
            return self._indicator_type_cache[name]

        session: Session = next(get_mysql_session())
        try:
            session.execute(
                text("""
                    INSERT INTO indicator_types (name, category, window_size, description)
                    VALUES (:name, :category, :window_size, :description)
                    ON DUPLICATE KEY UPDATE
                        category = VALUES(category),
                        window_size = VALUES(window_size),
                        description = VALUES(description);
                """),
                {
                    "name": name,
                    "category": category,
                    "window_size": window_size,
                    "description": description,
                },
            )
            session.commit()

            result = session.execute(
                text("SELECT id FROM indicator_types WHERE name = :name"),
                {"name": name},
            ).fetchone()
            indicator_id: int = result[0]
            self._indicator_type_cache[name] = indicator_id
            return indicator_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to upsert indicator type '{name}': {e}")
            raise DatabaseWriteError(f"Indicator type upsert failed: {e}") from e
        finally:
            session.close()

    # ------------------------------------------------------------------
    # WRITE: Indicator values (bulk)
    # ------------------------------------------------------------------

    def bulk_insert_indicator_values(
        self,
        records: List[Dict[str, Any]],
    ) -> int:
        """Bulk inserts computed indicator values into technical_indicator_values.

        Each record must have keys: ticker_id, indicator_type_id, timestamp, value.

        Args:
            records: List of dicts with indicator value data.

        Returns:
            The number of records inserted/updated.
        """
        if not records:
            return 0

        session: Session = next(get_mysql_session())
        try:
            query = text("""
                INSERT INTO technical_indicator_values
                    (ticker_id, indicator_type_id, timestamp, value)
                VALUES (:ticker_id, :indicator_type_id, :timestamp, :value)
                ON DUPLICATE KEY UPDATE
                    value = VALUES(value);
            """)
            session.execute(query, records)
            session.commit()
            logger.info(f"Bulk inserted {len(records)} indicator values into MySQL.")
            return len(records)
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to bulk insert indicator values: {e}")
            raise DatabaseWriteError(f"MySQL bulk insert failed: {e}") from e
        finally:
            session.close()
