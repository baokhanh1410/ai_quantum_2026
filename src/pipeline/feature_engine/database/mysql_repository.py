"""DuckDB & MySQL repository for reading raw OHLCV data and writing indicator results."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd
import duckdb
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.database.connection import get_mysql_session
from core.config.settings import settings
from core.utils.exceptions import DatabaseReadError, DatabaseWriteError

logger = logging.getLogger("feature_engine.database.mysql")


class MySQLRepository:
    """Repository for database CRUD operations on OHLCV, indicator, and macro tables.
    Prioritizes DuckDB (portfolio.duckdb) for complete analytical time-series,
    falling back to MySQL when necessary.
    """

    def __init__(self) -> None:
        self._indicator_type_cache: Dict[str, int] = {}
        self._macro_type_cache: Dict[str, int] = {}
        self._duckdb_path = settings.root_dir / "data" / "processed" / "portfolio.duckdb"

    def _get_duckdb_conn(self) -> duckdb.DuckDBPyConnection:
        """Helper to get a DuckDB connection."""
        return duckdb.connect(str(self._duckdb_path))

    # ------------------------------------------------------------------
    # READ: Raw data
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        start_date: str,
        end_date: str,
        ticker_symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetches OHLCV data from DuckDB (falling back to MySQL if DuckDB is absent)."""
        df = pd.DataFrame()

        # 1. Primary: DuckDB (Contains full 125 symbols)
        if self._duckdb_path.exists():
            try:
                conn = self._get_duckdb_conn()
                sym_clause = ""
                if ticker_symbols:
                    quoted = ", ".join([f"'{s}'" for s in ticker_symbols])
                    sym_clause = f"AND t.symbol IN ({quoted})"
                
                duck_query = f"""
                    SELECT
                        t.symbol,
                        o.timestamp,
                        o.open,
                        o.high,
                        o.low,
                        o.close,
                        o.close as adjusted_close,
                        o.volume,
                        t.exchange,
                        t.asset_class_id
                    FROM ohlcv o
                    JOIN tickers t ON o.ticker_id = t.id
                    WHERE o.timestamp >= '{start_date}' AND o.timestamp <= '{end_date}'
                    {sym_clause}
                    ORDER BY t.symbol, o.timestamp
                """
                df = conn.execute(duck_query).df()
                logger.info(f"Fetched {len(df)} OHLCV rows for {df['symbol'].nunique()} symbols from DuckDB.")
            except Exception as de:
                logger.warning(f"DuckDB OHLCV fetch failed ({de}). Falling back to MySQL...")

        # 2. Fallback: MySQL
        if df.empty:
            try:
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
                finally:
                    session.close()
            except Exception as e:
                logger.error(f"MySQL OHLCV fetch failed: {e}")

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def fetch_ticker_id_map(self) -> Dict[str, int]:
        """Returns a mapping of ticker symbol -> ticker_id."""
        if self._duckdb_path.exists():
            try:
                conn = self._get_duckdb_conn()
                df = conn.execute("SELECT symbol, id FROM tickers").df()
                return dict(zip(df["symbol"], df["id"]))
            except Exception as de:
                logger.warning(f"DuckDB fetch_ticker_id_map failed ({de}). Falling back to MySQL...")

        try:
            session: Session = next(get_mysql_session())
            try:
                result = session.execute(text("SELECT symbol, id FROM tickers"))
                return {row[0]: row[1] for row in result.fetchall()}
            finally:
                session.close()
        except Exception as e:
            logger.error(f"MySQL fetch_ticker_id_map failed: {e}")
        return {}

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
        """Upserts an indicator type and returns its ID."""
        if name in self._indicator_type_cache:
            return self._indicator_type_cache[name]

        if self._duckdb_path.exists():
            conn = self._get_duckdb_conn()
            res = conn.execute("SELECT id FROM indicator_types WHERE name = ?", [name]).fetchone()
            if res:
                indicator_id = res[0]
            else:
                max_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM indicator_types").fetchone()[0]
                conn.execute(
                    "INSERT INTO indicator_types (id, name, category, window_size, description) VALUES (?, ?, ?, ?, ?)",
                    [max_id, name, category, window_size, description]
                )
                indicator_id = max_id
            self._indicator_type_cache[name] = indicator_id
            return indicator_id

        try:
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
                    {"name": name, "category": category, "window_size": window_size, "description": description},
                )
                session.commit()
                result = session.execute(text("SELECT id FROM indicator_types WHERE name = :name"), {"name": name}).fetchone()
                indicator_id = result[0]
                self._indicator_type_cache[name] = indicator_id
                return indicator_id
            finally:
                session.close()
        except Exception as e:
            raise DatabaseWriteError(f"Upsert indicator_type failed for {name}: {e}") from e

    def bulk_insert_indicator_values(
        self,
        records: List[Dict[str, Any]],
    ) -> int:
        """Bulk inserts computed indicator values."""
        if not records:
            return 0

        if self._duckdb_path.exists():
            try:
                conn = self._get_duckdb_conn()
                df_rec = pd.DataFrame(records)
                conn.register('temp_tiv', df_rec)
                conn.execute("""
                    INSERT INTO technical_indicator_values (id, ticker_id, indicator_type_id, timestamp, value)
                    SELECT (SELECT COALESCE(MAX(id), 0) FROM technical_indicator_values) + row_number() OVER (), ticker_id, indicator_type_id, timestamp, value
                    FROM temp_tiv
                """)
                logger.info(f"Bulk inserted {len(records)} indicator values into DuckDB.")
                return len(records)
            except Exception as de:
                logger.warning(f"DuckDB bulk_insert_indicator_values failed ({de}). Falling back to MySQL...")

        try:
            session: Session = next(get_mysql_session())
            try:
                query = text("""
                    INSERT INTO technical_indicator_values (ticker_id, indicator_type_id, timestamp, value)
                    VALUES (:ticker_id, :indicator_type_id, :timestamp, :value)
                    ON DUPLICATE KEY UPDATE value = VALUES(value);
                """)
                session.execute(query, records)
                session.commit()
                logger.info(f"Bulk inserted {len(records)} indicator values into MySQL.")
                return len(records)
            finally:
                session.close()
        except Exception as e:
            logger.error(f"MySQL bulk_insert_indicator_values failed: {e}")

        return len(records)

    # ------------------------------------------------------------------
    # WRITE: Macro types & values
    # ------------------------------------------------------------------

    def upsert_macro_type(
        self,
        name: str,
        unit: str = "",
        description: str = "",
    ) -> int:
        """Upserts a macro type and returns its ID."""
        if name in self._macro_type_cache:
            return self._macro_type_cache[name]

        if self._duckdb_path.exists():
            conn = self._get_duckdb_conn()
            res = conn.execute("SELECT id FROM macro_types WHERE name = ?", [name]).fetchone()
            if res:
                macro_id = res[0]
            else:
                max_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM macro_types").fetchone()[0]
                conn.execute(
                    "INSERT INTO macro_types (id, name, unit, description) VALUES (?, ?, ?, ?)",
                    [max_id, name, unit, description]
                )
                macro_id = max_id
            self._macro_type_cache[name] = macro_id
            return macro_id

        try:
            session: Session = next(get_mysql_session())
            try:
                session.execute(
                    text("""
                        INSERT INTO macro_types (name, unit, description)
                        VALUES (:name, :unit, :description)
                        ON DUPLICATE KEY UPDATE unit = VALUES(unit), description = VALUES(description);
                    """),
                    {"name": name, "unit": unit, "description": description},
                )
                session.commit()
                result = session.execute(text("SELECT id FROM macro_types WHERE name = :name"), {"name": name}).fetchone()
                macro_id = result[0]
                self._macro_type_cache[name] = macro_id
                return macro_id
            finally:
                session.close()
        except Exception as e:
            raise DatabaseWriteError(f"Upsert macro_type failed for {name}: {e}") from e

    def bulk_insert_macro_values(
        self,
        records: List[Dict[str, Any]],
    ) -> int:
        """Bulk inserts computed macro values."""
        if not records:
            return 0

        if self._duckdb_path.exists():
            try:
                conn = self._get_duckdb_conn()
                df_rec = pd.DataFrame(records)
                conn.register('temp_mv', df_rec)
                conn.execute("""
                    INSERT INTO macro_values (id, macro_type_id, timestamp, value)
                    SELECT (SELECT COALESCE(MAX(id), 0) FROM macro_values) + row_number() OVER (), macro_type_id, CAST(timestamp AS DATE), value
                    FROM temp_mv
                """)
                logger.info(f"Bulk inserted {len(records)} macro values into DuckDB.")
                return len(records)
            except Exception as de:
                logger.warning(f"DuckDB bulk_insert_macro_values failed ({de}). Falling back to MySQL...")

        try:
            session: Session = next(get_mysql_session())
            try:
                query = text("""
                    INSERT INTO macro_values (macro_type_id, timestamp, value)
                    VALUES (:macro_type_id, :timestamp, :value)
                    ON DUPLICATE KEY UPDATE value = VALUES(value);
                """)
                session.execute(query, records)
                session.commit()
                logger.info(f"Bulk inserted {len(records)} macro values into MySQL.")
                return len(records)
            finally:
                session.close()
        except Exception as e:
            logger.error(f"MySQL bulk_insert_macro_values failed: {e}")

        return len(records)
