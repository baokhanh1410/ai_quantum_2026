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

    def _get_duckdb_conn(self, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        """Helper to get a DuckDB connection."""
        return duckdb.connect(str(self._duckdb_path), read_only=read_only)

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
                with self._get_duckdb_conn(read_only=True) as conn:
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
        """Returns a mapping of ticker symbol -> ticker_id (defaults to DuckDB if present, else MySQL)."""
        if self._duckdb_path.exists():
            return self.fetch_duckdb_ticker_map()
        return self.fetch_mysql_ticker_map()

    def fetch_duckdb_ticker_map(self) -> Dict[str, int]:
        """Returns symbol -> id map for DuckDB."""
        if self._duckdb_path.exists():
            try:
                with self._get_duckdb_conn(read_only=True) as conn:
                    df = conn.execute("SELECT symbol, id FROM tickers").df()
                return dict(zip(df["symbol"], df["id"]))
            except Exception as de:
                logger.warning(f"DuckDB fetch_duckdb_ticker_map failed ({de}).")
        return {}

    def fetch_mysql_ticker_map(self) -> Dict[str, int]:
        """Returns symbol -> id map for MySQL."""
        try:
            session: Session = next(get_mysql_session())
            try:
                result = session.execute(text("SELECT symbol, id FROM tickers"))
                return {row[0]: row[1] for row in result.fetchall()}
            finally:
                session.close()
        except Exception as e:
            logger.error(f"MySQL fetch_mysql_ticker_map failed: {e}")
        return {}

    def fetch_duckdb_indicator_type_map(self) -> Dict[str, int]:
        """Returns name -> id map for DuckDB indicator_types."""
        if self._duckdb_path.exists():
            try:
                with self._get_duckdb_conn(read_only=True) as conn:
                    df = conn.execute("SELECT name, id FROM indicator_types").df()
                return dict(zip(df["name"], df["id"]))
            except Exception as de:
                logger.warning(f"DuckDB fetch_duckdb_indicator_type_map failed ({de}).")
        return {}

    def fetch_mysql_indicator_type_map(self) -> Dict[str, int]:
        """Returns name -> id map for MySQL indicator_types."""
        try:
            session: Session = next(get_mysql_session())
            try:
                result = session.execute(text("SELECT name, id FROM indicator_types"))
                return {row[0]: row[1] for row in result.fetchall()}
            finally:
                session.close()
        except Exception as e:
            logger.error(f"MySQL fetch_mysql_indicator_type_map failed: {e}")
        return {}

    def fetch_duckdb_macro_type_map(self) -> Dict[str, int]:
        """Returns name -> id map for DuckDB macro_types."""
        if self._duckdb_path.exists():
            try:
                with self._get_duckdb_conn(read_only=True) as conn:
                    df = conn.execute("SELECT name, id FROM macro_types").df()
                return dict(zip(df["name"], df["id"]))
            except Exception as de:
                logger.warning(f"DuckDB fetch_duckdb_macro_type_map failed ({de}).")
        return {}

    def fetch_mysql_macro_type_map(self) -> Dict[str, int]:
        """Returns name -> id map for MySQL macro_types."""
        try:
            session: Session = next(get_mysql_session())
            try:
                result = session.execute(text("SELECT name, id FROM macro_types"))
                return {row[0]: row[1] for row in result.fetchall()}
            finally:
                session.close()
        except Exception as e:
            logger.error(f"MySQL fetch_mysql_macro_type_map failed: {e}")
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
        """Upserts an indicator type in DuckDB and MySQL and returns its ID."""
        if name in self._indicator_type_cache:
            return self._indicator_type_cache[name]

        indicator_id: Optional[int] = None

        if self._duckdb_path.exists():
            try:
                with self._get_duckdb_conn(read_only=False) as conn:
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
            except Exception as de:
                logger.warning(f"DuckDB upsert_indicator_type failed ({de}). Falling back to MySQL...")

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
                if result:
                    mysql_id = result[0]
                    if indicator_id is None:
                        indicator_id = mysql_id
            finally:
                session.close()
        except Exception as e:
            logger.error(f"MySQL upsert_indicator_type failed for {name}: {e}")
            if indicator_id is None:
                raise DatabaseWriteError(f"Upsert indicator_type failed for {name}: {e}") from e

        if indicator_id is not None:
            self._indicator_type_cache[name] = indicator_id
            return indicator_id

        raise DatabaseWriteError(f"Could not upsert indicator_type '{name}'")

    def bulk_insert_indicator_values(
        self,
        records: List[Dict[str, Any]],
    ) -> int:
        """Bulk inserts computed indicator values into DuckDB and MySQL."""
        if not records:
            return 0

        duckdb_success = False
        if self._duckdb_path.exists():
            try:
                duck_ticker_map = self.fetch_duckdb_ticker_map()
                duck_ind_map = self.fetch_duckdb_indicator_type_map()

                duck_records = []
                for r in records:
                    t_id = r.get("ticker_id") if "ticker_id" in r else duck_ticker_map.get(r.get("symbol"))
                    ind_id = r.get("indicator_type_id") if "indicator_type_id" in r else duck_ind_map.get(r.get("indicator_name"))
                    if t_id is not None and ind_id is not None:
                        duck_records.append({
                            "ticker_id": t_id,
                            "indicator_type_id": ind_id,
                            "timestamp": r["timestamp"],
                            "value": r["value"],
                        })

                if duck_records:
                    with self._get_duckdb_conn(read_only=False) as conn:
                        df_rec = pd.DataFrame(duck_records)
                        conn.register('temp_tiv', df_rec)
                        conn.execute("""
                            DELETE FROM technical_indicator_values
                            WHERE EXISTS (
                                SELECT 1 FROM temp_tiv t
                                WHERE technical_indicator_values.ticker_id = t.ticker_id
                                  AND technical_indicator_values.indicator_type_id = t.indicator_type_id
                                  AND technical_indicator_values.timestamp = t.timestamp
                            )
                        """)
                        conn.execute("""
                            INSERT INTO technical_indicator_values (id, ticker_id, indicator_type_id, timestamp, value)
                            SELECT (SELECT COALESCE(MAX(id), 0) FROM technical_indicator_values) + row_number() OVER (),
                                   ticker_id, indicator_type_id, timestamp, value
                            FROM temp_tiv
                        """)
                        conn.unregister('temp_tiv')
                    logger.info(f"Bulk inserted {len(duck_records)} indicator values into DuckDB.")
                    duckdb_success = True
            except Exception as de:
                logger.warning(f"DuckDB bulk_insert_indicator_values failed ({de}). Falling back to MySQL...")

        # Batch insert into MySQL using MySQL specific ticker_id and indicator_type_id
        mysql_ticker_map = self.fetch_mysql_ticker_map()
        mysql_ind_map = self.fetch_mysql_indicator_type_map()

        batch_size = 5000
        formatted_records = []
        import numpy as np
        for r in records:
            t_id = mysql_ticker_map.get(r.get("symbol")) if "symbol" in r else r.get("ticker_id")
            ind_id = mysql_ind_map.get(r.get("indicator_name")) if "indicator_name" in r else r.get("indicator_type_id")
            if t_id is None or ind_id is None:
                continue

            ts = r["timestamp"]
            if hasattr(ts, "strftime"):
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts_str = str(ts)
            val = r["value"]
            formatted_records.append({
                "ticker_id": t_id,
                "indicator_type_id": ind_id,
                "timestamp": ts_str,
                "value": float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else None,
            })

        if formatted_records:
            try:
                session: Session = next(get_mysql_session())
                try:
                    query = text("""
                        INSERT INTO technical_indicator_values (ticker_id, indicator_type_id, timestamp, value)
                        VALUES (:ticker_id, :indicator_type_id, :timestamp, :value)
                        ON DUPLICATE KEY UPDATE value = VALUES(value);
                    """)
                    for i in range(0, len(formatted_records), batch_size):
                        chunk = formatted_records[i : i + batch_size]
                        session.execute(query, chunk)
                        session.commit()
                    logger.info(f"Bulk inserted {len(formatted_records)} indicator values into MySQL.")
                except Exception as e:
                    session.rollback()
                    logger.error(f"MySQL bulk_insert_indicator_values failed: {e}")
                    if not duckdb_success:
                        raise DatabaseWriteError(f"MySQL bulk_insert_indicator_values failed: {e}") from e
                finally:
                    session.close()
            except Exception as e:
                if not duckdb_success:
                    raise DatabaseWriteError(f"Both DuckDB and MySQL bulk_insert_indicator_values failed: {e}") from e

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
        """Upserts a macro type in DuckDB and MySQL and returns its ID."""
        if name in self._macro_type_cache:
            return self._macro_type_cache[name]

        macro_id: Optional[int] = None

        if self._duckdb_path.exists():
            try:
                with self._get_duckdb_conn(read_only=False) as conn:
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
            except Exception as de:
                logger.warning(f"DuckDB upsert_macro_type failed ({de}). Falling back to MySQL...")

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
                if result:
                    mysql_id = result[0]
                    if macro_id is None:
                        macro_id = mysql_id
            finally:
                session.close()
        except Exception as e:
            logger.error(f"MySQL upsert_macro_type failed for {name}: {e}")
            if macro_id is None:
                raise DatabaseWriteError(f"Upsert macro_type failed for {name}: {e}") from e

        if macro_id is not None:
            self._macro_type_cache[name] = macro_id
            return macro_id

        raise DatabaseWriteError(f"Could not upsert macro_type '{name}'")

    def bulk_insert_macro_values(
        self,
        records: List[Dict[str, Any]],
    ) -> int:
        """Bulk inserts computed macro values into DuckDB and MySQL."""
        if not records:
            return 0

        duckdb_success = False
        if self._duckdb_path.exists():
            try:
                duck_macro_map = self.fetch_duckdb_macro_type_map()
                duck_records = []
                for r in records:
                    m_id = r.get("macro_type_id") if "macro_type_id" in r else duck_macro_map.get(r.get("macro_name"))
                    if m_id is not None:
                        duck_records.append({
                            "macro_type_id": m_id,
                            "timestamp": r["timestamp"],
                            "value": r["value"],
                        })

                if duck_records:
                    with self._get_duckdb_conn(read_only=False) as conn:
                        df_rec = pd.DataFrame(duck_records)
                        df_rec['timestamp'] = pd.to_datetime(df_rec['timestamp']).dt.date
                        conn.register('temp_mv', df_rec)
                        conn.execute("""
                            DELETE FROM macro_values
                            WHERE EXISTS (
                                SELECT 1 FROM temp_mv t
                                WHERE macro_values.macro_type_id = t.macro_type_id
                                  AND macro_values.timestamp = t.timestamp
                            )
                        """)
                        conn.execute("""
                            INSERT INTO macro_values (id, macro_type_id, timestamp, value)
                            SELECT (SELECT COALESCE(MAX(id), 0) FROM macro_values) + row_number() OVER (),
                                   macro_type_id, timestamp, value
                            FROM temp_mv
                        """)
                        conn.unregister('temp_mv')
                    logger.info(f"Bulk inserted {len(duck_records)} macro values into DuckDB.")
                    duckdb_success = True
            except Exception as de:
                logger.warning(f"DuckDB bulk_insert_macro_values failed ({de}). Falling back to MySQL...")

        mysql_macro_map = self.fetch_mysql_macro_type_map()
        batch_size = 5000
        formatted_macro = []
        import numpy as np
        for r in records:
            m_id = mysql_macro_map.get(r.get("macro_name")) if "macro_name" in r else r.get("macro_type_id")
            if m_id is None:
                continue

            ts = r["timestamp"]
            if hasattr(ts, "strftime"):
                ts_str = ts.strftime("%Y-%m-%d")
            else:
                ts_str = str(ts)[:10]
            val = r["value"]
            formatted_macro.append({
                "macro_type_id": m_id,
                "timestamp": ts_str,
                "value": float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else None,
            })

        if formatted_macro:
            try:
                session: Session = next(get_mysql_session())
                try:
                    query = text("""
                        INSERT INTO macro_values (macro_type_id, timestamp, value)
                        VALUES (:macro_type_id, :timestamp, :value)
                        ON DUPLICATE KEY UPDATE value = VALUES(value);
                    """)
                    for i in range(0, len(formatted_macro), batch_size):
                        chunk = formatted_macro[i : i + batch_size]
                        session.execute(query, chunk)
                        session.commit()
                    logger.info(f"Bulk inserted {len(formatted_macro)} macro values into MySQL.")
                except Exception as e:
                    session.rollback()
                    logger.error(f"MySQL bulk_insert_macro_values failed: {e}")
                    if not duckdb_success:
                        raise DatabaseWriteError(f"MySQL bulk_insert_macro_values failed: {e}") from e
                finally:
                    session.close()
            except Exception as e:
                if not duckdb_success:
                    raise DatabaseWriteError(f"Both DuckDB and MySQL bulk_insert_macro_values failed: {e}") from e

        return len(records)
