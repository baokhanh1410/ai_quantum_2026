"""Database repository for CRUD operations conforming to the 3NF database layout."""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from core.database.connection import get_mysql_session, get_duckdb_connection
from core.config.settings import settings
from core.utils.exceptions import DatabaseError

logger = logging.getLogger("data_engine.repository")

class DataRepository:
    """Repository handling persistence for AssetClasses, Tickers and OHLCV data."""

    def __init__(self):
        self._ticker_cache: Dict[str, int] = {}
        self._init_duckdb()
        self._init_mysql_asset_classes()
        self._init_duckdb_asset_classes()

    def _init_duckdb(self) -> None:
        """Initializes tables in DuckDB conforming to 3NF layout if they do not exist."""
        try:
            conn = get_duckdb_connection(read_only=False)
            try:
                has_symbol = False
                try:
                    columns = conn.execute("PRAGMA table_info('tickers')").fetchall()
                    has_symbol = any(c[1] == "symbol" for c in columns)
                except Exception:
                    pass
                    
                if not has_symbol:
                    logger.info("DuckDB tickers table is using old schema or doesn't exist.")
                    
                conn.execute("CREATE SEQUENCE IF NOT EXISTS asset_classes_id_seq;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS asset_classes (
                        id INTEGER DEFAULT nextval('asset_classes_id_seq') PRIMARY KEY,
                        name VARCHAR UNIQUE NOT NULL
                    );
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS asset_class_metadata (
                        asset_class_id INTEGER PRIMARY KEY REFERENCES asset_classes(id),
                        description VARCHAR,
                        settlement_type VARCHAR,
                        default_locked_days INTEGER DEFAULT 0,
                        price_limit_ratio DOUBLE,
                        default_lot_size INTEGER DEFAULT 100,
                        default_trading_fee DOUBLE DEFAULT 0.001,
                        allow_short BOOLEAN DEFAULT FALSE,
                        handler VARCHAR,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.execute("CREATE SEQUENCE IF NOT EXISTS tickers_id_seq;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tickers (
                        id INTEGER DEFAULT nextval('tickers_id_seq') PRIMARY KEY,
                        asset_class_id INTEGER NOT NULL REFERENCES asset_classes(id),
                        symbol VARCHAR UNIQUE NOT NULL,
                        name VARCHAR,
                        exchange VARCHAR,
                        active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.execute("CREATE SEQUENCE IF NOT EXISTS ohlcv_id_seq;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ohlcv (
                        id INTEGER DEFAULT nextval('ohlcv_id_seq') PRIMARY KEY,
                        ticker_id INTEGER REFERENCES tickers(id),
                        timestamp TIMESTAMP NOT NULL,
                        open DOUBLE,
                        high DOUBLE,
                        low DOUBLE,
                        close DOUBLE,
                        adjusted_close DOUBLE,
                        volume DOUBLE,
                        source VARCHAR,
                        data_quality VARCHAR,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (ticker_id, timestamp)
                    );
                """)
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"DuckDB tables init skipped or warning ({e}). Proceeding...")

    def _init_mysql_asset_classes(self) -> None:
        """Populates asset_classes (name only) and asset_class_metadata in MySQL using configurations in assets.yaml."""
        session = next(get_mysql_session())
        try:
            for class_name, config in settings.asset_class.items():
                class_id = config.asset_class_id
                desc = config.get("description", "")
                locked_days = config.get("locked_days", 0)
                settlement_type = f"T+{locked_days}"
                price_limit_ratio = config.get("price_limit_ratio", None)
                handler = config.get("handler", None)
                default_lot_size = 100 if class_id in [1, 2, 3] else 1
                default_trading_fee = 0.001 if class_id in [1, 2, 3] else 0.0
                
                session.execute(text("DELETE FROM asset_classes WHERE name = :name AND id != :id"), {"name": class_name, "id": class_id})
                session.execute(
                    text("""
                        INSERT INTO asset_classes (id, name)
                        VALUES (:id, :name)
                        ON DUPLICATE KEY UPDATE
                            name = VALUES(name);
                    """),
                    {"id": class_id, "name": class_name}
                )
                session.execute(
                    text("""
                        INSERT INTO asset_class_metadata (asset_class_id, description, settlement_type, default_locked_days, price_limit_ratio, default_lot_size, default_trading_fee, allow_short, handler)
                        VALUES (:id, :desc, :settlement, :locked_days, :price_limit, :lot_size, :fee, 0, :handler)
                        ON DUPLICATE KEY UPDATE
                            description = VALUES(description),
                            settlement_type = VALUES(settlement_type),
                            default_locked_days = VALUES(default_locked_days),
                            price_limit_ratio = VALUES(price_limit_ratio),
                            default_lot_size = VALUES(default_lot_size),
                            default_trading_fee = VALUES(default_trading_fee),
                            handler = VALUES(handler);
                    """),
                    {
                        "id": class_id,
                        "desc": desc,
                        "settlement": settlement_type,
                        "locked_days": locked_days,
                        "price_limit": price_limit_ratio,
                        "lot_size": default_lot_size,
                        "fee": default_trading_fee,
                        "handler": handler
                    }
                )
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to initialize MySQL asset classes metadata: {e}")
            raise DatabaseError(f"MySQL asset classes init failed: {e}") from e
        finally:
            session.close()



    def _init_duckdb_asset_classes(self) -> None:
        """Populates asset_classes and asset_class_metadata in DuckDB using configurations in assets.yaml."""
        try:
            conn = get_duckdb_connection(read_only=False)
            try:
                for class_name, config in settings.asset_class.items():
                    class_id = config.asset_class_id
                    desc = config.get("description", "")
                    locked_days = config.get("locked_days", 0)
                    settlement_type = f"T+{locked_days}"
                    price_limit_ratio = config.get("price_limit_ratio", None)
                    handler = config.get("handler", None)
                    default_lot_size = 100 if class_id in [1, 2, 3] else 1
                    default_trading_fee = 0.001 if class_id in [1, 2, 3] else 0.0
                    
                    try:
                        conn.execute("DELETE FROM asset_classes WHERE name = ? AND id != ?", (class_name, class_id))
                    except Exception:
                        pass
                    
                    conn.execute(
                        "INSERT INTO asset_classes (id, name) VALUES (?, ?) ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
                        (class_id, class_name)
                    )
                    conn.execute(
                        """
                        INSERT INTO asset_class_metadata (asset_class_id, description, settlement_type, default_locked_days, price_limit_ratio, default_lot_size, default_trading_fee, allow_short, handler)
                        VALUES (?, ?, ?, ?, ?, ?, ?, FALSE, ?)
                        ON CONFLICT (asset_class_id) DO UPDATE SET
                            description = EXCLUDED.description,
                            settlement_type = EXCLUDED.settlement_type,
                            default_locked_days = EXCLUDED.default_locked_days,
                            price_limit_ratio = EXCLUDED.price_limit_ratio,
                            default_lot_size = EXCLUDED.default_lot_size,
                            default_trading_fee = EXCLUDED.default_trading_fee,
                            handler = EXCLUDED.handler
                        """,
                        (class_id, desc, settlement_type, locked_days, price_limit_ratio, default_lot_size, default_trading_fee, handler)
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"DuckDB asset classes init skipped ({e}). Proceeding...")

    def _resolve_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """Resolves asset_class_id, exchange, and name for a symbol from settings."""
        # Fix: Ưu tiên tra cứu exchange từ ticker_exchange_map trong assets.yaml
        mapped_exchange = settings.ticker_exchange_map.get(symbol, None)

        vn = settings.apis.vnstock
        
        def in_list(val, attribute):
            if not hasattr(vn, attribute):
                return False
            attr = getattr(vn, attribute)
            if isinstance(attr, list):
                return val in attr
            return val == attr

        if in_list(symbol, "etf_symbols") or in_list(symbol, "hose_symbols"):
            return {"asset_class_id": 1, "exchange": mapped_exchange or "HOSE", "name": f"Stock {symbol} (HOSE)"}
        elif in_list(symbol, "hnx_symbols"):
            return {"asset_class_id": 2, "exchange": mapped_exchange or "HNX", "name": f"Stock {symbol} (HNX)"}
        elif in_list(symbol, "upcom_symbols"):
            return {"asset_class_id": 3, "exchange": mapped_exchange or "UPCOM", "name": f"Stock {symbol} (UPCOM)"}
            
        if symbol in ["SJC_BUY", "SJC_SELL"]:
            return {"asset_class_id": 5, "exchange": mapped_exchange or "SJC", "name": "Vàng SJC " + ("Mua" if "BUY" in symbol else "Bán")}
            
        if symbol.startswith("VNIBOR_"):
            return {"asset_class_id": 6, "exchange": mapped_exchange or "SBV", "name": f"Lãi suất liên ngân hàng SBV {symbol}"}
            
        for key, config in settings.macro_indices.items():
            sym = config.get("symbol")
            if isinstance(sym, list) and symbol in sym:
                return {"asset_class_id": 7, "exchange": mapped_exchange or "HOSE", "name": f"Chỉ số ngành HOSE {symbol}"}
            elif sym == symbol:
                return {"asset_class_id": 7, "exchange": mapped_exchange or "MACRO", "name": config.get("description", symbol)}
                
        return {"asset_class_id": 7, "exchange": mapped_exchange or "UNKNOWN", "name": f"Asset {symbol}"}

    def get_or_create_ticker_mysql(self, session: Session, symbol: str) -> int:
        """Retrieve ticker ID from MySQL or create it if not exists."""
        if symbol in self._ticker_cache:
            return self._ticker_cache[symbol]

        try:
            result = session.execute(
                text("SELECT id FROM tickers WHERE symbol = :symbol"), {"symbol": symbol}
            ).fetchone()

            if result:
                ticker_id = result[0]
            else:
                info = self._resolve_ticker_info(symbol)
                res = session.execute(
                    text("""
                        INSERT INTO tickers (asset_class_id, symbol, name, exchange, active)
                        VALUES (:asset_class_id, :symbol, :name, :exchange, 1)
                    """),
                    {
                        "asset_class_id": info["asset_class_id"],
                        "symbol": symbol,
                        "name": info["name"],
                        "exchange": info["exchange"]
                    }
                )
                session.commit()
                ticker_id = res.lastrowid
                
            self._ticker_cache[symbol] = ticker_id
            return ticker_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error getting/creating ticker {symbol} in MySQL: {e}")
            raise DatabaseError(f"MySQL ticker operation failed: {e}") from e

    def get_or_create_ticker_duckdb(self, symbol: str) -> int:
        """Retrieve ticker ID from DuckDB or create it if not exists."""
        try:
            conn = get_duckdb_connection(read_only=True)
            try:
                result = conn.execute("SELECT id FROM tickers WHERE symbol = ?", (symbol,)).fetchone()
                if result:
                    return result[0]
            finally:
                conn.close()
        except Exception:
            pass

        try:
            conn = get_duckdb_connection(read_only=False)
            try:
                info = self._resolve_ticker_info(symbol)
                conn.execute(
                    "INSERT INTO tickers (asset_class_id, symbol, name, exchange, active) VALUES (?, ?, ?, ?, TRUE)",
                    (info["asset_class_id"], symbol, info["name"], info["exchange"])
                )
                conn.commit()
                res = conn.execute("SELECT id FROM tickers WHERE symbol = ?", (symbol,)).fetchone()
                return res[0]
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"DuckDB ticker operation fallback ({e}). Defaulting ID to 1.")
            return 1

    def save_ohlcv_batch(self, records: List[Dict[str, Any]]) -> int:
        """Saves a batch of OHLCV records to both MySQL and DuckDB."""
        if not records:
            return 0

        ticker_ids_mysql = {}
        ticker_ids_duckdb = {}
        unique_tickers = {r["ticker_name"] for r in records}
        
        resolve_session = next(get_mysql_session())
        try:
            for sym in unique_tickers:
                ticker_ids_mysql[sym] = self.get_or_create_ticker_mysql(resolve_session, sym)
            resolve_session.commit()
        except Exception as e:
            resolve_session.rollback()
            logger.error(f"Failed to resolve MySQL ticker IDs: {e}")
            raise DatabaseError(f"MySQL ticker ID resolution failed: {e}") from e
        finally:
            resolve_session.close()

        for sym in unique_tickers:
            ticker_ids_duckdb[sym] = self.get_or_create_ticker_duckdb(sym)

        # Write to MySQL
        mysql_session = next(get_mysql_session())
        try:
            mysql_query = text("""
                INSERT INTO ohlcv (ticker_id, timestamp, open, high, low, close, adjusted_close, volume, source, data_quality)
                VALUES (:ticker_id, :timestamp, :open, :high, :low, :close, :adjusted_close, :volume, :source, :data_quality)
                ON DUPLICATE KEY UPDATE
                    open = VALUES(open),
                    high = VALUES(high),
                    low = VALUES(low),
                    close = VALUES(close),
                    adjusted_close = VALUES(adjusted_close),
                    volume = VALUES(volume),
                    source = VALUES(source),
                    data_quality = VALUES(data_quality);
            """)
            
            params = []
            for r in records:
                close_val = float(r["close"])
                adj_close = float(r.get("adjusted_close")) if r.get("adjusted_close") is not None else close_val
                
                params.append({
                    "ticker_id": ticker_ids_mysql[r["ticker_name"]],
                    "timestamp": r["timestamp"],
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": close_val,
                    "adjusted_close": adj_close,
                    "volume": int(r["volume"]),
                    "source": r.get("source") or "unknown",
                    "data_quality": r.get("data_quality") or "good"
                })
                
            mysql_session.execute(mysql_query, params)
            mysql_session.commit()
        except Exception as e:
            mysql_session.rollback()
            logger.error(f"Failed to upsert OHLCV batch to MySQL: {e}")
            raise DatabaseError(f"MySQL bulk upsert failed: {e}") from e
        finally:
            mysql_session.close()

        # Write to DuckDB safely (catch lock exception gracefully if locked)
        try:
            duck_conn = get_duckdb_connection(read_only=False)
            try:
                duck_query = """
                    INSERT INTO ohlcv (ticker_id, timestamp, open, high, low, close, adjusted_close, volume, source, data_quality)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (ticker_id, timestamp) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        adjusted_close = EXCLUDED.adjusted_close,
                        volume = EXCLUDED.volume,
                        source = EXCLUDED.source,
                        data_quality = EXCLUDED.data_quality;
                """
                duck_params = []
                for r in records:
                    ts = r["timestamp"]
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts)
                    close_val = float(r["close"])
                    adj_close = float(r.get("adjusted_close")) if r.get("adjusted_close") is not None else close_val
                    
                    duck_params.append((
                        ticker_ids_duckdb[r["ticker_name"]],
                        ts_str,
                        float(r["open"]),
                        float(r["high"]),
                        float(r["low"]),
                        close_val,
                        adj_close,
                        int(r["volume"]),
                        r.get("source") or "unknown",
                        r.get("data_quality") or "good"
                    ))
                    
                duck_conn.executemany(duck_query, duck_params)
                duck_conn.commit()
            finally:
                duck_conn.close()
        except Exception as e:
            logger.warning(f"DuckDB batch write skipped due to lock/error ({e}). Data persisted to MySQL successfully.")

        return len(records)
