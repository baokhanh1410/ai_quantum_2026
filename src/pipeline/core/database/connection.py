"""Centralized database connection manager for MySQL and DuckDB."""

import os
import logging
import urllib.parse
import duckdb
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from core.config.settings import settings
from core.utils.exceptions import DatabaseError

logger = logging.getLogger("core.database")

Base = declarative_base()

# Construct escaped MySQL URL
try:
    mysql_config = settings.database.mysql
    escaped_password = urllib.parse.quote_plus(str(mysql_config.password))
    
    # Using pymysql as default driver
    DATABASE_URL = (
        f"mysql+pymysql://{mysql_config.user}:{escaped_password}"
        f"@{mysql_config.host}:{mysql_config.port}/{mysql_config.database}"
    )
    
    # Setup SQLAlchemy engine with pooling
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,
        pool_pre_ping=True
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    logger.error(f"Failed to initialize MySQL engine: {e}")
    engine = None
    SessionLocal = None


def get_mysql_session():
    """Context manager / dependency for MySQL database sessions.

    Yields:
        A SQLAlchemy session object.
    """
    if SessionLocal is None:
        raise DatabaseError("MySQL database engine is not initialized.")
    session = SessionLocal()
    try:
        yield session
    except Exception as e:
        session.rollback()
        raise DatabaseError(f"MySQL database session error: {e}") from e
    finally:
        session.close()


def get_duckdb_connection(read_only: bool = False):
    """Creates and returns a connection to the local DuckDB database.

    Args:
        read_only: If True, opens connection in read-only mode for multi-process concurrency.

    Returns:
        A DuckDB Connection object.
    """
    try:
        db_path = str(settings.database.duckdb.path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return duckdb.connect(db_path, read_only=read_only)
    except Exception as e:
        if not read_only:
            try:
                logger.warning(f"DuckDB write connection failed ({e}), trying read-only mode...")
                return duckdb.connect(db_path, read_only=True)
            except Exception as ro_err:
                logger.warning(f"DuckDB read-only fallback connection failed: {ro_err}")
        logger.error(f"Failed to connect to DuckDB: {e}")
        raise DatabaseError(f"DuckDB connection error: {e}") from e
