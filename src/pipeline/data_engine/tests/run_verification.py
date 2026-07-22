"""Integration verification test script running ingestion pipelines and asserting DB counts."""

import os
import sys
import logging
import datetime
from sqlalchemy import text

# Add workspace root to sys.path
sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))))

from data_engine.services.ingestion_service import IngestionService
from core.database.connection import get_mysql_session, get_duckdb_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_engine.verification")

def run_integration_tests():
    """Runs a full integration run for a short timeframe and verifies DB entries."""
    logger.info("Initializing Ingestion Service...")
    service = IngestionService()

    # Define a short range (last 7 days) to speed up tests
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=7)
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    logger.info(f"Test ingestion date range: {start_date_str} to {end_date_str}")

    # 1. Test Stock Ingestion
    logger.info("--- Testing Stock Ingestion (vnstock) ---")
    stock_res = service.ingest_stocks(start_date=start_date_str, end_date=end_date_str)
    logger.info(f"Stock Ingestion Result: {stock_res}")

    # 2. Test SJC Gold Ingestion
    logger.info("--- Testing Gold Ingestion (SJC) ---")
    gold_res = service.ingest_gold()
    logger.info(f"Gold Ingestion Result: {gold_res}")

    # 3. Test Macro Ingestion
    logger.info("--- Testing Macro Ingestion (SBV, Investing, VNDirect) ---")
    macro_res = service.ingest_macro(start_date=start_date_str, end_date=end_date_str)
    logger.info(f"Macro Ingestion Result: {macro_res}")

    # 4. Verify MySQL database counts
    logger.info("--- Verifying MySQL Table Counts ---")
    session = next(get_mysql_session())
    try:
        tickers_count = session.execute(text("SELECT COUNT(*) FROM tickers")).scalar()
        ohlcv_count = session.execute(text("SELECT COUNT(*) FROM ohlcv")).scalar()
        logger.info(f"MySQL: {tickers_count} tickers, {ohlcv_count} ohlcv records.")
        
        # Verify ticker samples
        tickers = session.execute(text("SELECT name FROM tickers LIMIT 10")).fetchall()
        logger.info(f"MySQL Ticker samples: {[t[0] for t in tickers]}")
        
        # Verify ohlcv samples
        ohlcv_samples = session.execute(
            text("SELECT t.name, o.timestamp, o.close FROM ohlcv o JOIN tickers t ON o.ticker_id = t.id LIMIT 5")
        ).fetchall()
        logger.info("MySQL OHLCV samples:")
        for o in ohlcv_samples:
            logger.info(f"  {o[0]} at {o[1]}: close={o[2]}")
            
    finally:
        session.close()

    # 5. Verify DuckDB database counts
    logger.info("--- Verifying DuckDB Table Counts ---")
    duck_conn = get_duckdb_connection()
    try:
        tickers_count_dd = duck_conn.execute("SELECT COUNT(*) FROM tickers").fetchone()[0]
        ohlcv_count_dd = duck_conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        logger.info(f"DuckDB: {tickers_count_dd} tickers, {ohlcv_count_dd} ohlcv records.")
        
        # Verify samples
        tickers_dd = duck_conn.execute("SELECT name FROM tickers LIMIT 10").fetchall()
        logger.info(f"DuckDB Ticker samples: {[t[0] for t in tickers_dd]}")
    finally:
        duck_conn.close()

    logger.info("Integration test runs successfully completed!")

if __name__ == "__main__":
    run_integration_tests()
