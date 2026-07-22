"""Integration verification test for the Feature Engineering Engine."""

import os
import sys
import logging
import datetime

# Add workspace root to sys.path
sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))))

from sqlalchemy import text
from feature_engine.services.feature_pipeline_service import FeaturePipelineService
from core.database.connection import get_mysql_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("feature_engine.verification")


def run_integration_test():
    """Runs the feature pipeline for a short date range and verifies DB entries."""
    logger.info("Initializing Feature Pipeline Service...")
    service = FeaturePipelineService()

    # Use last 30 days for a meaningful test window
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    logger.info(f"Test date range: {start_str} to {end_str}")

    # Run the pipeline
    result = service.compute_features(start_date=start_str, end_date=end_str)
    logger.info(f"Pipeline result: {result}")

    # Verify MySQL indicator_types
    logger.info("--- Verifying MySQL indicator_types ---")
    session = next(get_mysql_session())
    try:
        indicator_count = session.execute(
            text("SELECT COUNT(*) FROM indicator_types")
        ).scalar()
        logger.info(f"Indicator types in DB: {indicator_count}")

        indicators = session.execute(
            text("SELECT name, category, window_size FROM indicator_types")
        ).fetchall()
        for ind in indicators:
            logger.info(f"  {ind[0]} ({ind[1]}, window={ind[2]})")

        # Verify technical_indicator_values
        tiv_count = session.execute(
            text("SELECT COUNT(*) FROM technical_indicator_values")
        ).scalar()
        logger.info(f"Total technical_indicator_values records: {tiv_count}")

        # Sample records
        samples = session.execute(
            text("""
                SELECT t.symbol, it.name, tiv.timestamp, tiv.value
                FROM technical_indicator_values tiv
                JOIN tickers t ON tiv.ticker_id = t.id
                JOIN indicator_types it ON tiv.indicator_type_id = it.id
                ORDER BY tiv.timestamp DESC
                LIMIT 10
            """)
        ).fetchall()
        logger.info("Sample indicator values:")
        for s in samples:
            logger.info(f"  {s[0]} | {s[1]} | {s[2]} | {s[3]:.6f}" if s[3] else f"  {s[0]} | {s[1]} | {s[2]} | NULL")
    finally:
        session.close()

    # Verify Parquet export
    features_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")),
        "data", "features"
    )
    if os.path.exists(features_dir):
        parquet_files = [f for f in os.listdir(features_dir) if f.endswith(".parquet")]
        logger.info(f"Parquet files exported: {len(parquet_files)}")
        for pf in parquet_files[-3:]:
            logger.info(f"  {pf}")
    else:
        logger.warning(f"Features directory not found: {features_dir}")

    logger.info("Integration test completed!")


if __name__ == "__main__":
    run_integration_test()
