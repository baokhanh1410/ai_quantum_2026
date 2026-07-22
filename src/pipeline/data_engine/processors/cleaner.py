"""Data cleaner handling timezone normalization, duplicate removal, and missing values."""

import logging
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger("data_engine.processors.cleaner")

class DataCleaner:
    """Cleaner for standardizing and removing bad records from datasets."""

    def clean_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cleans and standardizes raw parsed records.

        Args:
            records: List of dictionaries to clean.

        Returns:
            A cleaned list of records.
        """
        if not records:
            return []

        df = pd.DataFrame(records)
        
        # 1. Drop records with missing vital columns
        vital_cols = ["timestamp", "close", "ticker_name"]
        for col in vital_cols:
            if col not in df.columns:
                logger.error(f"Vital column {col} missing during data cleaning.")
                return []
                
        initial_count = len(df)
        df = df.dropna(subset=vital_cols)
        
        # Convert timestamp to datetime and remove timezone info
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

        # 2. Sort by timestamp to ensure chronological order
        df = df.sort_values(by="timestamp")

        # 3. Drop duplicate (timestamp, ticker_name) records keeping the last entry
        df = df.drop_duplicates(subset=["timestamp", "ticker_name"], keep="last")

        cleaned_count = len(df)
        if cleaned_count < initial_count:
            logger.info(f"Cleaner removed {initial_count - cleaned_count} invalid/duplicate records.")

        return df.to_dict(orient="records")
