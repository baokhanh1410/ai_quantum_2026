"""OHLCV normalization processor aligning volume and prices for single-value assets."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger("data_engine.processors.ohlcv")

class OHLCVProcessor:
    """Processor ensuring OHLCV price constraints and default volumes are met."""

    def process_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes and standardizes OHLCV attributes for a list of records.

        Args:
            records: Cleaned input dictionaries.

        Returns:
            Normalized list of records.
        """
        processed = []
        for r in records:
            # Create a shallow copy
            record = dict(r)
            
            # Ensure volume is present and positive
            vol = record.get("volume")
            if vol is None or (isinstance(vol, float) and vol != vol) or vol <= 0:
                record["volume"] = 1.0
                
            # Align prices
            o, h, l, c = record.get("open"), record.get("high"), record.get("low"), record.get("close")
            
            # If open, high, or low are missing, map them to close (single value asset)
            if o is None or h is None or l is None:
                record["open"] = c
                record["high"] = c
                record["low"] = c
            else:
                # Normal price bounds checks: high must be >= all, low must be <= all
                record["high"] = max(o, h, l, c)
                record["low"] = min(o, h, l, c)

            processed.append(record)
            
        return processed
