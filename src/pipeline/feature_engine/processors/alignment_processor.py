"""Alignment processor for synchronizing timelines to Vietnam stock trading days.

Constructs a Master Timeline from actual HOSE/HNX trading dates, then
left-joins macroeconomic and non-traditional asset data with forward fill.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from core.utils.exceptions import DataAlignmentError

logger = logging.getLogger("feature_engine.processors.alignment")

from core.config.settings import settings

# Determine target asset classes dynamically from features.yaml, fallback to default stocks (1, 2, 3)
try:
    _STOCK_ASSET_CLASS_IDS = set(settings.pipeline_settings.get("target_asset_class_ids", [1, 2, 3]))
except Exception:
    _STOCK_ASSET_CLASS_IDS = {1, 2, 3}


class AlignmentProcessor:
    """Aligns heterogeneous financial time-series to a master trading calendar.

    The Master Timeline is built from the union of distinct trading dates
    observed in the stock OHLCV data (HOSE / HNX / UPCOM). Macro and
    non-traditional assets are left-joined then forward-filled.
    """

    def build_master_timeline(self, ohlcv_df: pd.DataFrame) -> pd.DatetimeIndex:
        """Extracts the sorted set of unique stock trading dates.

        Args:
            ohlcv_df: Full OHLCV DataFrame containing an 'asset_class_id' column.

        Returns:
            Sorted DatetimeIndex of unique stock trading dates.

        Raises:
            DataAlignmentError: If no stock trading dates are found.
        """
        stock_mask = ohlcv_df["asset_class_id"].isin(_STOCK_ASSET_CLASS_IDS)
        stock_dates = ohlcv_df.loc[stock_mask, "timestamp"].dt.normalize().unique()

        if len(stock_dates) == 0:
            raise DataAlignmentError(
                "No stock trading dates found in OHLCV data. "
                "Cannot construct the Master Timeline."
            )

        timeline = pd.DatetimeIndex(sorted(stock_dates))
        logger.info(
            f"Master Timeline built: {len(timeline)} trading days "
            f"({timeline.min().date()} → {timeline.max().date()})"
        )
        return timeline

    def pivot_by_symbol(self, ohlcv_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Pivots the OHLCV DataFrame into per-symbol DataFrames indexed by date.

        Args:
            ohlcv_df: Full OHLCV DataFrame.

        Returns:
            Dictionary mapping symbol -> DataFrame with OHLCV columns
            indexed by normalized timestamp.
        """
        result: Dict[str, pd.DataFrame] = {}
        ohlcv_df = ohlcv_df.copy()
        ohlcv_df["timestamp"] = ohlcv_df["timestamp"].dt.normalize()

        for symbol, group in ohlcv_df.groupby("symbol"):
            df = group.set_index("timestamp").sort_index()
            # Drop duplicate dates (keep last)
            df = df[~df.index.duplicated(keep="last")]
            result[str(symbol)] = df

        return result

    def align_to_timeline(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        master_timeline: pd.DatetimeIndex,
        stock_symbols: List[str],
        macro_symbols: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """Aligns each symbol's DataFrame to the master timeline.

        Stock symbols are reindexed (NaN for non-trading days stays NaN).
        Macro/non-traditional symbols are left-joined then forward-filled.

        Args:
            symbol_data: Dictionary of symbol -> DataFrame from pivot_by_symbol.
            master_timeline: The master trading date index.
            stock_symbols: List of stock/ETF symbols (inner align).
            macro_symbols: List of macro/gold/bond symbols (ffill align).

        Returns:
            Dictionary of symbol -> aligned DataFrame.
        """
        aligned: Dict[str, pd.DataFrame] = {}

        for symbol, df in symbol_data.items():
            # Reindex to master timeline. This introduces NaNs for missing days.
            aligned_df = df.reindex(master_timeline)
            missing_mask = aligned_df["close"].isna()

            # We MUST forward-fill missing days. If we leave NaNs, recursive indicators 
            # like ADX and ATR will propagate NaN forever and destroy the data series.
            aligned_df = aligned_df.ffill()

            if symbol in stock_symbols:
                # For stocks, missing days mean halted/no-trade, so volume is 0
                if "volume" in aligned_df.columns:
                    aligned_df.loc[missing_mask, "volume"] = 0

            aligned[symbol] = aligned_df

        logger.info(
            f"Aligned {len(aligned)} symbols to master timeline "
            f"({len(master_timeline)} trading days)."
        )
        return aligned
