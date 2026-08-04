"""Feature Pipeline Service — the main orchestrator.

Coordinates the full feature engineering workflow:
1. Read raw OHLCV data from the database.
2. Build master timeline and align all symbols.

4. Calculate per-asset technical indicators.
5. Calculate cross-symbol macro indicators.
6. Optionally scale features via pre-fitted scaler.
7. Persist results to MySQL and export to Parquet.
"""

import datetime
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.config.settings import settings
from feature_engine.database.mysql_repository import MySQLRepository
from feature_engine.models.indicator_dto import IndicatorValueDTO
from feature_engine.processors.alignment_processor import AlignmentProcessor
from feature_engine.processors.calculator_processor import CalculatorProcessor
from feature_engine.processors.scaler_processor import ScalerProcessor
from core.utils.exceptions import (
    DataAlignmentError,
    DatabaseWriteError,
    FeatureCalculationError,
    FeatureEngineError,
)

logger = logging.getLogger("feature_engine.services.pipeline")

# Asset class IDs recognised as tradable stock/ETF/Sector Index (for indicator computation)
# Based on assets.yaml: 1=HOSE, 2=HNX, 3=UPCOM, 4=BOND_FUND, 7=SECTOR_INDEX
# Gold (5), Cash/SBV (6), Macro (8) are non-tradable macro assets
_STOCK_ASSET_CLASS_IDS = {1, 2, 3, 4, 7}  # HOSE, HNX, UPCOM, BOND_FUND, SECTOR_INDEX
# Asset class IDs for macro/non-traditional assets (forward-fill only, no per-asset technical indicators)
_MACRO_ASSET_CLASS_IDS = {5, 6, 8}  # GOLD, CASH, MACRO_INDEX


class FeaturePipelineService:
    """Orchestrates the end-to-end feature engineering pipeline.

    This service is the single entry point called by the API route.
    It does NOT contain SQL queries or raw indicator math — those are
    delegated to Repository and Processor layers respectively.
    """

    def __init__(self) -> None:
        self.mysql_repo = MySQLRepository()
        self.alignment = AlignmentProcessor()
        self.calculator = CalculatorProcessor()

        # ScalerProcessor initialised lazily based on pipeline_settings
        scale_enabled = settings.pipeline_settings.get("scale_features", False)
        self.scaler: Optional[ScalerProcessor] = ScalerProcessor() if scale_enabled else None

    def compute_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs the full feature engineering pipeline.

        Args:
            start_date: Start date (YYYY-MM-DD). Defaults to config or today.
            end_date: End date (YYYY-MM-DD). Defaults to config or today.

        Returns:
            Summary dict with status, record counts, and details.
        """
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = settings.system.get("start_date") or today_str
        if not end_date:
            end_date = settings.system.get("end_date") or today_str

        logger.info(f"Feature pipeline started: {start_date} → {end_date}")

        # ----- Step 1: Read raw OHLCV data -----
        logger.info("Step 1: Fetching raw OHLCV data from MySQL...")
        ohlcv_df = self.mysql_repo.fetch_ohlcv(start_date, end_date)

        if ohlcv_df.empty:
            logger.warning("No OHLCV data found for the given date range.")
            return {"status": "warning", "message": "No raw data found.", "total_records": 0}

        logger.info(f"Fetched {len(ohlcv_df)} OHLCV rows for {ohlcv_df['symbol'].nunique()} symbols.")

        # ----- Step 2: Build master timeline & align -----
        logger.info("Step 2: Building master timeline and aligning data...")
        master_timeline = self.alignment.build_master_timeline(ohlcv_df)
        symbol_data = self.alignment.pivot_by_symbol(ohlcv_df)

        # Classify symbols
        stock_symbols = self._classify_symbols(ohlcv_df, _STOCK_ASSET_CLASS_IDS)
        macro_symbols = self._classify_symbols(ohlcv_df, _MACRO_ASSET_CLASS_IDS)

        # ----- Step 3: Align all symbols to master trading calendar -----
        logger.info("Step 3: Aligning symbols to master timeline...")
        symbol_data = self.alignment.align_to_timeline(
            symbol_data, master_timeline, stock_symbols, macro_symbols
        )

        # ----- Step 4: Calculate per-asset technical indicators -----
        enabled_indicators = settings.get_enabled_indicators()
        logger.info(f"Step 4: Computing {len(enabled_indicators)} enabled indicators...")

        symbol_data = self.calculator.compute_asset_indicators(
            symbol_data, enabled_indicators, stock_symbols
        )

        # ----- Step 5: Calculate macro indicators -----
        logger.info("Step 5: Computing macro indicators...")
        macro_df = self.calculator.compute_macro_indicators(
            symbol_data, enabled_indicators, master_timeline
        )

        # ----- Step 6: Optional scaling -----
        if self.scaler is not None and self.scaler.is_available:
            logger.info("Step 6: Scaling features via pre-fitted scaler...")
            # Scale only the indicator columns for each stock symbol
            for symbol in stock_symbols:
                if symbol in symbol_data:
                    df = symbol_data[symbol]
                    indicator_cols = [
                        c for c in df.columns
                        if c not in ("symbol", "exchange", "asset_class_id",
                                     "open", "high", "low", "close",
                                     "adjusted_close", "volume")
                    ]
                    if indicator_cols:
                        symbol_data[symbol] = self.scaler.transform(df[indicator_cols])
            # Scale macro indicators
            if not macro_df.empty:
                macro_df = self.scaler.transform(macro_df)
        else:
            logger.info("Step 6: Skipping scaling (disabled or scaler not available).")

        # ----- Step 7: Drop NaN if configured -----
        drop_nan = settings.pipeline_settings.get("drop_nan", True)

        # ----- Step 8: Persist to MySQL & export Parquet -----
        logger.info("Step 8: Persisting results to MySQL and exporting Parquet...")
        total_saved = self._persist_results(
            symbol_data, macro_df, enabled_indicators,
            stock_symbols, master_timeline, drop_nan
        )

        self._export_parquet(symbol_data, macro_df, stock_symbols, master_timeline)

        logger.info(f"Feature pipeline completed: {total_saved} indicator values saved.")
        return {
            "status": "success",
            "total_records": total_saved,
            "date_range": {"start": start_date, "end": end_date},
            "timeline_days": len(master_timeline),
            "symbols_processed": len(symbol_data),
            "indicators_computed": len(enabled_indicators),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_symbols(
        ohlcv_df: pd.DataFrame,
        asset_class_ids: set,
    ) -> List[str]:
        """Returns unique symbols belonging to the given asset class IDs."""
        mask = ohlcv_df["asset_class_id"].isin(asset_class_ids)
        return sorted(ohlcv_df.loc[mask, "symbol"].unique().tolist())

    def _persist_results(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        macro_df: pd.DataFrame,
        indicators: List[Dict[str, Any]],
        stock_symbols: List[str],
        master_timeline: pd.DatetimeIndex,
        drop_nan: bool,
    ) -> int:
        """Writes computed indicator values into MySQL technical_indicator_values.

        Returns:
            Total number of records inserted.
        """
        ticker_id_map = self.mysql_repo.fetch_ticker_id_map()
        records: List[Dict[str, Any]] = []

        # ----- Asset-level technical indicators -----
        asset_indicators = [
            ind for ind in indicators if ind.get("category") != "macro"
        ]
        for ind_cfg in asset_indicators:
            name = ind_cfg["name"]
            window = ind_cfg.get("window_size", 14)
            col_name = f"{name}_{window}"

            indicator_type_id = self.mysql_repo.upsert_indicator_type(
                name=name,
                category=ind_cfg.get("category", ""),
                window_size=window,
                description=ind_cfg.get("description", ""),
            )

            for symbol in stock_symbols:
                if symbol not in symbol_data:
                    continue
                df = symbol_data[symbol]
                if col_name not in df.columns:
                    continue
                ticker_id = ticker_id_map.get(symbol)
                if ticker_id is None:
                    logger.warning(f"Ticker ID not found for '{symbol}', skipping.")
                    continue

                for ts, val in df[col_name].items():
                    if drop_nan and (val is None or (isinstance(val, float) and np.isnan(val))):
                        continue
                    records.append({
                        "ticker_id": ticker_id,
                        "indicator_type_id": indicator_type_id,
                        "timestamp": ts,
                        "value": float(val) if val is not None and not np.isnan(val) else None,
                    })

        total_inserted = 0
        if records:
            total_inserted += self.mysql_repo.bulk_insert_indicator_values(records)

        # ----- Macro indicators -----
        macro_records: List[Dict[str, Any]] = []
        macro_indicators = [ind for ind in indicators if ind.get("category") == "macro"]
        for ind_cfg in macro_indicators:
            name = ind_cfg["name"]

            macro_type_id = self.mysql_repo.upsert_macro_type(
                name=name,
                unit=ind_cfg.get("unit", ""),
                description=ind_cfg.get("description", ""),
            )

            if name not in macro_df.columns:
                continue

            for ts, val in macro_df[name].items():
                if drop_nan and (val is None or (isinstance(val, float) and np.isnan(val))):
                    continue
                macro_records.append({
                    "macro_type_id": macro_type_id,
                    "timestamp": ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10],
                    "value": float(val) if val is not None and not np.isnan(val) else None,
                })

        if macro_records:
            total_inserted += self.mysql_repo.bulk_insert_macro_values(macro_records)

        return total_inserted


    def _export_parquet(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        macro_df: pd.DataFrame,
        stock_symbols: List[str],
        master_timeline: pd.DatetimeIndex,
    ) -> None:
        """Exports the aligned feature matrix to a .parquet file.

        Output path: data/features/features_<date>.parquet
        """
        output_dir = settings.root_dir / "data" / "features"
        os.makedirs(output_dir, exist_ok=True)

        date_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"features_{date_stamp}.parquet"

        # Build a wide DataFrame: one row per trading day, columns per symbol+indicator
        frames: List[pd.DataFrame] = []

        for symbol in stock_symbols:
            if symbol not in symbol_data:
                continue
            df = symbol_data[symbol]
            # Select only indicator columns (not raw OHLCV)
            indicator_cols = [
                c for c in df.columns
                if c not in ("symbol", "exchange", "asset_class_id",
                             "open", "high", "low", "close",
                             "adjusted_close", "volume")
            ]
            if indicator_cols:
                renamed = df[indicator_cols].rename(
                    columns={c: f"{symbol}__{c}" for c in indicator_cols}
                )
                frames.append(renamed)

        if not macro_df.empty:
            frames.append(macro_df)

        if frames:
            combined = pd.concat(frames, axis=1)
            combined.index.name = "timestamp"
            combined.to_parquet(output_path, engine="pyarrow")
            logger.info(f"Exported features to {output_path} ({len(combined)} rows, {len(combined.columns)} cols).")
        else:
            logger.warning("No feature data to export to Parquet.")
