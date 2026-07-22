"""Scaler processor for feature normalization using pre-fitted scalers.

Loads a pickled scikit-learn StandardScaler and applies .transform() only.
Never calls .fit() or .fit_transform() to prevent data leakage in live
production or walk-forward evaluation scenarios.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.config.settings import settings
from core.utils.exceptions import ScalerNotFoundError

logger = logging.getLogger("feature_engine.processors.scaler")


class ScalerProcessor:
    """Loads a pre-fitted StandardScaler and transforms feature matrices.

    If the scaler file does not exist, the processor falls back to an
    identity transform (no scaling) and logs a warning, allowing the
    pipeline to continue in development/bootstrapping scenarios.
    """

    def __init__(self, scaler_path: Optional[str] = None) -> None:
        """Initialises the ScalerProcessor.

        Args:
            scaler_path: Absolute or relative path to the .pkl scaler file.
                         Falls back to pipeline_settings.scaler_path if None.
        """
        if scaler_path is None:
            scaler_path = settings.pipeline_settings.get("scaler_path", "")

        # Resolve relative paths against the project root
        resolved = Path(scaler_path)
        if not resolved.is_absolute():
            resolved = settings.root_dir / resolved

        self._scaler_path = resolved
        self._scaler = self._load_scaler()

    def _load_scaler(self):
        """Attempts to load the pickled scaler from disk.

        Returns:
            The loaded scaler object, or None if not found.
        """
        if not self._scaler_path.exists():
            logger.warning(
                f"Scaler file not found at {self._scaler_path}. "
                f"Scaling will be skipped (identity transform)."
            )
            return None

        try:
            with open(self._scaler_path, "rb") as f:
                scaler = pickle.load(f)
            logger.info(f"Loaded pre-fitted scaler from {self._scaler_path}")
            return scaler
        except Exception as e:
            raise ScalerNotFoundError(
                f"Failed to load scaler from {self._scaler_path}: {e}"
            ) from e

    @property
    def is_available(self) -> bool:
        """Whether a valid scaler has been loaded."""
        return self._scaler is not None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies .transform() using the pre-fitted scaler.

        Only numeric columns are scaled. Non-numeric columns are preserved
        unchanged. If the scaler is not available, returns the input unchanged.

        Args:
            df: DataFrame with numeric feature columns.

        Returns:
            DataFrame with scaled numeric columns.
        """
        if not self.is_available:
            logger.debug("No scaler loaded; returning data unscaled.")
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.warning("No numeric columns to scale.")
            return df

        try:
            scaled_values = self._scaler.transform(df[numeric_cols])
            result = df.copy()
            result[numeric_cols] = scaled_values
            logger.info(f"Scaled {len(numeric_cols)} feature columns.")
            return result
        except Exception as e:
            logger.error(f"Scaler transform failed: {e}")
            raise ScalerNotFoundError(
                f"Scaler .transform() failed — shape mismatch or corrupted pkl? {e}"
            ) from e
