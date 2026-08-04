"""Calculator processor for dynamically computing technical and macro indicators.

Uses a Dynamic Mapping Matrix to translate indicator names from features.yaml
into ``ta`` library function calls. Macro indicators (Yield Curve Slope, DXY
Log Return, SJC Premium, VNIBOR ON) are computed via custom handlers since
they operate across multiple symbols rather than on a single OHLCV DataFrame.

Note: Uses the ``ta`` library (pip install ta) which is compatible with
Python 3.9+, instead of ``pandas-ta`` which requires Python 3.12+.
"""

import logging
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, PercentagePriceOscillator
from ta.trend import ADXIndicator, CCIIndicator
from ta.volatility import AverageTrueRange

from core.utils.exceptions import FeatureCalculationError

logger = logging.getLogger("feature_engine.processors.calculator")


# ---------------------------------------------------------------------------
# Type alias for indicator compute functions
# Signature: (df: pd.DataFrame, window_size: int) -> pd.Series
# ---------------------------------------------------------------------------
IndicatorFn = Callable[[pd.DataFrame, int], pd.Series]


def _compute_rsi(df: pd.DataFrame, window_size: int) -> pd.Series:
    """Compute RSI using the ta library."""
    indicator = RSIIndicator(close=df["close"].astype(float), window=window_size)
    return indicator.rsi()


def _compute_ppo(df: pd.DataFrame, window_size: int) -> pd.Series:
    """Compute PPO (Percentage Price Oscillator) using the ta library.

    PPO = ((EMA_fast - EMA_slow) / EMA_slow) * 100
    fast = window_size // 2, slow = window_size
    """
    fast = max(window_size // 2, 1)
    slow = window_size
    indicator = PercentagePriceOscillator(
        close=df["close"].astype(float), window_slow=slow, window_fast=fast, window_sign=9
    )
    return indicator.ppo()


def _compute_cci(df: pd.DataFrame, window_size: int) -> pd.Series:
    """Compute CCI using the ta library."""
    indicator = CCIIndicator(
        high=df["high"].astype(float),
        low=df["low"].astype(float),
        close=df["close"].astype(float),
        window=window_size,
    )
    return indicator.cci()


def _compute_adx(df: pd.DataFrame, window_size: int) -> pd.Series:
    """Compute ADX using the ta library."""
    indicator = ADXIndicator(
        high=df["high"].astype(float),
        low=df["low"].astype(float),
        close=df["close"].astype(float),
        window=window_size,
    )
    return indicator.adx()


def _compute_atr(df: pd.DataFrame, window_size: int) -> pd.Series:
    """Compute ATR using the ta library."""
    indicator = AverageTrueRange(
        high=df["high"].astype(float),
        low=df["low"].astype(float),
        close=df["close"].astype(float),
        window=window_size,
    )
    return indicator.average_true_range()


def _compute_volatility(df: pd.DataFrame, window_size: int) -> pd.Series:
    """Compute rolling volatility as the standard deviation of log returns."""
    close = df["close"].astype(float)
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window=window_size).std()


# ---------------------------------------------------------------------------
# Dynamic Mapping Matrix:  indicator_name (str) -> compute function
# ---------------------------------------------------------------------------
_ASSET_INDICATOR_MAP: Dict[str, IndicatorFn] = {
    "RSI": _compute_rsi,
    "PPO": _compute_ppo,
    "CCI": _compute_cci,
    "ADX": _compute_adx,
    "ATR": _compute_atr,
    "VOLATILITY": _compute_volatility,
}


class CalculatorProcessor:
    """Dynamically computes technical and macro indicators from config.

    For each enabled indicator in features.yaml:
    - If category == 'trend' or 'volatility': apply the mapped ta library
      function on each asset's OHLCV DataFrame.
    - If category == 'macro': compute the cross-symbol macro feature.

    New indicators can be added by:
    1. Adding an entry in features.yaml.
    2. Registering the compute function in _ASSET_INDICATOR_MAP or
       handling it in _compute_macro_indicator.
    """

    def compute_asset_indicators(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        indicators: List[Dict[str, Any]],
        stock_symbols: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """Computes per-asset technical indicators for stock/ETF symbols.

        Args:
            symbol_data: Dictionary of symbol -> aligned OHLCV DataFrame.
            indicators: List of indicator config dicts from features.yaml
                        (only enabled, non-macro ones).
            stock_symbols: List of symbol names to compute indicators for.

        Returns:
            Dictionary of symbol -> DataFrame with new indicator columns appended.
        """
        asset_indicators = [
            ind for ind in indicators if ind.get("category") in ("trend", "volatility")
        ]

        if not asset_indicators:
            logger.info("No asset-level indicators enabled.")
            return symbol_data

        for symbol in stock_symbols:
            if symbol not in symbol_data:
                continue

            df = symbol_data[symbol].copy()

            # Filter non-NaN sub-dataframe to eliminate leading NaNs before calling `ta` library.
            # This prevents recursive EWM/smoothed moving averages in `ta` from starting on NaNs,
            # which would otherwise propagate NaNs permanently across the entire dataset.
            valid_mask = df["close"].notna()
            sub_df = df.loc[valid_mask]

            for ind_cfg in asset_indicators:
                name = ind_cfg["name"]
                window = ind_cfg.get("window_size", 14)
                col_name = f"{name}_{window}"
                df[col_name] = np.nan

                compute_fn = _ASSET_INDICATOR_MAP.get(name)
                if compute_fn is None:
                    logger.warning(
                        f"No mapping found for indicator '{name}'. "
                        f"Register it in _ASSET_INDICATOR_MAP to enable."
                    )
                    continue

                try:
                    valid_count = len(sub_df)
                    if valid_count < window + 1:
                        logger.debug(
                            f"Skipping {name} for {symbol}: "
                            f"only {valid_count} valid rows, need {window + 1}"
                        )
                        continue

                    series = compute_fn(sub_df, window)
                    df.loc[series.index, col_name] = series
                    logger.debug(f"Computed {col_name} for {symbol}")
                except Exception as e:
                    logger.warning(
                        f"Could not compute {name} for {symbol} "
                        f"({type(e).__name__}: {e}). Filling with NaN."
                    )
                    df[col_name] = np.nan

            symbol_data[symbol] = df

        logger.info(
            f"Computed {len(asset_indicators)} asset indicators "
            f"for {len(stock_symbols)} symbols."
        )
        return symbol_data

    def compute_macro_indicators(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        indicators: List[Dict[str, Any]],
        master_timeline: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Computes cross-symbol macroeconomic indicators.

        Args:
            symbol_data: Dictionary of symbol -> aligned DataFrame.
            indicators: List of indicator config dicts from features.yaml
                        (only enabled macro ones).
            master_timeline: The master trading date index.

        Returns:
            DataFrame indexed by master_timeline with one column per macro indicator.
        """
        macro_indicators = [ind for ind in indicators if ind.get("category") == "macro"]

        if not macro_indicators:
            logger.info("No macro indicators enabled.")
            return pd.DataFrame(index=master_timeline)

        macro_df = pd.DataFrame(index=master_timeline)

        for ind_cfg in macro_indicators:
            name = ind_cfg["name"]
            try:
                series = self._compute_macro_indicator(name, symbol_data, master_timeline)
                macro_df[name] = series
                logger.debug(f"Computed macro indicator: {name}")
            except FeatureCalculationError:
                raise
            except Exception as e:
                raise FeatureCalculationError(
                    f"Macro indicator '{name}' computation failed: {e}"
                ) from e

        logger.info(f"Computed {len(macro_indicators)} macro indicators.")
        return macro_df

    def _compute_macro_indicator(
        self,
        name: str,
        symbol_data: Dict[str, pd.DataFrame],
        master_timeline: pd.DatetimeIndex,
    ) -> pd.Series:
        """Dispatches macro indicator computation by name.

        Args:
            name: The indicator name from features.yaml.
            symbol_data: Dictionary of symbol -> aligned DataFrame.
            master_timeline: The master trading date index.

        Returns:
            pd.Series aligned to master_timeline.
        """
        if name == "YIELD_CURVE_SLOPE":
            return self._yield_curve_slope(symbol_data, master_timeline)
        elif name == "DXY_LOG_RETURN":
            return self._dxy_log_return(symbol_data, master_timeline)
        elif name == "SJC_PREMIUM":
            return self._sjc_premium(symbol_data, master_timeline)
        elif name == "VNIBOR_ON":
            return self._vnibor_on_passthrough(symbol_data, master_timeline)
        elif name == "VN3YT":
            return self._safe_close(symbol_data, "VN3YT", master_timeline)
        else:
            raise FeatureCalculationError(
                f"Unknown macro indicator '{name}'. "
                f"Add a handler in CalculatorProcessor._compute_macro_indicator."
            )

    # ------------------------------------------------------------------
    # Macro indicator implementations
    # ------------------------------------------------------------------

    def _yield_curve_slope(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        timeline: pd.DatetimeIndex,
    ) -> pd.Series:
        """Yield Curve Slope = VN10Y close - VN3Y close.

        Bond yield symbols in DB: VN10YT, VN3YT (mapped from TradingView).
        """
        vn10y = self._safe_close(symbol_data, "VN10YT", timeline)
        vn3y = self._safe_close(symbol_data, "VN3YT", timeline)
        return vn10y - vn3y

    def _dxy_log_return(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        timeline: pd.DatetimeIndex,
    ) -> pd.Series:
        """Log return of the DXY (US Dollar Index)."""
        dxy = self._safe_close(symbol_data, "DXY", timeline)
        return np.log(dxy / dxy.shift(1))

    def _sjc_premium(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        timeline: pd.DatetimeIndex,
    ) -> pd.Series:
        """SJC Premium = SJC Buy close - XAUUSD close * USDVND close / troy_oz_to_tael.

        SJC is quoted per tael (lượng). 1 troy ounce ≈ 0.829 tael.
        XAUUSD is USD/oz.  USDVND is VND/USD.
        World gold in VND per tael = XAUUSD * USDVND / 0.829
        """
        sjc_buy = self._safe_close(symbol_data, "SJC_BUY", timeline)
        xauusd = self._safe_close(symbol_data, "XAUUSD", timeline)
        usdvnd = self._safe_close(symbol_data, "USDVND", timeline)

        troy_oz_per_tael = 0.829
        world_gold_vnd_per_tael = (xauusd * usdvnd) / troy_oz_per_tael

        # SJC is stored in VND (e.g. 92_300_000), world gold is also VND
        premium = sjc_buy - world_gold_vnd_per_tael
        return premium

    def _vnibor_on_passthrough(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        timeline: pd.DatetimeIndex,
    ) -> pd.Series:
        """Passes through the VNIBOR ON daily rate."""
        return self._safe_close(symbol_data, "VNIBOR_ON", timeline)
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_close(
        symbol_data: Dict[str, pd.DataFrame],
        symbol: str,
        timeline: pd.DatetimeIndex,
    ) -> pd.Series:
        """Extracts the 'close' column for a symbol, aligned to the timeline.

        Returns NaN series if the symbol is missing.
        """
        if symbol in symbol_data:
            df = symbol_data[symbol]
            if "close" in df.columns:
                return df["close"].reindex(timeline).astype(float)
        logger.warning(f"Symbol '{symbol}' not found; returning NaN series.")
        return pd.Series(np.nan, index=timeline, name=symbol)
