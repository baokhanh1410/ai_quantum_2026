import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data: pd.DataFrame, features: List[str]):
        """
        Args:
            data (pd.DataFrame): Raw merged dataframe from DataQueryService
            features (List[str]): List of technical/macro indicators to use as states
        """
        self.data = data
        self.features = features
        
    def clean_data(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Cleans data by filling missing values, fixing zero OHLC prices, and broadcasting macro features.
        If start_date is specified, trims history prior to start_date after normalization.
        """
        if self.data.empty:
            return self.data
            
        df = self.data.copy()

        # 0. Fix zero prices in OHLC columns by replacing with NaN and backfilling/forward-filling
        ohlc_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
        for col in ohlc_cols:
            df[col] = df[col].replace(0, np.nan)
            
        if 'close' in df.columns:
            df['close'] = df.groupby('tic')['close'].transform(lambda s: s.ffill().bfill())
            if 'open' in df.columns:
                df['open'] = df['open'].fillna(df['close'])
            if 'high' in df.columns:
                df['high'] = df['high'].fillna(df['close'])
            if 'low' in df.columns:
                df['low'] = df['low'].fillna(df['close'])

        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            same_mask = (df['high'] == df['low']) | df['high'].isna() | df['low'].isna()
            df.loc[same_mask, 'high'] = df.loc[same_mask, 'close'] * 1.0001
            df.loc[same_mask, 'low'] = df.loc[same_mask, 'close'] * 0.9999

        # 1. Identify and broadcast macro features
        # Gold (SJC) is kept as a tradable asset in tradable_df
        MACRO_PREFIXES = ("DXY", "VN10YT", "VN3YT", "USDVND", "SJC_BUY", "VNIBOR", "XAUUSD")
        is_macro = df['tic'].str.startswith(MACRO_PREFIXES)
        
        macro_df = df[is_macro]
        tradable_df = df[~is_macro].copy()
        
        if not macro_df.empty and not tradable_df.empty:
            # Aggregate macro features by date (taking max ignores NaNs)
            macro_agg = macro_df.groupby('date').max().reset_index()
            
            # Drop columns that shouldn't be broadcasted
            cols_to_drop = ['tic', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'PPO', 'CCI', 'ADX', 'ATR', 'VOLATILITY'] 
            cols_to_drop = [c for c in cols_to_drop if c in macro_agg.columns]
            macro_agg = macro_agg.drop(columns=cols_to_drop)
            
            # Drop macro columns from tradable_df before merge to avoid _x, _y suffixes
            macro_cols = [c for c in macro_agg.columns if c != 'date']
            tradable_cols_to_drop = [c for c in macro_cols if c in tradable_df.columns]
            tradable_df = tradable_df.drop(columns=tradable_cols_to_drop)
            
            # Merge macro features into tradable assets
            tradable_df = pd.merge(tradable_df, macro_agg, on='date', how='left')
            
        # Scale Gold price from VND to a 1,000-point Price Index to match Sector Indices (~1,000-2,000 points)
        if 'SJC_SELL' in tradable_df['tic'].values:
            sjc_mask = tradable_df['tic'] == 'SJC_SELL'
            price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in tradable_df.columns]
            sjc_sub = tradable_df[sjc_mask].sort_values('date')
            if not sjc_sub.empty and 'close' in sjc_sub.columns:
                first_close = sjc_sub['close'].iloc[0]
                if first_close > 0:
                    tradable_df.loc[sjc_mask, price_cols] = (tradable_df.loc[sjc_mask, price_cols] / first_close) * 1000.0

        df = tradable_df
        
        if df.empty:
            return df
            
        # 2. Fill missing values per tradable ticker — forward fill ONLY (no bfill to prevent future leakage)
        df = df.sort_values(['date', 'tic'])
        cols_to_ffill = [c for c in df.columns if c != 'tic']
        df[cols_to_ffill] = df.groupby('tic')[cols_to_ffill].ffill()

        # 3. Clean initial 0 values (lookback warm-up period) in technical indicators by forward-filling per ticker
        #    NOTE: We use ffill here (not bfill) to avoid any future leakage. Values at the very start (no history)
        #          remain 0 and are handled by the fillna(0) step below.
        tech_cols = [c for c in ['ADX', 'ATR', 'RSI', 'PPO', 'CCI', 'VOLATILITY', 'YIELD_CURVE_SLOPE', 'DXY_LOG_RETURN', 'VN3YT'] if c in df.columns]
        for col in tech_cols:
            df[col] = df.groupby('tic')[col].transform(lambda s: s.replace(0, np.nan).ffill())

        # Fill any remaining NaNs with 0 (e.g. if an indicator is completely missing at start)
        df = df.fillna(0)

        # 3b. Rolling Z-Score Normalization for technical & macro features (per-ticker, window=30, clip=±3.0)
        #     Prevents State Drift caused by features with incompatible scales.
        #     Uses expanding window for initial warmup (< 30) and rolling window for win >= 30.
        ZSCORE_WINDOW = 30
        ZSCORE_CLIP = 3.0
        for col in tech_cols:
            if col in df.columns:
                def _zscore_normalize(s: pd.Series, win: int = ZSCORE_WINDOW, clip: float = ZSCORE_CLIP) -> pd.Series:
                    exp_mean = s.expanding(min_periods=1).mean()
                    exp_std = s.expanding(min_periods=1).std().fillna(1.0).replace(0, 1.0)
                    roll_mean = s.rolling(window=win, min_periods=1).mean()
                    roll_std = s.rolling(window=win, min_periods=1).std().fillna(1.0).replace(0, 1.0)
                    
                    n = len(s)
                    idx = np.arange(n)
                    mean = np.where(idx < win, exp_mean, roll_mean)
                    std = np.where(idx < win, exp_std, roll_std)
                    
                    z = (s - mean) / (std + 1e-8)
                    return pd.Series(z.clip(-clip, clip), index=s.index).fillna(0.0)
                df[col] = df.groupby('tic')[col].transform(_zscore_normalize)
        
        # 3.5 Dynamically compute Kritzman Turbulence Index if missing
        if "TURBULENCE" not in df.columns and "turbulence" not in df.columns:
            try:
                from feature_engine.processors.calculator_processor import CalculatorProcessor
                proc = CalculatorProcessor()
                master_timeline = pd.DatetimeIndex(pd.to_datetime(df['date'].unique()).sort_values())
                symbol_data = {}
                for tic_name, tic_df in df.groupby('tic'):
                    if 'close' in tic_df.columns:
                        s_df = tic_df.set_index(pd.to_datetime(tic_df['date']))[['close']].sort_index()
                        symbol_data[tic_name] = s_df
                
                if symbol_data:
                    turb_s = proc._compute_macro_indicator("TURBULENCE", symbol_data, master_timeline)
                    turb_df = pd.DataFrame({'date': master_timeline.strftime('%Y-%m-%d'), 'TURBULENCE': turb_s.values})
                    df = pd.merge(df, turb_df, on='date', how='left')
                    df['TURBULENCE'] = df['TURBULENCE'].fillna(0.0)
            except Exception as e:
                logger.warning(f"Could not calculate TURBULENCE in DataProcessor: {e}")
                df['TURBULENCE'] = 0.0

        # 4. Ensure all required features exist
        for feat in self.features:
            if feat not in df.columns:
                logger.warning(f"Feature {feat} not found in database. Filling with 0.")
                df[feat] = 0
                
        # 5. Keep OHLCV + TURBULENCE + configured features
        always_keep = ['tic', 'date', 'open', 'high', 'low', 'close', 'volume', 'TURBULENCE', 'turbulence']
        raw_keep = [c for c in always_keep + list(self.features) if c in df.columns]
        seen = set()
        keep_cols = [x for x in raw_keep if not (x in seen or seen.add(x))]
        df = df[keep_cols]

        # Re-sort to be strictly chronological for the environment
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)

        # Filter out warm-up historical rows prior to start_date if provided
        if start_date is not None:
            df = df[df['date'] >= start_date].reset_index(drop=True)
        
        return df
