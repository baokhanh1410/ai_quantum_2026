import pandas as pd
import logging
from typing import List

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
        
    def clean_data(self) -> pd.DataFrame:
        """
        Cleans data by filling missing values and broadcasting macro features.
        """
        if self.data.empty:
            return self.data
            
        df = self.data.copy()
        
        # 1. Identify and broadcast macro features
        MACRO_PREFIXES = ("DXY", "VN10YT", "VN3YT", "USDVND", "VNIBOR", "SJC", "XAUUSD")
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
            
        df = tradable_df
        
        if df.empty:
            return df
            
        # 2. Fill missing values per tradable ticker
        df = df.sort_values(['date', 'tic'])
        df = df.groupby('tic').apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
        
        # Fill any remaining NaNs with 0 (e.g. if an indicator is completely missing)
        df = df.fillna(0)
        
        # 3. Ensure all required features exist
        for feat in self.features:
            if feat not in df.columns:
                logger.warning(f"Feature {feat} not found in database. Filling with 0.")
                df[feat] = 0
                
        # Re-sort to be strictly chronological for the environment
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        return df
