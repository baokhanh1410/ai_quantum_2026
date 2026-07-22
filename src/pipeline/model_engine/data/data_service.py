import pandas as pd
import logging
from sqlalchemy import create_engine, text
import urllib.parse
from core.config.settings import settings
from core.config.settings import MODEL_CONFIG

logger = logging.getLogger(__name__)

class DataQueryService:
    def __init__(self):
        mysql_config = settings.database.mysql
        escaped_password = urllib.parse.quote_plus(str(mysql_config.password))
        self.db_url = (
            f"mysql+pymysql://{mysql_config.user}:{escaped_password}"
            f"@{mysql_config.host}:{mysql_config.port}/{mysql_config.database}"
        )
        self.engine = create_engine(self.db_url)
        
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Query OHLCV and technical indicators, then merge them.
        Uses parameterized queries to prevent SQL injection.
        """
        logger.info(f"Fetching data from {start_date} to {end_date}...")
        
        # Get target asset class IDs (from config — integers, safe to interpolate)
        target_asset_ids = MODEL_CONFIG.get("target_asset_class_ids", [1, 2, 3])
        asset_ids_str = ",".join(map(str, target_asset_ids))

        # Get target features
        features = MODEL_CONFIG.get("features", [])
        feature_filter = ""
        feature_params: dict = {}
        if features:
            placeholders = ",".join([f":feat_{i}" for i in range(len(features))])
            feature_filter = f"AND ind.name IN ({placeholders})"
            for i, f in enumerate(features):
                feature_params[f"feat_{i}"] = f

        # 1. Fetch OHLCV data — parameterized for start/end dates
        query_ohlcv = text(f"""
            SELECT 
                t.symbol as tic,
                o.timestamp as date,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume
            FROM ohlcv o
            JOIN tickers t ON o.ticker_id = t.id
            WHERE o.timestamp >= :start_date AND o.timestamp <= :end_date
              AND t.asset_class_id IN ({asset_ids_str})
            ORDER BY o.timestamp ASC, t.symbol ASC
        """)
        
        # 2. Fetch technical/macro indicators — parameterized for start/end dates
        query_indicators = text(f"""
            SELECT 
                t.symbol as tic,
                ti.timestamp as date,
                ind.name as indicator_name,
                ti.value
            FROM technical_indicator_values ti
            JOIN tickers t ON ti.ticker_id = t.id
            JOIN indicator_types ind ON ti.indicator_type_id = ind.id
            WHERE ti.timestamp >= :start_date AND ti.timestamp <= :end_date
              AND t.asset_class_id IN ({asset_ids_str})
              {feature_filter}
        """)
        
        # Shared params for date range; merge feature params for indicator query
        date_params = {"start_date": start_date, "end_date": end_date}
        indicator_params = {**date_params, **feature_params}
        
        with self.engine.connect() as conn:
            df_ohlcv = pd.read_sql(query_ohlcv, conn, params=date_params)
            df_ind = pd.read_sql(query_indicators, conn, params=indicator_params)
        
        if df_ohlcv.empty:
            logger.warning(f"No OHLCV data found for {start_date} - {end_date}")
            return pd.DataFrame()
            
        if not df_ind.empty:
            # Normalize dates to string before merging to strip off time components (e.g. 07:00:00 vs 00:00:00)
            df_ohlcv['date'] = pd.to_datetime(df_ohlcv['date']).dt.strftime('%Y-%m-%d')
            df_ind['date'] = pd.to_datetime(df_ind['date']).dt.strftime('%Y-%m-%d')
            
            # Pivot indicators
            df_ind_pivot = df_ind.pivot_table(
                index=['date', 'tic'], 
                columns='indicator_name', 
                values='value'
            ).reset_index()
            
            # Merge
            df_merged = pd.merge(df_ohlcv, df_ind_pivot, on=['date', 'tic'], how='left')
        else:
            logger.warning("No indicator data found. Returning OHLCV only.")
            df_merged = df_ohlcv
            
        # Ensure date is string/datetime and sort properly
        df_merged['date'] = pd.to_datetime(df_merged['date']).dt.strftime('%Y-%m-%d')
        df_merged = df_merged.sort_values(['date', 'tic']).reset_index(drop=True)
        
        return df_merged

    def fetch_symbol_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches OHLCV data for a specific ticker symbol (e.g. 'VN30' benchmark)
        regardless of target_asset_class_ids configuration.
        """
        query = text("""
            SELECT 
                t.symbol as tic,
                o.timestamp as date,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume
            FROM ohlcv o
            JOIN tickers t ON o.ticker_id = t.id
            WHERE t.symbol = :symbol
              AND o.timestamp >= :start_date 
              AND o.timestamp <= :end_date
            ORDER BY o.timestamp ASC
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"symbol": symbol, "start_date": start_date, "end_date": end_date})
            
        if not df.empty:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = df.sort_values(['date', 'tic']).reset_index(drop=True)
            
        return df

