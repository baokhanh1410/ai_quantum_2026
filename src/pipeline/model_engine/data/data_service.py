import pandas as pd
import logging
from pathlib import Path
from sqlalchemy import create_engine, text
import urllib.parse
import duckdb
from core.config.settings import settings, MODEL_CONFIG, reload_settings

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
        self.duckdb_path = settings.root_dir / "data" / "processed" / "portfolio.duckdb"

    def _execute_query(self, query_str: str, params: dict) -> pd.DataFrame:
        """Executes query prioritizing DuckDB for full dataset, falling back to MySQL."""
        df = pd.DataFrame()
        if self.duckdb_path.exists():
            try:
                duck_conn = duckdb.connect(str(self.duckdb_path))
                duck_query = query_str
                for k, v in params.items():
                    if isinstance(v, str):
                        duck_query = duck_query.replace(f":{k}", f"'{v}'")
                    else:
                        duck_query = duck_query.replace(f":{k}", str(v))
                df = duck_conn.execute(duck_query).df()
            except Exception as de:
                logger.warning(f"DuckDB query failed ({de}). Falling back to MySQL...")

        if df.empty:
            try:
                with self.engine.connect() as conn:
                    df = pd.read_sql(text(query_str), conn, params=params)
            except Exception as e:
                logger.error(f"MySQL query failed: {e}")

        return df

    def fetch_data(self, start_date: str, end_date: str, lookback_days: int = 45) -> pd.DataFrame:
        """
        Query OHLCV and technical indicators, then merge them.
        Uses parameterized queries with DuckDB priority.
        If lookback_days > 0, fetches history prior to start_date for indicator warm-up.
        """
        query_start_date = start_date
        if lookback_days > 0:
            query_start_date = (pd.to_datetime(start_date) - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        logger.info(f"Fetching data from {query_start_date} to {end_date} (lookback_days={lookback_days})...")
        
        reload_settings()
        model_cfg = settings.model_engine.to_dict() if settings.model_engine else {}
        target_asset_ids = model_cfg.get("target_asset_class_ids", [1, 2, 3])
        asset_ids_str = ",".join(map(str, target_asset_ids))

        features = model_cfg.get("features", [])
        feature_filter = ""
        if features:
            placeholders = ",".join([f"'{f}'" for f in features])
            feature_filter = f"AND ind.name IN ({placeholders})"

        query_ohlcv = f"""
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
        """
        
        query_indicators = f"""
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
        """

        macro_feature_filter = ""
        if features:
            placeholders = ",".join([f"'{f}'" for f in features])
            macro_feature_filter = f"AND mt.name IN ({placeholders})"

        query_macro = f"""
            SELECT 
                mv.timestamp as date,
                mt.name as macro_name,
                mv.value
            FROM macro_values mv
            JOIN macro_types mt ON mv.macro_type_id = mt.id
            WHERE mv.timestamp >= :start_date AND mv.timestamp <= :end_date
              {macro_feature_filter}
        """
        
        date_params = {"start_date": query_start_date, "end_date": end_date}
        
        df_ohlcv = self._execute_query(query_ohlcv, date_params)
        df_ind = self._execute_query(query_indicators, date_params)
        df_macro = self._execute_query(query_macro, date_params)
        
        if df_ohlcv.empty:
            logger.warning(f"No OHLCV data found for {start_date} - {end_date}")
            return pd.DataFrame()
            
        df_merged = df_ohlcv.copy()
        df_merged['date'] = pd.to_datetime(df_merged['date']).dt.strftime('%Y-%m-%d')

        if not df_ind.empty:
            df_ind['date'] = pd.to_datetime(df_ind['date']).dt.strftime('%Y-%m-%d')
            df_ind_pivot = df_ind.pivot_table(
                index=['date', 'tic'], 
                columns='indicator_name', 
                values='value'
            ).reset_index()
            df_merged = pd.merge(df_merged, df_ind_pivot, on=['date', 'tic'], how='left')

        if not df_macro.empty:
            df_macro['date'] = pd.to_datetime(df_macro['date']).dt.strftime('%Y-%m-%d')
            df_macro_pivot = df_macro.pivot_table(
                index=['date'],
                columns='macro_name',
                values='value'
            ).reset_index()

            overlap_cols = [c for c in df_macro_pivot.columns if c != 'date' and c in df_merged.columns]
            if overlap_cols:
                df_merged = df_merged.drop(columns=overlap_cols)

            df_merged = pd.merge(df_merged, df_macro_pivot, on=['date'], how='left')

        df_merged['date'] = pd.to_datetime(df_merged['date']).dt.strftime('%Y-%m-%d')
        df_merged = df_merged.sort_values(['date', 'tic']).reset_index(drop=True)
        
        return df_merged

    def fetch_symbol_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches OHLCV data for a specific ticker symbol (e.g. 'VN30' benchmark).
        """
        query = f"""
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
        """
        params = {"symbol": symbol, "start_date": start_date, "end_date": end_date}
        df = self._execute_query(query, params)
            
        if not df.empty:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = df.sort_values(['date', 'tic']).reset_index(drop=True)
            
        return df
