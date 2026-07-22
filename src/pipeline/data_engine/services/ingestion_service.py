"""Service orchestrating client data fetches, handlers, processors, and repositories."""

import logging
import datetime
from typing import List, Dict, Any, Optional
from core.config.settings import settings
from data_engine.api.vnstock_client import VNStockClient
from data_engine.api.sbv_client import SBVClient
from data_engine.api.gold_client import GoldClient
from data_engine.api.vndirect_client import VNDirectClient
from data_engine.api.yahoo_finance_client import YahooFinanceClient
from data_engine.api.tradingview_client import TradingViewClient

from data_engine.handlers.vnstock_handler import VNStockHandler
from data_engine.handlers.sbv_handler import SBVHandler
from data_engine.handlers.gold_handler import GoldHandler
from data_engine.handlers.vndirect_handler import VNDirectHandler
from data_engine.handlers.yahoo_finance_handler import YahooFinanceHandler
from data_engine.handlers.tradingview_handler import TradingViewHandler

from data_engine.processors.cleaner import DataCleaner
from data_engine.processors.ohlcv_processor import OHLCVProcessor
from data_engine.processors.validator import DataValidator
from data_engine.database.repository import DataRepository

logger = logging.getLogger("data_engine.services.ingestion")

class IngestionService:
    """Orchestrator for financial and macroeconomic ingestion pipelines."""

    def __init__(self):
        # Clients
        self.vnstock_client = VNStockClient()
        self.sbv_client = SBVClient()
        self.gold_client = GoldClient()
        self.vndirect_client = VNDirectClient()
        self.yahoo_finance_client = YahooFinanceClient()
        self.tradingview_client = TradingViewClient()

        # Handlers
        self.vnstock_handler = VNStockHandler()
        self.sbv_handler = SBVHandler()
        self.gold_handler = GoldHandler()
        self.vndirect_handler = VNDirectHandler()
        self.yahoo_finance_handler = YahooFinanceHandler()
        self.tradingview_handler = TradingViewHandler()

        # Processors & Repository
        self.cleaner = DataCleaner()
        self.ohlcv_processor = OHLCVProcessor()
        self.validator = DataValidator()
        self.repo = DataRepository()

    def _run_pipeline(self, raw_records: List[Dict[str, Any]], source_name: str) -> int:
        """Helper to run the cleaning, processing, validating, and saving steps."""
        if not raw_records:
            return 0
        for r in raw_records:
            r["source"] = source_name
        cleaned = self.cleaner.clean_records(raw_records)
        processed = self.ohlcv_processor.process_records(cleaned)
        validated = self.validator.validate_records(processed)
        return self.repo.save_ohlcv_batch(validated)

    def ingest_stocks(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """Ingests historical stocks and ETFs pricing from vnstock.

        Args:
            start_date: Start date string (YYYY-MM-DD). Defaults to 2018-01-01.
            end_date: End date string (YYYY-MM-DD). Defaults to today.

        Returns:
            A status dictionary showing count of records ingested per symbol.
        """
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = settings.system.get("start_date") or today_str
        if not end_date:
            end_date = settings.system.get("end_date") or today_str

        vnstock_config = settings.apis.vnstock
        
        # Collect symbols
        symbols = []
        if hasattr(vnstock_config, "etf_symbols"):
            symbols.extend(list(vnstock_config.etf_symbols))
        if hasattr(vnstock_config, "hnx_symbols"):
            symbols.extend(list(vnstock_config.hnx_symbols))
        if hasattr(vnstock_config, "upcom_symbols"):
            symbols.extend(list(vnstock_config.upcom_symbols))
            
        # HOSE_symbols can be a list or a string like "VN100" (which we might expand, or treat as a single ticker)
        if hasattr(vnstock_config, "hose_symbols"):
            hose = vnstock_config.hose_symbols
            if isinstance(hose, list):
                symbols.extend(hose)
            elif isinstance(hose, str) and hose:
                symbols.append(hose)

        results = {}
        total_saved = 0
        for symbol in symbols:
            try:
                # Use type="index" for known indexes to avoid vnstock library errors
                symbol_type = "index" if symbol in ["VN100", "VN30", "VNINDEX"] else vnstock_config.get("type", "stock")
                raw_df = self.vnstock_client.fetch_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    resolution=vnstock_config.get("resolution", "1D"),
                    type=symbol_type
                )
                raw_records = self.vnstock_handler.format_data(raw_df, symbol)
                saved = self._run_pipeline(raw_records, "vnstock")
                results[symbol] = saved
                total_saved += saved
            except Exception as e:
                logger.error(f"Failed to ingest stock {symbol}: {e}")
                results[symbol] = f"Error: {e}"

        return {"status": "success", "total_records": total_saved, "details": results}

    def ingest_gold(self) -> Dict[str, Any]:
        """Ingests SJC Gold price histories (buy and sell tickers)."""
        try:
            csv_content = self.gold_client.fetch_gold_csv()
            raw_records = self.gold_handler.format_data(csv_content)
            
            # Filter SJC data to start from 2018-01-01
            cutoff = datetime.datetime(2018, 1, 1)
            filtered_records = [r for r in raw_records if r["timestamp"] >= cutoff]
            
            saved = self._run_pipeline(filtered_records, "sjc_crawler")
            return {"status": "success", "total_records": saved, "details": {"SJC_BUY_SELL": saved}}
        except Exception as e:
            logger.error(f"Failed SJC Gold ingestion: {e}")
            return {"status": "error", "message": str(e)}

    def ingest_macro(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """Ingests macro indicators (SBV rates, Investing TVC indicators, VNDirect sector indices)."""
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = settings.system.get("start_date") or today_str
        if not end_date:
            end_date = settings.system.get("end_date") or today_str

        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        
        results = {}
        total_saved = 0

        # 1. Ingest SBV Interbank rates
        try:
            # We crawl pages until we hit records before start_date
            sbv_records = []
            page = 1
            should_stop = False
            while not should_stop:
                data = self.sbv_client.fetch_rates(page=page, page_size=100)
                parsed = self.sbv_handler.format_data(data)
                if not parsed:
                    break
                
                # Check if we should stop paging (records sorted descending)
                min_dt = min(r["timestamp"] for r in parsed)
                if min_dt < start_dt:
                    should_stop = True
                    # Keep only those within range
                    parsed = [r for r in parsed if r["timestamp"] >= start_dt]
                
                sbv_records.extend(parsed)
                page += 1
                
            saved = self._run_pipeline(sbv_records, "sbv_crawler")
            results["SBV_rates"] = saved
            total_saved += saved
        except Exception as e:
            logger.error(f"Failed to ingest SBV rates: {e}")
            results["SBV_rates"] = f"Error: {e}"

        # 2. Ingest Yahoo Finance & TradingView metrics
        from_ts = int(start_dt.timestamp())
        to_ts = int(end_dt.timestamp())

        # Map each index details format from assets.yaml
        formats = {}
        for index_name, index_config in settings.macro_indices.__dict__.items():
            if hasattr(index_config, "symbol"):
                sym = index_config.symbol
                if isinstance(sym, str):
                    formats[sym] = index_config.get("data_format", "ohlcv")

        # 2a. Yahoo Finance (DXY, USDVND, XAUUSD)
        if hasattr(settings.apis, "yahoo_finance"):
            yf_config = settings.apis.yahoo_finance
            yf_symbols = yf_config.get("symbols", {})
            if hasattr(yf_symbols, "to_dict"):
                yf_symbols = yf_symbols.to_dict()
            else:
                yf_symbols = dict(yf_symbols)

            for db_symbol, yf_symbol in yf_symbols.items():
                try:
                    fmt = formats.get(db_symbol, "ohlcv")
                    logger.info(f"Ingesting {db_symbol} from Yahoo Finance as {yf_symbol}")
                    raw_json = self.yahoo_finance_client.fetch_history(yf_symbol, from_ts, to_ts)
                    raw_records = self.yahoo_finance_handler.format_data(raw_json, db_symbol, data_format=fmt)
                    saved = self._run_pipeline(raw_records, "yahoo_finance")
                    results[f"Yahoo_{db_symbol}"] = saved
                    total_saved += saved
                except Exception as e:
                    logger.error(f"Failed to ingest Yahoo Finance symbol {db_symbol} ({yf_symbol}): {e}")
                    results[f"Yahoo_{db_symbol}"] = f"Error: {e}"

        # 2b. TradingView (VN10Y, VN3Y) -> mapped to VN10YT, VN3YT in database
        if hasattr(settings.apis, "tradingview"):
            tv_config = settings.apis.tradingview
            tv_symbols = tv_config.get("symbols", {})
            if hasattr(tv_symbols, "to_dict"):
                tv_symbols = tv_symbols.to_dict()
            else:
                tv_symbols = dict(tv_symbols)

            db_symbol_map = {
                "VN10Y": "VN10YT",
                "VN3Y": "VN3YT"
            }

            for tv_key, tv_item in tv_symbols.items():
                db_symbol = db_symbol_map.get(tv_key, tv_key)
                try:
                    fmt = formats.get(db_symbol, "ohlcv")
                    if hasattr(tv_item, "to_dict"):
                        tv_item = tv_item.to_dict()
                    else:
                        tv_item = dict(tv_item)

                    tv_symbol_code = tv_item.get("symbol", tv_key)
                    tv_exchange = tv_item.get("exchange", "TVC")

                    logger.info(f"Ingesting {db_symbol} from TradingView as {tv_exchange}:{tv_symbol_code}")
                    raw_df = self.tradingview_client.fetch_history(tv_symbol_code, tv_exchange, from_ts, to_ts)
                    raw_records = self.tradingview_handler.format_data(raw_df, db_symbol, data_format=fmt)
                    saved = self._run_pipeline(raw_records, "tradingview")
                    results[f"TradingView_{db_symbol}"] = saved
                    total_saved += saved
                except Exception as e:
                    logger.error(f"Failed to ingest TradingView symbol {db_symbol} ({tv_key}): {e}")
                    results[f"TradingView_{db_symbol}"] = f"Error: {e}"

        # 3. Ingest VNDirect Sector Indices
        vndirect_config = settings.apis.vndirect_sector_indices
        sector_symbols = list(vndirect_config.get("symbols", []))
        for symbol in sector_symbols:
            try:
                from_ts = int(start_dt.timestamp())
                to_ts = int(end_dt.timestamp())
                
                raw_json = self.vndirect_client.fetch_history(symbol, from_ts, to_ts)
                raw_records = self.vndirect_handler.format_data(raw_json, symbol)
                saved = self._run_pipeline(raw_records, "vndirect_sector_indices")
                results[f"VNDirect_{symbol}"] = saved
                total_saved += saved
            except Exception as e:
                logger.error(f"Failed to ingest VNDirect sector index {symbol}: {e}")
                results[f"VNDirect_{symbol}"] = f"Error: {e}"

        return {"status": "success", "total_records": total_saved, "details": results}

    def ingest_all(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """Ingests all sources sequentially."""
        logger.info("Executing full ingestion sync pipeline...")
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = settings.system.get("start_date") or today_str
        if not end_date:
            end_date = settings.system.get("end_date") or today_str
            
        stock_res = self.ingest_stocks(start_date, end_date)
        gold_res = self.ingest_gold()
        macro_res = self.ingest_macro(start_date, end_date)

        return {
            "status": "success",
            "pipelines": {
                "stocks": stock_res,
                "gold": gold_res,
                "macro": macro_res
            }
        }
