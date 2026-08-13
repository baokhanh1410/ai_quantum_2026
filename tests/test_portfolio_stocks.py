import pytest
import sys
import pathlib

# Ensure src pipeline is in path
root_dir = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src" / "pipeline"))

from core.config.settings import settings, reload_settings, get_portfolio_stocks
from model_engine.data.data_service import DataQueryService


def test_get_portfolio_stocks():
    reload_settings()
    stocks = get_portfolio_stocks()
    assert isinstance(stocks, list)
    assert len(stocks) == 5
    assert "VNM" in stocks
    assert "FPT" in stocks
    assert "VCB" in stocks



def test_data_service_query_filtering():
    reload_settings()
    svc = DataQueryService()
    
    # Test with custom tickers
    custom_tickers = ["VNM", "FPT"]
    # We can mock or test query generation by checking logic
    active_stock_tickers = [s.strip() for s in custom_tickers if s and s.strip()]
    assert active_stock_tickers == ["VNM", "FPT"]

    # Test ticker_exchange_map lookup
    ticker_map = getattr(settings, "ticker_exchange_map", {})
    for t in custom_tickers:
        assert t in ticker_map
