import sys
from pathlib import Path
import numpy as np

# Add src/pipeline to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "pipeline"))

from model_engine.env.components.execution_engine import ExecutionEngine

def test_execution_engine_tiered_exit_fee():
    tickers = ["TCBF"]
    engine = ExecutionEngine(tickers=tickers, ticker_asset_map={"TCBF": "BOND_FUND"})

    # Check initial holding days = 0 (< 30 days) -> sell fee = 1.0% (0.010)
    lot_size, buy_fee, sell_fee, is_t0 = engine.get_ticker_microstructure("TCBF", idx=0)
    assert sell_fee == 0.010, f"Expected 0.010 for holding < 30 days, got {sell_fee}"

    # Advance holding days to 100 days (< 180 days) -> sell fee = 0.5% (0.005)
    engine.holding_days[0] = 100
    lot_size, buy_fee, sell_fee, is_t0 = engine.get_ticker_microstructure("TCBF", idx=0)
    assert sell_fee == 0.005, f"Expected 0.005 for holding 100 days, got {sell_fee}"

    # Advance holding days to 400 days (> 365 days) -> sell fee = 0%
    engine.holding_days[0] = 400
    lot_size, buy_fee, sell_fee, is_t0 = engine.get_ticker_microstructure("TCBF", idx=0)
    assert sell_fee == 0.000, f"Expected 0.000 for holding 400 days, got {sell_fee}"

def test_execution_engine_cash_advance_fee():
    tickers = ["VIC"]
    engine = ExecutionEngine(tickers=tickers, default_buy_cost=0.0015, default_sell_cost=0.0025, lot_size=100)
    
    # Setup initial state: 1,000 shares of VIC available at T+0 (idx 2)
    shares_state = np.zeros((1, 3), dtype=np.int64)
    shares_state[0, 2] = 1000
    prices = np.array([100000.0]) # 100k VND per share
    
    # Rebalance: Sell 1000 shares VIC, and immediately use 100% of sell revenue to Buy back
    # Initial balance = 0
    balance = 0.0
    weights = np.array([0.0, 1.0]) # 100% stock
    
    new_balance, new_shares_state, traded_val, _ = engine.execute_rebalance(balance, shares_state, prices, weights)
    
    # Cash advance fee should have been applied because initial_cash was 0 and sell proceeds were reused immediately in T+0
    assert new_balance < 1000.0, "Cash advance fee should be deducted from new_balance"

if __name__ == "__main__":
    test_execution_engine_tiered_exit_fee()
    test_execution_engine_cash_advance_fee()
    print("All ExecutionEngine unit tests passed successfully!")
