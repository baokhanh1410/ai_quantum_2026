import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "pipeline"))

from model_engine.env.stock_trading_env import StockTradingEnv


def _make_env(n_dates: int = 10, n_tickers: int = 2, initial_balance: float = 1_000_000_000.0):
    """Helper: build a minimal StockTradingEnv with synthetic data."""
    dates = pd.date_range(start="2024-01-02", periods=n_dates, freq="B").strftime("%Y-%m-%d").tolist()
    tickers = [f"TIC{i}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": d, "tic": t,
                "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0,
                "volume": 100000.0, "TURBULENCE": 1.0,
                "RSI": 0.0, "PPO": 0.0, "CCI": 0.0, "ADX": 0.0,
                "ATR": 0.0, "VOLATILITY": 0.0,
                "YIELD_CURVE_SLOPE": 0.0, "DXY_LOG_RETURN": 0.0, "VN3YT": 0.0,
            })
    df = pd.DataFrame(rows)
    features = ["RSI", "PPO", "CCI", "ADX", "ATR", "VOLATILITY", "YIELD_CURVE_SLOPE", "DXY_LOG_RETURN", "VN3YT"]
    env = StockTradingEnv(df=df, features=features, initial_balance=initial_balance)
    return env, dates


def _run_full_episode(env):
    """Run all-cash agent through one complete evaluation episode."""
    n_tickers = env.num_stocks
    obs, _ = env.reset()
    done = False
    while not done:
        action = np.zeros(n_tickers + 1, dtype=np.float32)
        action[0] = 1.0  # 100% cash
        obs, reward, done, _, _ = env.step(action)
    return env


def test_stock_trading_env():
    """Original smoke test."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    tickers = ["VIC", "TCBF"]

    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "tic": t,
                "close": 100000.0 if t == "VIC" else 10000.0,
                "feature1": 1.0,
                "feature2": 0.5,
            })

    df = pd.DataFrame(rows)
    features = ["feature1", "feature2"]

    env = StockTradingEnv(df=df, features=features, initial_balance=1_000_000_000.0)

    obs, info = env.reset()
    assert obs is not None
    assert len(obs) > 0

    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    assert obs is not None
    assert isinstance(reward, (float, np.floating, int))
    print(f"StockTradingEnv test passed! Initial NAV: 1B, Step Reward: {reward:.4f}")


# ── Fix: Evaluation Date Integrity Tests (fix-date-memory-duplication) ────────

def test_date_memory_no_duplicates():
    """
    Spec: evaluation-date-integrity
    Scenario: No duplicate on first date
    date_memory SHALL contain no duplicate values after a full evaluation episode.
    """
    n_dates = 15
    env, _ = _make_env(n_dates=n_dates)
    env = _run_full_episode(env)

    date_mem = env.date_memory
    assert len(date_mem) == len(set(date_mem)), (
        f"date_memory has duplicate dates! "
        f"Duplicates: {[d for d in set(date_mem) if date_mem.count(d) > 1]}"
    )
    print(f"[PASS] test_date_memory_no_duplicates: {len(date_mem)} unique dates, no duplicates")


def test_date_memory_length_equals_trading_days():
    """
    Spec: evaluation-date-integrity
    Scenario: Number of rows equals number of trading days
    len(date_memory) SHALL be <= n_dates and >= n_dates - 1 (env stops one step before end_step).
    """
    n_dates = 20
    env, dates = _make_env(n_dates=n_dates)
    env = _run_full_episode(env)

    date_mem = env.date_memory
    assert len(date_mem) <= n_dates, (
        f"date_memory length ({len(date_mem)}) exceeds total trading dates ({n_dates})"
    )
    assert len(date_mem) >= n_dates - 1, (
        f"date_memory length ({len(date_mem)}) is too short (expected ~{n_dates})"
    )
    print(f"[PASS] test_date_memory_length_equals_trading_days: {len(date_mem)} entries for {n_dates} trading dates")


def test_asset_memory_date_memory_same_length():
    """
    Spec: evaluation-date-integrity
    Scenario: Asset memory and date memory lengths match
    asset_memory[0] is the initial NAV (no date), then each step appends to both.
    After fix: len(date_memory) == len(asset_memory) - 1 (initial NAV has no date entry).
    The important property is that date_memory has no duplicates and aligns with
    asset_memory[1:] so that df_account can be constructed without off-by-one errors.
    """
    n_dates = 12
    env, _ = _make_env(n_dates=n_dates)
    env = _run_full_episode(env)

    # After fix: asset_memory[0] = initial_balance (no date), rest aligned with date_memory
    assert len(env.asset_memory) == len(env.date_memory) + 1, (
        f"Expected len(asset_memory) == len(date_memory) + 1, "
        f"got len(asset_memory)={len(env.asset_memory)}, len(date_memory)={len(env.date_memory)}"
    )
    # Verify initial balance is correctly stored as first element
    assert env.asset_memory[0] == env.initial_balance, (
        f"asset_memory[0] should be initial_balance={env.initial_balance}, "
        f"got {env.asset_memory[0]}"
    )
    print(f"[PASS] test_asset_memory_date_memory_same_length: asset_memory={len(env.asset_memory)}, date_memory={len(env.date_memory)} (+1 for initial NAV)")


if __name__ == "__main__":
    test_stock_trading_env()
    test_date_memory_no_duplicates()
    test_date_memory_length_equals_trading_days()
    test_asset_memory_date_memory_same_length()
    print("\nAll tests passed!")
