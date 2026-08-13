"""Unit tests for Kritzman Financial Turbulence calculation and Gym Environment Circuit Breaker."""

import numpy as np
import pandas as pd
import pytest

from feature_engine.processors.calculator_processor import CalculatorProcessor
from model_engine.env.stock_trading_env import StockTradingEnv


def test_kritzman_turbulence_calculation():
    """Verify CalculatorProcessor._kritzman_turbulence returns valid non-negative values."""
    processor = CalculatorProcessor()
    dates = pd.date_range("2023-01-01", periods=100)

    np.random.seed(42)
    df_a = pd.DataFrame({"close": np.cumprod(1 + np.random.normal(0, 0.02, 100)) * 100}, index=dates)
    df_b = pd.DataFrame({"close": np.cumprod(1 + np.random.normal(0, 0.02, 100)) * 50}, index=dates)
    symbol_data = {"STOCK_A": df_a, "STOCK_B": df_b}

    turb_series = processor._compute_macro_indicator("TURBULENCE", symbol_data, dates)

    assert isinstance(turb_series, pd.Series)
    assert len(turb_series) == 100
    assert (turb_series >= 0.0).all()
    # Ensure turbulence is computed (not all zeros) after minimum window period
    assert (turb_series.iloc[15:] > 0.0).any()


def test_stock_trading_env_turbulence_circuit_breaker():
    """Verify StockTradingEnv shifts weights to 100% Cash when turbulence breaches threshold."""
    dates = pd.date_range("2023-01-01", periods=5)
    
    # Create test dataframe with a turbulence breach on date index 2
    records = []
    for d in dates:
        records.append({
            "date": d,
            "tic": "STOCK_A",
            "close": 100.0,
            "TURBULENCE": 150.0 if d == dates[2] else 10.0,
            "RSI": 50.0
        })
    df = pd.DataFrame(records)

    env = StockTradingEnv(
        df=df,
        features=["RSI"],
        turbulence_threshold=100.0,
        initial_balance=1_000_000,
    )

    env.reset()
    # Step 1 (date 0): Normal turbulence (10.0)
    obs, reward, done, truncated, info = env.step(np.array([0.0, 10.0], dtype=np.float32))
    assert env.action_memory[-1][0] < 1.0

    # Step 2 (date 1): Normal turbulence (10.0)
    obs, reward, done, truncated, info = env.step(np.array([0.0, 10.0], dtype=np.float32))
    assert env.action_memory[-1][0] < 1.0

    # Step 3 (date 2): Turbulence Breach (150.0 > 100.0) -> Should force actions[0] = 10.0 (100% Cash)
    obs, reward, done, truncated, info = env.step(np.array([0.0, 10.0], dtype=np.float32))
    actual_weights = env.action_memory[-1]
    assert actual_weights[0] >= 0.95  # Portfolio shifted to Cash by Circuit Breaker
