"""Metrics Analyzer module for evaluating DRL portfolio strategies.

Computes financial and risk metrics: Cumulative Return, Annualized Return,
Annualized Volatility, Sharpe Ratio, Sortino Ratio, Max Drawdown (MDD),
Calmar Ratio, Win Rate, and Profit Factor.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple


class MetricsAnalyzer:
    """Calculates comprehensive financial performance and risk metrics."""

    def __init__(self, risk_free_rate: float = 0.03, trading_days: int = 252):
        """
        Args:
            risk_free_rate: Annual risk-free rate (e.g. 0.03 for 3%).
            trading_days: Number of trading days per year (default 252).
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def _get_values(self, df_account: pd.DataFrame) -> np.ndarray:
        """Helper to extract portfolio/benchmark series values from 'account_value' or 'close'."""
        if "account_value" in df_account.columns:
            return df_account["account_value"].values
        elif "close" in df_account.columns:
            return df_account["close"].values
        return np.array([])

    def compute_returns(self, df_account: pd.DataFrame) -> pd.Series:
        """Extracts or computes daily returns from account_value or close."""
        if "daily_return" in df_account.columns:
            return df_account["daily_return"].fillna(0)
        col = "account_value" if "account_value" in df_account.columns else "close"
        if col in df_account.columns:
            return df_account[col].pct_change(1).fillna(0)
        return pd.Series(dtype=float)

    def compute_cumulative_return(self, df_account: pd.DataFrame) -> float:
        """Calculates cumulative total return: (End_Value / Start_Value) - 1."""
        values = self._get_values(df_account)
        if len(values) == 0 or values[0] == 0:
            return 0.0
        return (values[-1] / values[0]) - 1.0

    def compute_annualized_return(self, df_account: pd.DataFrame) -> float:
        """Calculates annualized return (CAGR)."""
        values = self._get_values(df_account)
        n_days = len(values)
        if n_days <= 1 or values[0] == 0:
            return 0.0
        total_return = values[-1] / values[0]
        years = n_days / self.trading_days
        if years <= 0:
            return 0.0
        return float(total_return ** (1.0 / years) - 1.0)

    def compute_annualized_volatility(self, df_account: pd.DataFrame) -> float:
        """Calculates annualized volatility of daily returns."""
        returns = self.compute_returns(df_account)
        return float(returns.std() * np.sqrt(self.trading_days))

    def compute_sharpe_ratio(self, df_account: pd.DataFrame) -> float:
        """Calculates annualized Sharpe Ratio: (Annualized Return - Risk Free Rate) / Annualized Volatility."""
        ann_return = self.compute_annualized_return(df_account)
        ann_vol = self.compute_annualized_volatility(df_account)
        if ann_vol == 0 or np.isnan(ann_vol):
            return 0.0
        return float((ann_return - self.risk_free_rate) / ann_vol)

    def compute_sortino_ratio(self, df_account: pd.DataFrame) -> float:
        """Calculates annualized Sortino Ratio: (Annualized Return - Risk Free Rate) / Annualized Downside Volatility."""
        ann_return = self.compute_annualized_return(df_account)
        returns = self.compute_returns(df_account)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            # Trả về giá trị hữu hạn (không dùng inf để tránh lỗi JSON/display)
            return min((ann_return - self.risk_free_rate) / 1e-8, 999.0) if ann_return > self.risk_free_rate else 0.0
        ann_downside_vol = float(downside_returns.std() * np.sqrt(self.trading_days))
        if ann_downside_vol == 0 or np.isnan(ann_downside_vol):
            return 0.0
        return float((ann_return - self.risk_free_rate) / ann_downside_vol)

    def compute_max_drawdown(self, df_account: pd.DataFrame) -> Tuple[float, Optional[str], Optional[str], pd.Series]:
        """Calculates Max Drawdown (MDD) percentage, peak date, trough date, and drawdown series.

        Returns:
            Tuple of (mdd_pct, peak_date, trough_date, drawdown_series)
        """
        values = self._get_values(df_account)
        if len(values) == 0:
            return 0.0, None, None, pd.Series(dtype=float)

        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax
        mdd_pct = float(np.min(drawdown))  # Negative number e.g. -0.15 for -15%

        dates = df_account["date"].values if "date" in df_account.columns else np.arange(len(values))
        trough_idx = int(np.argmin(drawdown))
        peak_idx = int(np.argmax(values[:trough_idx + 1])) if trough_idx >= 0 else 0

        peak_date = str(dates[peak_idx]) if len(dates) > peak_idx else None
        trough_date = str(dates[trough_idx]) if len(dates) > trough_idx else None

        dd_series = pd.Series(drawdown, index=df_account.index)
        return mdd_pct, peak_date, trough_date, dd_series

    def compute_calmar_ratio(self, df_account: pd.DataFrame) -> float:
        """Calculates Calmar Ratio = Annualized Return / |Max Drawdown|."""
        ann_return = self.compute_annualized_return(df_account)
        mdd_pct, _, _, _ = self.compute_max_drawdown(df_account)
        if abs(mdd_pct) == 0:
            return 0.0
        return float(ann_return / abs(mdd_pct))

    def compute_win_rate(self, df_account: pd.DataFrame) -> float:
        """Calculates win rate (% of trading days with positive return)."""
        returns = self.compute_returns(df_account)
        non_zero_returns = returns[returns != 0]
        if len(non_zero_returns) == 0:
            return 0.0
        win_days = (non_zero_returns > 0).sum()
        return float(win_days / len(non_zero_returns))

    def compute_profit_factor(self, df_account: pd.DataFrame) -> float:
        """Calculates Profit Factor = Gross Profit / Gross Loss."""
        returns = self.compute_returns(df_account)
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return float(gains / losses)

    def compute_all_metrics(
        self,
        df_account: pd.DataFrame,
        benchmark_account: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Calculates a comprehensive summary dictionary of all metrics.

        Args:
            df_account: Agent's account history with 'account_value' and 'date'.
            benchmark_account: Optional benchmark account history (e.g. VN30).

        Returns:
            Dictionary with formatted metrics.
        """
        cum_ret = self.compute_cumulative_return(df_account)
        ann_ret = self.compute_annualized_return(df_account)
        ann_vol = self.compute_annualized_volatility(df_account)
        sharpe = self.compute_sharpe_ratio(df_account)
        sortino = self.compute_sortino_ratio(df_account)
        mdd_pct, peak_d, trough_d, _ = self.compute_max_drawdown(df_account)
        calmar = self.compute_calmar_ratio(df_account)
        win_rate = self.compute_win_rate(df_account)
        profit_factor = self.compute_profit_factor(df_account)

        metrics = {
            "cumulative_return": cum_ret,
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": mdd_pct,
            "max_drawdown_peak_date": peak_d,
            "max_drawdown_trough_date": trough_d,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "trading_days": len(df_account),
        }

        if benchmark_account is not None and not benchmark_account.empty:
            bm_cum = self.compute_cumulative_return(benchmark_account)
            bm_ann = self.compute_annualized_return(benchmark_account)
            bm_vol = self.compute_annualized_volatility(benchmark_account)
            bm_sharpe = self.compute_sharpe_ratio(benchmark_account)
            bm_mdd, _, _, _ = self.compute_max_drawdown(benchmark_account)

            metrics["benchmark"] = {
                "cumulative_return": bm_cum,
                "annualized_return": bm_ann,
                "annualized_volatility": bm_vol,
                "sharpe_ratio": bm_sharpe,
                "max_drawdown": bm_mdd,
                "alpha": ann_ret - bm_ann,
                "excess_cumulative_return": cum_ret - bm_cum,
            }

        return metrics

    def calculate_all(
        self,
        df_account: pd.DataFrame,
        benchmark_account: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Alias của compute_all_metrics() — tương thích ngược với Streamlit 3_Analysis.py."""
        return self.compute_all_metrics(df_account, benchmark_account)
