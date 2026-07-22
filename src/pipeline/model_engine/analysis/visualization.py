"""Visualization module for DRL Portfolio Evaluation.

Renders high-quality plots for Agent NAV vs Benchmark (VN30), Drawdown (Underwater Plot),
Action Distributions per ticker, and a comprehensive summary dashboard.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model_engine.analysis.metrics_analyzer import MetricsAnalyzer


class ModelVisualizer:
    """Renders and saves evaluation charts for DRL trading strategies."""

    def __init__(self, output_dir: Optional[str] = None):
        """
        Args:
            output_dir: Directory path where plots will be saved as PNGs.
                        Defaults to data/reports.
        """
        if output_dir is None:
            # Resolve relative to project root
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../data/reports"))
        
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.analyzer = MetricsAnalyzer()

        # Modern styling configuration
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"
        plt.rcParams["axes.edgecolor"] = "#cccccc"
        plt.rcParams["axes.linewidth"] = 0.8

    def plot_nav_vs_benchmark(
        self,
        df_agent: pd.DataFrame,
        df_benchmark: Optional[pd.DataFrame] = None,
        benchmark_name: str = "VN30",
        title: str = "Cumulative NAV Performance: DRL Agent vs Benchmark",
        save_name: str = "nav_vs_benchmark.png"
    ) -> str:
        """Plots normalized NAV curve of DRL Agent vs Benchmark.

        Args:
            df_agent: DataFrame with 'date' and 'account_value'
            df_benchmark: Optional DataFrame with 'date' and 'close' or 'account_value'
            benchmark_name: Label for benchmark (default VN30)
            title: Chart title
            save_name: PNG filename to save

        Returns:
            Absolute file path of the saved PNG.
        """
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

        # Convert to Cumulative Return Percentage (%) from base 1.0
        dates_agent = pd.to_datetime(df_agent["date"])
        nav_agent = (df_agent["account_value"].values / df_agent["account_value"].values[0] - 1.0) * 100.0

        ax.plot(dates_agent, nav_agent, label="DRL Agent Strategy", color="#1f77b4", linewidth=2.2)

        # Plot Benchmark if provided
        if df_benchmark is not None and not df_benchmark.empty:
            bm_col = "account_value" if "account_value" in df_benchmark.columns else "close"
            df_agent_dates = pd.DataFrame({"date": df_agent["date"].astype(str)})
            df_bm_dates = df_benchmark.copy()
            df_bm_dates["date"] = df_bm_dates["date"].astype(str)
            
            merged_bm = pd.merge(df_agent_dates, df_bm_dates[["date", bm_col]], on="date", how="left").ffill().bfill()
            bm_values = merged_bm[bm_col].values
            if len(bm_values) > 0 and bm_values[0] > 0:
                nav_bm = (bm_values / bm_values[0] - 1.0) * 100.0
                ax.plot(dates_agent, nav_bm, label=f"Benchmark ({benchmark_name})", color="#ff7f0e", linestyle="--", linewidth=1.8)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Cumulative Return (%)", fontsize=11)
        ax.yaxis.set_major_formatter("{x:.2f}%")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1)
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
        fig.tight_layout()

        save_path = self.output_dir / save_name
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)

    def plot_underwater(
        self,
        df_agent: pd.DataFrame,
        title: str = "Underwater Plot (Drawdown %)",
        save_name: str = "underwater_plot.png"
    ) -> str:
        """Plots drawdown percentage over time (Underwater plot).

        Returns:
            Absolute file path of the saved PNG.
        """
        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)

        dates = pd.to_datetime(df_agent["date"])
        mdd_pct, peak_d, trough_d, dd_series = self.analyzer.compute_max_drawdown(df_agent)

        dd_percent = dd_series.values * 100.0  # Convert to percentage e.g. -15%

        ax.fill_between(dates, dd_percent, 0, color="#d62728", alpha=0.35)
        ax.plot(dates, dd_percent, color="#d62728", linewidth=1.2, label=f"Drawdown (Max: {mdd_pct*100:.2f}%)")

        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Drawdown (%)", fontsize=10)
        ax.yaxis.set_major_formatter("{x:.1f}%")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.legend(loc="lower left", frameon=True, facecolor="white")
        fig.tight_layout()

        save_path = self.output_dir / save_name
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)

    def plot_action_distribution(
        self,
        df_actions: pd.DataFrame,
        title: str = "Agent Action Distribution per Asset",
        save_name: str = "action_distribution.png"
    ) -> str:
        """Plots action distribution histograms / boxplots for each asset.

        Args:
            df_actions: DataFrame with 'date' and action values [-1.0, 1.0] per ticker column.

        Returns:
            Absolute file path of the saved PNG.
        """
        ticker_cols = [c for c in df_actions.columns if c != "date"]
        if not ticker_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No action data available", ha="center", va="center")
            save_path = self.output_dir / save_name
            fig.savefig(save_path)
            plt.close(fig)
            return str(save_path)

        n_tickers = len(ticker_cols)
        cols_per_row = min(n_tickers, 4)
        rows = int(np.ceil(n_tickers / cols_per_row))

        fig, axes = plt.subplots(rows, cols_per_row, figsize=(4 * cols_per_row, 3 * rows), dpi=150, squeeze=False)
        axes_flat = axes.flatten()

        for idx, ticker in enumerate(ticker_cols):
            ax = axes_flat[idx]
            vals = df_actions[ticker].dropna().values
            
            # Draw histogram and kde-like curve
            ax.hist(vals, bins=20, range=(-1.0, 1.0), color="#2ca02c", alpha=0.6, edgecolor="white")
            ax.set_title(f"{ticker}", fontsize=11, fontweight="bold")
            ax.set_xlim(-1.1, 1.1)
            ax.axvline(0, color="#888888", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Action (-1 Sell, +1 Buy)", fontsize=8)

        # Hide extra subplots
        for idx in range(n_tickers, len(axes_flat)):
            axes_flat[idx].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()

        save_path = self.output_dir / save_name
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)

    def plot_summary_dashboard(
        self,
        df_agent: pd.DataFrame,
        df_actions: pd.DataFrame,
        df_benchmark: Optional[pd.DataFrame] = None,
        benchmark_name: str = "VN30",
        save_name: str = "summary_dashboard.png"
    ) -> Tuple[str, Dict[str, Any]]:
        """Renders a complete 2x2 Evaluation Summary Dashboard containing:
        1. NAV Performance vs Benchmark
        2. Drawdown Underwater Plot
        3. Action Distribution Boxplot
        4. Metrics Summary Text Card

        Returns:
            Tuple of (saved_png_path, metrics_dict)
        """
        metrics = self.analyzer.compute_all_metrics(df_agent, df_benchmark)

        fig = plt.figure(figsize=(16, 10), dpi=150)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.3, wspace=0.25)

        # 1. NAV Curve vs Benchmark
        ax1 = fig.add_subplot(gs[0, 0])
        dates_agent = pd.to_datetime(df_agent["date"])
        nav_agent = (df_agent["account_value"].values / df_agent["account_value"].values[0] - 1.0) * 100.0
        ax1.plot(dates_agent, nav_agent, label="DRL Agent", color="#1f77b4", linewidth=2)

        if df_benchmark is not None and not df_benchmark.empty:
            bm_col = "account_value" if "account_value" in df_benchmark.columns else "close"
            df_agent_dates = pd.DataFrame({"date": df_agent["date"].astype(str)})
            df_bm_dates = df_benchmark.copy()
            df_bm_dates["date"] = df_bm_dates["date"].astype(str)
            
            merged_bm = pd.merge(df_agent_dates, df_bm_dates[["date", bm_col]], on="date", how="left").ffill().bfill()
            bm_vals = merged_bm[bm_col].values
            if len(bm_vals) > 0 and bm_vals[0] > 0:
                nav_bm = (bm_vals / bm_vals[0] - 1.0) * 100.0
                ax1.plot(dates_agent, nav_bm, label=f"Benchmark ({benchmark_name})", color="#ff7f0e", linestyle="--", linewidth=1.5)

        ax1.set_title("Cumulative Return (%) vs Benchmark", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Return (%)", fontsize=9)
        ax1.yaxis.set_major_formatter("{x:.2f}%")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.axhline(0.0, color="#888888", linestyle=":", linewidth=0.8)
        ax1.legend(loc="upper left", fontsize=8)

        # 2. Drawdown Plot
        ax2 = fig.add_subplot(gs[0, 1])
        _, _, _, dd_series = self.analyzer.compute_max_drawdown(df_agent)
        dd_percent = dd_series.values * 100.0
        ax2.fill_between(dates_agent, dd_percent, 0, color="#d62728", alpha=0.4)
        ax2.plot(dates_agent, dd_percent, color="#d62728", linewidth=1.0)
        ax2.set_title(f"Underwater Plot (Max Drawdown: {metrics['max_drawdown']*100:.2f}%)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Drawdown (%)", fontsize=9)
        ax2.yaxis.set_major_formatter("{x:.1f}%")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # 3. Action Distribution Boxplot per Ticker
        ax3 = fig.add_subplot(gs[1, 0])
        ticker_cols = [c for c in df_actions.columns if c != "date"]
        if ticker_cols:
            data_to_plot = [df_actions[c].dropna().values for c in ticker_cols]
            bp = ax3.boxplot(data_to_plot, labels=ticker_cols, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("#2ca02c")
                patch.set_alpha(0.6)
            ax3.set_ylim(-1.1, 1.1)
            ax3.set_title("Action Distribution per Asset (-1 Sell, +1 Buy)", fontsize=12, fontweight="bold")
            ax3.set_ylabel("Action Intensity", fontsize=9)
            ax3.axhline(0, color="#888888", linestyle=":", linewidth=0.8)
            ax3.tick_params(axis="x", rotation=45, labelsize=8)
        else:
            ax3.text(0.5, 0.5, "No Actions Recorded", ha="center", va="center")

        # 4. Metrics Text Card
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")

        card_text = (
            "STRATEGY PERFORMANCE REPORT\n"
            "-----------------------------------------------\n"
            f"• Cumulative Return:      {metrics['cumulative_return']*100:8.2f}%\n"
            f"• Annualized Return:      {metrics['annualized_return']*100:8.2f}%\n"
            f"• Annualized Volatility:  {metrics['annualized_volatility']*100:8.2f}%\n"
            f"• Sharpe Ratio:           {metrics['sharpe_ratio']:8.2f}\n"
            f"• Sortino Ratio:          {metrics['sortino_ratio']:8.2f}\n"
            f"• Max Drawdown:           {metrics['max_drawdown']*100:8.2f}%\n"
            f"• Calmar Ratio:           {metrics['calmar_ratio']:8.2f}\n"
            f"• Win Rate:               {metrics['win_rate']*100:8.2f}%\n"
            f"• Profit Factor:          {metrics['profit_factor']:8.2f}\n"
        )

        if "benchmark" in metrics:
            bm = metrics["benchmark"]
            card_text += (
                "-----------------------------------------------\n"
                "📌 VS BENCHMARK (VN30):\n"
                f"• Benchmark Return:       {bm['cumulative_return']*100:8.2f}%\n"
                f"• Excess Alpha:           {bm['alpha']*100:8.2f}%\n"
                f"• Benchmark Sharpe:       {bm['sharpe_ratio']:8.2f}\n"
            )

        ax4.text(
            0.05, 0.95, card_text,
            transform=ax4.transAxes,
            fontsize=10,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#dddddd")
        )

        fig.suptitle("DRL Portfolio Strategy Evaluation Dashboard", fontsize=16, fontweight="bold", y=0.98)

        save_path = self.output_dir / save_name
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return str(save_path), metrics
