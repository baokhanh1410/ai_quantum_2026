"""Charts Component — Plotly interactive chart wrappers for Streamlit demo.

Provides:
- plot_close_price_plotly(): OHLCV line chart for a single ticker
- plot_action_histogram_plotly(): Action distribution histogram per ticker
- render_metrics_cards(): KPI cards using st.metric()
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional


def plot_close_price_plotly(
    df: pd.DataFrame,
    ticker: str,
    title: Optional[str] = None,
) -> go.Figure:
    """Renders an interactive Plotly line chart of closing prices for one ticker.

    Args:
        df: DataFrame with columns ['date', 'tic', 'close', ...].
        ticker: Ticker symbol to plot.
        title: Optional chart title.

    Returns:
        Plotly Figure object (caller passes to st.plotly_chart).
    """
    df_t = df[df["tic"] == ticker].copy()
    df_t["date"] = pd.to_datetime(df_t["date"])
    df_t = df_t.sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_t["date"],
        y=df_t["close"],
        mode="lines",
        name=ticker,
        line=dict(color="#4C72B0", width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Giá đóng cửa: %{y:,.0f} VND<extra></extra>",
    ))

    fig.update_layout(
        title=title or f"Giá đóng cửa — {ticker}",
        xaxis_title="Ngày",
        yaxis_title="Giá (VND)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=20, t=50, b=40),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def plot_action_histogram_plotly(
    df_actions: pd.DataFrame,
    max_cols: int = 4,
) -> Optional[go.Figure]:
    """Renders an interactive histogram of action values per ticker/CASH.

    Args:
        df_actions: DataFrame with 'date' + one column per ticker/CASH.
        max_cols: Max tickers to display per row.

    Returns:
        Plotly Figure or None if no ticker data.
    """
    ticker_cols = [c for c in df_actions.columns if c != "date"]
    if not ticker_cols:
        return None

    # Melt to long format
    df_long = df_actions.melt(id_vars="date", value_vars=ticker_cols,
                               var_name="Ticker", value_name="Weight")

    fig = px.histogram(
        df_long,
        x="Weight",
        facet_col="Ticker",
        facet_col_wrap=max_cols,
        nbins=30,
        title="Phân phối Tỷ trọng Hành động (Action Weight Distribution)",
        labels={"Weight": "Tỷ trọng [0.0, 1.0]", "count": "Tần suất"},
        template="plotly_white",
        height=max(300, 200 * ((len(ticker_cols) - 1) // max_cols + 1)),
    )
    fig.update_layout(
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def plot_cash_allocation_plotly(df_actions: pd.DataFrame) -> Optional[go.Figure]:
    """Renders a stacked area chart showing portfolio weight allocation over time (CASH vs Stocks).

    Args:
        df_actions: DataFrame with 'date' + action columns ('CASH', tickers...).

    Returns:
        Plotly Figure.
    """
    cols = [c for c in df_actions.columns if c != "date"]
    if not cols:
        return None

    df = df_actions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    fig = go.Figure()
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for i, col in enumerate(cols):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df[col] * 100.0,
            mode="lines",
            stackgroup="one",
            name=col,
            line=dict(width=0.5, color=color),
            hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{col}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        title="Biến động Tỷ trọng Danh mục qua Thời gian (Portfolio Weight Allocation: CASH vs Stocks)",
        xaxis_title="Ngày",
        yaxis_title="Tỷ trọng (%)",
        yaxis_range=[0, 100],
        hovermode="x unified",
        template="plotly_white",
        height=400,
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig



def plot_nav_plotly(
    df_account: pd.DataFrame,
    df_benchmark: Optional[pd.DataFrame] = None,
    benchmark_name: str = "VN30",
) -> go.Figure:
    """Interactive Plotly NAV curve — Agent vs Benchmark.

    Args:
        df_account: DataFrame with ['date', 'account_value'].
        df_benchmark: Optional DataFrame with ['date', 'close'] or ['date', 'account_value'].
        benchmark_name: Label for benchmark series.

    Returns:
        Plotly Figure.
    """
    df_a = df_account.copy()
    df_a["date"] = pd.to_datetime(df_a["date"])
    df_a = df_a.sort_values("date")

    base = df_a["account_value"].iloc[0]
    nav_agent = (df_a["account_value"] / base - 1.0) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_a["date"], y=nav_agent,
        mode="lines", name="DRL Agent",
        line=dict(color="#1f77b4", width=2.5),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Agent: %{y:.2f}%<extra></extra>",
    ))

    if df_benchmark is not None and not df_benchmark.empty:
        bm_col = "account_value" if "account_value" in df_benchmark.columns else "close"
        df_bm = df_benchmark.copy()
        df_bm["date"] = pd.to_datetime(df_bm["date"])
        # Align benchmark to agent dates
        merged = pd.merge(
            df_a[["date"]], df_bm[["date", bm_col]],
            on="date", how="left"
        ).ffill().bfill()

        bm_vals = merged[bm_col].values
        if len(bm_vals) > 0 and bm_vals[0] > 0:
            nav_bm = (bm_vals / bm_vals[0] - 1.0) * 100.0
            fig.add_trace(go.Scatter(
                x=df_a["date"], y=nav_bm,
                mode="lines", name=benchmark_name,
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{benchmark_name}: %{{y:.2f}}%<extra></extra>",
            ))

    fig.add_hline(y=0, line_dash="dot", line_color="#888888", line_width=1)
    fig.update_layout(
        title="NAV Curve: DRL Agent vs Benchmark",
        xaxis_title="Ngày",
        yaxis_title="Cumulative Return (%)",
        yaxis_tickformat=".2f",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


def plot_drawdown_plotly(df_account: pd.DataFrame) -> go.Figure:
    """Interactive Plotly drawdown (underwater) chart.

    Args:
        df_account: DataFrame with ['date', 'account_value'].

    Returns:
        Plotly Figure.
    """
    df = df_account.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    values = df["account_value"].values
    running_max = np.maximum.accumulate(values)
    # Avoid division by zero
    drawdown = np.where(running_max > 0, (values / running_max - 1.0) * 100.0, 0.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=drawdown,
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="#d62728", width=1.5),
        fillcolor="rgba(214, 39, 40, 0.25)",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Drawdown: %{y:.2f}%<extra></extra>",
    ))

    fig.add_hline(y=0, line_color="#444444", line_width=0.8)
    fig.update_layout(
        title="Underwater Plot (Drawdown %)",
        xaxis_title="Ngày",
        yaxis_title="Drawdown (%)",
        yaxis_tickformat=".1f",
        hovermode="x unified",
        template="plotly_white",
        height=300,
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def render_metrics_cards(metrics: Dict[str, Any]) -> None:
    """Render KPI metric cards using st.metric() in a grid layout.

    Args:
        metrics: Dictionary from MetricsAnalyzer.calculate_all().
    """
    col1, col2, col3, col4 = st.columns(4)

    cum_ret = metrics.get("cumulative_return", 0.0)
    sharpe  = metrics.get("sharpe_ratio", 0.0)
    sortino = metrics.get("sortino_ratio", 0.0)
    mdd     = metrics.get("max_drawdown", 0.0)
    calmar  = metrics.get("calmar_ratio", 0.0)
    win     = metrics.get("win_rate", 0.0)
    ann_ret = metrics.get("annualized_return", 0.0)
    ann_vol = metrics.get("annualized_volatility", 0.0)

    bm = metrics.get("benchmark", {})
    bm_cum = bm.get("cumulative_return", None)
    alpha   = bm.get("alpha", None)

    with col1:
        st.metric(
            "📈 Lợi nhuận tích lũy",
            f"{cum_ret*100:.2f}%",
            delta=f"vs BM: {(cum_ret - bm_cum)*100:+.2f}%" if bm_cum is not None else None,
        )
        st.metric("📅 LN hàng năm", f"{ann_ret*100:.2f}%")

    with col2:
        st.metric(
            "⚡ Sharpe Ratio",
            f"{sharpe:.3f}",
            delta=f"Alpha: {alpha*100:+.2f}%" if alpha is not None else None,
        )
        st.metric("🎯 Sortino Ratio", f"{sortino:.3f}")

    with col3:
        st.metric("📉 Max Drawdown", f"{mdd*100:.2f}%")
        st.metric("🔄 Calmar Ratio", f"{calmar:.3f}")

    with col4:
        st.metric("🏆 Win Rate", f"{win*100:.1f}%")
        st.metric("📊 Biến động / năm", f"{ann_vol*100:.2f}%")

    if bm:
        st.caption(
            f"Benchmark ({metrics.get('benchmark_name', 'VN30')}): "
            f"Return={bm.get('cumulative_return', 0)*100:.2f}%  |  "
            f"Sharpe={bm.get('sharpe_ratio', 0):.3f}  |  "
            f"Max DD={bm.get('max_drawdown', 0)*100:.2f}%"
        )
