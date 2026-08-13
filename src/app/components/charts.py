"""Charts Component — Plotly interactive chart wrappers for Streamlit demo.

Provides executive, modern, and publication-ready Plotly charts with:
- Crisp percentage formatting & clear financial units (%)
- Clean legend positioning without overlap
- Curated color palettes for Multi-Asset / Sector tickers
- Interactive hover tooltips
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional

# Curated palette for assets & sectors
TICKER_COLORS = {
    "CASH": "#10B981",    # Emerald Green
    "VNCOND": "#3B82F6",  # Royal Blue
    "VNCONS": "#8B5CF6",  # Violet / Purple
    "VNENE": "#F59E0B",   # Amber / Gold
    "VNFIN": "#EF4444",   # Coral Red
    "VNHEAL": "#EC4899",  # Pink
    "VNIND": "#6366F1",   # Indigo
    "VNIT": "#06B6D4",    # Cyan
    "VNMAT": "#F97316",   # Orange
    "VNREAL": "#84CC16",  # Lime Green
    "VNUTI": "#14B8A6",   # Teal
    "SJC_SELL": "#D97706",# Dark Amber / Gold
}

DEFAULT_PALETTE = [
    "#10B981", "#3B82F6", "#8B5CF6", "#F59E0B", "#EF4444",
    "#EC4899", "#6366F1", "#06B6D4", "#F97316", "#84CC16",
    "#14B8A6", "#D97706", "#64748B", "#475569"
]


def _get_ticker_color(ticker: str, index: int = 0) -> str:
    """Returns assigned color for a ticker symbol or fallback from palette."""
    return TICKER_COLORS.get(ticker, DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)])


def plot_close_price_plotly(
    df: pd.DataFrame,
    ticker: str,
    title: Optional[str] = None,
) -> go.Figure:
    """Renders an interactive Plotly line chart of closing prices for one ticker."""
    df_t = df[df["tic"] == ticker].copy()
    df_t["date"] = pd.to_datetime(df_t["date"])
    df_t = df_t.sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_t["date"],
        y=df_t["close"],
        mode="lines",
        name=ticker,
        line=dict(color=_get_ticker_color(ticker), width=2.2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Giá đóng cửa: <b>%{y:,.2f} VND</b><extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title or f"Biến động Giá đóng cửa — {ticker}", font=dict(size=16, weight="bold")),
        xaxis=dict(title="Ngày", showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(title="Giá (VND)", tickformat=",", showgrid=True, gridcolor="#F3F4F6"),
        hovermode="x unified",
        template="plotly_white",
        height=400,
        margin=dict(l=60, r=30, t=50, b=40),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def plot_action_histogram_plotly(
    df_actions: pd.DataFrame,
    max_cols: int = 4,
) -> Optional[go.Figure]:
    """Renders an interactive histogram of action values per ticker/CASH with % units."""
    ticker_cols = [c for c in df_actions.columns if c != "date"]
    if not ticker_cols:
        return None

    # Convert weights from fraction [0.0, 1.0] to percentage [0%, 100%]
    df_pct = df_actions.copy()
    for col in ticker_cols:
        df_pct[col] = df_pct[col] * 100.0

    df_long = df_pct.melt(
        id_vars="date",
        value_vars=ticker_cols,
        var_name="Ticker",
        value_name="Weight_Pct"
    )

    fig = px.histogram(
        df_long,
        x="Weight_Pct",
        color="Ticker",
        facet_col="Ticker",
        facet_col_wrap=max_cols,
        nbins=25,
        title="Phân phối Tỷ trọng Hành động (Trục X: Tỷ trọng phân bổ % | Trục Y: Số phiên)",
        color_discrete_map=TICKER_COLORS,
        template="plotly_white",
        height=max(360, 230 * ((len(ticker_cols) - 1) // max_cols + 1)),
    )

    # Clean facet titles (strip "Ticker=") and format font cleanly
    fig.for_each_annotation(
        lambda a: a.update(
            text=f"<b>{a.text.split('=')[-1]}</b>",
            font=dict(size=12, color="#1F2937")
        )
    )

    # Clear individual subplot axis titles to prevent 12 overlapping axis title labels!
    fig.update_xaxes(
        title_text="",       # Clear individual subplot X-axis text title
        ticksuffix="%",      # Display % suffix on ticks (0%, 20%, 40%...)
        showgrid=True,
        gridcolor="#F3F4F6",
        tickfont=dict(size=9.5),
        range=[0, 105],
    )
    fig.update_yaxes(
        title_text="",       # Clear individual subplot Y-axis text title
        showgrid=True,
        gridcolor="#F3F4F6",
        tickfont=dict(size=9.5),
    )

    fig.update_traces(
        marker=dict(line=dict(color="rgba(0,0,0,0.15)", width=0.5)),
        hovertemplate="<b>Tỷ trọng: %{x:.1f}%</b><br>Số phiên: <b>%{y} phiên</b><extra></extra>",
    )

    fig.update_layout(
        font=dict(family="Inter, sans-serif"),
        title=dict(font=dict(size=15, weight="bold")),
        showlegend=False,
        margin=dict(l=50, r=30, t=75, b=50),
    )
    return fig


def plot_cash_allocation_plotly(df_actions: pd.DataFrame) -> Optional[go.Figure]:
    """Renders a stacked area chart showing portfolio weight allocation over time (CASH vs Stocks) with % units."""
    cols = [c for c in df_actions.columns if c != "date"]
    if not cols:
        return None

    df = df_actions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    fig = go.Figure()

    for i, col in enumerate(cols):
        color = _get_ticker_color(col, i)
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df[col] * 100.0,
            mode="lines",
            stackgroup="one",
            name=col,
            line=dict(width=0.6, color=color),
            fillcolor=color,
            hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br><span style='color:{color}'>■</span> {col}: <b>%{{y:.2f}}%</b><extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="Biến động Tỷ trọng Danh mục qua Thời gian (Portfolio Weight Allocation: CASH vs Sectors)",
            font=dict(size=16, weight="bold")
        ),
        xaxis=dict(title="Ngày", showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(
            title="Tỷ trọng Phân bổ (%)",
            ticksuffix="%",
            range=[0, 100.5],
            showgrid=True,
            gridcolor="#F3F4F6",
        ),
        hovermode="x unified",
        template="plotly_white",
        height=490,
        font=dict(family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            title=dict(text="<b>Tài sản / Ngành:</b>", font=dict(size=11)),
            font=dict(size=11),
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#E5E7EB",
            borderwidth=1,
        ),
        margin=dict(l=60, r=30, t=65, b=110),
    )
    return fig


def plot_nav_plotly(
    df_account: pd.DataFrame,
    df_benchmark: Optional[pd.DataFrame] = None,
    benchmark_name: str = "VN30",
) -> go.Figure:
    """Interactive Plotly NAV curve — Agent vs Benchmark with % units."""
    df_a = df_account.copy()
    df_a["date"] = pd.to_datetime(df_a["date"])
    df_a = df_a.sort_values("date")

    base = df_a["account_value"].iloc[0]
    nav_agent = (df_a["account_value"] / base - 1.0) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_a["date"],
        y=nav_agent,
        mode="lines",
        name="DRL Agent (Proposed)",
        line=dict(color="#2563EB", width=2.8),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br><span style='color:#2563EB'>■</span> DRL Agent: <b>%{y:+.2f}%</b><extra></extra>",
    ))

    if df_benchmark is not None and not df_benchmark.empty:
        bm_col = "account_value" if "account_value" in df_benchmark.columns else "close"
        df_bm = df_benchmark.copy()
        
        # Standardize date format for clean alignment
        df_a_dates = pd.DataFrame({"date": pd.to_datetime(df_a["date"]).dt.strftime("%Y-%m-%d")})
        df_bm["date_str"] = pd.to_datetime(df_bm["date"]).dt.strftime("%Y-%m-%d")

        merged = pd.merge(
            df_a_dates, df_bm[["date_str", bm_col]].rename(columns={"date_str": "date"}),
            on="date", how="left"
        ).ffill().bfill()

        bm_vals = merged[bm_col].values
        if len(bm_vals) > 0 and not np.isnan(bm_vals[0]) and bm_vals[0] > 0:
            nav_bm = (bm_vals / bm_vals[0] - 1.0) * 100.0
            fig.add_trace(go.Scatter(
                x=df_a["date"],
                y=nav_bm,
                mode="lines",
                name=f"Benchmark ({benchmark_name})",
                line=dict(color="#F59E0B", width=2.0, dash="dash"),
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br><span style='color:#F59E0B'>■</span> {benchmark_name}: <b>%{{y:+.2f}}%</b><extra></extra>",
            ))

    fig.add_hline(y=0, line_dash="dot", line_color="#9CA3AF", line_width=1)
    fig.update_layout(
        title=dict(text=f"Đường cong NAV: DRL Agent vs Benchmark ({benchmark_name})", font=dict(size=16, weight="bold")),
        xaxis=dict(title="Ngày", showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(
            title="Lợi nhuận tích lũy (%)",
            ticksuffix="%",
            tickformat="+.1f",
            showgrid=True,
            gridcolor="#F3F4F6"
        ),
        hovermode="x unified",
        template="plotly_white",
        height=450,
        font=dict(family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E5E7EB",
            borderwidth=1
        ),
        margin=dict(l=60, r=30, t=65, b=40),
    )
    return fig


def plot_drawdown_plotly(df_account: pd.DataFrame) -> go.Figure:
    """Interactive Plotly drawdown (underwater) chart with % units."""
    df = df_account.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    values = df["account_value"].values
    running_max = np.maximum.accumulate(values)
    drawdown = np.where(running_max > 0, (values / running_max - 1.0) * 100.0, 0.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=drawdown,
        mode="lines",
        fill="tozeroy",
        name="Drawdown (%)",
        line=dict(color="#EF4444", width=1.8),
        fillcolor="rgba(239, 68, 68, 0.20)",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br><span style='color:#EF4444'>■</span> Drawdown: <b>%{y:.2f}%</b><extra></extra>",
    ))

    fig.add_hline(y=0, line_color="#6B7280", line_width=1)
    fig.update_layout(
        title=dict(text="Underwater Plot — Biến động Sụt giảm Tài sản (Max Drawdown %)", font=dict(size=15, weight="bold")),
        xaxis=dict(title="Ngày", showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(
            title="Sụt giảm (%)",
            ticksuffix="%",
            tickformat=".1f",
            showgrid=True,
            gridcolor="#F3F4F6"
        ),
        hovermode="x unified",
        template="plotly_white",
        height=320,
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


def render_metrics_cards(metrics: Dict[str, Any]) -> None:
    """Render KPI metric cards using st.metric() in a clean 4-column grid layout."""
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
            f"{cum_ret*100:+.2f}%",
            delta=f"vs BM: {(cum_ret - bm_cum)*100:+.2f}%" if bm_cum is not None else None,
        )
        st.metric("📅 Lợi nhuận quy năm", f"{ann_ret*100:+.2f}%")

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
        st.metric("🏆 Win Rate (Phiên thắng)", f"{win*100:.1f}%")
        st.metric("📊 Biến động quy năm", f"{ann_vol*100:.2f}%")

    if bm:
        st.caption(
            f"💡 **Benchmark ({metrics.get('benchmark_name', 'VN30')})**: "
            f"Return = {bm.get('cumulative_return', 0)*100:+.2f}%  |  "
            f"Sharpe = {bm.get('sharpe_ratio', 0):.3f}  |  "
            f"Max DD = {bm.get('max_drawdown', 0)*100:.2f}%"
        )


def plot_turbulence_plotly(
    df: pd.DataFrame,
    threshold: float = 100.0,
    turbulence_type: str = "cooldown_period",
    ewma_span: int = 10,
    threshold_trigger: Optional[float] = None,
    threshold_exit: Optional[float] = None,
) -> Optional[go.Figure]:
    """Interactive Plotly chart of Kritzman Financial Turbulence Index with emergency threshold lines."""
    turb_col = next((c for c in ["TURBULENCE", "turbulence"] if c in df.columns), None)
    if turb_col is None or df.empty:
        return None

    df_t = df.copy()
    if "date" in df_t.columns:
        df_t["date"] = pd.to_datetime(df_t["date"])
        # Take single row per date if df has multiple tickers
        df_t = df_t.groupby("date", as_index=False)[turb_col].mean().sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_t["date"],
        y=df_t[turb_col],
        mode="lines",
        name="Turbulence Thô (Raw)",
        line=dict(color="#8B5CF6", width=2.0),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br><span style='color:#8B5CF6'>■</span> Turbulence Thô: <b>%{y:.2f}</b><extra></extra>",
    ))

    if turbulence_type in ("ewma_smoothed", "dual_threshold", "ewma_dual_threshold"):
        df_t["ewma_turb"] = df_t[turb_col].ewm(span=ewma_span, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=df_t["date"],
            y=df_t["ewma_turb"],
            mode="lines",
            name=f"EWMA làm mịn (Span={ewma_span})",
            line=dict(color="#F59E0B", width=2.5),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br><span style='color:#F59E0B'>■</span> EWMA Turbulence: <b>%{y:.2f}</b><extra></extra>",
        ))

    trig = threshold_trigger if threshold_trigger is not None else threshold
    exit_trig = threshold_exit if threshold_exit is not None else (trig * 0.45)

    if turbulence_type in ("dual_threshold", "ewma_dual_threshold"):
        fig.add_hline(
            y=trig,
            line_dash="dash",
            line_color="#EF4444",
            line_width=1.8,
            annotation_text=f"🔴 Trigger ON ({trig:.0f})",
            annotation_position="top right",
            annotation_font=dict(color="#EF4444", size=11, weight="bold")
        )
        fig.add_hline(
            y=exit_trig,
            line_dash="dash",
            line_color="#10B981",
            line_width=1.8,
            annotation_text=f"🟢 Exit OFF ({exit_trig:.0f})",
            annotation_position="bottom right",
            annotation_font=dict(color="#10B981", size=11, weight="bold")
        )
    else:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#EF4444",
            line_width=1.8,
            annotation_text=f"Ngưỡng Phanh Khẩn cấp ({threshold:.0f})",
            annotation_position="top right",
            annotation_font=dict(color="#EF4444", size=11, weight="bold")
        )

    type_labels = {
        "static": "Static Phanh 1 Ngày",
        "cooldown_period": "Cooldown Giữ Phanh N Ngày",
        "ewma_smoothed": f"EWMA làm mịn (Span={ewma_span})",
        "adaptive_percentile": "Adaptive Percentile Dynamic Threshold",
        "dual_threshold": f"Dual Threshold Band ({trig:.0f} / {exit_trig:.0f})",
        "ewma_dual_threshold": f"EWMA Dual Threshold Band ({trig:.0f} / {exit_trig:.0f})",
    }
    label_str = type_labels.get(turbulence_type, turbulence_type)

    fig.update_layout(
        title=dict(
            text=f"🛡️ Chỉ số Rủi ro Hệ thống Kritzman Turbulence — Chiến lược: [{label_str}]",
            font=dict(size=14, weight="bold")
        ),
        xaxis=dict(title="Ngày", showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(title="Turbulence Score", showgrid=True, gridcolor="#F3F4F6"),
        hovermode="x unified",
        template="plotly_white",
        height=380,
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=60, r=30, t=55, b=40),
    )
    return fig

