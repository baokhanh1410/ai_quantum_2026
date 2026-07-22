"""Page 3 — Analysis: Phân tích kết quả backtest đầy đủ."""
import sys
import pathlib

_APP_DIR = pathlib.Path(__file__).parent.parent
_PIPELINE_DIR = _APP_DIR.parent / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


import streamlit as st
import pandas as pd
import numpy as np

from components.state import (
    init_state, render_sidebar_status,
    get_state, set_state,
    KEY_TRAINED_MODEL, KEY_MODEL_NAME,
    KEY_VAL_DATA, KEY_BENCHMARK_DF,
    KEY_DF_ACCOUNT, KEY_DF_ACTIONS, KEY_DF_SHARES,
    has_model, has_analysis,
)
from components.charts import (
    render_metrics_cards,
    plot_action_histogram_plotly,
    plot_nav_plotly,
    plot_drawdown_plotly,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phân tích — AI Quantum 2026",
    page_icon="📈",
    layout="wide",
)

init_state()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 Phân tích Kết quả")

    if has_model() and has_analysis():
        if st.button("🔄 Tính lại Evaluation", use_container_width=True):
            # Clear only analysis results to force re-run
            set_state(KEY_DF_ACCOUNT, None)
            set_state(KEY_DF_ACTIONS, None)
            set_state(KEY_DF_SHARES,  None)
            st.rerun()

    render_sidebar_status()

# ─── Main ────────────────────────────────────────────────────────────────────
st.title("📈 Phân tích Kết quả Backtest")
model_name = get_state(KEY_MODEL_NAME, "DRL Agent")
st.caption(f"Model: **{model_name}** · Evaluation trên Validation Set")

# ─── Guard: model must be trained ────────────────────────────────────────────
if not has_model():
    st.warning("⚠️ Chưa có model. Vui lòng sang **Trang 2 — Huấn luyện** để train trước.")
    st.page_link("pages/2_Train_Model.py", label="→ Đi đến Trang Huấn luyện", icon="🧠")
    st.stop()

# ─── Auto-run evaluation if not already done ─────────────────────────────────
if not has_analysis():
    val_data = get_state(KEY_VAL_DATA)
    model    = get_state(KEY_TRAINED_MODEL)

    if val_data is None or val_data.empty:
        st.error("❌ Không có dữ liệu Validation. Quay lại Trang 1 để tải dữ liệu.")
        st.stop()

    with st.spinner("⏳ Đang chạy evaluation trên Validation Set..."):
        try:
            from model_engine.env.stock_trading_env import StockTradingEnv
            from model_engine.models.drl_models import DRLEnsembleStrategy
            from core.config.settings import MODEL_CONFIG, MARKET_CONFIG

            feature_cols = [
                c for c in val_data.columns
                if c not in ["tic", "date", "open", "high", "low", "close", "volume"]
            ]

            env_kwargs = {
                "initial_balance": MODEL_CONFIG.get("initial_balance", 1_000_000_000),
                "buy_cost_pct":    MARKET_CONFIG.get("transaction_costs", {}).get("brokerage_fee_buy", 0.0015),
                "sell_cost_pct":   (
                    MARKET_CONFIG.get("transaction_costs", {}).get("brokerage_fee_sell", 0.0015)
                    + MARKET_CONFIG.get("transaction_costs", {}).get("personal_income_tax_sell", 0.001)
                ),
                "lot_size":        MARKET_CONFIG.get("trading_rules", {}).get("lot_size", 100),
                "feature_names":   feature_cols,
            }

            strategy = DRLEnsembleStrategy(
                env_train_class=StockTradingEnv,
                env_kwargs=env_kwargs,
                train_data=val_data,   # dummy train_data, only val_data used for eval
                val_data=val_data,
            )

            df_account, df_actions, df_shares = strategy.evaluate_and_get_trajectory(model, val_data)

            set_state(KEY_DF_ACCOUNT, df_account)
            set_state(KEY_DF_ACTIONS, df_actions)
            set_state(KEY_DF_SHARES,  df_shares)

        except Exception as e:
            st.error(f"❌ Lỗi evaluation:\n\n```\n{e}\n```")
            import traceback
            with st.expander("Chi tiết lỗi"):
                st.code(traceback.format_exc())
            st.stop()

# ─── Load results ─────────────────────────────────────────────────────────────
df_account = get_state(KEY_DF_ACCOUNT)
df_actions = get_state(KEY_DF_ACTIONS)
df_shares  = get_state(KEY_DF_SHARES)
bm_df      = get_state(KEY_BENCHMARK_DF)

if df_account is None or df_account.empty:
    st.error("Không có dữ liệu evaluation. Thử nhấn 'Tính lại Evaluation'.")
    st.stop()

# ─── Metrics Cards ────────────────────────────────────────────────────────────
st.subheader("📊 Tóm tắt Hiệu suất")

try:
    from model_engine.analysis.metrics_analyzer import MetricsAnalyzer
    analyzer = MetricsAnalyzer()

    # Align benchmark to agent date range
    bm_aligned = None
    if bm_df is not None and not bm_df.empty:
        agent_dates = set(df_account["date"].astype(str).tolist())
        bm_filtered = bm_df[bm_df["date"].astype(str).isin(agent_dates)].copy()
        if not bm_filtered.empty:
            bm_aligned = bm_filtered

    metrics = analyzer.calculate_all(df_account, bm_aligned)
    render_metrics_cards(metrics)

except Exception as e:
    st.warning(f"Không thể tính metrics chi tiết: {e}")

st.divider()

# ─── NAV Curve ────────────────────────────────────────────────────────────────
st.subheader("📈 NAV Curve — Agent vs Benchmark (VN30)")

fig_nav = plot_nav_plotly(df_account, bm_df, benchmark_name="VN30")
st.plotly_chart(fig_nav, use_container_width=True)

# Also render Matplotlib version for export quality
with st.expander("🖼️ Xem bản Matplotlib (chất lượng cao)"):
    try:
        from model_engine.analysis.visualization import ModelVisualizer
        viz = ModelVisualizer()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig_mpl, ax = plt.subplots(figsize=(12, 5), dpi=120)
        dates_a = pd.to_datetime(df_account["date"])
        nav_a = (df_account["account_value"].values / df_account["account_value"].values[0] - 1.0) * 100
        ax.plot(dates_a, nav_a, label="DRL Agent", color="#1f77b4", linewidth=2)

        if bm_df is not None and not bm_df.empty:
            bm_col = "close" if "close" in bm_df.columns else "account_value"
            merged = pd.merge(df_account[["date"]], bm_df[["date", bm_col]].rename(columns={"date": "date"}), on="date", how="left").ffill().bfill()
            bm_v = merged[bm_col].values
            if len(bm_v) > 0 and bm_v[0] > 0:
                nav_bm = (bm_v / bm_v[0] - 1.0) * 100
                ax.plot(dates_a, nav_bm, label="VN30", color="#ff7f0e", linestyle="--", linewidth=1.8)

        ax.axhline(0, color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Return (%)")
        ax.set_title(f"NAV Curve — {model_name} vs VN30")
        ax.yaxis.set_major_formatter("{x:.2f}%")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.legend(); fig_mpl.tight_layout()
        st.pyplot(fig_mpl)
        plt.close(fig_mpl)
    except Exception as e2:
        st.caption(f"Matplotlib render: {e2}")

st.divider()

# ─── Drawdown ────────────────────────────────────────────────────────────────
st.subheader("📉 Underwater Plot (Drawdown)")
fig_dd = plot_drawdown_plotly(df_account)
st.plotly_chart(fig_dd, use_container_width=True)

st.divider()

# ─── Portfolio Weight Allocation Over Time ──────────────────────────────────
st.subheader("💵 Biến động Tỷ trọng Danh mục qua Thời gian (CASH vs Stocks)")

if df_actions is not None and not df_actions.empty:
    from components.charts import plot_cash_allocation_plotly
    fig_cash = plot_cash_allocation_plotly(df_actions)
    if fig_cash is not None:
        st.plotly_chart(fig_cash, use_container_width=True)

st.divider()

# ─── Action Distribution ──────────────────────────────────────────────────────
st.subheader("🎯 Phân phối Tỷ trọng Hành động (Action Weight Distribution)")

if df_actions is not None and not df_actions.empty:
    fig_action = plot_action_histogram_plotly(df_actions)
    if fig_action is not None:
        st.plotly_chart(fig_action, use_container_width=True)
    else:
        st.info("Không có dữ liệu action.")
else:
    st.info("Không có dữ liệu action để hiển thị.")


st.divider()

# ─── Portfolio Holdings Over Time ─────────────────────────────────────────────
if df_shares is not None and not df_shares.empty:
    with st.expander("📋 Lịch sử Nắm giữ Cổ phiếu (Shares)"):
        st.dataframe(df_shares.tail(30), use_container_width=True, hide_index=True)

# ─── Download ────────────────────────────────────────────────────────────────
st.subheader("💾 Tải xuống Báo cáo")
col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    csv_nav = df_account.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Tải NAV Trajectory (CSV)",
        data=csv_nav,
        file_name=f"nav_trajectory_{model_name}.csv",
        mime="text/csv",
    )

with col_dl2:
    if df_actions is not None and not df_actions.empty:
        csv_actions = df_actions.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Tải Action History (CSV)",
            data=csv_actions,
            file_name=f"actions_{model_name}.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption("AI Quantum 2026 · NEU · Deep Reinforcement Learning for Vietnam Stock Market")
