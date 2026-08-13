"""Page 2 — Train Model: Cấu hình và huấn luyện DRL Agent."""
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
    get_state, set_state, clear_model_state,
    KEY_TRAIN_DATA, KEY_VAL_DATA,
    KEY_TRAINED_MODEL, KEY_MODEL_NAME,
    KEY_DF_ACCOUNT, KEY_DF_ACTIONS, KEY_DF_SHARES,
    KEY_TRAIN_START, KEY_TRAIN_END, KEY_VAL_START, KEY_VAL_END,
    has_data,
)
from components.config_panel import render_config_panel, render_config_summary_card

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Huấn luyện — AI Quantum 2026",
    page_icon="🧠",
    layout="wide",
)

init_state()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Cấu hình Huấn luyện")

    algo = st.selectbox(
        "Thuật toán",
        ["A2C", "PPO", "DDPG", "All (Ensemble)"],
        help="Ensemble sẽ train cả 3 và chọn model tốt nhất theo Sharpe Ratio",
        key="train_algo",
    )

    try:
        from core.config.settings import MODEL_CONFIG
        default_timesteps = MODEL_CONFIG.get("training_settings", {}).get("total_timesteps", 20000)
        default_ep_len = MODEL_CONFIG.get("episode_length", 60)
    except Exception:
        default_timesteps = 20000
        default_ep_len = 60

    timesteps = st.number_input(
        "Total Timesteps",
        min_value=1000,
        max_value=500000,
        value=int(default_timesteps),
        step=5000,
        help="Tổng số bước thời gian (timesteps) để huấn luyện mô hình DRL",
        key="train_timesteps",
    )

    ep_len = st.number_input(
        "Episode Length (Trading Days)",
        min_value=20,
        max_value=500,
        value=int(default_ep_len),
        step=10,
        help="Độ dài episode khi reset() ngẫu nhiên trong lúc train (Rolling Horizon)",
        key="train_ep_len",
    )

    st.divider()
    model_cfg = render_config_panel(in_sidebar=True)
    st.divider()
    train_btn = st.button("▶ Bắt đầu Huấn luyện", type="primary", use_container_width=True)
    if st.button("🗑️ Xóa Model", use_container_width=True):
        clear_model_state()
        st.rerun()

    render_sidebar_status()

# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("🧠 Huấn luyện DRL Agent")
st.caption("Chọn thuật toán và cấu hình timesteps, sau đó nhấn Bắt đầu Huấn luyện.")

# ─── Guard: data must be loaded ───────────────────────────────────────────────
if not has_data():
    st.warning("⚠️ Chưa có dữ liệu. Vui lòng sang **Trang 1 — Dữ liệu** để tải dữ liệu trước.")
    st.page_link("pages/1_Data_Overview.py", label="→ Đi đến Trang Dữ liệu", icon="📊")
    st.stop()

train_data = get_state(KEY_TRAIN_DATA)
val_data   = get_state(KEY_VAL_DATA)

# ─── Config Summary ────────────────────────────────────────────────────────
st.subheader("📋 Cấu hình Training")

cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
cfg_col1.metric("Thuật toán", algo)
cfg_col2.metric("Timesteps", f"{timesteps:,}")
cfg_col3.metric("Train Period", f"{get_state(KEY_TRAIN_START, '?')} → {get_state(KEY_TRAIN_END, '?')}")
cfg_col4.metric("Episode Length", f"{ep_len} ngày (Random Start)")

render_config_summary_card()

tickers = sorted(train_data["tic"].unique().tolist()) if "tic" in train_data.columns else []
train_dates = train_data["date"].nunique() if "date" in train_data.columns else 0

c1, c2 = st.columns(2)
c1.info(f"**Train data:** {len(train_data):,} rows · {len(tickers)} tickers · {train_dates:,} trading days")

try:
    from core.config.settings import MARKET_CONFIG, settings
    mc = MARKET_CONFIG.get("transaction_costs", {})
    lot = MARKET_CONFIG.get("trading_rules", {}).get("lot_size", 100)
    c2.info(
        f"**Market Constraints:** Lô {lot} cp · "
        f"Mua {mc.get('brokerage_fee_buy',0)*100:.2f}% · "
        f"Bán {(mc.get('brokerage_fee_sell',0)+mc.get('personal_income_tax_sell',0))*100:.2f}% · "
        f"T+2 Settlement"
    )
    t_map = getattr(settings, "ticker_exchange_map", {})
except Exception:
    c2.info("**Market Constraints:** T+2 · Lot 100 · Fee 0.15%/0.25%")
    t_map = {}

stock_tics = [t for t in tickers if t_map.get(t, "HOSE") not in ("GOLD", "BOND")]
defensive_tics = [t for t in tickers if t_map.get(t, "") in ("GOLD", "BOND")]

with st.expander(f"📌 Danh sách tài sản trong model ({len(tickers)} tài sản: {len(stock_tics)} cổ phiếu, {len(defensive_tics)} phòng thủ)"):
    st.markdown(f"**Cổ phiếu ({len(stock_tics)}):** " + ", ".join([f"`{t}`" for t in stock_tics]))
    if defensive_tics:
        st.markdown(f"**Tài sản phòng thủ ({len(defensive_tics)}):** " + ", ".join([f"`{t}`" for t in defensive_tics]))

st.divider()


# ─── Train ─────────────────────────────────────────────────────────────────
if train_btn:
    clear_model_state()
    with st.spinner(f"⏳ Đang huấn luyện {algo}... (timesteps={timesteps:,}, ep_length={ep_len}) — vui lòng đợi"):
        try:
            from model_engine.env.stock_trading_env import StockTradingEnv
            from model_engine.models.drl_models import DRLEnsembleStrategy
            from core.config.settings import MODEL_CONFIG, MARKET_CONFIG

            # Build env_kwargs — đọc turbulence_settings từ MODEL_CONFIG
            feature_cols = [
                c for c in train_data.columns
                if c not in ["tic", "date", "open", "high", "low", "close", "volume"]
            ]
            turb_cfg = model_cfg.get("turbulence_settings", {})
            reward_cfg = model_cfg.get("reward_settings", {})
            raw_trig = turb_cfg.get("threshold_trigger")
            raw_thresh = turb_cfg.get("threshold")
            if raw_trig is not None:
                turb_threshold = float(raw_trig)
            elif raw_thresh is not None:
                turb_threshold = float(raw_thresh)
            else:
                turb_threshold = 100.0

            initial_balance = int(model_cfg.get("initial_balance", 1_000_000_000))


            env_kwargs = {
                "features":             feature_cols,
                "initial_balance":      initial_balance,
                "turbulence_threshold": turb_threshold,
                "episode_length":       ep_len,
                "random_start":         True,
                "reward_settings":      reward_cfg,
                "turbulence_settings":  turb_cfg,
            }

            strategy = DRLEnsembleStrategy(
                env_train_class=StockTradingEnv,
                env_kwargs=env_kwargs,
                train_data=train_data,
                val_data=val_data,
                total_timesteps=timesteps,
            )

            # Gọi trực tiếp DRLEnsembleStrategy methods để đảm bảo checkpoint & TensorBoard được lưu
            if algo == "A2C":
                selected_model = strategy.train_a2c()
                selected_name = "A2C"

            elif algo == "PPO":
                selected_model = strategy.train_ppo()
                selected_name = "PPO"

            elif algo == "DDPG":
                selected_model = strategy.train_ddpg()
                selected_name = "DDPG"

            else:  # All (Ensemble)
                selected_name, selected_model = strategy.train_and_select(total_timesteps=timesteps)

            # Evaluate on val set to get Sharpe preview
            df_account, df_actions, df_shares = strategy.evaluate_and_get_trajectory(
                selected_model, val_data
            )

            # Save to session state
            set_state(KEY_TRAINED_MODEL, selected_model)
            set_state(KEY_MODEL_NAME,    selected_name)
            set_state(KEY_DF_ACCOUNT,    df_account)
            set_state(KEY_DF_ACTIONS,    df_actions)
            set_state(KEY_DF_SHARES,     df_shares)

            # Quick Sharpe preview
            _trading_days = float(MARKET_CONFIG.get("market_settings", {}).get("trading_days_per_year", 252))
            returns = df_account["daily_return"]
            quick_sharpe = float((_trading_days ** 0.5) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0
            cum_return = float((df_account["account_value"].iloc[-1] / df_account["account_value"].iloc[0]) - 1.0) * 100

            st.success(
                f"✅ **{selected_name}** huấn luyện hoàn tất!\n\n"
                f"📈 Cumulative Return (Val): **{cum_return:+.2f}%** · "
                f"Sharpe: **{quick_sharpe:.3f}**"
            )

        except Exception as e:
            st.error(f"❌ Lỗi khi huấn luyện:\n\n```\n{e}\n```")
            import traceback
            with st.expander("Chi tiết lỗi"):
                st.code(traceback.format_exc())
            st.stop()

# ─── Show Results if Model Exists ─────────────────────────────────────────────
model_name  = get_state(KEY_MODEL_NAME)
df_account  = get_state(KEY_DF_ACCOUNT)

if model_name and df_account is not None and not df_account.empty:
    st.subheader(f"📊 Kết quả Sơ bộ — {model_name}")

    _trading_days = float(MARKET_CONFIG.get("market_settings", {}).get("trading_days_per_year", 252))
    returns = df_account["daily_return"]
    cum_ret = float((df_account["account_value"].iloc[-1] / df_account["account_value"].iloc[0]) - 1.0) * 100
    quick_sharpe = float((_trading_days ** 0.5) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0

    mdd_vals = df_account["account_value"].values
    running_max = np.maximum.accumulate(mdd_vals)
    dd = np.where(running_max > 0, mdd_vals / running_max - 1.0, 0.0)
    mdd_pct = float(dd.min()) * 100

    r1, r2, r3 = st.columns(3)
    r1.metric("📈 Cumulative Return (Val)", f"{cum_ret:+.2f}%")
    r2.metric("⚡ Sharpe Ratio (Val)",       f"{quick_sharpe:.3f}")
    r3.metric("📉 Max Drawdown (Val)",        f"{mdd_pct:.2f}%")

    st.divider()
    st.page_link("pages/3_Analysis.py", label="→ Xem Phân tích Đầy đủ", icon="📈")
else:
    st.info("💡 Nhấn **Bắt đầu Huấn luyện** ở sidebar để train model.")
