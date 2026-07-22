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

    ep_len = st.number_input(
        "Episode Length (Trading Days)",
        min_value=20,
        max_value=500,
        value=60,
        step=10,
        help="Độ dài episode khi reset() ngẫu nhiên trong lúc train (Rolling Horizon)",
        key="train_ep_len",
    )

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

tickers = sorted(train_data["tic"].unique().tolist()) if "tic" in train_data.columns else []
train_dates = train_data["date"].nunique() if "date" in train_data.columns else 0

c1, c2 = st.columns(2)
c1.info(f"**Train data:** {len(train_data):,} rows · {len(tickers)} tickers · {train_dates:,} trading days")

try:
    from core.config.settings import MARKET_CONFIG
    mc = MARKET_CONFIG.get("transaction_costs", {})
    lot = MARKET_CONFIG.get("trading_rules", {}).get("lot_size", 100)
    c2.info(
        f"**Market Constraints:** Lô {lot} cp · "
        f"Mua {mc.get('brokerage_fee_buy',0)*100:.2f}% · "
        f"Bán {(mc.get('brokerage_fee_sell',0)+mc.get('personal_income_tax_sell',0))*100:.2f}% · "
        f"T+2 Settlement"
    )
except Exception:
    c2.info("**Market Constraints:** T+2 · Lot 100 · Fee 0.15%/0.25%")

st.divider()

# ─── Train ─────────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner(f"⏳ Đang huấn luyện {algo}... (timesteps={timesteps:,}, ep_length={ep_len}) — vui lòng đợi"):
        try:
            from model_engine.env.stock_trading_env import StockTradingEnv
            from model_engine.models.drl_models import DRLEnsembleStrategy
            from core.config.settings import MODEL_CONFIG, MARKET_CONFIG

            # Build env_kwargs consistent with notebook.py
            feature_cols = [
                c for c in train_data.columns
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
                "episode_length":  ep_len,
                "random_start":    True,
            }

            strategy = DRLEnsembleStrategy(
                env_train_class=StockTradingEnv,
                env_kwargs=env_kwargs,
                train_data=train_data,
                val_data=val_data,
            )

            # Override timesteps in config temporarily
            import stable_baselines3 as sb3

            if algo == "A2C":
                from stable_baselines3 import A2C
                from stable_baselines3.common.vec_env import DummyVecEnv
                env = strategy._make_env(train_data, is_eval=False)
                from core.config.settings import MODEL_CONFIG as MC
                model = A2C(
                    "MlpPolicy", env,
                    learning_rate=MC.get("algorithms", {}).get("a2c", {}).get("learning_rate", 0.0007),
                    n_steps=MC.get("algorithms", {}).get("a2c", {}).get("n_steps", 5),
                    ent_coef=MC.get("algorithms", {}).get("a2c", {}).get("ent_coef", 0.01),
                    verbose=0,
                )
                model.learn(total_timesteps=timesteps)
                selected_name = "A2C"
                selected_model = model

            elif algo == "PPO":
                from stable_baselines3 import PPO
                env = strategy._make_env(train_data, is_eval=False)
                from core.config.settings import MODEL_CONFIG as MC
                model = PPO(
                    "MlpPolicy", env,
                    learning_rate=MC.get("algorithms", {}).get("ppo", {}).get("learning_rate", 0.00025),
                    n_steps=MC.get("algorithms", {}).get("ppo", {}).get("n_steps", 2048),
                    batch_size=MC.get("algorithms", {}).get("ppo", {}).get("batch_size", 64),
                    ent_coef=MC.get("algorithms", {}).get("ppo", {}).get("ent_coef", 0.01),
                    verbose=0,
                )
                model.learn(total_timesteps=timesteps)
                selected_name = "PPO"
                selected_model = model

            elif algo == "DDPG":
                from stable_baselines3 import DDPG
                env = strategy._make_env(train_data, is_eval=False)
                from core.config.settings import MODEL_CONFIG as MC
                model = DDPG(
                    "MlpPolicy", env,
                    learning_rate=MC.get("algorithms", {}).get("ddpg", {}).get("learning_rate", 0.001),
                    batch_size=MC.get("algorithms", {}).get("ddpg", {}).get("batch_size", 128),
                    buffer_size=MC.get("algorithms", {}).get("ddpg", {}).get("buffer_size", 50000),
                    verbose=0,
                )
                model.learn(total_timesteps=timesteps)
                selected_name = "DDPG"
                selected_model = model


            else:  # All (Ensemble)
                selected_name, selected_model = strategy.train_and_select()

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
            returns = df_account["daily_return"]
            quick_sharpe = float((252 ** 0.5) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0
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

    returns = df_account["daily_return"]
    cum_ret = float((df_account["account_value"].iloc[-1] / df_account["account_value"].iloc[0]) - 1.0) * 100
    quick_sharpe = float((252 ** 0.5) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0
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
