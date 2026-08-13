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

# Fix: Import MARKET_CONFIG và MODEL_CONFIG ở module scope để tránh NameError
# khi page được load sau khi model đã train từ session trước (has_analysis() = True)
from core.config.settings import MODEL_CONFIG, MARKET_CONFIG

from components.state import (
    init_state, render_sidebar_status,
    get_state, set_state,
    KEY_TRAINED_MODEL, KEY_MODEL_NAME,
    KEY_VAL_DATA, KEY_BENCHMARK_DF,
    KEY_DF_ACCOUNT, KEY_DF_ACTIONS, KEY_DF_SHARES,
    has_model, has_analysis,
)
from components.config_panel import render_config_panel, render_config_summary_card
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

    st.divider()
    model_cfg = render_config_panel(in_sidebar=True)
    render_sidebar_status()

# ─── Main ────────────────────────────────────────────────────────────────────
st.title("📈 Phân tích Kết quả Backtest")
model_name = get_state(KEY_MODEL_NAME, "DRL Agent")
st.caption(f"Model: **{model_name}** · Evaluation trên Validation Set")
render_config_summary_card()

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
            # MODEL_CONFIG, MARKET_CONFIG đã được import ở top-level scope

            feature_cols = [
                c for c in val_data.columns
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
                "reward_settings":      reward_cfg,
                "turbulence_settings":  turb_cfg,
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
st.divider()

# ─── Systemic Risk & Kritzman Turbulence Index ─────────────────────────────
st.subheader("🛡️ Chỉ số Rủi ro Hệ thống (Kritzman Turbulence Index)")

val_data_turb = get_state(KEY_VAL_DATA)
if val_data_turb is not None and not val_data_turb.empty:
    from components.charts import plot_turbulence_plotly
    model_turb_cfg = MODEL_CONFIG.get("turbulence_settings", {})
    market_risk_ctrls = MARKET_CONFIG.get("risk_controls", {})
    market_turb_cfg = market_risk_ctrls.get("turbulence_settings", {})
    
    turb_cfg = model_turb_cfg if model_turb_cfg else market_turb_cfg
    turb_type = turb_cfg.get("turbulence_type", "cooldown_period")
    turb_threshold = float(turb_cfg.get("threshold", 100.0))
    raw_trig = turb_cfg.get("threshold_trigger")
    raw_exit = turb_cfg.get("threshold_exit")
    threshold_trigger = float(raw_trig) if raw_trig is not None else turb_threshold
    threshold_exit = float(raw_exit) if raw_exit is not None else (threshold_trigger * 0.75)
    ewma_span = int(turb_cfg.get("ewma_span", 10))
    cooldown_steps = int(turb_cfg.get("cooldown_steps", 5))


    fig_turb = plot_turbulence_plotly(
        val_data_turb,
        threshold=turb_threshold,
        turbulence_type=turb_type,
        ewma_span=ewma_span,
        threshold_trigger=threshold_trigger,
        threshold_exit=threshold_exit,
    )
    
    if fig_turb is not None:
        st.plotly_chart(fig_turb, use_container_width=True)
        
        turb_col = next((c for c in ["TURBULENCE", "turbulence"] if c in val_data_turb.columns), None)
        if turb_col:
            max_turb = float(val_data_turb[turb_col].max())
            eval_thresh = threshold_trigger if turb_type in ("dual_threshold", "ewma_dual_threshold") else turb_threshold
            breach_count = int((val_data_turb.groupby("date")[turb_col].mean() > eval_thresh).sum())
            
            type_names = {
                "static": "Phanh 1 Ngày",
                "cooldown_period": f"Cooldown ({cooldown_steps} phiên)",
                "ewma_smoothed": f"EWMA làm mịn ({ewma_span} phiên)",
                "adaptive_percentile": "Adaptive Percentile (P90)",
                "dual_threshold": f"Dual Threshold ({threshold_trigger:.0f} / {threshold_exit:.0f})",
                "ewma_dual_threshold": f"EWMA Dual Threshold ({threshold_trigger:.0f} / {threshold_exit:.0f})",
            }
            strategy_name = type_names.get(turb_type, turb_type)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("📊 Turbulence Cao Nhất", f"{max_turb:.2f}")
            mc2.metric("🎯 Chiến lược Phanh Rủi ro", strategy_name, delta=f"Ngưỡng: {turb_threshold:.0f}")
            if breach_count > 0:
                mc3.metric("⚠️ Số phiên bùng nổ Sốc", f"{breach_count} phiên", delta="⚠️ Kích hoạt Phanh 100% Cash", delta_color="inverse")
            else:
                mc3.metric("✅ Trạng thái Hệ thống", "An toàn (0 phiên)", delta="🟢 Trong tầm kiểm soát")

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

st.divider()

# ─── SHAP Explainable AI (XAI) ─────────────────────────────────────────────
st.subheader("🧠 Giải thích Mô hình SHAP (Explainable AI - XAI)")
st.caption("Phân tích tầm quan trọng của các chỉ báo kỹ thuật & vĩ mô tới quyết định phân bổ danh mục của Agent.")

val_data = get_state(KEY_VAL_DATA)
model    = get_state(KEY_TRAINED_MODEL)

if model is not None and val_data is not None and not val_data.empty:
    with st.expander("🔍 Cấu hình & Chạy phân tích SHAP Feature Attribution", expanded=("shap_vals" not in st.session_state)):
        all_dates_list = list(val_data["date"].unique())
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            nsamples_shap = st.slider("Số lượng mẫu tính toán SHAP:", min_value=20, max_value=200, value=50, step=10,
                                      help="Số mẫu Monte-Carlo. Giá trị lớn hơn cho kết quả chính xác hơn nhưng tốn thời gian hơn.")
            run_shap_btn = st.button("🚀 Chạy Phân tích SHAP", use_container_width=True)

        with col_s2:
            sample_mode = st.radio(
                "📅 Phạm vi Ngày giao dịch (Date Range):",
                options=["Trải đều Cả năm (Tháng 1 -> Tháng 12)", "Khoảng ngày Cụ thể", "Toàn bộ Ngày trong Validation"],
                index=0,
                horizontal=True
            )
            shap_start_d = all_dates_list[0]
            shap_end_d = all_dates_list[-1]
            if sample_mode == "Khoảng ngày Cụ thể":
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    shap_start_d = st.selectbox("Từ ngày:", options=all_dates_list, index=0)
                with col_d2:
                    shap_end_d = st.selectbox("Đến ngày:", options=all_dates_list, index=len(all_dates_list)-1)

        if run_shap_btn:
            with st.spinner("⏳ Đang thu thập dữ liệu quan sát và tính toán SHAP cho toàn bộ giai đoạn đã chọn..."):
                try:
                    from model_engine.analysis.shap_explainer import SHAPExplainer
                    from model_engine.env.stock_trading_env import StockTradingEnv
                    # MODEL_CONFIG, MARKET_CONFIG đã được import ở top-level scope
                    
                    feature_cols = [c for c in val_data.columns if c not in ["tic", "date", "open", "high", "low", "close", "volume"]]
                    tickers = list(val_data["tic"].unique()) if "tic" in val_data.columns else ["Asset"]
                    target_names = ["CASH"] + [f"Action_{t}" for t in tickers]

                    # Chạy mô phỏng để thu thập toàn bộ quan sát của tập Validation
                    turb_threshold = MARKET_CONFIG.get("risk_controls", {}).get("default_turbulence_threshold", 100.0)
                    env_eval = StockTradingEnv(
                        df=val_data,
                        features=feature_cols,
                        initial_balance=MODEL_CONFIG.get("initial_balance", 1_000_000_000),
                        turbulence_threshold=turb_threshold,
                    )

                    all_obs_list = []
                    obs, _ = env_eval.reset()
                    all_obs_list.append(obs)
                    done = False
                    while not done:
                        act, _ = model.predict(obs, deterministic=True)
                        obs, _, done, _, _ = env_eval.step(act)
                        all_obs_list.append(obs)

                    all_obs_arr = np.array(all_obs_list, dtype=np.float32)
                    all_dates_arr = np.array(env_eval.dates[:len(all_obs_arr)])

                    # Chọn danh sách chỉ mục ngày giao dịch dựa theo Chế độ đã chọn
                    if sample_mode == "Khoảng ngày Cụ thể":
                        s_idx = np.where(all_dates_arr >= shap_start_d)[0]
                        e_idx = np.where(all_dates_arr <= shap_end_d)[0]
                        if len(s_idx) > 0 and len(e_idx) > 0:
                            valid_indices = list(range(s_idx[0], e_idx[-1] + 1))
                        else:
                            valid_indices = list(range(len(all_dates_arr)))
                        if len(valid_indices) > 50:
                            selected_indices = np.linspace(valid_indices[0], valid_indices[-1], 50, dtype=int)
                        else:
                            selected_indices = valid_indices
                    elif sample_mode == "Toàn bộ Ngày trong Validation":
                        if len(all_dates_arr) > 60:
                            selected_indices = np.linspace(0, len(all_dates_arr) - 1, 60, dtype=int)
                        else:
                            selected_indices = list(range(len(all_dates_arr)))
                    else: # Trải đều Cả năm (Tháng 1 -> Tháng 12)
                        n_points = min(40, len(all_dates_arr))
                        selected_indices = np.linspace(0, len(all_dates_arr) - 1, n_points, dtype=int)

                    X_eval = all_obs_arr[selected_indices]
                    eval_dates = all_dates_arr[selected_indices]

                    feature_names = ["CASH_WEIGHT"]
                    for t in tickers:
                        feature_names.append(f"{t}_W_T0")
                    for t in tickers:
                        feature_names.append(f"{t}_W_T1")
                    for t in tickers:
                        feature_names.append(f"{t}_W_T2")
                    for t in tickers:
                        for f in feature_cols:
                            feature_names.append(f"{t}_{f}")

                    explainer = SHAPExplainer(
                        model=model,
                        feature_names=feature_names,
                        target_names=target_names
                    )

                    shap_vals = explainer.compute_shap_values(X_eval, nsamples=nsamples_shap)
                    
                    # Store in session state for persistence across Streamlit reruns
                    st.session_state["shap_vals"] = shap_vals
                    st.session_state["shap_x_eval"] = X_eval
                    st.session_state["shap_eval_dates"] = eval_dates
                    st.session_state["shap_feature_names"] = feature_names
                    st.session_state["shap_target_names"] = target_names
                    st.rerun()

                except Exception as ex_shap:
                    st.error(f"❌ Không thể thực thi SHAP Analysis: {ex_shap}")
                    import traceback
                    st.code(traceback.format_exc())

    # Displays stored SHAP results persistently
    if "shap_vals" in st.session_state and st.session_state["shap_vals"] is not None:
        try:
            from model_engine.analysis.shap_explainer import SHAPExplainer
            import matplotlib.pyplot as plt

            shap_vals = st.session_state["shap_vals"]
            X_eval = st.session_state["shap_x_eval"]
            eval_dates = st.session_state["shap_eval_dates"]
            feature_names = st.session_state["shap_feature_names"]
            target_names = st.session_state["shap_target_names"]

            explainer = SHAPExplainer(
                model=model,
                feature_names=feature_names,
                target_names=target_names
            )

            st.success("✅ Kết quả SHAP Analysis đã sẵn sàng!")

            # Allow user to select which action target output to analyze (CASH, Stocks, etc.)
            col_t1, col_t2 = st.columns([1, 1])
            with col_t1:
                selected_target_idx = st.selectbox(
                    "🎯 Chọn Đầu ra Tài sản để Giải thích:",
                    options=list(range(len(target_names))),
                    format_func=lambda i: f"Tỷ trọng: {target_names[i]}",
                    key="shap_target_selectbox"
                )

            # 1. Display Summary Bar Chart
            st.markdown(f"#### 📊 Tầm quan trọng của Đặc trưng đến Đầu ra: **[{target_names[selected_target_idx]}]**")
            fig_shap_bar = explainer.plot_summary_bar(shap_vals, X_eval, action_idx=selected_target_idx, max_display=10)
            st.pyplot(fig_shap_bar)
            plt.close(fig_shap_bar)

            # 2. Display Dataframe
            st.markdown(f"#### 📋 Bảng Xếp hạng Mức độ Đóng góp đến **[{target_names[selected_target_idx]}]**")
            df_shap_imp = explainer.get_feature_importance_df(shap_vals, action_idx=selected_target_idx)
            st.dataframe(df_shap_imp, use_container_width=True, hide_index=True)

            # 3. Waterfall for sample date
            st.markdown(f"#### 🌊 Giải thích Lệnh Giao dịch theo Ngày (Target: **[{target_names[selected_target_idx]}]**)")
            date_idx = st.selectbox("Chọn ngày giao dịch trong tập Validation:", range(len(eval_dates)),
                                   format_func=lambda i: str(eval_dates[i]), key="shap_date_selectbox")
            fig_waterfall = explainer.plot_trade_waterfall(
                shap_vals, X_eval, sample_idx=date_idx, action_idx=selected_target_idx, date_str=str(eval_dates[date_idx])
            )
            st.pyplot(fig_waterfall)
            plt.close(fig_waterfall)

        except Exception as render_ex:
            st.error(f"Lỗi hiển thị kết quả SHAP: {render_ex}")

else:
    st.info("Cần huấn luyện mô hình và có dữ liệu Validation để chạy phân tích SHAP.")

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

