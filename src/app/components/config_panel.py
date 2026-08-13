"""Model Testing & Environment Configuration Panel Component.

Provides unified UI controls for customizing:
- All 5 turbulence risk management types (static, cooldown_period, ewma_smoothed, adaptive_percentile, dual_threshold)
- All 4 reward strategies (net_return, sortino, drawdown_penalty, excess_return) and penalty coefficients
- Backtest timeline and initial portfolio balance settings
- Standalone Main Page Summary Overview Card (`render_config_summary_card`)
- Synchronization with st.session_state['model_config']
"""

import streamlit as st
import yaml
import pathlib
from typing import Dict, Any, Optional

DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent.parent.parent.parent / "config" / "model.yaml"

def load_default_config() -> Dict[str, Any]:
    """Load default configuration dictionary from model.yaml."""
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)
            raw_engine_cfg = full_cfg.get("model_engine", {})
            # Sanitize string keys (strip trailing spaces)
            if "reward_settings" in raw_engine_cfg and "reward_type" in raw_engine_cfg["reward_settings"]:
                raw_engine_cfg["reward_settings"]["reward_type"] = str(raw_engine_cfg["reward_settings"]["reward_type"]).strip()
            if "turbulence_settings" in raw_engine_cfg and "turbulence_type" in raw_engine_cfg["turbulence_settings"]:
                raw_engine_cfg["turbulence_settings"]["turbulence_type"] = str(raw_engine_cfg["turbulence_settings"]["turbulence_type"]).strip()
            return raw_engine_cfg
    # Default fallback dictionary
    return {
        "test_start_date": "2024-01-01",
        "test_end_date": "2024-12-31",
        "initial_balance": 1_000_000_000,
        "episode_length": 60,
        "reward_settings": {
            "reward_type": "excess_return",
            "scale_factor": 10.0,
            "alpha_multiplier": 2.0,
            "mdd_threshold": 0.02,
            "mdd_penalty_coef": 5.0,
            "infeasibility_coef": 1.0,
        },
        "turbulence_settings": {
            "turbulence_type": "dual_threshold",
            "threshold": 100.0,
            "threshold_trigger": 80.0,
            "threshold_exit": 36.0,
            "use_ewma": True,
            "cooldown_steps": 10,
            "ewma_span": 10,
            "rolling_window": 252,
            "percentile": 90.0,
            "force_sell_on_turbulence": True,
        }
    }


def init_config_state() -> Dict[str, Any]:
    """Initialize st.session_state['model_config'] if not present."""
    if "model_config" not in st.session_state or st.session_state["model_config"] is None:
        st.session_state["model_config"] = load_default_config()
    return st.session_state["model_config"]


def render_config_panel(in_sidebar: bool = True) -> Dict[str, Any]:
    """Render interactive configuration panel for turbulence, reward, and test parameters.
    
    Args:
        in_sidebar: If True, renders panel in Streamlit sidebar, otherwise in main page.
        
    Returns:
        Updated model configuration dictionary.
    """
    cfg = init_config_state()
    container = st.sidebar if in_sidebar else st.container()

    with container:
        st.markdown("### ⚙️ Cấu Hình Tham Số Testing")
        
        # 1. Expander for Turbulence Strategy
        with st.expander("🛡️ Phanh Rủi Ro (Turbulence)", expanded=False):
            turb_cfg = cfg.get("turbulence_settings", {})
            turb_types = ["static", "cooldown_period", "ewma_smoothed", "adaptive_percentile", "dual_threshold"]
            current_turb_type = str(turb_cfg.get("turbulence_type", "dual_threshold")).strip()
            turb_type_idx = turb_types.index(current_turb_type) if current_turb_type in turb_types else 4

            selected_turb_type = st.selectbox(
                "Loại Turbulence Strategy",
                options=turb_types,
                index=turb_type_idx,
                help="Static: Ngưỡng cố định | Cooldown: Khóa giao dịch K phiên | EWMA: Làm mịn | Adaptive: Bách phân vị | Dual: Băng thông Hysteresis",
                key="cfg_turb_type",
            )
            turb_cfg["turbulence_type"] = selected_turb_type

            # Dynamic parameters based on turbulence type
            if selected_turb_type in ["static", "cooldown_period", "ewma_smoothed"]:
                turb_cfg["threshold"] = st.number_input(
                    "Ngưỡng phanh khẩn cấp (Threshold)",
                    min_value=1.0, max_value=500.0,
                    value=float(turb_cfg.get("threshold", 80.0)),
                    step=5.0,
                    key="cfg_turb_threshold",
                )

            if selected_turb_type == "cooldown_period":
                turb_cfg["cooldown_steps"] = st.number_input(
                    "Số phiên khóa giao dịch (Cooldown Steps)",
                    min_value=1, max_value=60,
                    value=int(turb_cfg.get("cooldown_steps", 10)),
                    key="cfg_turb_cooldown",
                )

            elif selected_turb_type == "ewma_smoothed":
                turb_cfg["ewma_span"] = st.number_input(
                    "Chu kỳ làm mịn EWMA (Days)",
                    min_value=2, max_value=50,
                    value=int(turb_cfg.get("ewma_span", 10)),
                    key="cfg_turb_ewma_span",
                )

            elif selected_turb_type == "adaptive_percentile":
                turb_cfg["rolling_window"] = st.number_input(
                    "Cửa sổ lịch sử cuộn (Rolling Window)",
                    min_value=30, max_value=504,
                    value=int(turb_cfg.get("rolling_window", 252)),
                    key="cfg_turb_window",
                )
                turb_cfg["percentile"] = st.slider(
                    "Ngưỡng bách phân vị (Percentile %)",
                    min_value=50.0, max_value=99.9,
                    value=float(turb_cfg.get("percentile", 90.0)),
                    step=0.5,
                    key="cfg_turb_percentile",
                )

            elif selected_turb_type == "dual_threshold":
                raw_trig = turb_cfg.get("threshold_trigger")
                raw_exit = turb_cfg.get("threshold_exit")
                default_trig = float(raw_trig) if raw_trig is not None else 16.0
                default_exit = float(raw_exit) if raw_exit is not None else 12.0

                turb_cfg["threshold_trigger"] = st.number_input(
                    "Ngưỡng phanh ON (Trigger Threshold)",
                    min_value=1.0, max_value=300.0,
                    value=default_trig,
                    step=1.0,
                    key="cfg_turb_trigger",
                )
                turb_cfg["threshold_exit"] = st.number_input(
                    "Ngưỡng dỡ phanh OFF (Exit Threshold)",
                    min_value=1.0, max_value=200.0,
                    value=default_exit,
                    step=1.0,
                    key="cfg_turb_exit",
                )
                turb_cfg["use_ewma"] = st.checkbox(
                    "Sử dụng tín hiệu làm mịn EWMA",
                    value=bool(turb_cfg.get("use_ewma", True)),
                    key="cfg_turb_use_ewma",
                )

            turb_cfg["force_sell_on_turbulence"] = st.checkbox(
                "Bắt buộc bán 100% Tiền mặt khi vi phạm phanh",
                value=bool(turb_cfg.get("force_sell_on_turbulence", True)),
                key="cfg_turb_force_sell",
            )
            cfg["turbulence_settings"] = turb_cfg

        # 2. Expander for Reward Function Configuration
        with st.expander("🎯 Hàm Thưởng (Reward Function)", expanded=False):
            reward_cfg = cfg.get("reward_settings", {})
            reward_types = ["net_return", "sortino", "drawdown_penalty", "excess_return"]
            current_reward_type = str(reward_cfg.get("reward_type", "excess_return")).strip()
            reward_type_idx = reward_types.index(current_reward_type) if current_reward_type in reward_types else 3

            selected_reward_type = st.selectbox(
                "Loại Hàm Thưởng (Reward Type)",
                options=reward_types,
                index=reward_type_idx,
                help="net_return: Lợi nhuận ròng | sortino: Sortino Ratio | drawdown_penalty: Phạt MDD | excess_return: Lợi nhuận vượt chênh",
                key="cfg_reward_type",
            )
            reward_cfg["reward_type"] = selected_reward_type

            reward_cfg["scale_factor"] = st.number_input(
                "Hệ số nhân tỉ lệ (Scale Factor)",
                min_value=0.1, max_value=100.0,
                value=float(reward_cfg.get("scale_factor", 10.0)),
                step=1.0,
                key="cfg_reward_scale",
            )

            if selected_reward_type == "excess_return":
                reward_cfg["alpha_multiplier"] = st.number_input(
                    "Hệ số Alpha Multiplier",
                    min_value=0.1, max_value=10.0,
                    value=float(reward_cfg.get("alpha_multiplier", 2.0)),
                    step=0.5,
                    key="cfg_reward_alpha",
                )

            if selected_reward_type in ["drawdown_penalty", "excess_return"]:
                reward_cfg["mdd_threshold"] = st.number_input(
                    "Ngưỡng sụt giảm tối đa cho phép (MDD Threshold)",
                    min_value=0.005, max_value=0.20,
                    value=float(reward_cfg.get("mdd_threshold", 0.01)),
                    step=0.005,
                    format="%.3f",
                    key="cfg_reward_mdd_thresh",
                )
                reward_cfg["mdd_penalty_coef"] = st.number_input(
                    "Hệ số phạt MDD (MDD Penalty Coef)",
                    min_value=0.0, max_value=100.0,
                    value=float(reward_cfg.get("mdd_penalty_coef", 10.0)),
                    step=0.5,
                    key="cfg_reward_mdd_coef",
                )

            reward_cfg["infeasibility_coef"] = st.number_input(
                "Hệ số phạt vi phạm T+2 / Lô (Infeasibility Coef)",
                min_value=0.0, max_value=10.0,
                value=float(reward_cfg.get("infeasibility_coef", 1.0)),
                step=0.1,
                key="cfg_reward_infeasibility",
            )
            cfg["reward_settings"] = reward_cfg

        # 3. Expander for Portfolio & Timeline Settings
        with st.expander("📅 Danh Mục & Thời Gian", expanded=False):
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                cfg["initial_balance"] = st.number_input(
                    "Vốn ban đầu (VND)",
                    min_value=100_000_000, max_value=100_000_000_000,
                    value=int(cfg.get("initial_balance", 1_000_000_000)),
                    step=100_000_000,
                    key="cfg_initial_balance",
                )
            with col_b2:
                cfg["episode_length"] = st.number_input(
                    "Độ dài Episode (Days)",
                    min_value=20, max_value=252,
                    value=int(cfg.get("episode_length", 60)),
                    step=10,
                    key="cfg_episode_length",
                )

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                cfg["test_start_date"] = st.text_input(
                    "Ngày bắt đầu Test",
                    value=str(cfg.get("test_start_date", "2024-01-01")),
                    key="cfg_test_start",
                )
            with col_d2:
                cfg["test_end_date"] = st.text_input(
                    "Ngày kết thúc Test",
                    value=str(cfg.get("test_end_date", "2024-12-31")),
                    key="cfg_test_end",
                )

        # Save to st.session_state
        st.session_state["model_config"] = cfg
        return cfg


def render_config_summary_card(cfg: Optional[Dict[str, Any]] = None, title: str = "📋 Tổng Quan Cấu Hình Model & Environment Hiện Tại") -> None:
    """Render a spacious, beautiful summary card on the MAIN PAGE showing all active model parameters."""
    if cfg is None:
        cfg = init_config_state()

    turb_cfg = cfg.get("turbulence_settings", {})
    reward_cfg = cfg.get("reward_settings", {})

    t_type = str(turb_cfg.get("turbulence_type", "N/A")).strip()
    r_type = str(reward_cfg.get("reward_type", "N/A")).strip()

    st.markdown(f"### {title}")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"#### 🛡️ Phanh Turbulence: `{t_type}`")
        if t_type == "dual_threshold":
            trig_disp = turb_cfg.get('threshold_trigger')
            exit_disp = turb_cfg.get('threshold_exit')
            st.markdown(f"- **Trigger ON:** `{trig_disp if trig_disp is not None else 'Auto (P90)'}`")
            st.markdown(f"- **Exit OFF:** `{exit_disp if exit_disp is not None else 'Auto (P70)'}`")
            st.markdown(f"- **Làm mịn EWMA:** `{turb_cfg.get('use_ewma')}`")

        elif t_type == "adaptive_percentile":
            st.markdown(f"- **Rolling Window:** `{turb_cfg.get('rolling_window')} ngày`")
            st.markdown(f"- **Percentile:** `{turb_cfg.get('percentile')}%`")
        elif t_type == "cooldown_period":
            st.markdown(f"- **Threshold:** `{turb_cfg.get('threshold')}`")
            st.markdown(f"- **Cooldown:** `{turb_cfg.get('cooldown_steps')} phiên`")
        elif t_type == "ewma_smoothed":
            st.markdown(f"- **Threshold:** `{turb_cfg.get('threshold')}`")
            st.markdown(f"- **EWMA Span:** `{turb_cfg.get('ewma_span')} ngày`")
        else:
            st.markdown(f"- **Threshold:** `{turb_cfg.get('threshold')}`")
        
        st.markdown(f"- **Ép bán 100% Cash:** `{'✅ Có' if turb_cfg.get('force_sell_on_turbulence') else '❌ Không'}`")

    with col2:
        st.markdown(f"#### 🎯 Hàm Thưởng Reward: `{r_type}`")
        st.markdown(f"- **Scale Factor:** `{reward_cfg.get('scale_factor')}`")
        if r_type in ["drawdown_penalty", "excess_return"]:
            mdd_th = float(reward_cfg.get('mdd_threshold', 0.01))
            st.markdown(f"- **MDD Threshold:** `{mdd_th:.3f}` ({mdd_th*100:.1f}%)")
            st.markdown(f"- **MDD Penalty Coef:** `{reward_cfg.get('mdd_penalty_coef')}`")
        if r_type == "excess_return":
            st.markdown(f"- **Alpha Multiplier:** `{reward_cfg.get('alpha_multiplier')}`")
        st.markdown(f"- **Infeasibility Coef (T+2):** `{reward_cfg.get('infeasibility_coef')}`")

    with col3:
        st.markdown(f"#### 💼 Danh Mục & Khung Thời Gian")
        init_bal = float(cfg.get('initial_balance', 1_000_000_000))
        st.markdown(f"- **Vốn ban đầu:** `{init_bal/1e9:.2f} Tỷ VND`")
        st.markdown(f"- **Độ dài Episode:** `{cfg.get('episode_length')} ngày`")
        st.markdown(f"- **Giai đoạn Test:** `{cfg.get('test_start_date')} → {cfg.get('test_end_date')}`")

    with st.expander("🔍 Xem Cấu Hình Chi Tiết (Raw JSON)", expanded=False):
        st.json(cfg)
