"""Session State Management for AI Quantum 2026 Demo App.

Provides a centralized, typed schema for all shared state between pages.
All keys are defined as constants to prevent typos across pages.
"""
import streamlit as st
import pandas as pd
from typing import Optional, Any

# ─────────────────────────── Key Constants ────────────────────────────
KEY_TRAIN_DATA    = "train_data"
KEY_VAL_DATA      = "val_data"
KEY_BENCHMARK_DF  = "benchmark_df"
KEY_TRAINED_MODEL = "trained_model"
KEY_MODEL_NAME    = "model_name"
KEY_DF_ACCOUNT    = "df_account"
KEY_DF_ACTIONS    = "df_actions"
KEY_DF_SHARES     = "df_shares"
KEY_TRAIN_START   = "cfg_train_start"
KEY_TRAIN_END     = "cfg_train_end"
KEY_VAL_START     = "cfg_val_start"
KEY_VAL_END       = "cfg_val_end"

_ALL_KEYS = [
    KEY_TRAIN_DATA, KEY_VAL_DATA, KEY_BENCHMARK_DF,
    KEY_TRAINED_MODEL, KEY_MODEL_NAME,
    KEY_DF_ACCOUNT, KEY_DF_ACTIONS, KEY_DF_SHARES,
    KEY_TRAIN_START, KEY_TRAIN_END, KEY_VAL_START, KEY_VAL_END,
]


def init_state() -> None:
    """Initialize all session state keys with None defaults (idempotent)."""
    for key in _ALL_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None


def get_state(key: str, default: Any = None) -> Any:
    """Get a value from session state safely."""
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """Set a value in session state."""
    st.session_state[key] = value


def clear_model_state() -> None:
    """Clear model + analysis results, keeping data loaded."""
    for key in [KEY_TRAINED_MODEL, KEY_MODEL_NAME, KEY_DF_ACCOUNT, KEY_DF_ACTIONS, KEY_DF_SHARES]:
        st.session_state[key] = None


def clear_all_state() -> None:
    """Full reset of all session state."""
    for key in _ALL_KEYS:
        st.session_state[key] = None


def has_data() -> bool:
    """True if training data has been loaded."""
    return (
        KEY_TRAIN_DATA in st.session_state
        and st.session_state[KEY_TRAIN_DATA] is not None
        and not st.session_state[KEY_TRAIN_DATA].empty
    )


def has_model() -> bool:
    """True if a model has been trained and stored."""
    return (
        KEY_TRAINED_MODEL in st.session_state
        and st.session_state[KEY_TRAINED_MODEL] is not None
    )


def has_analysis() -> bool:
    """True if evaluation trajectory has been computed."""
    return (
        KEY_DF_ACCOUNT in st.session_state
        and st.session_state[KEY_DF_ACCOUNT] is not None
        and not st.session_state[KEY_DF_ACCOUNT].empty
    )


def render_sidebar_status() -> None:
    """Render a status indicator in the sidebar showing current pipeline state."""
    st.sidebar.divider()
    st.sidebar.markdown("#### 📊 Trạng thái Pipeline")

    data_ok  = has_data()
    model_ok = has_model()
    result_ok = has_analysis()

    st.sidebar.markdown(
        f"{'✅' if data_ok  else '⬜'} **Dữ liệu** đã tải\n\n"
        f"{'✅' if model_ok else '⬜'} **Model** đã huấn luyện\n\n"
        f"{'✅' if result_ok else '⬜'} **Phân tích** đã tính"
    )

    if model_ok:
        st.sidebar.caption(f"Model: **{get_state(KEY_MODEL_NAME, 'N/A')}**")
