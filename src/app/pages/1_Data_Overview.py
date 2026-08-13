"""Page 1 — Data Overview: Tải và xem tổng quan dữ liệu thị trường."""
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
from components.state import (
    init_state, render_sidebar_status,
    set_state, get_state,
    KEY_TRAIN_DATA, KEY_VAL_DATA, KEY_BENCHMARK_DF,
    KEY_TRAIN_START, KEY_TRAIN_END, KEY_VAL_START, KEY_VAL_END,
    KEY_SELECTED_STOCKS,
)
from components.charts import plot_close_price_plotly
from components.config_panel import render_config_panel, render_config_summary_card

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dữ liệu — AI Quantum 2026",
    page_icon="📊",
    layout="wide",
)

init_state()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình Dữ liệu")

    # Load defaults from MODEL_CONFIG
    try:
        from core.config.settings import MODEL_CONFIG
        default_ts = MODEL_CONFIG.get("train_start_date", "2024-01-01")
        default_te = MODEL_CONFIG.get("train_end_date",   "2024-12-31")
        default_vs = MODEL_CONFIG.get("val_start_date",   "2025-01-01")
        default_ve = MODEL_CONFIG.get("val_end_date",     "2025-12-31")
    except Exception:
        default_ts, default_te = "2024-01-01", "2024-12-31"
        default_vs, default_ve = "2025-01-01", "2025-12-31"

    train_start = st.date_input("Train: Từ ngày", value=pd.to_datetime(default_ts), key="di_ts")
    train_end   = st.date_input("Train: Đến ngày", value=pd.to_datetime(default_te), key="di_te")
    st.divider()
    val_start   = st.date_input("Val: Từ ngày", value=pd.to_datetime(default_vs), key="di_vs")
    val_end     = st.date_input("Val: Đến ngày", value=pd.to_datetime(default_ve), key="di_ve")

    st.divider()
    st.markdown("### 📈 Cổ phiếu Danh mục")
    try:
        from core.config.settings import get_portfolio_stocks, settings
        yaml_stocks = get_portfolio_stocks()
        ticker_map = getattr(settings, "ticker_exchange_map", {})
        all_known = sorted(list(set(yaml_stocks + list(ticker_map.keys()))))
        stock_options = [s for s in all_known if s not in ["GOLD", "TCBF", "VBF", "CASH", "VN30"]]
    except Exception:
        yaml_stocks = ["VNM", "FPT", "VCB", "VHM", "VIC", "TCB", "MBB", "HPG", "MWG", "PNJ"]
        stock_options = yaml_stocks
        ticker_map = {}

    selected_stocks = st.multiselect(
        "Mã cổ phiếu đầu tư (Session)",
        options=stock_options,
        default=yaml_stocks,
        help="Danh sách mã cổ phiếu muốn đưa vào danh mục. Mặc định đọc từ config/market.yaml.",
        key="ms_portfolio_stocks",
    )
    set_state(KEY_SELECTED_STOCKS, selected_stocks)

    if not selected_stocks:
        st.info("💡 Chưa chọn mã cổ phiếu: Hệ thống sẽ tự động tải TOÀN BỘ cổ phiếu theo asset_class_ids trong model.yaml.")
    else:
        missing = [s for s in selected_stocks if s not in ticker_map]
        if missing:
            st.warning(f"⚠️ Các mã sau chưa có trong config/assets.yaml: {missing} (sẽ dùng HOSE ±7% mặc định)")

    st.divider()
    load_btn = st.button("📥 Tải Dữ liệu", type="primary", use_container_width=True)
    if st.button("🗑️ Xóa Cache", use_container_width=True):
        from components.state import clear_all_state
        clear_all_state()
        st.rerun()

    st.divider()
    render_config_panel(in_sidebar=True)
    render_sidebar_status()

# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("📊 Tổng quan Dữ liệu Thị trường")
st.caption("Tải dữ liệu OHLCV + chỉ báo kỹ thuật từ database, sau đó cấu hình ở Trang Huấn luyện.")
render_config_summary_card()

# ─── Load Data ───────────────────────────────────────────────────────────────
if load_btn:
    ts_str = str(train_start)
    te_str = str(train_end)
    vs_str = str(val_start)
    ve_str = str(val_end)
    sel_stocks = get_state(KEY_SELECTED_STOCKS, None)

    with st.spinner("⏳ Đang tải dữ liệu từ database..."):
        try:
            from model_engine.data.data_service import DataQueryService
            from model_engine.data.processor import DataProcessor
            from core.config.settings import MODEL_CONFIG

            svc = DataQueryService()
            lookback = int(MODEL_CONFIG.get("lookback_days", 45))
            features = MODEL_CONFIG.get(
                "features",
                ["RSI", "PPO", "CCI", "ADX", "ATR", "VOLATILITY", "YIELD_CURVE_SLOPE", "DXY_LOG_RETURN", "VN3YT"]
            )

            # 1. Truy vấn dữ liệu thô kèm lịch sử nung nóng
            raw_train_df = svc.fetch_data(ts_str, te_str, lookback_days=lookback, stock_tickers=sel_stocks)
            raw_val_df   = svc.fetch_data(vs_str, ve_str, lookback_days=lookback, stock_tickers=sel_stocks)

            bm_df        = svc.fetch_symbol_data("VN30", ts_str, ve_str)

            # 2. Tiền xử lý: Quảng bá biến vĩ mô, chuẩn hóa Rolling Z-Score và lọc bỏ warm-up history
            train_df = DataProcessor(raw_train_df, features).clean_data(start_date=ts_str)
            val_df   = DataProcessor(raw_val_df, features).clean_data(start_date=vs_str)

            set_state(KEY_TRAIN_DATA,   train_df)
            set_state(KEY_VAL_DATA,     val_df)
            set_state(KEY_BENCHMARK_DF, bm_df)
            set_state(KEY_TRAIN_START,  ts_str)
            set_state(KEY_TRAIN_END,    te_str)
            set_state(KEY_VAL_START,    vs_str)
            set_state(KEY_VAL_END,      ve_str)

            st.success(f"✅ Đã tải và tiền xử lý Z-Score: Train={len(train_df):,} dòng ({train_df['tic'].nunique() if not train_df.empty else 0} Tickers) | Val={len(val_df):,} dòng | VN30={len(bm_df):,} ngày")

        except Exception as e:
            st.error(f"❌ Lỗi kết nối hoặc xử lý dữ liệu:\n\n```\n{e}\n```")
            st.info("💡 Kiểm tra cấu hình tại `config/api.yaml` và đảm bảo MySQL/DuckDB đang hoạt động.")
            st.stop()

# ─── Display if Data is Loaded ───────────────────────────────────────────────
train_data = get_state(KEY_TRAIN_DATA)

if train_data is None or (hasattr(train_data, 'empty') and train_data.empty):
    st.info("👈 Cấu hình khoảng thời gian ở sidebar và nhấn **Tải Dữ liệu** để bắt đầu.")
    st.stop()

val_data    = get_state(KEY_VAL_DATA)
bm_data     = get_state(KEY_BENCHMARK_DF)

# ─── Summary Cards ───────────────────────────────────────────────────────────
tickers = sorted(train_data["tic"].unique().tolist()) if "tic" in train_data.columns else []
train_days = train_data["date"].nunique() if "date" in train_data.columns else 0
val_days   = val_data["date"].nunique()   if val_data is not None and "date" in val_data.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("📋 Số Tickers", len(tickers))
c2.metric("📅 Ngày Train", f"{train_days:,}")
c3.metric("📅 Ngày Validation", f"{val_days:,}")
c4.metric("📈 VN30 Benchmark", f"{len(bm_data):,} ngày" if bm_data is not None and not bm_data.empty else "N/A")

st.divider()

# ─── Price Chart ──────────────────────────────────────────────────────────────
st.subheader("📈 Biểu đồ Giá Đóng cửa")

tab1, tab2 = st.tabs(["📊 Train Set", "📊 Validation Set"])

with tab1:
    if tickers:
        sel_ticker_train = st.selectbox(
            "Chọn Ticker (Train)",
            tickers,
            key="sel_train_ticker",
        )
        fig_train = plot_close_price_plotly(train_data, sel_ticker_train)
        st.plotly_chart(fig_train, use_container_width=True)
    else:
        st.warning("Không tìm thấy cột 'tic' trong dữ liệu.")

with tab2:
    if val_data is not None and not val_data.empty and tickers:
        val_tickers = sorted(val_data["tic"].unique().tolist())
        sel_ticker_val = st.selectbox(
            "Chọn Ticker (Val)",
            val_tickers,
            key="sel_val_ticker",
        )
        fig_val = plot_close_price_plotly(val_data, sel_ticker_val)
        st.plotly_chart(fig_val, use_container_width=True)

st.divider()

# ─── Ticker List ─────────────────────────────────────────────────────────────
st.subheader("📋 Danh sách Tickers")
cols = st.columns(min(len(tickers), 6))
for i, t in enumerate(tickers):
    cols[i % len(cols)].markdown(f"`{t}`")

st.divider()

# ─── Data Preview ─────────────────────────────────────────────────────────────
st.subheader("🔍 Preview Dữ liệu Train (20 dòng đầu)")
st.dataframe(
    train_data.head(20),
    use_container_width=True,
    hide_index=True,
)

# Feature columns available
feature_cols = [c for c in train_data.columns if c not in ["tic", "date", "open", "high", "low", "close", "volume"]]
if feature_cols:
    with st.expander(f"📊 Chỉ báo kỹ thuật có sẵn ({len(feature_cols)} features)"):
        st.write(feature_cols)
