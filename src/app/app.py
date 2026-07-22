"""AI Quantum 2026 — Streamlit Demo App Entry Point.

Run from project root:
    streamlit run src/app/app.py
"""
import sys
import pathlib

# ── sys.path injection ──────────────────────────────────────────────────────
_APP_DIR = pathlib.Path(__file__).parent
_PIPELINE_DIR = _APP_DIR.parent / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st
from components.state import init_state, render_sidebar_status


# ─────────────────────────── Page Config ────────────────────────────────────
st.set_page_config(
    page_title="AI Quantum 2026",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Global CSS ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4C72B0, #55A868, #C44E52);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1.1rem;
    color: #666;
    margin-bottom: 2rem;
}
.engine-card {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    border-left: 4px solid #4C72B0;
    margin-bottom: 0.8rem;
}
.step-card {
    background: white;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #dee2e6;
    text-align: center;
    height: 100%;
}
.step-num {
    font-size: 2rem;
    font-weight: 700;
    color: #4C72B0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Init State ─────────────────────────────────────
init_state()

# ─────────────────────────── Sidebar ────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
    st.markdown("### AI Quantum 2026")
    st.caption("Deep Reinforcement Learning\ncho Thị trường Chứng khoán Việt Nam")
    render_sidebar_status()

# ─────────────────────────── Hero Section ───────────────────────────────────
st.markdown('<div class="hero-title">🤖 AI Quantum 2026</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Hệ thống giao dịch tự động sử dụng Deep Reinforcement Learning '
    'cho Thị trường Chứng khoán Việt Nam (HOSE/HNX) — T+2 Settlement · Lot 100 · Realistic Fees</div>',
    unsafe_allow_html=True
)

st.divider()

# ─────────────────────────── Architecture Overview ──────────────────────────
st.subheader("🏗️ Kiến trúc Hệ thống 3 Engine")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("""
<div class="engine-card">
<b>⚙️ Data Engine</b><br/>
<small style="color:#555">Thu thập dữ liệu tự động từ API thị trường, lưu vào MySQL database qua schedule trigger N8N.
Bao gồm: OHLCV, chỉ số vĩ mô (DXY, VNIBOR), dữ liệu VN30.</small>
</div>
""", unsafe_allow_html=True)

with col_b:
    st.markdown("""
<div class="engine-card" style="border-left-color: #55A868">
<b>🔬 Feature Engine</b><br/>
<small style="color:#555">Tự động tính toán chỉ báo kỹ thuật (RSI, PPO, CCI, ADX, ATR, VOLATILITY)
và chỉ số vĩ mô (YIELD_CURVE_SLOPE, DXY_LOG_RETURN) — lưu vào database.</small>
</div>
""", unsafe_allow_html=True)

with col_c:
    st.markdown("""
<div class="engine-card" style="border-left-color: #C44E52">
<b>🧠 Model Engine</b><br/>
<small style="color:#555">Huấn luyện và đánh giá agent DRL (A2C/PPO/DDPG) trong môi trường
mô phỏng thị trường VN với ràng buộc T+2, lô 100 cp, phí 0.15%/0.25%.</small>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────── 3-Step Guide ───────────────────────────────────
st.subheader("🚀 Hướng dẫn sử dụng Demo")

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown("""
<div class="step-card">
<div class="step-num">1️⃣</div>
<h4>Dữ liệu</h4>
<p style="color:#666;font-size:0.9rem">Cấu hình khoảng thời gian Train/Validation,
tải dữ liệu OHLCV và chỉ báo kỹ thuật từ database.
Xem biểu đồ giá và thống kê tổng quan.</p>
</div>
""", unsafe_allow_html=True)

with s2:
    st.markdown("""
<div class="step-card">
<div class="step-num">2️⃣</div>
<h4>Huấn luyện</h4>
<p style="color:#666;font-size:0.9rem">Chọn thuật toán DRL (A2C, PPO, DDPG hoặc Ensemble),
cấu hình timesteps, bắt đầu huấn luyện agent và xem kết quả sơ bộ.</p>
</div>
""", unsafe_allow_html=True)

with s3:
    st.markdown("""
<div class="step-card">
<div class="step-num">3️⃣</div>
<h4>Phân tích</h4>
<p style="color:#666;font-size:0.9rem">Xem kết quả backtest đầy đủ: NAV Curve vs VN30,
Drawdown, Action Distribution, và bảng metrics tài chính (Sharpe, Sortino, Calmar...).</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────── Quick Links ────────────────────────────────────
st.subheader("📋 Thông tin Cấu hình Hiện tại")

try:
    from core.config.settings import MODEL_CONFIG, MARKET_CONFIG
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        st.markdown("**Model Config**")
        st.json({
            "train_period": f"{MODEL_CONFIG.get('train_start_date')} → {MODEL_CONFIG.get('train_end_date')}",
            "val_period":   f"{MODEL_CONFIG.get('val_start_date')} → {MODEL_CONFIG.get('val_end_date')}",
            "initial_balance": f"{MODEL_CONFIG.get('initial_balance', 0):,} VND",
            "features": MODEL_CONFIG.get("features", []),
        })
    with cfg_col2:
        st.markdown("**Market Config**")
        mc = MARKET_CONFIG.get("transaction_costs", {})
        st.json({
            "buy_fee":   f"{mc.get('brokerage_fee_buy', 0)*100:.2f}%",
            "sell_fee":  f"{(mc.get('brokerage_fee_sell', 0) + mc.get('personal_income_tax_sell', 0))*100:.2f}%",
            "lot_size":  MARKET_CONFIG.get("trading_rules", {}).get("lot_size", 100),
            "settlement": "T+2",
        })
except Exception as e:
    st.warning(f"⚠️ Không thể đọc config: {e}")

st.markdown("---")
st.caption("AI Quantum 2026 · NEU · Deep Reinforcement Learning for Vietnam Stock Market")
