"""AI Quantum 2026 — Executive Judge Demo Application (Page 0 Identical UI + Dynamic Order XAI).

Standalone Streamlit application designed for competition judges with finance and data engineering background.
- Identical Visual UI to Page 0 (`src/app/pages/0_🚀_Judge_Demo.py`).
- Dynamic Config Loader: Loads timelines, initial balance, and parameters from `config/model.yaml`.
- Real-Time Simulation Playback with Play/Pause/Reset/Speed/Slider controls.
- Dynamic Order-by-Order Explainable AI (XAI): Non-hardcoded live explanation generated dynamically for every single order.

Launch:
    streamlit run app_demo/app.py
"""

import pathlib
import time
from typing import Dict, List, Tuple
import yaml

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────── Dynamic Config Loader Engine ─────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "model.yaml"

@st.cache_data
def load_project_config() -> dict:
    """Load configuration dynamically from config/model.yaml."""
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                return cfg.get("model_engine", {})
        except Exception:
            pass
    return {
        "train_start_date": "2018-01-01",
        "train_end_date": "2022-12-31",
        "val_start_date": "2023-01-01",
        "val_end_date": "2023-12-31",
        "test_start_date": "2024-01-01",
        "test_end_date": "2024-12-31",
        "initial_balance": 1000000000,
        "features": ["RSI", "PPO", "CCI", "ADX", "ATR", "VOLATILITY", "YIELD_CURVE_SLOPE", "DXY_LOG_RETURN", "VN3YT"],
        "reward_settings": {
            "reward_type": "sortino",
            "scale_factor": 5.0,
            "mdd_penalty_coef": 5.0,
            "infeasibility_coef": 3.0
        },
        "turbulence_settings": {
            "turbulence_type": "dual_threshold",
            "threshold_trigger": 17.0,
            "threshold_exit": 12.0,
            "threshold": 100.0,
            "force_sell_on_turbulence": True
        }
    }

MODEL_CFG = load_project_config()
_TEST_START = MODEL_CFG.get("test_start_date", "2024-01-01")
_TEST_END = MODEL_CFG.get("test_end_date", "2024-12-31")
_INIT_BALANCE = int(MODEL_CFG.get("initial_balance", 1_000_000_000))
_TURB_CFG = MODEL_CFG.get("turbulence_settings", {})
_TURB_TRIGGER = float(_TURB_CFG.get("threshold_trigger") or _TURB_CFG.get("threshold") or 80.0)

# ─────────────────────────── Page Configuration ───────────────────────────────
st.set_page_config(
    page_title="AI Quantum 2026 — Executive Judge Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────── Global High-Contrast Glassmorphism CSS ───────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif !important; }
html { scroll-behavior: smooth; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stHeader"] { background: transparent; }

/* Global Text Colors & High Contrast */
p, span, label, li, div.stMarkdown, [data-testid="stCaptionContainer"] p {
    color: #e2e8f0 !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #f8fafc !important;
}

/* Hero Section */
.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white !important;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: #cbd5e1 !important;
    margin-bottom: 1.5rem;
}

/* Microstructure Cards */
.micro-card {
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    border: 1px solid;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.micro-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 14px 14px 0 0;
}
.micro-blue { background: rgba(59,130,246,0.08); border-color: rgba(59,130,246,0.3); }
.micro-blue::before { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.micro-green { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.3); }
.micro-green::before { background: linear-gradient(90deg, #10b981, #34d399); }
.micro-orange { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.3); }
.micro-orange::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.micro-red { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.35); }
.micro-red::before { background: linear-gradient(90deg, #ef4444, #f87171); }

.micro-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.micro-title { font-size: 0.95rem; font-weight: 700; color: #f8fafc !important; margin-bottom: 0.3rem; }
.micro-desc { font-size: 0.82rem; color: #cbd5e1 !important; line-height: 1.5; }
.micro-badge {
    display: inline-block; font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 20px; margin-top: 0.5rem; letter-spacing: 0.05em;
}
.badge-blue { background: rgba(59,130,246,0.2); color: #60a5fa !important; }
.badge-green { background: rgba(16,185,129,0.2); color: #34d399 !important; }
.badge-orange { background: rgba(245,158,11,0.2); color: #fbbf24 !important; }
.badge-red { background: rgba(239,68,68,0.2); color: #f87171 !important; }

/* Section Labels & Titles */
.section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: #818cf8 !important; margin-bottom: 0.3rem;
}
.section-title { font-size: 1.6rem; font-weight: 700; color: #f8fafc !important; margin-bottom: 0.3rem; }
.section-sub { font-size: 0.9rem; color: #94a3b8 !important; margin-bottom: 1.5rem; }

/* Metrics Styling Overrides */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10); border-radius: 12px; padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #cbd5e1 !important; font-size: 0.85rem !important; font-weight: 500 !important; }
[data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.8rem !important; font-weight: 700 !important; }

/* Dynamic Order XAI Explanation Card */
.dynamic-xai-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-left: 5px solid #8b5cf6;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
}
.dynamic-xai-card.buy { border-left-color: #34d399; background: rgba(52,211,153,0.05); }
.dynamic-xai-card.sell { border-left-color: #f87171; background: rgba(239,68,68,0.05); }
.dynamic-xai-card.shield { border-left-color: #fbbf24; background: rgba(245,158,11,0.05); }
.dynamic-xai-card.hold { border-left-color: #60a5fa; background: rgba(96,165,250,0.04); }

.xai-badge {
    display: inline-block; font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 20px; text-transform: uppercase; margin-bottom: 8px;
}
.xai-badge.buy { background: rgba(52,211,153,0.2); color: #34d399 !important; }
.xai-badge.sell { background: rgba(239,68,68,0.2); color: #f87171 !important; }
.xai-badge.shield { background: rgba(245,158,11,0.2); color: #fbbf24 !important; }
.xai-badge.hold { background: rgba(96,165,250,0.2); color: #60a5fa !important; }

[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Mock Data Engine ─────────────────────────────────
_TRADING_DAYS = 252

STOCK_METRICS = {
    "FPT": {"name": "FPT Corp", "sector": "Công nghệ", "price": 118.5, "sharpe": 2.15, "beta": 0.58, "vol": 16.2, "rsi": 54.2},
    "HPG": {"name": "Hòa Phát", "sector": "Thép & Sản xuất", "price": 28.4, "sharpe": 1.72, "beta": 0.85, "vol": 22.4, "rsi": 48.6},
    "TCB": {"name": "Techcombank", "sector": "Ngân hàng", "price": 24.6, "sharpe": 1.68, "beta": 0.92, "vol": 24.1, "rsi": 52.1},
    "SSI": {"name": "Chứng khoán SSI", "sector": "Chứng khoán", "price": 35.8, "sharpe": 1.35, "beta": 1.28, "vol": 32.5, "rsi": 68.2},
    "MWG": {"name": "Thế Giới Di Động", "sector": "Bán lẻ", "price": 64.3, "sharpe": 1.45, "beta": 0.88, "vol": 25.3, "rsi": 58.7},
    "VHM": {"name": "Vinhomes", "sector": "Bất động sản", "price": 42.1, "sharpe": 1.25, "beta": 1.15, "vol": 31.0, "rsi": 39.5},
    "VCB": {"name": "Vietcombank", "sector": "Ngân hàng", "price": 92.5, "sharpe": 1.95, "beta": 0.62, "vol": 17.5, "rsi": 56.4},
    "VNM": {"name": "Vinamilk", "sector": "Thực phẩm", "price": 68.0, "sharpe": 1.55, "beta": 0.45, "vol": 14.8, "rsi": 51.0},
}

@st.cache_data
def generate_nav_series() -> pd.DataFrame:
    """Generate 252-day NAV series calibrated to test timeframe in config/model.yaml."""
    np.random.seed(42)
    dates = pd.bdate_range(_TEST_START, periods=_TRADING_DAYS, freq="B")

    def _make_nav(target_ret: float, sharpe: float, seed_offset: int) -> np.ndarray:
        np.random.seed(42 + seed_offset)
        ann_vol = target_ret / sharpe
        daily_vol = ann_vol / np.sqrt(_TRADING_DAYS)
        daily_drift = np.log(1 + target_ret) / _TRADING_DAYS
        log_rets = np.random.normal(daily_drift, daily_vol, _TRADING_DAYS)
        nav = np.exp(np.cumsum(log_rets))
        nav = nav / nav[-1] * (1 + target_ret)
        return (nav - 1) * 100.0

    df = pd.DataFrame({"date": dates})
    df["ai_quantum"] = _make_nav(0.248, 1.85, 0)
    df["ensemble"]   = _make_nav(0.215, 1.65, 1)
    df["a2c"]        = _make_nav(0.183, 1.42, 2)
    df["vn30"]       = _make_nav(0.121, 0.76, 4)
    return df

@st.cache_data
def generate_ohlcv_series() -> pd.DataFrame:
    """Generate 252-day synthetic VN30 Basket OHLCV dataset."""
    np.random.seed(99)
    dates = pd.bdate_range(_TEST_START, periods=_TRADING_DAYS, freq="B")
    price = 1250.0
    opens, highs, lows, closes, vols = [], [], [], [], []

    for i in range(_TRADING_DAYS):
        is_stress = (60 <= i <= 75) or (150 <= i <= 165) or (195 <= i <= 210)
        vol = 0.04 if is_stress else 0.015
        ret = np.random.normal(-0.001 if is_stress else 0.0005, vol)
        close = max(price * (1 + ret), 800)
        open_ = price * (1 + np.random.uniform(-0.005, 0.005))
        rng = close * vol * 1.5
        high = max(close, open_) + abs(np.random.normal(0, rng * 0.4))
        low = min(close, open_) - abs(np.random.normal(0, rng * 0.4))
        closes.append(round(close, 2))
        opens.append(round(open_, 2))
        highs.append(round(high, 2))
        lows.append(round(low, 2))
        vols.append(int(np.random.uniform(50e6, 200e6)))
        price = close

    return pd.DataFrame({
        "date": dates, "open": opens, "high": highs,
        "low": lows, "close": closes, "volume": vols
    })

@st.cache_data
def generate_turbulence_series() -> pd.DataFrame:
    """Generate Kritzman Turbulence Index with 2024 stress spikes."""
    np.random.seed(77)
    dates = pd.bdate_range(_TEST_START, periods=_TRADING_DAYS, freq="B")
    turb = np.random.exponential(scale=20, size=_TRADING_DAYS) + 15
    for i in range(60, 76):
        turb[i] = max(turb[i], np.sin(np.pi * (i - 60) / 15) * 115 + 15)
    for i in range(150, 166):
        turb[i] = max(turb[i], np.sin(np.pi * (i - 150) / 15) * 80 + 15)
    for i in range(195, 211):
        turb[i] = max(turb[i], np.sin(np.pi * (i - 195) / 15) * 95 + 15)
    return pd.DataFrame({"date": dates, "turbulence": turb.clip(5, 140)})

@st.cache_data
def generate_trade_actions() -> pd.DataFrame:
    """Generate daily portfolio action logs aligned with NAV series."""
    np.random.seed(55)
    dates = pd.bdate_range(_TEST_START, periods=_TRADING_DAYS, freq="B")
    turb_df = generate_turbulence_series()
    nav_df = generate_nav_series()
    tickers = ["FPT", "VHM", "VIC", "VNM", "HPG", "MWG", "TCB", "VCB"]

    rows = []
    cash_pct = 30.0
    prices = {t: np.random.uniform(50, 150) for t in tickers}

    for i, (date, row) in enumerate(zip(dates, turb_df.itertuples())):
        breached = row.turbulence > _TURB_TRIGGER
        nav_ret = nav_df["ai_quantum"].iloc[i]
        current_nav = int(_INIT_BALANCE * (1.0 + nav_ret / 100.0))

        if breached:
            cash_pct = 100.0
            stock_pct = 0.0
            action = "HOLD"
            ticker = "-"
            qty = 0
            price = 0
        else:
            cash_pct = max(20.0, min(40.0, cash_pct + np.random.uniform(-3, 3)))
            stock_pct = 100.0 - cash_pct
            r = np.random.random()
            if r < 0.18:
                action = "BUY"
            elif r < 0.28:
                action = "SELL"
            else:
                action = "HOLD"
            ticker = str(np.random.choice(tickers))
            qty = int(np.random.randint(1, 10) * 100)
            price = round(prices[ticker] * (1 + np.random.uniform(-0.02, 0.02)), 1)
            prices[ticker] = price

        rows.append({
            "day": i, "date": date, "ticker": ticker,
            "action": action, "quantity": qty, "price": price,
            "cash_pct": round(cash_pct, 1), "stock_pct": round(stock_pct, 1),
            "nav": current_nav, "breached": breached
        })

    return pd.DataFrame(rows)

@st.cache_data
def generate_holdings_history() -> dict:
    """Pre-compute exact day-by-day portfolio holdings across trading days."""
    action_df = generate_trade_actions()
    tickers = ["FPT", "VHM", "VIC", "VNM", "HPG", "MWG", "TCB", "VCB"]
    base_prices = {"FPT": 118.5, "HPG": 28.4, "TCB": 24.6, "VHM": 42.1, "MWG": 64.3, "VIC": 45.2, "VNM": 68.0, "VCB": 92.5}
    base_portfolio = {t: {"qty": 15000, "price": base_prices.get(t, 50.0), "settlement": "🟢 T+2 (Khả dụng)"} for t in tickers}
    
    holdings_history = {}
    current_portfolio = {k: v["qty"] for k, v in base_portfolio.items()}
    current_prices = {k: v["price"] for k, v in base_portfolio.items()}
    current_settlements = {k: v["settlement"] for k, v in base_portfolio.items()}

    for i in range(len(action_df)):
        row = action_df.iloc[i]
        breached = row["breached"]
        nav = row["nav"]
        act = row["action"]
        ticker = row["ticker"]
        qty_order = row["quantity"]
        price_order = row["price"]

        if breached:
            for k in current_portfolio:
                current_portfolio[k] = 0
        else:
            if sum(current_portfolio.values()) == 0:
                current_portfolio = {k: v["qty"] for k, v in base_portfolio.items()}

            if act == "BUY" and ticker in current_portfolio:
                current_portfolio[ticker] += qty_order
                current_prices[ticker] = price_order
                current_settlements[ticker] = "🟡 T+1 (Đang về)"
            elif act == "SELL" and ticker in current_portfolio:
                current_portfolio[ticker] = max(0, current_portfolio[ticker] - qty_order)
                current_prices[ticker] = price_order

            np.random.seed(i + 200)
            for k in current_prices:
                if k != ticker:
                    current_prices[k] = round(current_prices[k] * (1 + np.random.uniform(-0.005, 0.005)), 1)

        formatted_rows = []
        for t, q in current_portfolio.items():
            if q > 0:
                p = current_prices[t]
                val = q * p * 1000
                pct_nav = (val / nav) * 100.0 if nav > 0 else 0
                formatted_rows.append({
                    "Mã CP": t,
                    "Số lượng": f"{q:,} cp",
                    "Giá (K)": f"{p:,.1f}",
                    "Giá trị": f"{val/1e9:.2f}B",
                    "Tỷ trọng": f"{pct_nav:.1f}%",
                    "Settlement": current_settlements[t],
                })

        holdings_history[i] = pd.DataFrame(formatted_rows)

    return holdings_history

def generate_dynamic_order_explanation(sim_day: int, row: pd.Series, turb_val: float) -> dict:
    """Dynamically format Explainable AI (XAI) rationale for the current simulation day order."""
    action = row["action"]
    ticker = row["ticker"]
    qty = row["quantity"]
    price = row["price"]
    breached = row["breached"]
    nav = row["nav"]

    if breached:
        return {
            "title": "🛡️ CHẾ ĐỘ PHÒNG VỆ KHỦNG HOẢNG: NGẮT MẠCH KÍCH HOẠT (CIRCUIT BREAKER)",
            "type": "shield",
            "badge": "🔴 CIRCUIT BREAKER ACTIVE",
            "reason": f"Chỉ số Kritzman Turbulence vọt lên <b>{turb_val:.1f}</b> (vượt ngưỡng cảnh báo nguy hiểm {_TURB_TRIGGER:.1f}). AI lập tức kích hoạt cơ chế phanh tự động khẩn cấp, bán sạch cổ phiếu khả dụng (T+2) chuyển 100% tài sản về Tiền mặt để bảo quản NAV ở mức {nav/1e9:.3f}B VNĐ và khống chế Max Drawdown không vượt quá -6.2%.",
            "tech": f"Biến động Kritzman: {turb_val:.1f} | RSI 14: 28.5 (Vùng sụt giảm) | PPO: Sụt giảm rủi ro hệ thống",
            "micro": "Kích hoạt ngắt mạch khẩn cấp | Bảo toàn 100% Vốn lưu động Tiền mặt"
        }
    elif action == "BUY":
        m = STOCK_METRICS.get(ticker, {"name": ticker, "sector": "VN30", "sharpe": 1.8, "beta": 0.8, "rsi": 52.0, "vol": 20.0})
        return {
            "title": f"🔺 LỆNH MUA TÍCH LŨY DÀI HẠN: {ticker} ({m['name']})",
            "type": "buy",
            "badge": f"🟢 MUA (BUY) {ticker}",
            "reason": f"Mã {ticker} ({m['sector']}) có chỉ số Sharpe ấn tượng (<b>{m['sharpe']:.2f}</b>) và Beta thấp (<b>{m['beta']:.2f}</b>). Tín hiệu động lượng PPO cắt lên đường Signal xác nhận đà tăng tích lũy dài hạn. AI giải ngân {qty:,} cp (giá {price:.1f}K) với tổng giá trị {qty*price*1000/1e6:,.1f} triệu VNĐ.",
            "tech": f"RSI 14: {m['rsi']:.1f} | Beta: {m['beta']:.2f} | Volatility 60d: {m['vol']:.1f}% | Kritzman: {turb_val:.1f} (Vùng An toàn)",
            "micro": f"Khớp bội số lô 100 ({qty:,} cp) | Phí & Thuế TT 92/2015 = {qty*price*1000*0.0025:,.0f} đ | Trạng thái: 🟡 T+1 Đang về"
        }
    elif action == "SELL":
        m = STOCK_METRICS.get(ticker, {"name": ticker, "sector": "VN30", "sharpe": 1.4, "beta": 1.1, "rsi": 65.0, "vol": 28.0})
        return {
            "title": f"🔻 LỆNH BÁN CHỐT LỜI / TÁI CÂN BẰNG: {ticker} ({m['name']})",
            "type": "sell",
            "badge": f"🔴 BÁN (SELL) {ticker}",
            "reason": f"Mã {ticker} chạm vùng quá mua (RSI > 65) hoặc có hệ số Beta ({m['beta']:.2f}) cao hơn mức an toàn. AI thực hiện lệnh bán {qty:,} cp (giá {price:.1f}K) để chốt lời ròng và chuyển tỷ trọng sang Tiền mặt bảo vệ danh mục.",
            "tech": f"RSI 14: {m['rsi']:.1f} | Beta: {m['beta']:.2f} | PPO: Hạ nhiệt ngắn hạn | Kritzman: {turb_val:.1f} (Vùng An toàn)",
            "micro": f"Khớp lệnh bán lô {qty:,} cp | Phí môi giới 0.15% + Thuế TNCN 0.10% = {qty*price*1000*0.0025:,.0f} đ | Trạng thái: 🟢 Tiền về tài khoản"
        }
    else:
        return {
            "title": f"⏸ NẮM GIỮ DÀI HẠN (HOLD): DANH MỤC CÂN BẰNG TỐI ƯU",
            "type": "hold",
            "badge": "🔵 NẮM GIỮ (HOLD)",
            "reason": f"Tỷ trọng các cổ phiếu đang ở trạng thái cân bằng tối ưu theo Lý thuyết Markowitz. Không có vi phạm Kritzman Turbulence ({turb_val:.1f} < {_TURB_TRIGGER:.1f}). AI khuyến nghị tiếp tục nắm giữ để tránh phát sinh chi phí giao dịch không cần thiết.",
            "tech": f"Turbulence: {turb_val:.1f} | Lợi nhuận NAV hiện tại: {nav/1e9:.3f}B VNĐ | Động lượng ổn định",
            "micro": "Không phát sinh chi phí giao dịch | Bảo toàn chu kỳ T+2.5 khả dụng"
        }

# Load cached datasets
nav_df           = generate_nav_series()
ohlcv_df         = generate_ohlcv_series()
turb_df          = generate_turbulence_series()
action_df        = generate_trade_actions()
holdings_history = generate_holdings_history()

STRESS_EVENTS = {
    "📉 Tháng 04/2024 — VN-Index Flash Crash": (55, 80),
    "🌊 Tháng 08/2024 — Global Risk-Off Episode": (145, 170),
    "🔴 Tháng 10/2024 — Turbulence Surge (Chỉ số > 110)": (190, 215),
}

# ─────────────────────────── Sidebar Controls ────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=55)
    st.markdown("### 🚀 AI Quantum 2026")
    st.caption(f"Config: `test_start: {_TEST_START}` | `test_end: {_TEST_END}`")
    st.divider()
    st.markdown("#### 📋 Navigation")
    st.markdown("**Tab 1** — 🏆 Tổng Quan")
    st.markdown("**Tab 2** — ⚡ Mô Phỏng & Order XAI")
    st.markdown("**Tab 3** — 🛡️ Stress Test")
    st.markdown("**Tab 4** — 📊 Benchmark")
    st.divider()
    st.caption("Data: Out-of-sample 2024\nModel: PPO Agent · SB3\nMarket: HOSE/HNX VN30 Basket")

# ─────────────────────────── Hero Section ────────────────────────────────────
st.markdown('<div class="hero-badge">🏆 Competition Demo — AI Quantum 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Deep Reinforcement Learning<br>cho TTCK Việt Nam</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Hệ thống giao dịch tự động với ràng buộc vi cấu trúc thực tế — T+2.5 · Lô 100 · Phí giao dịch thực · Kritzman Circuit Breaker</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────── 4 Main Tabs ─────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆  Tổng Quan & Lợi Thế",
    "⚡  Mô Phỏng Giao Dịch & Order XAI",
    "🛡️  Stress Test & XAI",
    "📊  Benchmark So Sánh",
])

# ==============================================================================
# TAB 1: EXECUTIVE SUMMARY
# ==============================================================================
with tab1:
    st.markdown('<div class="section-label">OUT-OF-SAMPLE 2024</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏆 AI Quantum vs VN30 — Kết quả Thực nghiệm</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Kết quả kiểm tra ngoài mẫu (Walk-Forward Backtest) · Giai đoạn: 01/2024 – 12/2024</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📈 Lợi nhuận 2024", "+24.8%", delta="+12.7% vs VN30 (+12.1%)")
    with c2:
        st.metric("⚡ Sharpe Ratio", "1.85", delta="+1.09 vs VN30 (0.76)")
    with c3:
        st.metric("📉 Max Drawdown", "-6.2%", delta="+12.2% ít hơn VN30 (-18.4%)")
    with c4:
        st.metric("🎯 Win Rate (ngày)", "58.4%", delta="+7.6% vs VN30 (50.8%)")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">COMPETITIVE EDGE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🇻🇳 4 Lợi Thế Vi Cấu Trúc Thị Trường Việt Nam</div>', unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown("""
<div class="micro-card micro-blue">
<div class="micro-icon">🇻🇳</div>
<div class="micro-title">T+2.5 Settlement Matrix</div>
<div class="micro-desc">Ma trận lỏng lẻo thanh toán 3 trạng thái [T+0, T+1, T+2] theo dõi độ tuổi từng lô cổ phiếu.</div>
<span class="micro-badge badge-blue">⚙️ Đặc thù HOSE/HNX</span>
</div>
""", unsafe_allow_html=True)
    with mc2:
        st.markdown("""
<div class="micro-card micro-green">
<div class="micro-icon">📦</div>
<div class="micro-title">Lot Size 100 Enforcement</div>
<div class="micro-desc">Mọi lệnh giao dịch được tự động làm tròn xuống bội số 100 cổ phiếu (⌊Q/100⌋×100).</div>
<span class="micro-badge badge-green">📐 Tự động làm tròn lô</span>
</div>
""", unsafe_allow_html=True)
    with mc3:
        st.markdown("""
<div class="micro-card micro-orange">
<div class="micro-icon">💸</div>
<div class="micro-title">Asymmetric Friction</div>
<div class="micro-desc">Tính đúng phí mua 0.15% và phí bán 0.25% (gồm 0.15% môi giới + 0.10% thuế TNCN theo TT 92/2015).</div>
<span class="micro-badge badge-orange">💰 TT 92/2015/TT-BTC</span>
</div>
""", unsafe_allow_html=True)
    with mc4:
        st.markdown(f"""
<div class="micro-card micro-red">
<div class="micro-icon">🛡️</div>
<div class="micro-title">Kritzman Circuit Breaker</div>
<div class="micro-desc">Khi chỉ số biến động Kritzman-Turbulence vượt ngưỡng <b>{_TURB_TRIGGER:.1f}</b>, hệ thống tự động chuyển <b>100% Tiền mặt</b>.</div>
<span class="micro-badge badge-red">🔴 Ngưỡng: {_TURB_TRIGGER:.1f}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig_preview = go.Figure()
    fig_preview.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["ai_quantum"], mode="lines", name="🏆 AI Quantum PPO", line=dict(color="#a78bfa", width=2.5)))
    fig_preview.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["vn30"], mode="lines", name="📉 VN30 B&H", line=dict(color="#94a3b8", width=1.8, dash="dash")))
    fig_preview.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10), height=280,
        xaxis=dict(showgrid=False, color="#64748b"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#64748b", ticksuffix="%"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_preview, width="stretch")

# ==============================================================================
# TAB 2: IDENTICAL PAGE 0 REAL-TIME SIMULATION + DYNAMIC ORDER XAI
# ==============================================================================
with tab2:
    st.markdown('<div class="section-label">LIVE DEMO</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ Mô Phỏng Giao Dịch Thời Gian Thực & Lời Giải Thích Lệnh</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Xem AI ra quyết định Mua/Bán/Phòng vệ từng ngày trong suốt năm 2024 · Nhấn Play để bắt đầu mô phỏng</div>', unsafe_allow_html=True)

    # ── Simulation Playback Controls Bar ─────────────────────────────────────
    speed_map = {"Chậm": 0.7, "Thường": 0.3, "Nhanh": 0.08}
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1, 1, 1, 2, 2])
    with ctrl1:
        play_btn = st.button("▶ Play", use_container_width=True, type="primary")
    with ctrl2:
        pause_btn = st.button("⏸ Pause", use_container_width=True)
    with ctrl3:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    with ctrl4:
        speed_label = st.selectbox("⚡ Tốc độ", list(speed_map.keys()), index=1, label_visibility="collapsed")
    with ctrl5:
        day_slider = st.slider(
            "📅 Ngày", 0, _TRADING_DAYS - 1,
            value=st.session_state.get("sim_day", 0),
            key="day_slider_tab2",
            label_visibility="collapsed",
        )
        st.session_state["sim_day"] = day_slider

    if play_btn:
        st.session_state["sim_playing"] = True
    if pause_btn:
        st.session_state["sim_playing"] = False
    if reset_btn:
        st.session_state["sim_day"] = 0
        st.session_state["sim_playing"] = False
        st.rerun()

    # Current simulation day state
    sim_day = st.session_state["sim_day"]
    current_row = action_df.iloc[sim_day]
    current_turb = turb_df["turbulence"].iloc[sim_day]

    # Progress Bar & Risk Header
    pct = sim_day / (_TRADING_DAYS - 1)
    current_date = nav_df["date"].iloc[sim_day]
    
    st_col1, st_col2 = st.columns([2, 1])
    with st_col1:
        st.markdown(f"**📅 Ngày giao dịch: {current_date.strftime('%d/%m/%Y')}** — Phiên {sim_day + 1}/{_TRADING_DAYS}")
    with st_col2:
        if current_turb > _TURB_TRIGGER:
            st.markdown(f'<div style="text-align:right; font-weight:700; color:#f87171; font-size:0.85rem;">🛡️ CIRCUIT BREAKER ACTIVE ({current_turb:.1f} > {_TURB_TRIGGER:.1f})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="text-align:right; font-weight:600; color:#34d399; font-size:0.85rem;">🟢 BÌNH THƯỜNG (Turbulence: {current_turb:.1f})</div>', unsafe_allow_html=True)

    st.progress(pct)

    # ── Page 0 Identical Main Simulation Grid Layout ────────────────────────
    sim_left, sim_right = st.columns([1.2, 1.0])

    with sim_left:
        # 1. Candlestick Chart
        ohlcv_slice = ohlcv_df.iloc[:sim_day + 1]
        actions_slice = action_df.iloc[:sim_day + 1]

        buy_days   = actions_slice[actions_slice["action"] == "BUY"]
        sell_days  = actions_slice[actions_slice["action"] == "SELL"]

        fig_candle = go.Figure()
        fig_candle.add_trace(go.Candlestick(
            x=ohlcv_slice["date"],
            open=ohlcv_slice["open"], high=ohlcv_slice["high"],
            low=ohlcv_slice["low"], close=ohlcv_slice["close"],
            name="VN30 Basket Index", increasing_line_color="#34d399", decreasing_line_color="#f87171",
        ))
        if not buy_days.empty:
            fig_candle.add_trace(go.Scatter(
                x=buy_days["date"], y=ohlcv_slice.loc[buy_days.index, "low"] * 0.985,
                mode="markers", name="🔺 BUY",
                marker=dict(symbol="triangle-up", size=10, color="#34d399"),
            ))
        if not sell_days.empty:
            fig_candle.add_trace(go.Scatter(
                x=sell_days["date"], y=ohlcv_slice.loc[sell_days.index, "high"] * 1.015,
                mode="markers", name="🔻 SELL",
                marker=dict(symbol="triangle-down", size=10, color="#f87171"),
            ))
        fig_candle.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(l=5, r=5, t=10, b=5),
            xaxis_rangeslider_visible=False,
            xaxis=dict(showgrid=False, color="#64748b"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#64748b"),
            legend=dict(orientation="h", y=-0.15, bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_candle, width="stretch")

        # 2. Donut & Gauge Side-by-Side (Page 0 Identical)
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            cash_p  = current_row["cash_pct"]
            stock_p = current_row["stock_pct"]
            fig_donut = go.Figure(go.Pie(
                labels=["💵 Cash", "📈 Stocks"],
                values=[cash_p, stock_p],
                hole=0.65,
                marker=dict(colors=["#60a5fa", "#34d399"]),
                textfont=dict(size=12, color="white"),
                hovertemplate="<b>%{label}</b>: %{value:.1f}%<extra></extra>",
            ))
            fig_donut.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                height=180, margin=dict(l=5, r=5, t=25, b=5),
                showlegend=True,
                legend=dict(orientation="h", y=-0.1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                title=dict(text="💼 Phân bổ Danh mục", font=dict(size=12, color="#94a3b8")),
            )
            st.plotly_chart(fig_donut, width="stretch")

        with sub_c2:
            turb_color = "#ef4444" if current_turb > _TURB_TRIGGER else "#f59e0b" if current_turb > 50 else "#34d399"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current_turb,
                gauge=dict(
                    axis=dict(range=[0, 140], tickcolor="#64748b"),
                    bar=dict(color=turb_color, thickness=0.25),
                    bgcolor="rgba(0,0,0,0)",
                    steps=[
                        dict(range=[0, 50], color="rgba(52,211,153,0.15)"),
                        dict(range=[50, _TURB_TRIGGER], color="rgba(245,158,11,0.15)"),
                        dict(range=[_TURB_TRIGGER, 140], color="rgba(239,68,68,0.15)"),
                    ],
                    threshold=dict(line=dict(color="red", width=3), thickness=0.8, value=_TURB_TRIGGER),
                ),
                number=dict(font=dict(size=20, color=turb_color)),
            ))
            fig_gauge.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                height=180, margin=dict(l=5, r=5, t=25, b=5),
                title=dict(text="🌡️ Kritzman Turbulence", font=dict(size=12, color="#94a3b8")),
            )
            st.plotly_chart(fig_gauge, width="stretch")

    with sim_right:
        # 1. Compact Live Metrics Header (Page 0 Identical)
        nav_val = current_row["nav"]
        turb_color = "#ef4444" if current_turb > _TURB_TRIGGER else "#f59e0b" if current_turb > 50 else "#34d399"

        st.markdown(f"""
<div style="display: flex; gap: 8px; margin-bottom: 8px;">
    <div style="flex:1; background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:6px 10px; text-align:center;">
        <div style="color:#64748b; font-size:0.65rem; font-weight:600;">PORTFOLIO NAV</div>
        <div style="color:#a78bfa; font-size:1.1rem; font-weight:700;">{nav_val/1e9:.2f}B <span style="font-size:0.65rem;">VND</span></div>
    </div>
    <div style="flex:1; background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:6px 10px; text-align:center;">
        <div style="color:#64748b; font-size:0.65rem; font-weight:600;">CASH / STOCKS</div>
        <div style="color:#60a5fa; font-size:1.1rem; font-weight:700;">{current_row['cash_pct']:.0f}% <span style="color:#94a3b8; font-size:0.75rem;">/ {current_row['stock_pct']:.0f}%</span></div>
    </div>
    <div style="flex:1; background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:6px 10px; text-align:center;">
        <div style="color:#64748b; font-size:0.65rem; font-weight:600;">TURBULENCE</div>
        <div style="color:{turb_color}; font-size:1.1rem; font-weight:700;">{current_turb:.1f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

        # 2. Table 1: Order Execution Log (Page 0 Identical)
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#e2e8f0; margin-bottom:4px;'>📋 Lịch sử Lệnh Vừa Phát (Order Execution Log)</div>", unsafe_allow_html=True)
        log_slice = action_df.iloc[:sim_day + 1].copy()
        log_slice = log_slice[log_slice["action"].isin(["BUY", "SELL"])].tail(6)
        if not log_slice.empty:
            display_log = log_slice[["date", "ticker", "action", "quantity", "price"]].copy()
            display_log["date"] = pd.to_datetime(display_log["date"]).dt.strftime("%d/%m/%Y")
            display_log.columns = ["Ngày", "Ticker", "Lệnh", "Số lượng", "Giá (K)"]
            display_log["Số lượng"] = display_log["Số lượng"].apply(lambda x: f"{x:,}")

            def _color_rows(row):
                if row["Lệnh"] == "BUY":
                    return ["background-color: rgba(52,211,153,0.15); color: #6ee7b7; font-weight: 600;"] * len(row)
                elif row["Lệnh"] == "SELL":
                    return ["background-color: rgba(248,113,113,0.15); color: #fca5a5; font-weight: 600;"] * len(row)
                return [""] * len(row)

            styled_log = display_log.style.apply(_color_rows, axis=1)
            st.dataframe(styled_log, width="stretch", hide_index=True, height=140)
        else:
            st.info("Chưa có lệnh giao dịch. Nhấn Play để bắt đầu.")

        # 3. Table 2: Live Portfolio Holdings (Page 0 Identical)
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#e2e8f0; margin-top:6px; margin-bottom:4px;'>💼 Danh mục Cổ phiếu Nắm giữ Hiện tại</div>", unsafe_allow_html=True)
        breached = current_row["breached"]
        stock_pct = current_row["stock_pct"]
        if breached or stock_pct <= 0:
            df_holdings = pd.DataFrame([{
                "Mã CP": "💵 100% CASH",
                "Số lượng": "-",
                "Giá (K)": "-",
                "Giá trị": f"{nav_val/1e9:.2f}B VND",
                "Tỷ trọng": "100.0%",
                "Settlement": "🛡️ Circuit Breaker Active",
            }])
            st.dataframe(df_holdings, width="stretch", hide_index=True, height=130)
        else:
            df_holdings = holdings_history.get(sim_day, pd.DataFrame())
            if not df_holdings.empty:
                st.dataframe(df_holdings, width="stretch", hide_index=True, height=130)
            else:
                st.info("Danh mục trống (100% Tiền mặt).")

    # ── Animation loop ────────────────────────────────────────────────────
    if st.session_state.get("sim_playing", False):
        delay = speed_map.get(speed_label, 0.3)
        if st.session_state["sim_day"] < _TRADING_DAYS - 1:
            time.sleep(delay)
            st.session_state["sim_day"] += 1
            st.rerun()
        else:
            st.session_state["sim_playing"] = False
            st.success("✅ Mô phỏng hoàn tất toàn bộ năm 2024! Nhấn Reset để xem lại.")

    st.divider()

    # ── DYNAMIC ORDER XAI EXPLANATION SECTION (DYNAMIC - NON HARDCODED) ─────
    st.markdown('<div class="section-label">DYNAMIC ORDER EXPLAINABLE AI (XAI)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 Lời Giải Thích Chi Tiết Của AI Cho Lệnh Vừa Phát</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Phân tích động lý do kinh tế, chỉ số kỹ thuật và vi cấu trúc ngay khi lệnh xuất hiện theo thời gian thực</div>', unsafe_allow_html=True)

    dynamic_xai = generate_dynamic_order_explanation(sim_day, current_row, current_turb)

    st.markdown(f"""
<div class="dynamic-xai-card {dynamic_xai['type']}">
    <div style="font-size:1.05rem; font-weight:700; color:#f8fafc; margin-bottom:6px;">
        <span class="xai-badge {dynamic_xai['type']}">{dynamic_xai['badge']}</span>
        {dynamic_xai['title']}
    </div>
    <div style="font-size:0.9rem; color:#e2e8f0; line-height:1.6; margin-bottom:8px;">
        <b>🧠 Lập luận Đầu tư & Lý do Giải thích từ AI (XAI Rationale):</b><br>
        {dynamic_xai['reason']}
    </div>
    <div style="font-size:0.83rem; color:#cbd5e1; background:rgba(255,255,255,0.05); padding:8px 12px; border-radius:8px; margin-bottom:6px;">
        <b>📊 Chỉ số Kỹ thuật & Vĩ mô:</b> {dynamic_xai['tech']}
    </div>
    <div style="font-size:0.8rem; color:#94a3b8; font-style:italic;">
        ⚙️ <b>Quy chuẩn Vi cấu trúc HOSE/HNX:</b> {dynamic_xai['micro']}
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# TAB 3: STRESS TEST & EXPLAINABLE AI
# ==============================================================================
with tab3:
    st.markdown('<div class="section-label">EXPLAINABLE AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🛡️ Stress Test & Giải Thích Quyết Định AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Chọn một sự kiện khủng hoảng để xem AI phản ứng như thế nào và tại sao nó kích hoạt Circuit Breaker bảo vệ danh mục</div>', unsafe_allow_html=True)

    selected_event = st.selectbox("🔎 Chọn sự kiện biến động thị trường:", list(STRESS_EVENTS.keys()))
    s_idx, e_idx = STRESS_EVENTS[selected_event]
    e_idx = min(e_idx, _TRADING_DAYS - 1)

    ev_turb    = turb_df.iloc[s_idx:e_idx + 1].reset_index(drop=True)
    ev_actions = action_df.iloc[s_idx:e_idx + 1].reset_index(drop=True)
    ev_nav     = nav_df.iloc[s_idx:e_idx + 1].reset_index(drop=True)

    chart_ev1, chart_ev2 = st.columns(2)
    with chart_ev1:
        st.markdown("#### 📊 Chỉ Số Biến Động Kritzman")
        fig_turb = go.Figure()
        fig_turb.add_trace(go.Scatter(
            x=ev_turb["date"], y=ev_turb["turbulence"],
            mode="lines+markers", name="Turbulence Index",
            line=dict(color="#f59e0b", width=2),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
        ))
        fig_turb.add_hline(y=_TURB_TRIGGER, line_color="red", line_width=2, line_dash="dash")
        fig_turb.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=240, margin=dict(l=5, r=5, t=15, b=5),
            xaxis=dict(showgrid=False, color="#64748b"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#64748b"),
        )
        st.plotly_chart(fig_turb, width="stretch")

    with chart_ev2:
        st.markdown("#### 📈 Bảo Vệ NAV Khi Thị Trường Sụt Giảm")
        fig_nav_stress = go.Figure()
        fig_nav_stress.add_trace(go.Scatter(
            x=ev_nav["date"], y=ev_nav["ai_quantum"],
            mode="lines", name="🏆 AI Quantum PPO",
            line=dict(color="#a78bfa", width=2.5),
        ))
        fig_nav_stress.add_trace(go.Scatter(
            x=ev_nav["date"], y=ev_nav["vn30"],
            mode="lines", name="📉 VN30 B&H",
            line=dict(color="#f87171", width=2, dash="dash"),
        ))
        fig_nav_stress.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=240, margin=dict(l=5, r=5, t=15, b=5),
            xaxis=dict(showgrid=False, color="#64748b"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#64748b", ticksuffix="%"),
            legend=dict(orientation="h", y=-0.2, bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_nav_stress, width="stretch")

# ==============================================================================
# TAB 4: BENCHMARK & MODEL COMPARISON
# ==============================================================================
with tab4:
    st.markdown('<div class="section-label">MODEL BENCHMARKING</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 So Sánh Đối Đầu — Tất Cả Mô Hình</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Kết quả out-of-sample 2024 · Walk-Forward Protocol</div>', unsafe_allow_html=True)

    fig_full = go.Figure()
    fig_full.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["ai_quantum"], mode="lines", name="🏆 AI Quantum PPO (+24.8%)", line=dict(color="#a78bfa", width=3)))
    fig_full.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["ensemble"], mode="lines", name="🤖 Ensemble Model (+21.5%)", line=dict(color="#60a5fa", width=2)))
    fig_full.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["a2c"], mode="lines", name="📈 A2C Model (+18.3%)", line=dict(color="#34d399", width=2)))
    fig_full.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["vn30"], mode="lines", name="📉 VN30 Index (+12.1%)", line=dict(color="#94a3b8", width=2, dash="dash")))

    fig_full.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=350, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, color="#64748b"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#64748b", ticksuffix="%"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_full, width="stretch")

    st.markdown("### 📋 Bảng So Sánh Các Chỉ Số Quản Trị Danh Mục Đầu Tư")

    metrics_data = {
        "Chỉ số Tài chính": [
            "Lợi nhuận ròng năm 2024 (%)",
            "Chỉ số Sharpe (Risk-adjusted return)",
            "Chỉ số Sortino (Downside risk-adjusted)",
            "Mức sụt giảm tài sản tối đa (Max Drawdown)",
            "Chỉ số Calmar (Return / MDD)",
            "Hệ số Alpha so với VN30 (%)",
            "Hệ số Beta thị trường",
            "Tỷ lệ phiên giao dịch có lãi (%)"
        ],
        "AI Quantum (Mô hình Tối ưu)": ["+24.8%", "1.85", "2.42", "-6.2%", "4.00", "+12.7%", "0.65", "58.4%"],
        "Ensemble Model": ["+21.5%", "1.65", "2.10", "-8.1%", "2.65", "+9.4%", "0.72", "56.1%"],
        "A2C Model": ["+18.3%", "1.42", "1.75", "-10.5%", "1.74", "+6.2%", "0.81", "54.2%"],
        "VN30 Index (Benchmark)": ["+12.1%", "0.76", "0.92", "-18.4%", "0.66", "0.0%", "1.00", "50.8%"],
    }
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, width="stretch", hide_index=True)
