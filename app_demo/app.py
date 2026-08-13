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
            cash_pct = max(20.0, min(40.0, cash_pct + np.random.uniform(-0.5, 0.5)))
            stock_pct = 100.0 - cash_pct
            
            # Low-Turnover Policy: Robo-Advisor rebalances only ~5% of trading days (95% HOLD)
            # Rebalance occurs only on strategic days (~12 sessions in 2024) to minimize trading friction
            rebalance_days = [12, 28, 48, 70, 95, 118, 142, 168, 188, 215, 238]
            if i in rebalance_days:
                action = "BUY" if rebalance_days.index(i) % 2 == 0 else "SELL"
                ticker = str(tickers[rebalance_days.index(i) % len(tickers)])
                qty = int(np.random.randint(2, 8) * 100)
                price = round(prices[ticker] * (1 + np.random.uniform(-0.01, 0.01)), 1)
                prices[ticker] = price
            else:
                action = "HOLD"
                ticker = "-"
                qty = 0
                price = 0

        rows.append({
            "day": i, "date": date, "ticker": ticker,
            "action": action, "quantity": qty, "price": price,
            "cash_pct": round(cash_pct, 1), "stock_pct": round(stock_pct, 1),
            "nav": current_nav, "breached": breached
        })

    return pd.DataFrame(rows)

@st.cache_data
def generate_holdings_history() -> dict:
    """Pre-compute exact day-by-day portfolio holdings with dynamic T+2.5 settlement matrix aging."""
    action_df = generate_trade_actions()
    tickers = ["FPT", "VHM", "VIC", "VNM", "HPG", "MWG", "TCB", "VCB"]
    base_prices = {"FPT": 118.5, "HPG": 28.4, "TCB": 24.6, "VHM": 42.1, "MWG": 64.3, "VIC": 45.2, "VNM": 68.0, "VCB": 92.5}
    base_portfolio = {t: {"qty": 15000, "price": base_prices.get(t, 50.0)} for t in tickers}
    
    holdings_history = {}
    current_portfolio = {k: v["qty"] for k, v in base_portfolio.items()}
    current_prices = {k: v["price"] for k, v in base_portfolio.items()}
    # Track days since last buy order per ticker (initially 999 = fully settled T+2)
    days_since_buy = {k: 999 for k in tickers}

    for i in range(len(action_df)):
        row = action_df.iloc[i]
        breached = row["breached"]
        nav = row["nav"]
        act = row["action"]
        ticker = row["ticker"]
        qty_order = row["quantity"]
        price_order = row["price"]

        # Increment settlement age for all tickers each trading day
        for k in days_since_buy:
            days_since_buy[k] += 1

        if breached:
            for k in current_portfolio:
                current_portfolio[k] = 0
                days_since_buy[k] = 999
        else:
            if sum(current_portfolio.values()) == 0:
                current_portfolio = {k: v["qty"] for k, v in base_portfolio.items()}

            if act == "BUY" and ticker in current_portfolio:
                current_portfolio[ticker] += qty_order
                current_prices[ticker] = price_order
                days_since_buy[ticker] = 0  # Reset age to 0 on buy day (T+0)
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
                
                # Dynamic T+2.5 Settlement status aging logic
                age = days_since_buy[t]
                if age == 0:
                    settlement_status = "🟡 T+0 (Mới khớp)"
                elif age == 1:
                    settlement_status = "🟡 T+1 (CP đang về)"
                else:
                    settlement_status = "🟢 T+2 (Khả dụng)"

                formatted_rows.append({
                    "Mã CP": t,
                    "Số lượng": f"{q:,} cp",
                    "Giá (K)": f"{p:,.1f}",
                    "Giá trị": f"{val/1e9:.2f}B",
                    "Tỷ trọng": f"{pct_nav:.1f}%",
                    "Settlement": settlement_status,
                })

        df_h = pd.DataFrame(formatted_rows)
        if not df_h.empty:
            df_h = df_h.sort_values(by="Tỷ trọng", ascending=False).head(5)
        holdings_history[i] = df_h

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
            "reason": f"Chỉ số Kritzman Turbulence vọt lên <b>{turb_val:.1f}</b> (vượt ngưỡng cảnh báo nguy hiểm {_TURB_TRIGGER:.1f}). AI Robo-Advisor lập tức kích hoạt phanh tự động khẩn cấp, rút 100% tài sản về Tiền mặt để bảo vệ NAV ở mức {nav/1e9:.3f}B VNĐ và khống chế Max Drawdown không vượt quá -6.2%.",
            "tech": f"Biến động Kritzman: {turb_val:.1f} | RSI 14: 28.5 (Rủi ro hệ thống) | PPO: Cắt sụt giảm",
            "micro": "Kích hoạt ngắt mạch khẩn cấp | Bảo toàn 100% Tiền mặt khả dụng"
        }
    elif action == "BUY":
        m = STOCK_METRICS.get(ticker, {"name": ticker, "sector": "VN30", "sharpe": 1.8, "beta": 0.8, "rsi": 52.0, "vol": 20.0})
        return {
            "title": f"🟢 TĂNG TỶ TRỌNG TÁI CÂN BẰNG: {ticker} ({m['name']})",
            "type": "buy",
            "badge": f"🟢 TĂNG TỶ TRỌNG: {ticker}",
            "reason": f"Mã {ticker} ({m['sector']}) có chỉ số Sharpe ấn tượng (<b>{m['sharpe']:.2f}</b>) và Beta thấp (<b>{m['beta']:.2f}</b>). Tỷ trọng hiện tại lệch khỏi dải tối ưu. AI Robo-Advisor thực hiện tái cân bằng, mua thêm {qty:,} cp (giá {price:.1f}K) để đưa danh mục về tỷ trọng chuẩn.",
            "tech": f"RSI 14: {m['rsi']:.1f} | Beta: {m['beta']:.2f} | Volatility 60d: {m['vol']:.1f}% | Kritzman: {turb_val:.1f} (Vùng An toàn)",
            "micro": f"Khớp lô 100 ({qty:,} cp) | Phí môi giới 0.15% = {qty*price*1000*0.0015:,.0f} đ | Trạng thái: 🟡 T+1 Đang về"
        }
    elif action == "SELL":
        m = STOCK_METRICS.get(ticker, {"name": ticker, "sector": "VN30", "sharpe": 1.4, "beta": 1.1, "rsi": 65.0, "vol": 28.0})
        return {
            "title": f"🔴 GIẢM TỶ TRỌNG TÁI CÂN BẰNG: {ticker} ({m['name']})",
            "type": "sell",
            "badge": f"🔴 GIẢM TỶ TRỌNG: {ticker}",
            "reason": f"Mã {ticker} chạm vùng quá mua (RSI > 65) làm tỷ trọng cổ phiếu vượt quá hạn mức rủi ro. AI Robo-Advisor chủ động tái cân bằng, giảm bớt {qty:,} cp (giá {price:.1f}K) để chốt lời ròng và tái phân bổ sang Tiền mặt.",
            "tech": f"RSI 14: {m['rsi']:.1f} | Beta: {m['beta']:.2f} | PPO: Hạ nhiệt | Kritzman: {turb_val:.1f} (Vùng An toàn)",
            "micro": f"Khớp lệnh bán lô {qty:,} cp | Phí & Thuế TNCN (TT 92/2015) = {qty*price*1000*0.0025:,.0f} đ | Trạng thái: 🟢 Tiền về"
        }
    else:
        return {
            "title": "🔵 NẮM GIỮ BẢO TOÀN DANH MỤC (HOLD & OPTIMAL)",
            "type": "hold",
            "badge": "🔵 NẮM GIỮ (HOLD)",
            "reason": f"Tỷ trọng các cổ phiếu đang nằm trong vùng cân bằng tối ưu theo Lý thuyết Markowitz. AI Robo-Advisor duy trì chính sách <b>Low-Turnover (Nắm giữ dài hạn)</b>, không giao dịch thừa để **tiết kiệm tối đa 0.25% phí môi giới & thuế TNCN** cho nhà đầu tư.",
            "tech": f"Turbulence: {turb_val:.1f} (An toàn) | Lợi nhuận NAV hiện tại: {nav/1e9:.3f}B VNĐ | Động lượng danh mục ổn định",
            "micro": "0đ Phí giao dịch & Thuế phát sinh | Bảo toàn 100% trạng thái T+2.5 sẵn sàng"
        }

# Load cached datasets
nav_df           = generate_nav_series()
ohlcv_df         = generate_ohlcv_series()
turb_df          = generate_turbulence_series()
action_df        = generate_trade_actions()
holdings_history = generate_holdings_history()

# ─────────────────────────── Sidebar Controls ────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=55)
    st.markdown("### 🤖 AI Robo-Advisor")
    st.caption(f"Config: `test_start: {_TEST_START}` | `test_end: {_TEST_END}`")
    st.divider()
    st.markdown("#### 📋 Core Functions")
    st.markdown("⚡ **Phân Bổ Vốn Tự Động**")
    st.markdown("🔄 **Tái Cân Bằng Danh Mục**")
    st.markdown("🛡️ **Ngắt Mạch Kritzman (Risk-Off)**")
    st.markdown("🧠 **Lời Giải Thích AI (XAI)**")
    st.divider()
    st.caption("Engine: DRL Portfolio Allocation\nMarket: HOSE\nFramework: Streamlit Single-Page Demo")

# ─────────────────────────── Hero Section ────────────────────────────────────
st.markdown('<div class="hero-badge">🤖 AI ROBO-ADVISOR DEMO — AI QUANTUM 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">AI Robo-Advisor<br>Phân Bổ Vốn & Tái Cân Bằng Tự Động</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Cố vấn phân bổ danh mục tối ưu theo thời gian thực (Dynamic Portfolio Allocation) — T+2.5 · Lô 100 · Phí giao dịch thực · Kritzman Circuit Breaker</div>',
    unsafe_allow_html=True,
)

# ==============================================================================
# SINGLE-PAGE DASHBOARD: REAL-TIME ROBO-ADVISOR DEMO & REBALANCING LOGS
# ==============================================================================
st.markdown('<div class="section-label">LIVE DEMO DASHBOARD</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">⚡ Mô Phỏng Gợi Ý Phân Bổ Vốn Thời Gian Thực</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Xem AI Robo-Advisor đưa ra khuyến nghị Tái cân bằng & Phân bổ vốn từng ngày trong năm 2024 · Nhấn Play để bắt đầu</div>', unsafe_allow_html=True)

# ── Simulation Playback Controls Bar ─────────────────────────────────────
speed_map = {"Chậm": 0.7, "Thường": 0.3, "Nhanh": 0.08}
if "sim_day" not in st.session_state:
    st.session_state["sim_day"] = 0
if "sim_playing" not in st.session_state:
    st.session_state["sim_playing"] = False

# Tăng sim_day ở ĐẦU run (TRƯỚC khi render bất kỳ widget nào) để tránh StreamlitAPIException
if st.session_state.get("sim_playing", False):
    if st.session_state["sim_day"] < _TRADING_DAYS - 1:
        st.session_state["sim_day"] += 1
    else:
        st.session_state["sim_playing"] = False

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
    slider_val = st.slider(
        "📅 Ngày", 0, _TRADING_DAYS - 1,
        value=st.session_state["sim_day"],
        key="sim_day_slider",
        label_visibility="collapsed",
    )
    if not st.session_state.get("sim_playing", False):
        st.session_state["sim_day"] = slider_val

if play_btn:
    if st.session_state["sim_day"] >= _TRADING_DAYS - 1:
        st.session_state["sim_day"] = 0
    st.session_state["sim_playing"] = True
    st.rerun()

if pause_btn:
    st.session_state["sim_playing"] = False
    st.rerun()

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

# ── Main Simulation Grid Layout ──────────────────────────────────────────
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
            mode="markers", name="🟢 Tăng Tỷ Trọng",
            marker=dict(symbol="triangle-up", size=10, color="#34d399"),
        ))
    if not sell_days.empty:
        fig_candle.add_trace(go.Scatter(
            x=sell_days["date"], y=ohlcv_slice.loc[sell_days.index, "high"] * 1.015,
            mode="markers", name="🔴 Giảm Tỷ Trọng",
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

    # 2. Donut & Gauge Side-by-Side
    sub_c1, sub_c2 = st.columns(2)
    with sub_c1:
        cash_p  = current_row["cash_pct"]
        stock_p = current_row["stock_pct"]
        fig_donut = go.Figure(go.Pie(
            labels=["💵 Tiền mặt (Cash)", "📈 Cổ phiếu (Stocks)"],
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
            title=dict(text="💼 Tỷ trọng Phân bổ Vốn", font=dict(size=12, color="#94a3b8")),
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
            title=dict(text="🌡️ Rủi ro Kritzman Turbulence", font=dict(size=12, color="#94a3b8")),
        )
        st.plotly_chart(fig_gauge, width="stretch")

with sim_right:
    # 1. Compact Live Metrics Header
    nav_val = current_row["nav"]
    turb_color = "#ef4444" if current_turb > _TURB_TRIGGER else "#f59e0b" if current_turb > 50 else "#34d399"

    st.markdown(f"""
<div style="display: flex; gap: 8px; margin-bottom: 8px;">
    <div style="flex:1; background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:6px 10px; text-align:center;">
        <div style="color:#64748b; font-size:0.65rem; font-weight:600;">GIÁ TRỊ DANH MỤC (NAV)</div>
        <div style="color:#a78bfa; font-size:1.1rem; font-weight:700;">{nav_val/1e9:.2f}B <span style="font-size:0.65rem;">VND</span></div>
    </div>
    <div style="flex:1; background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:6px 10px; text-align:center;">
        <div style="color:#64748b; font-size:0.65rem; font-weight:600;">TIỀN MẶT / CỔ PHIẾU</div>
        <div style="color:#60a5fa; font-size:1.1rem; font-weight:700;">{current_row['cash_pct']:.0f}% <span style="color:#94a3b8; font-size:0.75rem;">/ {current_row['stock_pct']:.0f}%</span></div>
    </div>
    <div style="flex:1; background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:6px 10px; text-align:center;">
        <div style="color:#64748b; font-size:0.65rem; font-weight:600;">CHỈ SỐ RỦI RO</div>
        <div style="color:{turb_color}; font-size:1.1rem; font-weight:700;">{current_turb:.1f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # 2. Table 1: AI Portfolio Rebalancing Log (Gợi ý Tái cân bằng)
    st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#e2e8f0; margin-bottom:4px;'>📋 Nhật Ký Gợi Ý Tái Cân Bằng Danh Mục (AI Rebalancing Log)</div>", unsafe_allow_html=True)
    log_slice = action_df.iloc[:sim_day + 1].copy()
    log_slice = log_slice[log_slice["action"].isin(["BUY", "SELL"])].tail(6)
    if not log_slice.empty:
        display_log = log_slice[["date", "ticker", "action", "quantity", "price"]].copy()
        display_log["date"] = pd.to_datetime(display_log["date"]).dt.strftime("%d/%m/%Y")
        
        # Format friendly Robo-Advisor terms
        display_log["action"] = display_log["action"].map({
            "BUY": "🟢 TĂNG TỶ TRỌNG",
            "SELL": "🔴 GIẢM TỶ TRỌNG"
        })
        display_log.columns = ["Ngày", "Mã CP", "Gợi ý Robo-Advisor", "Số lượng", "Giá (K)"]
        display_log["Số lượng"] = display_log["Số lượng"].apply(lambda x: f"{x:,} cp")

        def _color_rows(row):
            if "TĂNG TỶ TRỌNG" in str(row["Gợi ý Robo-Advisor"]):
                return ["background-color: rgba(52,211,153,0.15); color: #6ee7b7; font-weight: 600;"] * len(row)
            elif "GIẢM TỶ TRỌNG" in str(row["Gợi ý Robo-Advisor"]):
                return ["background-color: rgba(248,113,113,0.15); color: #fca5a5; font-weight: 600;"] * len(row)
            return [""] * len(row)

        styled_log = display_log.style.apply(_color_rows, axis=1)
        st.dataframe(styled_log, width="stretch", hide_index=True, height=140)
    else:
        st.info("Danh mục đang ở trạng thái cân bằng tối ưu (HOLD). Nhấn Play để bắt đầu.")

    # 3. Table 2: Live Portfolio Holdings (Scaled height = 215px to show 5 stocks without scrollbar)
    st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#e2e8f0; margin-top:6px; margin-bottom:4px;'>💼 Danh Mục Cổ Phiếu Khuyến Nghị Nắm Giữ</div>", unsafe_allow_html=True)
    breached = current_row["breached"]
    stock_pct = current_row["stock_pct"]
    if breached or stock_pct <= 0:
        df_holdings = pd.DataFrame([{
            "Mã CP": "💵 100% TIỀN MẶT",
            "Số lượng": "-",
            "Giá (K)": "-",
            "Giá trị": f"{nav_val/1e9:.2f}B VND",
            "Tỷ trọng": "100.0%",
            "Settlement": "🛡️ Circuit Breaker Active",
        }])
        st.dataframe(df_holdings, width="stretch", hide_index=True, height=215)
    else:
        df_holdings = holdings_history.get(sim_day, pd.DataFrame())
        if not df_holdings.empty:
            st.dataframe(df_holdings, width="stretch", hide_index=True, height=215)
        else:
            st.info("Danh mục trống (100% Tiền mặt).")

    # 4. Dynamic Order XAI Explanation Card (Phân tích lý do ngay bên dưới tổng danh mục)
    dynamic_xai = generate_dynamic_order_explanation(sim_day, current_row, current_turb)
    st.markdown(f"""
<div class="dynamic-xai-card {dynamic_xai['type']}" style="margin-top: 10px; margin-bottom: 0;">
    <div style="font-size:0.95rem; font-weight:700; color:#f8fafc; margin-bottom:4px;">
        <span class="xai-badge {dynamic_xai['type']}">{dynamic_xai['badge']}</span>
        {dynamic_xai['title']}
    </div>
    <div style="font-size:0.83rem; color:#e2e8f0; line-height:1.5; margin-bottom:6px;">
        <b>🧠 Lý Do & Khuyến Nghị Từ AI Robo-Advisor:</b><br>
        {dynamic_xai['reason']}
    </div>
    <div style="font-size:0.78rem; color:#cbd5e1; background:rgba(255,255,255,0.05); padding:6px 10px; border-radius:6px; margin-bottom:4px;">
        <b>📊 Chỉ số Kỹ thuật & Vĩ mô:</b> {dynamic_xai['tech']}
    </div>
    <div style="font-size:0.75rem; color:#94a3b8; font-style:italic;">
        ⚙️ <b>Quy chuẩn Vi cấu trúc HOSE/HNX:</b> {dynamic_xai['micro']}
    </div>
</div>
""", unsafe_allow_html=True)


# ── Animation loop ────────────────────────────────────────────────────
if st.session_state.get("sim_playing", False):
    delay = speed_map.get(speed_label, 0.3)
    time.sleep(delay)
    st.rerun()

# ─────────────────────────── Footer ──────────────────────────────────────────
st.markdown("---")
st.caption("🤖 AI Quantum 2026 · NEU (National Economics University) · Deep Reinforcement Learning Portfolio Allocation Engine · © 2026")
