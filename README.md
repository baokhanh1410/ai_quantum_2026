# AI Quantum 2026: Deep Reinforcement Learning for Vietnam Stock Market

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit%20%7C%20Stable--Baselines3-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end quantitative trading framework leveraging Deep Reinforcement Learning (DRL) tailored for the Vietnam Stock Market (HOSE) microstructures.

---

## 🌟 Key Features & Architecture

The architecture is divided into three modular engines:

```
                          ┌──────────────────────────┐
                          │   1. Data Engine         │
                          │ (Automated API Fetching) │
                          └────────────┬─────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │   2. Feature Engine      │
                          │ (Technical & Macro Inds) │
                          └────────────┬─────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │   3. Model Engine        │
                          │ (DRL & Active Cash Alloc)│
                          └────────────┬─────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │ Streamlit Web Dashboard  │
                          │ (Data → Train → Analysis)│
                          └──────────────────────────┘
```

1. **Data Engine**: Automated market & macro data pipeline fetching OHLCV, sector indices (`VNFIN`, `VNREAL`...), and macroeconomic indicators (`VNIBOR_ON`, `DXY`, `Yield Curve`).
2. **Feature Engine**: Calculates stationarity-tested technical indicators (`RSI`, `PPO`, `CCI`, `ADX`, `ATR`, `Volatility`).
3. **Model Engine**:
   - **Active Portfolio Weight Allocation**: $(N + 1)$-dimensional action space $[w_{\text{cash}}, w_{\text{stock}_1}, \dots, w_{\text{stock}_N}]$ ensuring dynamic shifting into Cash during market downturns.
   - **Rolling Horizon Training Window**: Episode randomization to mitigate financial concept drift.
   - **Vietnam Market Microstructure Enforcements**:
     - **T+2 Settlement Liquidity Matrix**: $3$-state matrix $[T+2, T+1, T+0]$ preventing illegal $T+0/T+1$ selling.
     - **Lot Size 100**: Floor rounding to integer multiples of 100 shares.
     - **Asymmetric Transaction Frictions**: $0.15\%$ Buy fee vs $0.25\%$ Sell fee ($0.15\%$ brokerage + $0.10\%$ Personal Income Tax).
     - **Risk-Free Cash Yield**: Overnight cash balance accrual based on `VNIBOR_ON`.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/baokhanh1410/ai_quantum_2026.git
cd ai_quantum_2026

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 🚀 Interactive Streamlit Web App

Launch the multi-page Streamlit web dashboard:

```bash
streamlit run src/app/app.py
```

### Dashboard Workflows:
- **Page 1 — Data Overview**: Query OHLCV data, inspect tickers, and view interactive Plotly price charts.
- **Page 2 — Model Training**: Select DRL algorithms (**PPO**, **A2C**, **DDPG**, or **Ensemble Strategy**), configure total timesteps, and train agents with real-time execution feedback.
- **Page 3 — Performance Analysis**:
  - Interactive **NAV Curve** vs VN30 Benchmark.
  - Financial Metrics Summary (**Sharpe**, **Sortino**, **Max Drawdown**, **Alpha**, **Win Rate**, **Calmar**).
  - **Portfolio Weight Allocation Chart**: Track dynamic Cash vs Stock allocation over time.

---

## 📁 Repository Structure

```
.
├── config/                      # YAML Configuration Layer
│   ├── assets.yaml              # Asset class definitions & settlement locks
│   ├── config.yaml              # Global project & API endpoints settings
│   ├── features.yaml            # Technical & macro indicators catalog
│   ├── market.yaml              # Vietnam market rules (T+2, Lot 100, fees)
│   ├── model.yaml               # DRL Hyperparameters & training timelines
│   └── sector_mapping.yaml      # ICB sector to HOSE index mapping
├── database/                    # Database DDL Schemas
│   └── schema.sql               # MySQL relational database tables DDL
├── src/
│   ├── app/                     # Streamlit Interactive Web Application
│   │   ├── app.py               # Application Entry Point & System Overview
│   │   ├── components/          # State management & Plotly chart renderers
│   │   └── pages/               # 1_Data_Overview, 2_Train_Model, 3_Analysis
│   └── pipeline/                # Core 3-Engine Architecture
│       ├── core/                # Core database connection & settings loader
│       ├── data_engine/         # 1. Automated Data Ingestion Engine
│       │   ├── api/             # Market API Clients (vnstock, SBV, TradingView, Yahoo)
│       │   ├── handlers/        # Data format normalization handlers
│       │   ├── processors/      # OHLCV validation & cleaning
│       │   └── services/        # Ingestion pipeline orchestrator
│       ├── feature_engine/      # 2. Technical & Macro Feature Engine
│       │   ├── database/        # DuckDB & MySQL feature storage repositories
│       │   ├── processors/      # Technical indicator calculators (RSI, PPO, CCI...)
│       │   └── services/        # Feature calculation pipeline service
│       └── model_engine/        # 3. Deep Reinforcement Learning Engine
│           ├── analysis/        # MetricsAnalyzer (Sharpe, MDD, Alpha) & Visualizer
│           ├── data/            # DataQueryService for DRL environment
│           ├── env/             # StockTradingEnv (T+2, Lot 100, Cash Allocation)
│           ├── models/          # DRLEnsembleStrategy (PPO, A2C, DDPG)
│           └── notebook.py      # Standalone training & backtesting pipeline
├── DATABASE.md                  # Database Architecture Documentation
├── MODEL.md                     # Model Engine & DRL Technical Documentation
├── README.md                    # Project Master Guide
└── requirements.txt             # Python Dependencies
```


---

