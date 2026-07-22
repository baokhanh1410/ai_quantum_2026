#!/usr/bin/env python
# coding: utf-8

# # AI Model Engine - Interactive Training & Analysis
# Notebook này giúp bạn tải dữ liệu, kiểm tra tính dừng (stationarity), tiến hành huấn luyện RL Agent một cách độc lập không cần server, và thực hiện kiểm định đánh giá hiệu năng chiến lược (Sharpe Ratio, Max Drawdown, NAV vs VN30, Action Distribution).

# In[1]:


import sys
import os
from pathlib import Path
import pandas as pd
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)

# Tự động tìm và thêm thư mục src/pipeline vào sys.path bất kể script chạy ở đâu
current_dir = Path(os.getcwd()).resolve()
possible_paths = [
    current_dir,
    current_dir / "src" / "pipeline",
    current_dir.parent / "src" / "pipeline",
    current_dir.parent.parent / "src" / "pipeline",
    current_dir.parent.parent.parent / "src" / "pipeline"
]
for p in possible_paths:
    if p.exists() and (p / "model_engine").exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
        break


# ## 1. Load Data

# In[2]:


from model_engine.data.data_service import DataQueryService
from model_engine.data.processor import DataProcessor
from core.config.settings import MODEL_CONFIG

train_start = MODEL_CONFIG.get("train_start_date", "2018-06-25")
train_end = MODEL_CONFIG.get("train_end_date", "2020-12-31")

features = MODEL_CONFIG.get("features", [])
query_svc = DataQueryService()
train_raw = query_svc.fetch_data(train_start, train_end)

processor = DataProcessor(train_raw, features)
train_data = processor.clean_data()

print("Data shape after processing:", train_data.shape)
print(train_data.tail(15))


# ## 2. Stationarity Check
# Kiểm tra xem các features có đạt tiêu chuẩn dừng (Stationary) cho Machine Learning hay không.

# In[3]:


from model_engine.analysis.stationarity import StationarityTester

tester = StationarityTester(significance_level=0.05)
# Check 3 features for demo purposes
features_to_check = features[:3] if len(features) >= 3 else features

print(f"Checking features: {features_to_check}")
report = tester.analyze_features(train_data, features_to_check, groupby_col='tic')

if not report.empty:
    non_stationary = report[~report['is_strictly_stationary']]
    print(f"Có {len(non_stationary)} / {len(report)} chuỗi không dừng.")
    print(non_stationary.head(10))
else:
    print("Không có dữ liệu stationarity report.")


# ## 3. Train DRL Ensemble

# In[4]:


from model_engine.models.drl_models import DRLEnsembleStrategy
from model_engine.env.stock_trading_env import StockTradingEnv

# Create validation split for early stopping/selection
val_start = MODEL_CONFIG.get("val_start_date", "2021-01-01")
val_end = MODEL_CONFIG.get("val_end_date", "2021-12-31")
val_raw = query_svc.fetch_data(val_start, val_end)
val_data = DataProcessor(val_raw, features).clean_data()

env_kwargs = {
    "features": features,
    "initial_balance": MODEL_CONFIG.get("initial_balance", 1_000_000_000),
    "transaction_cost_pct": MODEL_CONFIG.get("transaction_cost_pct", 0.0015),
    "turbulence_threshold": MODEL_CONFIG.get("turbulence_threshold", 150),
    "episode_length": MODEL_CONFIG.get("episode_length", 60),
    "random_start": MODEL_CONFIG.get("random_start", True),
}

strategy = DRLEnsembleStrategy(StockTradingEnv, env_kwargs, train_data, val_data)
# Chạy 3 thuật toán A2C, PPO, DDPG và chọn thuật toán tốt nhất
best_name, best_model = strategy.train_and_select()

print(f"🎉 Huấn luyện thành công. Mô hình chiến thắng: {best_name}")


# ## 4. Strategy Evaluation & Financial Metrics

# In[5]:


from model_engine.analysis.metrics_analyzer import MetricsAnalyzer

# Trích xuất lịch sử giao dịch và tài sản của mô hình chiến thắng
df_account, df_actions, df_shares = strategy.evaluate_and_get_trajectory(best_model, val_data)

# Truy vấn trực tiếp dữ liệu VN30 Benchmark từ database bất kể asset class id
vn30_df = query_svc.fetch_symbol_data("VN30", val_start, val_end)
if not vn30_df.empty:
    print(f"✓ Đã tải thành công {len(vn30_df)} phiên dữ liệu VN30 Benchmark.")
else:
    print("⚠️ Không tìm thấy dữ liệu VN30 Benchmark trong khoảng thời gian validation.")

analyzer = MetricsAnalyzer(risk_free_rate=0.03)
metrics = analyzer.compute_all_metrics(df_account, benchmark_account=vn30_df)

print("\n📊 BẢNG CHỈ SỐ KIỂM ĐỊNH HIỆU NĂNG TÀI CHÍNH:")
print(f"• Cumulative Return:      {metrics['cumulative_return']*100:8.2f}%")
print(f"• Annualized Return:      {metrics['annualized_return']*100:8.2f}%")
print(f"• Annualized Volatility:  {metrics['annualized_volatility']*100:8.2f}%")
print(f"• Sharpe Ratio:           {metrics['sharpe_ratio']:8.2f}")
print(f"• Sortino Ratio:          {metrics['sortino_ratio']:8.2f}")
print(f"• Max Drawdown:           {metrics['max_drawdown']*100:8.2f}% (Peak: {metrics['max_drawdown_peak_date']}, Trough: {metrics['max_drawdown_trough_date']})")
print(f"• Calmar Ratio:           {metrics['calmar_ratio']:8.2f}")
print(f"• Win Rate:               {metrics['win_rate']*100:8.2f}%")
print(f"• Profit Factor:          {metrics['profit_factor']:8.2f}")
print(f"• Trading Days:           {metrics['trading_days']:8d}")

if 'benchmark' in metrics:
    bm = metrics['benchmark']
    print("\n📌 SO SÁNH VỚI CHỈ SỐ VN30 BENCHMARK:")
    print(f"• VN30 Return:            {bm['cumulative_return']*100:8.2f}%")
    print(f"• Excess Return (Alpha):  {bm['alpha']*100:8.2f}%")
    print(f"• VN30 Sharpe Ratio:      {bm['sharpe_ratio']:8.2f}")


# ## 5. Visualizations & Dashboard

# In[6]:


from model_engine.analysis.visualization import ModelVisualizer

visualizer = ModelVisualizer()

# 1. NAV Curve vs Benchmark
nav_path = visualizer.plot_nav_vs_benchmark(df_account, df_benchmark=vn30_df, benchmark_name="VN30", save_name="notebook_nav_vs_vn30.png")

# 2. Underwater / Drawdown Plot
underwater_path = visualizer.plot_underwater(df_account, save_name="notebook_underwater.png")

# 3. Action Distribution per Ticker
action_path = visualizer.plot_action_distribution(df_actions, save_name="notebook_action_dist.png")

# 4. Full Evaluation Dashboard
dashboard_path, _ = visualizer.plot_summary_dashboard(df_account, df_actions, df_benchmark=vn30_df, benchmark_name="VN30", save_name="notebook_summary_dashboard.png")

print("\n🖼️ ĐÃ TẠO VÀ LƯU CÁC BIỂU ĐỒ PHÂN TÍCH THÀNH CÔNG:")
print(f"• Dashboard Plot:   {dashboard_path}")
print(f"• NAV Curve:        {nav_path}")
print(f"• Underwater Plot:  {underwater_path}")
print(f"• Action Dist Plot: {action_path}")
