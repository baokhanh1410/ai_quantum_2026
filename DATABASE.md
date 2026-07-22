# Database Schema Design (3NF)

## 1. Tổng quan

Cơ sở dữ liệu được thiết kế theo chuẩn **Third Normal Form (3NF)** nhằm:

* Loại bỏ dư thừa dữ liệu.
* Giảm chi phí cập nhật.
* Dễ dàng mở rộng thêm tài sản, chỉ báo và dữ liệu vĩ mô.
* Tách biệt tầng lưu trữ dữ liệu (OLTP) với tầng Feature Engineering phục vụ Reinforcement Learning.

> **Lưu ý**
>
> MySQL chỉ lưu dữ liệu đã chuẩn hóa.
>


---

# 2. Asset Management

## 2.1 asset_classes

Lưu các nhóm tài sản.

| Column              | Type        | Key    | Description                  |
| ------------------- | ----------- | ------ | ---------------------------- |
| id                  | TINYINT     | PK     | ID nhóm tài sản              |
| name                | VARCHAR(50) | UNIQUE | STOCK, BOND_FUND, GOLD, CASH |
| description         | TEXT        |        | Mô tả                        |
| settlement_type     | VARCHAR(20) |        | T+0, T+1, T+2                |
| default_locked_days | TINYINT     |        | Số ngày khóa mặc định        |

---

## 2.2 tickers

Lưu toàn bộ tài sản.

| Column         | Type         | Key                   | Description                 |
| -------------- | ------------ | --------------------- | --------------------------- |
| id             | SMALLINT     | PK                    | ID tài sản                  |
| asset_class_id | TINYINT      | FK → asset_classes.id | Nhóm tài sản                |
| symbol         | VARCHAR(20)  | UNIQUE                | Mã giao dịch                |
| name           | VARCHAR(255) |                       | Tên tài sản                 |
| exchange       | VARCHAR(50)  |                       | HOSE, HNX, UPCOM, TCBF, SJC |
| active         | BOOLEAN      |                       | Trạng thái giao dịch        |
| created_at     | TIMESTAMP    |                       | Thời gian tạo               |

---

## 2.3 ticker_metadata

Thông tin vi cấu trúc của từng tài sản.

| Column           | Type          | Key    |
| ---------------- | ------------- | ------ |
| ticker_id        | SMALLINT      | PK, FK |
| settlement_days  | TINYINT       |        |
| liquidity_type   | VARCHAR(30)   |        |
| lot_size         | INT           |        |
| tick_size        | DECIMAL(12,6) |        |
| price_limit_up   | FLOAT         |        |
| price_limit_down | FLOAT         |        |
| trading_fee      | FLOAT         |        |
| allow_short      | BOOLEAN       |        |

---

## 2.4 stock_sector_mappings

Bảng ánh xạ cổ phiếu thuộc về (các) nhóm ngành nào (N-N mapping).
Cho phép một cổ phiếu lẻ (ví dụ FPT) ánh xạ vào nhiều rổ chỉ số ngành/thị trường (ví dụ VNIT, VN30).

| Column           | Type          | Key    | Description |
| ---------------- | ------------- | ------ | ----------- |
| stock_ticker_id  | SMALLINT      | PK, FK | ID của cổ phiếu lẻ (từ bảng tickers) |
| sector_ticker_id | SMALLINT      | PK, FK | ID của chỉ số ngành (từ bảng tickers) |
| weight           | DECIMAL(5,4)  |        | (Optional) Tỷ trọng của cổ phiếu trong rổ |
| created_at       | TIMESTAMP     |        | Thời gian tạo |

---

# 3. Market Data

## 3.1 ohlcv

Lưu dữ liệu giá lịch sử.

| Column         | Type          | Key |
| -------------- | ------------- | --- |
| id             | BIGINT        | PK  |
| ticker_id      | SMALLINT      | FK  |
| timestamp      | TIMESTAMP     |     |
| open           | DECIMAL(18,6) |     |
| high           | DECIMAL(18,6) |     |
| low            | DECIMAL(18,6) |     |
| close          | DECIMAL(18,6) |     |
| adjusted_close | DECIMAL(18,6) |     |
| volume         | BIGINT        |     |
| source         | VARCHAR(100)  |     |
| data_quality   | VARCHAR(30)   |     |
| created_at     | TIMESTAMP     |     |

---

# 4. Technical Indicators

## 4.1 indicator_types

Danh mục các chỉ báo kỹ thuật.

| Column      | Type        | Key    |
| ----------- | ----------- | ------ |
| id          | SMALLINT    | PK     |
| name        | VARCHAR(50) | UNIQUE |
| category    | VARCHAR(50) |        |
| window_size | SMALLINT    |        |
| description | TEXT        |        |

Ví dụ:

* RSI
* PPO
* CCI
* ADX
* ATR
* Rolling Volatility
* Log Return
* Normalized Volume

---

## 4.2 technical_indicator_values

Giá trị từng chỉ báo theo từng tài sản.

| Column            | Type      | Key |
| ----------------- | --------- | --- |
| id                | BIGINT    | PK  |
| ticker_id         | SMALLINT  | FK  |
| indicator_type_id | SMALLINT  | FK  |
| timestamp         | TIMESTAMP |     |
| value             | DOUBLE    |     |
| created_at        | TIMESTAMP |     |

---

# 5. Macro Data

## 5.1 macro_types

Danh mục biến vĩ mô.

| Column      | Type        | Key    |
| ----------- | ----------- | ------ |
| id          | SMALLINT    | PK     |
| name        | VARCHAR(50) | UNIQUE |
| unit        | VARCHAR(30) |        |
| description | TEXT        |        |

Ví dụ

* Interbank Rate
* Risk Free Rate
* DXY
* USD/VND
* Gold Premium
* Government Bond Yield 3Y
* Government Bond Yield 10Y
* Yield Curve Slope
* Inflation

---

## 5.2 macro_values

Giá trị dữ liệu vĩ mô.

| Column        | Type      | Key |
| ------------- | --------- | --- |
| id            | BIGINT    | PK  |
| macro_type_id | SMALLINT  | FK  |
| timestamp     | DATE      |     |
| value         | DOUBLE    |     |
| created_at    | TIMESTAMP |     |

---

# 6. Sector Data

## 6.1 sector_types

Danh mục ngành.

| Column      | Type         |
| ----------- | ------------ |
| id          | SMALLINT     |
| name        | VARCHAR(100) |
| description | TEXT         |

---

## 6.2 sector_values

Thông tin chỉ số ngành.

| Column         | Type     |
| -------------- | -------- |
| id             | BIGINT   |
| sector_type_id | SMALLINT |
| timestamp      | DATE     |
| sector_return  | DOUBLE   |
| volume         | BIGINT   |

---

# 7. User

## 7.1 users

| Column        | Type         |
| ------------- | ------------ |
| id            | INT          |
| username      | VARCHAR(50)  |
| password_hash | VARCHAR(255) |
| email         | VARCHAR(100) |
| status        | VARCHAR(20)  |
| created_at    | TIMESTAMP    |

---

## 7.2 user_wallets

| Column          | Type          |
| --------------- | ------------- |
| wallet_id       | INT           |
| user_id         | INT           |
| initial_capital | DECIMAL(18,2) |
| current_cash    | DECIMAL(18,2) |
| created_at      | TIMESTAMP     |

---

# 8. Portfolio

## 8.1 portfolio_positions

Vị thế hiện tại của danh mục.

| Column             | Type          |
| ------------------ | ------------- |
| wallet_id          | INT           |
| ticker_id          | SMALLINT      |
| quantity_available | DECIMAL(18,6) |
| average_cost       | DECIMAL(18,6) |
| market_value       | DECIMAL(18,6) |
| unrealized_pnl     | DECIMAL(18,6) |
| realized_pnl       | DECIMAL(18,6) |
| updated_at         | TIMESTAMP     |

---

## 8.2 position_locks

Mô phỏng cơ chế thanh khoản T+0, T+1 và T+2.

| Column      | Type          |
| ----------- | ------------- |
| id          | BIGINT        |
| wallet_id   | INT           |
| ticker_id   | SMALLINT      |
| quantity    | DECIMAL(18,6) |
| lock_reason | VARCHAR(50)   |
| lock_start  | DATE          |
| unlock_date | DATE          |
| status      | VARCHAR(20)   |
| created_at  | TIMESTAMP     |

---

## 8.3 portfolio_snapshots

Lưu trạng thái danh mục theo thời gian.

| Column               | Type          |
| -------------------- | ------------- |
| id                   | BIGINT        |
| wallet_id            | INT           |
| snapshot_time        | TIMESTAMP     |
| total_nav            | DECIMAL(18,2) |
| cash                 | DECIMAL(18,2) |
| invested_value       | DECIMAL(18,2) |
| pnl_percentage       | DOUBLE        |
| sharpe_ratio         | DOUBLE        |
| max_drawdown         | DOUBLE        |
| portfolio_volatility | DOUBLE        |

---

## 8.4 order_history

Lưu lịch sử giao dịch.

| Column          | Type               |
| --------------- | ------------------ |
| id              | BIGINT             |
| wallet_id       | INT                |
| ticker_id       | SMALLINT           |
| order_type      | ENUM('BUY','SELL') |
| quantity        | DECIMAL(18,6)      |
| order_price     | DECIMAL(18,6)      |
| execution_price | DECIMAL(18,6)      |
| fee             | DECIMAL(18,6)      |
| slippage        | DOUBLE             |
| status          | VARCHAR(30)        |
| created_at      | TIMESTAMP          |

---

# 9. Reinforcement Learning

## 9.1 rl_dataset_versions

Quản lý phiên bản dữ liệu huấn luyện.

| Column      | Type        |
| ----------- | ----------- |
| id          | INT         |
| version     | VARCHAR(30) |
| description | TEXT        |
| created_at  | TIMESTAMP   |

---

## 9.2 rl_model_trainings

Thông tin huấn luyện mô hình.

| Column             | Type         |
| ------------------ | ------------ |
| id                 | INT          |
| model_name         | VARCHAR(100) |
| algorithm          | VARCHAR(50)  |
| dataset_version    | VARCHAR(30)  |
| hyperparameters    | JSON         |
| train_start_date   | DATE         |
| train_end_date     | DATE         |
| final_sharpe_ratio | DOUBLE       |
| checkpoint_path    | VARCHAR(255) |
| created_at         | TIMESTAMP    |

---

## 9.3 rl_episodes

Thông tin từng Episode.

| Column         | Type      |
| -------------- | --------- |
| id             | BIGINT    |
| training_id    | INT       |
| episode_number | INT       |
| start_date     | DATE      |
| end_date       | DATE      |
| total_reward   | DOUBLE    |
| sharpe_ratio   | DOUBLE    |
| max_drawdown   | DOUBLE    |
| created_at     | TIMESTAMP |

---

## 9.4 rl_steps

Thông tin từng bước trong Episode.

| Column     | Type      |
| ---------- | --------- |
| id         | BIGINT    |
| episode_id | BIGINT    |
| step_index | INT       |
| timestamp  | TIMESTAMP |
| reward     | DOUBLE    |
| done       | BOOLEAN   |
| created_at | TIMESTAMP |

---

## 9.5 rl_action_allocations

Phân bổ trọng số danh mục tại mỗi bước.

| Column           | Type     |
| ---------------- | -------- |
| id               | BIGINT   |
| step_id          | BIGINT   |
| ticker_id        | SMALLINT |
| portfolio_weight | DOUBLE   |

Ví dụ:

| Step | Ticker | Weight |
| ---- | ------ | ------ |
| 150  | FPT    | 0.20   |
| 150  | VCB    | 0.15   |
| 150  | TCBF   | 0.35   |
| 150  | GOLD   | 0.20   |
| 150  | CASH   | 0.10   |

---

## 9.6 reward_types

Danh mục thành phần Reward.

| Column      | Type        |
| ----------- | ----------- |
| id          | SMALLINT    |
| name        | VARCHAR(50) |
| description | TEXT        |

Ví dụ:

* Return Reward
* Risk Penalty
* Liquidity Penalty
* Transaction Cost
* Drawdown Penalty

---

## 9.7 reward_values

Giá trị Reward của từng thành phần.

| Column         | Type     |
| -------------- | -------- |
| id             | BIGINT   |
| step_id        | BIGINT   |
| reward_type_id | SMALLINT |
| value          | DOUBLE   |

---

# 10. System

## 10.1 system_config

Thông tin cấu hình hệ thống.

| Column      | Type         |
| ----------- | ------------ |
| id          | INT          |
| key         | VARCHAR(100) |
| value       | TEXT         |
| description | TEXT         |

---

# 11. Quan hệ giữa các bảng

```text
asset_classes
      │
      └──────── tickers
                    │
          ┌─────────┴──────────────┐
          │                        │
          ▼                        ▼
ticker_metadata                 ohlcv
                                     │
                 ┌───────────────────┴────────────────────┐
                 ▼                                        ▼
        technical_indicator_values                 indicator_types
                 │
                 ▼
           (ETL Pipeline)
                 │
                 ▼
      Feature Vector (DuckDB / Parquet)

macro_types
      │
      ▼
macro_values

sector_types
      │
      ▼
sector_values

users
      │
user_wallets
      │
portfolio_positions
      │
position_locks
      │
portfolio_snapshots
      │
order_history

rl_dataset_versions
      │
rl_model_trainings
      │
rl_episodes
      │
rl_steps
      ├──────────── rl_action_allocations
      └──────────── reward_values
                           │
                           ▼
                     reward_types
```

# 12. Luồng dữ liệu

```text
Raw Data Sources
       │
       ▼
OHLCV + Macro + Sector
       │
       ▼
Indicator Calculation
       │
       ▼
technical_indicator_values
       │
       ▼
ETL Pipeline
(DuckDB / Parquet)
       │
       ▼
State Vector
       │
       ▼
RL Agent
       │
       ▼
Action
       │
       ▼
Portfolio + Orders + Position Locks
       │
       ▼
Reward Calculation
       │
       ▼
Reward Values
```
