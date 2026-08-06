-- =====================================================================
-- Database Schema (3NF) - RL Portfolio Trading System
-- Generated from Database_Schema_Data_Dictionary
-- Engine: InnoDB | Charset: utf8mb4
-- =====================================================================

CREATE DATABASE IF NOT EXISTS ai_quantum_2026
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE ai_quantum_2026;

SET FOREIGN_KEY_CHECKS = 0;

-- =====================================================================
-- 2. ASSET MANAGEMENT
-- =====================================================================

CREATE TABLE asset_classes (
    id                   TINYINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name                 VARCHAR(50) NOT NULL UNIQUE COMMENT 'STOCK_HOSE, STOCK_HNX, STOCK_UPCOM, BOND_FUND, GOLD, CASH, SECTOR_INDEX, MACRO_INDEX'
) ENGINE=InnoDB;

-- Initial Seed Data for asset_classes
INSERT INTO asset_classes (id, name) VALUES
(1, 'STOCK_HOSE'),
(2, 'STOCK_HNX'),
(3, 'STOCK_UPCOM'),
(4, 'BOND_FUND'),
(5, 'GOLD'),
(6, 'CASH'),
(7, 'SECTOR_INDEX'),
(8, 'MACRO_INDEX')
ON DUPLICATE KEY UPDATE 
    name = VALUES(name);

CREATE TABLE asset_class_metadata (
    asset_class_id      TINYINT UNSIGNED PRIMARY KEY,
    description         TEXT,
    settlement_type     VARCHAR(20) COMMENT 'T+0, T+1, T+2',
    default_locked_days TINYINT UNSIGNED DEFAULT 0,
    price_limit_ratio   FLOAT COMMENT 'Bien do gia tran/san: 0.07 (HOSE), 0.10 (HNX), 0.15 (UPCOM), NULL (Gold/Bond)',
    default_lot_size    INT UNSIGNED DEFAULT 100 COMMENT 'Lo giao dich mac dinh: 100 hoac 1',
    default_trading_fee FLOAT DEFAULT 0.001 COMMENT 'Phi giao dich mac dinh',
    allow_short         BOOLEAN DEFAULT FALSE COMMENT 'Cho phep ban khong hay khong',
    handler             VARCHAR(50) COMMENT 'Module crawler/API handler',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_ac_metadata_class
        FOREIGN KEY (asset_class_id) REFERENCES asset_classes(id)
        ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB;

-- Initial Seed Data for asset_class_metadata
INSERT INTO asset_class_metadata (asset_class_id, description, settlement_type, default_locked_days, price_limit_ratio, default_lot_size, default_trading_fee, allow_short, handler) VALUES
(1, 'HOSE Listed Equities',                'T+2', 2, 0.07, 100, 0.001, FALSE, 'vnstock'),
(2, 'HNX Listed Equities',                 'T+2', 2, 0.10, 100, 0.001, FALSE, 'vnstock'),
(3, 'UPCoM Listed Equities',               'T+2', 2, 0.15, 100, 0.001, FALSE, 'vnstock'),
(4, 'Bond Fund Certificates',              'T+2', 2, NULL, 1,   0.000, FALSE, 'techcom_capital'),
(5, 'SJC Physical Gold',                   'T+0', 0, NULL, 1,   0.000, FALSE, 'sjc_crawler'),
(6, 'Cash / Risk-Free Interest',           'T+0', 0, 0.00, 1,   0.000, FALSE, 'sbv_crawler'),
(7, 'HOSE Sector & Market Indices',        'T+2', 2, 0.07, 1,   0.000, FALSE, 'vndirect'),
(8, 'Macro Indicators, FX Rates & Yields', 'T+0', 0, NULL, 1,   0.000, FALSE, 'yahoo_finance')
ON DUPLICATE KEY UPDATE 
    description = VALUES(description),
    settlement_type = VALUES(settlement_type),
    default_locked_days = VALUES(default_locked_days),
    price_limit_ratio = VALUES(price_limit_ratio),
    default_lot_size = VALUES(default_lot_size),
    default_trading_fee = VALUES(default_trading_fee),
    allow_short = VALUES(allow_short),
    handler = VALUES(handler);




CREATE TABLE tickers (
    id              SMALLINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    asset_class_id  TINYINT UNSIGNED NOT NULL,
    symbol          VARCHAR(20) NOT NULL UNIQUE,
    name            VARCHAR(255),
    exchange        VARCHAR(50) COMMENT 'HOSE, HNX, UPCOM, TCBF, SJC',
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_tickers_asset_class
        FOREIGN KEY (asset_class_id) REFERENCES asset_classes(id)
        ON UPDATE CASCADE ON DELETE RESTRICT
) ENGINE=InnoDB;

CREATE TABLE ticker_metadata (
    ticker_id          SMALLINT UNSIGNED PRIMARY KEY,
    settlement_days     TINYINT UNSIGNED,
    liquidity_type      VARCHAR(30),
    lot_size            INT UNSIGNED,
    tick_size           DECIMAL(12,6),
    price_limit_up      FLOAT,
    price_limit_down    FLOAT,
    trading_fee         FLOAT,
    allow_short         BOOLEAN DEFAULT FALSE,
    CONSTRAINT fk_metadata_ticker
        FOREIGN KEY (ticker_id) REFERENCES tickers(id)
        ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE stock_sector_mappings (
    stock_ticker_id   SMALLINT UNSIGNED,
    sector_ticker_id  SMALLINT UNSIGNED,
    weight            DECIMAL(5,4) DEFAULT 1.0,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_ticker_id, sector_ticker_id),
    CONSTRAINT fk_mapping_stock
        FOREIGN KEY (stock_ticker_id) REFERENCES tickers(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_mapping_sector
        FOREIGN KEY (sector_ticker_id) REFERENCES tickers(id)
        ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB;

-- =====================================================================
-- 3. MARKET DATA
-- =====================================================================

CREATE TABLE ohlcv (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ticker_id       SMALLINT UNSIGNED NOT NULL,
    timestamp       TIMESTAMP NOT NULL,
    open            DECIMAL(18,6),
    high            DECIMAL(18,6),
    low             DECIMAL(18,6),
    close           DECIMAL(18,6),
    adjusted_close  DECIMAL(18,6),
    volume          BIGINT UNSIGNED,
    source          VARCHAR(100),
    data_quality    VARCHAR(30),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_ohlcv_ticker
        FOREIGN KEY (ticker_id) REFERENCES tickers(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE KEY uq_ohlcv_ticker_time (ticker_id, timestamp)
) ENGINE=InnoDB;

-- =====================================================================
-- 4. TECHNICAL INDICATORS
-- =====================================================================

CREATE TABLE indicator_types (
    id            SMALLINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name          VARCHAR(50) NOT NULL UNIQUE COMMENT 'RSI, PPO, CCI, ADX, ATR, Rolling Volatility, Log Return, Normalized Volume',
    category      VARCHAR(50),
    window_size   SMALLINT UNSIGNED,
    description   TEXT
) ENGINE=InnoDB;

CREATE TABLE technical_indicator_values (
    id                  BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ticker_id           SMALLINT UNSIGNED NOT NULL,
    indicator_type_id   SMALLINT UNSIGNED NOT NULL,
    timestamp           TIMESTAMP NOT NULL,
    value               DOUBLE,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_tiv_ticker
        FOREIGN KEY (ticker_id) REFERENCES tickers(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_tiv_indicator_type
        FOREIGN KEY (indicator_type_id) REFERENCES indicator_types(id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    UNIQUE KEY uq_tiv (ticker_id, indicator_type_id, timestamp)
) ENGINE=InnoDB;

-- =====================================================================
-- 5. MACRO DATA
-- =====================================================================

CREATE TABLE macro_types (
    id            SMALLINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name          VARCHAR(50) NOT NULL UNIQUE COMMENT 'Interbank Rate, Risk Free Rate, DXY, USD/VND, Gold Premium, Gov Bond Yield 3Y/10Y, Yield Curve Slope, Inflation',
    unit          VARCHAR(30),
    description   TEXT
) ENGINE=InnoDB;

CREATE TABLE macro_values (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    macro_type_id   SMALLINT UNSIGNED NOT NULL,
    timestamp       DATE NOT NULL,
    value           DOUBLE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_macro_values_type
        FOREIGN KEY (macro_type_id) REFERENCES macro_types(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE KEY uq_macro_values (macro_type_id, timestamp)
) ENGINE=InnoDB;

-- =====================================================================
-- 6. SECTOR DATA
-- =====================================================================

CREATE TABLE sector_types (
    id            SMALLINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    description   TEXT
) ENGINE=InnoDB;

CREATE TABLE sector_values (
    id               BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    sector_type_id   SMALLINT UNSIGNED NOT NULL,
    timestamp        DATE NOT NULL,
    sector_return    DOUBLE,
    volume           BIGINT UNSIGNED,
    CONSTRAINT fk_sector_values_type
        FOREIGN KEY (sector_type_id) REFERENCES sector_types(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE KEY uq_sector_values (sector_type_id, timestamp)
) ENGINE=InnoDB;

-- =====================================================================
-- 7. USER
-- =====================================================================

CREATE TABLE users (
    id             INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    username       VARCHAR(50) NOT NULL UNIQUE,
    password_hash  VARCHAR(255) NOT NULL,
    email          VARCHAR(100) NOT NULL UNIQUE,
    status         VARCHAR(20) DEFAULT 'ACTIVE',
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE user_wallets (
    wallet_id         INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id           INT UNSIGNED NOT NULL,
    initial_capital   DECIMAL(18,2) NOT NULL DEFAULT 0,
    current_cash      DECIMAL(18,2) NOT NULL DEFAULT 0,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_wallet_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB;

-- =====================================================================
-- 8. PORTFOLIO
-- =====================================================================

CREATE TABLE portfolio_positions (
    wallet_id             INT UNSIGNED NOT NULL,
    ticker_id             SMALLINT UNSIGNED NOT NULL,
    quantity_available    DECIMAL(18,6) DEFAULT 0,
    average_cost          DECIMAL(18,6) DEFAULT 0,
    market_value          DECIMAL(18,6) DEFAULT 0,
    unrealized_pnl        DECIMAL(18,6) DEFAULT 0,
    realized_pnl          DECIMAL(18,6) DEFAULT 0,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (wallet_id, ticker_id),
    CONSTRAINT fk_position_wallet
        FOREIGN KEY (wallet_id) REFERENCES user_wallets(wallet_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_position_ticker
        FOREIGN KEY (ticker_id) REFERENCES tickers(id)
        ON UPDATE CASCADE ON DELETE RESTRICT
) ENGINE=InnoDB;

CREATE TABLE position_locks (
    id            BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    wallet_id     INT UNSIGNED NOT NULL,
    ticker_id     SMALLINT UNSIGNED NOT NULL,
    quantity      DECIMAL(18,6) NOT NULL,
    lock_reason   VARCHAR(50) COMMENT 'Mo phong co che thanh khoan T+0, T+1, T+2',
    lock_start    DATE NOT NULL,
    unlock_date   DATE NOT NULL,
    status        VARCHAR(20) DEFAULT 'LOCKED',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_lock_wallet
        FOREIGN KEY (wallet_id) REFERENCES user_wallets(wallet_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_lock_ticker
        FOREIGN KEY (ticker_id) REFERENCES tickers(id)
        ON UPDATE CASCADE ON DELETE RESTRICT
) ENGINE=InnoDB;

CREATE TABLE portfolio_snapshots (
    id                     BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    wallet_id              INT UNSIGNED NOT NULL,
    snapshot_time          TIMESTAMP NOT NULL,
    total_nav              DECIMAL(18,2),
    cash                   DECIMAL(18,2),
    invested_value         DECIMAL(18,2),
    pnl_percentage         DOUBLE,
    sharpe_ratio           DOUBLE,
    max_drawdown           DOUBLE,
    portfolio_volatility   DOUBLE,
    CONSTRAINT fk_snapshot_wallet
        FOREIGN KEY (wallet_id) REFERENCES user_wallets(wallet_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE KEY uq_snapshot (wallet_id, snapshot_time)
) ENGINE=InnoDB;

CREATE TABLE order_history (
    id                BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    wallet_id         INT UNSIGNED NOT NULL,
    ticker_id         SMALLINT UNSIGNED NOT NULL,
    order_type        ENUM('BUY','SELL') NOT NULL,
    quantity          DECIMAL(18,6) NOT NULL,
    order_price       DECIMAL(18,6),
    execution_price   DECIMAL(18,6),
    fee               DECIMAL(18,6) DEFAULT 0,
    slippage          DOUBLE DEFAULT 0,
    status            VARCHAR(30) DEFAULT 'PENDING',
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_order_wallet
        FOREIGN KEY (wallet_id) REFERENCES user_wallets(wallet_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_order_ticker
        FOREIGN KEY (ticker_id) REFERENCES tickers(id)
        ON UPDATE CASCADE ON DELETE RESTRICT
) ENGINE=InnoDB;

-- =====================================================================
-- 9. REINFORCEMENT LEARNING
-- =====================================================================

CREATE TABLE rl_dataset_versions (
    id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    version       VARCHAR(30) NOT NULL UNIQUE,
    description   TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE rl_model_trainings (
    id                   INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    model_name           VARCHAR(100) NOT NULL,
    algorithm            VARCHAR(50),
    dataset_version      VARCHAR(30),
    hyperparameters      JSON,
    train_start_date     DATE,
    train_end_date       DATE,
    final_sharpe_ratio   DOUBLE,
    checkpoint_path      VARCHAR(255),
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_training_dataset_version
        FOREIGN KEY (dataset_version) REFERENCES rl_dataset_versions(version)
        ON UPDATE CASCADE ON DELETE RESTRICT
) ENGINE=InnoDB;

CREATE TABLE rl_episodes (
    id               BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    training_id      INT UNSIGNED NOT NULL,
    episode_number   INT UNSIGNED NOT NULL,
    start_date       DATE,
    end_date         DATE,
    total_reward     DOUBLE,
    sharpe_ratio     DOUBLE,
    max_drawdown     DOUBLE,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_episode_training
        FOREIGN KEY (training_id) REFERENCES rl_model_trainings(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE KEY uq_episode (training_id, episode_number)
) ENGINE=InnoDB;

CREATE TABLE rl_steps (
    id            BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    episode_id    BIGINT UNSIGNED NOT NULL,
    step_index    INT UNSIGNED NOT NULL,
    timestamp     TIMESTAMP,
    reward        DOUBLE,
    done          BOOLEAN DEFAULT FALSE,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_step_episode
        FOREIGN KEY (episode_id) REFERENCES rl_episodes(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE KEY uq_step (episode_id, step_index)
) ENGINE=InnoDB;

CREATE TABLE rl_action_allocations (
    id                  BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    step_id             BIGINT UNSIGNED NOT NULL,
    ticker_id           SMALLINT UNSIGNED NOT NULL,
    portfolio_weight    DOUBLE NOT NULL,
    CONSTRAINT fk_alloc_step
        FOREIGN KEY (step_id) REFERENCES rl_steps(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_alloc_ticker
        FOREIGN KEY (ticker_id) REFERENCES tickers(id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    UNIQUE KEY uq_alloc (step_id, ticker_id)
) ENGINE=InnoDB;

CREATE TABLE reward_types (
    id            SMALLINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name          VARCHAR(50) NOT NULL UNIQUE COMMENT 'Return Reward, Risk Penalty, Liquidity Penalty, Transaction Cost, Drawdown Penalty',
    description   TEXT
) ENGINE=InnoDB;

CREATE TABLE reward_values (
    id               BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    step_id          BIGINT UNSIGNED NOT NULL,
    reward_type_id   SMALLINT UNSIGNED NOT NULL,
    value            DOUBLE,
    CONSTRAINT fk_reward_step
        FOREIGN KEY (step_id) REFERENCES rl_steps(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT fk_reward_type
        FOREIGN KEY (reward_type_id) REFERENCES reward_types(id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    UNIQUE KEY uq_reward (step_id, reward_type_id)
) ENGINE=InnoDB;

-- =====================================================================
-- 10. SYSTEM
-- =====================================================================

CREATE TABLE system_config (
    id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    `key`         VARCHAR(100) NOT NULL UNIQUE,
    value         TEXT,
    description   TEXT
) ENGINE=InnoDB;

SET FOREIGN_KEY_CHECKS = 1;

-- =====================================================================
-- Recommended indexes for common query patterns
-- =====================================================================
CREATE INDEX idx_ohlcv_timestamp ON ohlcv (timestamp);
CREATE INDEX idx_tiv_timestamp ON technical_indicator_values (timestamp);
CREATE INDEX idx_macro_values_timestamp ON macro_values (timestamp);
CREATE INDEX idx_sector_values_timestamp ON sector_values (timestamp);
CREATE INDEX idx_order_history_wallet ON order_history (wallet_id, created_at);
CREATE INDEX idx_position_locks_unlock ON position_locks (unlock_date, status);
CREATE INDEX idx_rl_steps_episode ON rl_steps (episode_id, timestamp);