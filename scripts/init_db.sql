-- PostgreSQL 初始化腳本
-- RL Market Making 完整資料庫結構
-- 版本: 2.0
-- 日期: 2025-01-16

-- ============================================================
-- 1. 幣種管理表
-- ============================================================

-- 支援的交易對
CREATE TABLE IF NOT EXISTS symbols (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) UNIQUE NOT NULL,
    base_currency VARCHAR(20) NOT NULL,
    quote_currency VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) DEFAULT 'binance',
    is_active BOOLEAN DEFAULT TRUE,
    min_trade_amount DOUBLE PRECISION,
    max_trade_amount DOUBLE PRECISION,
    tick_size DOUBLE PRECISION,
    lot_size DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE symbols IS '交易對基本資訊';
COMMENT ON COLUMN symbols.symbol IS '交易對代碼，例如 BTCUSDT';
COMMENT ON COLUMN symbols.is_active IS '是否啟用交易';
COMMENT ON COLUMN symbols.metadata IS '其他交易所特定的設定';

-- ============================================================
-- 2. 市場資料表
-- ============================================================

-- OHLCV 歷史資料
CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL,
    timeframe VARCHAR(10) NOT NULL,  -- 1m, 5m, 15m, 1h, 1d
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    quote_volume DOUBLE PRECISION,
    trades_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol_id, timestamp, timeframe)
);

COMMENT ON TABLE market_data IS 'OHLCV K線歷史資料';
COMMENT ON COLUMN market_data.timeframe IS 'K線週期';
COMMENT ON COLUMN market_data.quote_volume IS '以計價貨幣計算的成交量';

-- ============================================================
-- 3. 訓練執行表
-- ============================================================

-- 訓練執行主表
CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) UNIQUE NOT NULL,
    symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    symbol VARCHAR(50) NOT NULL,  -- 保留字串欄位以防 symbol 被刪除
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status VARCHAR(50) NOT NULL,  -- running, completed, failed, stopped
    algorithm VARCHAR(50),  -- PPO, A2C, SAC, etc.
    total_timesteps BIGINT,
    total_episodes INTEGER,
    best_reward DOUBLE PRECISION,
    final_reward DOUBLE PRECISION,
    best_pnl DOUBLE PRECISION,
    final_pnl DOUBLE PRECISION,
    config JSONB,
    hyperparameters JSONB,
    environment_config JSONB,
    training_data_info JSONB,  -- 訓練資料範圍等資訊
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE training_runs IS '訓練執行記錄';
COMMENT ON COLUMN training_runs.run_id IS '唯一執行 ID，格式: run_{symbol}_{timestamp}_v{version}';
COMMENT ON COLUMN training_runs.status IS '執行狀態';
COMMENT ON COLUMN training_runs.training_data_info IS '訓練資料的時間範圍、數量等資訊';
COMMENT ON COLUMN training_runs.training_data_info IS '訓練資料的時間範圍、數量等資訊';

-- ============================================================
-- 4. 訓練 Episodes 表
-- ============================================================

-- Episodes 詳細記錄
CREATE TABLE IF NOT EXISTS episodes (
    id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL REFERENCES training_runs(run_id) ON DELETE CASCADE,
    episode_num INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timesteps INTEGER,
    episode_reward DOUBLE PRECISION,
    episode_length INTEGER,
    episode_pnl DOUBLE PRECISION,
    total_trades INTEGER,
    profitable_trades INTEGER,
    win_rate DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    sortino_ratio DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    max_drawdown_pct DOUBLE PRECISION,
    total_fees DOUBLE PRECISION,
    avg_trade_duration DOUBLE PRECISION,
    metrics JSONB,  -- 額外的詳細指標
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, episode_num)
);

COMMENT ON TABLE episodes IS 'Episodes 訓練詳細記錄';
COMMENT ON COLUMN episodes.timesteps IS '該 episode 的 timesteps 數量';
COMMENT ON COLUMN episodes.metrics IS '其他自定義指標，如 inventory, positions 等';

-- ============================================================
-- 5. 模型管理表
-- ============================================================

-- 訓練出的模型
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL REFERENCES training_runs(run_id) ON DELETE CASCADE,
    symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    symbol VARCHAR(50) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_path VARCHAR(500) NOT NULL,
    model_type VARCHAR(50),  -- PPO, A2C, etc.
    framework VARCHAR(50) DEFAULT 'stable-baselines3',
    version VARCHAR(50),
    file_size_bytes BIGINT,
    training_episodes INTEGER,
    training_timesteps BIGINT,
    performance_metrics JSONB,
    config JSONB,
    is_best BOOLEAN DEFAULT FALSE,
    is_deployed BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE models IS '訓練模型註冊表';
COMMENT ON COLUMN models.is_best IS '是否為該幣種的最佳模型';
COMMENT ON COLUMN models.is_deployed IS '是否已部署到生產環境';
COMMENT ON COLUMN models.performance_metrics IS '模型表現指標';

-- ============================================================
-- 6. 回測結果表
-- ============================================================

-- 回測執行記錄
CREATE TABLE IF NOT EXISTS backtest_runs (
    id SERIAL PRIMARY KEY,
    backtest_id VARCHAR(255) UNIQUE NOT NULL,
    model_id INTEGER REFERENCES models(id) ON DELETE CASCADE,
    symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    symbol VARCHAR(50) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    data_start_date TIMESTAMP NOT NULL,
    data_end_date TIMESTAMP NOT NULL,
    initial_balance DOUBLE PRECISION NOT NULL,
    final_balance DOUBLE PRECISION,
    total_pnl DOUBLE PRECISION,
    total_pnl_pct DOUBLE PRECISION,
    total_trades INTEGER,
    profitable_trades INTEGER,
    losing_trades INTEGER,
    win_rate DOUBLE PRECISION,
    avg_profit DOUBLE PRECISION,
    avg_loss DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    sortino_ratio DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    max_drawdown_pct DOUBLE PRECISION,
    total_fees DOUBLE PRECISION,
    avg_trade_duration DOUBLE PRECISION,
    metrics JSONB,
    trades_detail JSONB,  -- 所有交易的詳細記錄
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE backtest_runs IS '回測執行結果';
COMMENT ON COLUMN backtest_runs.profit_factor IS '盈虧比：總盈利/總虧損';
COMMENT ON COLUMN backtest_runs.trades_detail IS '所有交易的詳細記錄（JSON 陣列）';

-- ============================================================
-- 7. 交易記錄表（可選，用於詳細分析）
-- ============================================================

-- 單筆交易記錄
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    backtest_id VARCHAR(255) REFERENCES backtest_runs(backtest_id) ON DELETE CASCADE,
    trade_id VARCHAR(255) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- buy, sell
    order_type VARCHAR(20),  -- market, limit
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_price DOUBLE PRECISION,
    quantity DOUBLE PRECISION NOT NULL,
    pnl DOUBLE PRECISION,
    pnl_pct DOUBLE PRECISION,
    fees DOUBLE PRECISION,
    duration_seconds INTEGER,
    status VARCHAR(20),  -- open, closed
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE trades IS '交易記錄（用於詳細分析）';
COMMENT ON COLUMN trades.side IS '交易方向';
COMMENT ON COLUMN trades.metadata IS '其他交易相關資訊';

-- ============================================================
-- 8. 系統日誌表
-- ============================================================

-- 系統事件日誌
CREATE TABLE IF NOT EXISTS system_logs (
    id BIGSERIAL PRIMARY KEY,
    log_level VARCHAR(20) NOT NULL,  -- DEBUG, INFO, WARNING, ERROR, CRITICAL
    component VARCHAR(100),  -- training, dashboard, api, etc.
    message TEXT NOT NULL,
    details JSONB,
    run_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE system_logs IS '系統事件日誌';

-- ============================================================
-- 索引優化
-- ============================================================

-- symbols 表索引
CREATE INDEX IF NOT EXISTS idx_symbols_symbol ON symbols(symbol);
CREATE INDEX IF NOT EXISTS idx_symbols_active ON symbols(is_active) WHERE is_active = TRUE;

-- market_data 表索引
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_timeframe ON market_data(timeframe);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp DESC);

-- training_runs 表索引
CREATE INDEX IF NOT EXISTS idx_training_runs_symbol ON training_runs(symbol);
CREATE INDEX IF NOT EXISTS idx_training_runs_symbol_id ON training_runs(symbol_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_start_time ON training_runs(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_training_runs_algorithm ON training_runs(algorithm);

-- episodes 表索引
CREATE INDEX IF NOT EXISTS idx_episodes_run_id ON episodes(run_id);
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_reward ON episodes(episode_reward DESC);

-- models 表索引
CREATE INDEX IF NOT EXISTS idx_models_run_id ON models(run_id);
CREATE INDEX IF NOT EXISTS idx_models_symbol ON models(symbol);
CREATE INDEX IF NOT EXISTS idx_models_symbol_id ON models(symbol_id);
CREATE INDEX IF NOT EXISTS idx_models_is_best ON models(is_best) WHERE is_best = TRUE;
CREATE INDEX IF NOT EXISTS idx_models_is_deployed ON models(is_deployed) WHERE is_deployed = TRUE;
CREATE INDEX IF NOT EXISTS idx_models_created ON models(created_at DESC);

-- backtest_runs 表索引
CREATE INDEX IF NOT EXISTS idx_backtest_model_id ON backtest_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_backtest_symbol ON backtest_runs(symbol);
CREATE INDEX IF NOT EXISTS idx_backtest_created ON backtest_runs(created_at DESC);

-- trades 表索引
CREATE INDEX IF NOT EXISTS idx_trades_backtest_id ON trades(backtest_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time DESC);

-- system_logs 表索引
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(log_level);
CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component);
CREATE INDEX IF NOT EXISTS idx_system_logs_created ON system_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_run_id ON system_logs(run_id);

-- ============================================================
-- 觸發器：自動更新 updated_at
-- ============================================================

-- 創建或替換觸發器函數
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 為相關表添加觸發器
DROP TRIGGER IF EXISTS update_symbols_updated_at ON symbols;
CREATE TRIGGER update_symbols_updated_at
    BEFORE UPDATE ON symbols
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_training_runs_updated_at ON training_runs;
CREATE TRIGGER update_training_runs_updated_at
    BEFORE UPDATE ON training_runs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_models_updated_at ON models;
CREATE TRIGGER update_models_updated_at
    BEFORE UPDATE ON models
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 實用視圖
-- ============================================================

-- 最新訓練執行概覽
CREATE OR REPLACE VIEW v_latest_training_runs AS
SELECT 
    tr.run_id,
    tr.symbol,
    s.base_currency,
    s.quote_currency,
    tr.algorithm,
    tr.start_time,
    tr.end_time,
    tr.status,
    tr.best_reward,
    tr.final_pnl,
    tr.total_episodes,
    COUNT(e.id) as episode_count,
    MAX(e.episode_reward) as max_episode_reward,
    AVG(e.episode_reward) as avg_episode_reward,
    AVG(e.win_rate) as avg_win_rate
FROM training_runs tr
LEFT JOIN symbols s ON tr.symbol_id = s.id
LEFT JOIN episodes e ON tr.run_id = e.run_id
GROUP BY tr.id, tr.run_id, tr.symbol, s.base_currency, s.quote_currency,
         tr.algorithm, tr.start_time, tr.end_time, tr.status, 
         tr.best_reward, tr.final_pnl, tr.total_episodes
ORDER BY tr.start_time DESC;

-- 各幣種表現統計
CREATE OR REPLACE VIEW v_symbol_performance AS
SELECT 
    s.symbol,
    s.base_currency,
    s.quote_currency,
    COUNT(DISTINCT tr.run_id) as total_runs,
    COUNT(DISTINCT CASE WHEN tr.status = 'completed' THEN tr.run_id END) as completed_runs,
    COUNT(DISTINCT CASE WHEN tr.status = 'failed' THEN tr.run_id END) as failed_runs,
    AVG(tr.final_pnl) as avg_pnl,
    MAX(tr.final_pnl) as best_pnl,
    MIN(tr.final_pnl) as worst_pnl,
    AVG(tr.total_episodes) as avg_episodes,
    COUNT(m.id) as total_models,
    COUNT(CASE WHEN m.is_best THEN 1 END) as best_models
FROM symbols s
LEFT JOIN training_runs tr ON s.id = tr.symbol_id
LEFT JOIN models m ON s.id = m.symbol_id
WHERE s.is_active = TRUE
GROUP BY s.id, s.symbol, s.base_currency, s.quote_currency
ORDER BY total_runs DESC;

-- 模型表現排行
CREATE OR REPLACE VIEW v_model_leaderboard AS
SELECT 
    m.id,
    m.model_name,
    m.symbol,
    m.model_type,
    m.is_best,
    m.is_deployed,
    m.created_at,
    tr.final_pnl as training_pnl,
    tr.best_reward as training_reward,
    (m.performance_metrics->>'sharpe_ratio')::DOUBLE PRECISION as sharpe_ratio,
    (m.performance_metrics->>'win_rate')::DOUBLE PRECISION as win_rate,
    COUNT(br.id) as backtest_count,
    AVG(br.total_pnl) as avg_backtest_pnl
FROM models m
LEFT JOIN training_runs tr ON m.run_id = tr.run_id
LEFT JOIN backtest_runs br ON m.id = br.model_id
GROUP BY m.id, m.model_name, m.symbol, m.model_type, m.is_best, 
         m.is_deployed, m.created_at, tr.final_pnl, tr.best_reward,
         m.performance_metrics
ORDER BY training_pnl DESC NULLS LAST;

-- 市場資料統計
CREATE OR REPLACE VIEW v_market_data_stats AS
SELECT 
    s.symbol,
    md.timeframe,
    COUNT(*) as data_points,
    MIN(md.timestamp) as earliest_data,
    MAX(md.timestamp) as latest_data,
    AVG(md.volume) as avg_volume,
    AVG(md.high - md.low) as avg_price_range
FROM market_data md
JOIN symbols s ON md.symbol_id = s.id
GROUP BY s.symbol, md.timeframe
ORDER BY s.symbol, md.timeframe;

-- ============================================================
-- 初始資料
-- ============================================================

-- 插入常用交易對
INSERT INTO symbols (symbol, base_currency, quote_currency, exchange, is_active)
VALUES 
    ('BTCUSDT', 'BTC', 'USDT', 'binance', TRUE),
    ('ETHUSDT', 'ETH', 'USDT', 'binance', TRUE),
    ('BNBUSDT', 'BNB', 'USDT', 'binance', TRUE),
    ('SOLUSDT', 'SOL', 'USDT', 'binance', TRUE),
    ('ADAUSDT', 'ADA', 'USDT', 'binance', TRUE)
ON CONFLICT (symbol) DO NOTHING;

-- ============================================================
-- 資料庫版本資訊
-- ============================================================

CREATE TABLE IF NOT EXISTS db_version (
    id SERIAL PRIMARY KEY,
    version VARCHAR(20) NOT NULL,
    description TEXT,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO db_version (version, description)
VALUES ('2.0', '完整重新設計資料庫結構，支援市場資料、訓練、回測等完整功能')
ON CONFLICT DO NOTHING;

-- 授予權限（如果需要）
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rl_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rl_user;
-- GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO rl_user;
