-- SQLite 初始化腳本
-- RL Market Making 完整資料庫結構 (SQLite版本)
-- 版本: 2.0
-- 日期: 2025-01-16

-- ============================================================
-- 1. 幣種管理表
-- ============================================================

CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    base_currency TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    exchange TEXT DEFAULT 'binance',
    is_active INTEGER DEFAULT 1,
    min_trade_amount REAL,
    max_trade_amount REAL,
    tick_size REAL,
    lot_size REAL,
    metadata TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 2. 市場資料表
-- ============================================================

CREATE TABLE IF NOT EXISTS market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timeframe TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    quote_volume REAL,
    trades_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE CASCADE,
    UNIQUE(symbol_id, timestamp, timeframe)
);

-- ============================================================
-- 3. 訓練執行表
-- ============================================================

CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    symbol_id INTEGER,
    symbol TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status TEXT NOT NULL,
    algorithm TEXT,
    total_timesteps INTEGER,
    total_episodes INTEGER,
    best_reward REAL,
    final_reward REAL,
    best_pnl REAL,
    final_pnl REAL,
    config TEXT,  -- JSON
    hyperparameters TEXT,  -- JSON
    environment_config TEXT,  -- JSON
    training_data_info TEXT,  -- JSON
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE SET NULL
);

-- ============================================================
-- 4. 訓練 Episodes 表
-- ============================================================

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    episode_num INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timesteps INTEGER,
    episode_reward REAL,
    episode_length INTEGER,
    episode_pnl REAL,
    total_trades INTEGER,
    profitable_trades INTEGER,
    win_rate REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    max_drawdown_pct REAL,
    total_fees REAL,
    avg_trade_duration REAL,
    metrics TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE,
    UNIQUE(run_id, episode_num)
);

-- ============================================================
-- 5. 模型管理表
-- ============================================================

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    symbol_id INTEGER,
    symbol TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_path TEXT NOT NULL,
    model_type TEXT,
    framework TEXT DEFAULT 'stable-baselines3',
    version TEXT,
    file_size_bytes INTEGER,
    training_episodes INTEGER,
    training_timesteps INTEGER,
    performance_metrics TEXT,  -- JSON
    config TEXT,  -- JSON
    is_best INTEGER DEFAULT 0,
    is_deployed INTEGER DEFAULT 0,
    deployed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE SET NULL
);

-- ============================================================
-- 6. 回測結果表
-- ============================================================

CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id TEXT UNIQUE NOT NULL,
    model_id INTEGER,
    symbol_id INTEGER,
    symbol TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    data_start_date TIMESTAMP NOT NULL,
    data_end_date TIMESTAMP NOT NULL,
    initial_balance REAL NOT NULL,
    final_balance REAL,
    total_pnl REAL,
    total_pnl_pct REAL,
    total_trades INTEGER,
    profitable_trades INTEGER,
    losing_trades INTEGER,
    win_rate REAL,
    avg_profit REAL,
    avg_loss REAL,
    profit_factor REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    max_drawdown_pct REAL,
    total_fees REAL,
    avg_trade_duration REAL,
    metrics TEXT,  -- JSON
    trades_detail TEXT,  -- JSON
    config TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
    FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE SET NULL
);

-- ============================================================
-- 7. 交易記錄表
-- ============================================================

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id TEXT,
    trade_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT,
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity REAL NOT NULL,
    pnl REAL,
    pnl_pct REAL,
    fees REAL,
    duration_seconds INTEGER,
    status TEXT,
    metadata TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (backtest_id) REFERENCES backtest_runs(backtest_id) ON DELETE CASCADE
);

-- ============================================================
-- 8. 系統日誌表
-- ============================================================

CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_level TEXT NOT NULL,
    component TEXT,
    message TEXT NOT NULL,
    details TEXT,  -- JSON
    run_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 索引優化
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_symbols_symbol ON symbols(symbol);
CREATE INDEX IF NOT EXISTS idx_symbols_active ON symbols(is_active) WHERE is_active = 1;

CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_timeframe ON market_data(timeframe);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_training_runs_symbol ON training_runs(symbol);
CREATE INDEX IF NOT EXISTS idx_training_runs_symbol_id ON training_runs(symbol_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_start_time ON training_runs(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_training_runs_algorithm ON training_runs(algorithm);

CREATE INDEX IF NOT EXISTS idx_episodes_run_id ON episodes(run_id);
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_reward ON episodes(episode_reward DESC);

CREATE INDEX IF NOT EXISTS idx_models_run_id ON models(run_id);
CREATE INDEX IF NOT EXISTS idx_models_symbol ON models(symbol);
CREATE INDEX IF NOT EXISTS idx_models_symbol_id ON models(symbol_id);
CREATE INDEX IF NOT EXISTS idx_models_is_best ON models(is_best) WHERE is_best = 1;
CREATE INDEX IF NOT EXISTS idx_models_is_deployed ON models(is_deployed) WHERE is_deployed = 1;
CREATE INDEX IF NOT EXISTS idx_models_created ON models(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_backtest_model_id ON backtest_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_backtest_symbol ON backtest_runs(symbol);
CREATE INDEX IF NOT EXISTS idx_backtest_created ON backtest_runs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_trades_backtest_id ON trades(backtest_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time DESC);

CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(log_level);
CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component);
CREATE INDEX IF NOT EXISTS idx_system_logs_created ON system_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_run_id ON system_logs(run_id);

-- ============================================================
-- 觸發器：自動更新 updated_at
-- ============================================================

CREATE TRIGGER IF NOT EXISTS update_symbols_updated_at
AFTER UPDATE ON symbols
FOR EACH ROW
BEGIN
    UPDATE symbols SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_training_runs_updated_at
AFTER UPDATE ON training_runs
FOR EACH ROW
BEGIN
    UPDATE training_runs SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_models_updated_at
AFTER UPDATE ON models
FOR EACH ROW
BEGIN
    UPDATE models SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;

-- ============================================================
-- 實用視圖
-- ============================================================

CREATE VIEW IF NOT EXISTS v_latest_training_runs AS
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
GROUP BY tr.id
ORDER BY tr.start_time DESC;

CREATE VIEW IF NOT EXISTS v_symbol_performance AS
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
    COUNT(CASE WHEN m.is_best = 1 THEN 1 END) as best_models
FROM symbols s
LEFT JOIN training_runs tr ON s.id = tr.symbol_id
LEFT JOIN models m ON s.id = m.symbol_id
WHERE s.is_active = 1
GROUP BY s.id
ORDER BY total_runs DESC;

CREATE VIEW IF NOT EXISTS v_model_leaderboard AS
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
    json_extract(m.performance_metrics, '$.sharpe_ratio') as sharpe_ratio,
    json_extract(m.performance_metrics, '$.win_rate') as win_rate,
    COUNT(br.id) as backtest_count,
    AVG(br.total_pnl) as avg_backtest_pnl
FROM models m
LEFT JOIN training_runs tr ON m.run_id = tr.run_id
LEFT JOIN backtest_runs br ON m.id = br.model_id
GROUP BY m.id
ORDER BY training_pnl DESC;

CREATE VIEW IF NOT EXISTS v_market_data_stats AS
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
