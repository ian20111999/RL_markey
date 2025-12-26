-- PostgreSQL 測試資料插入腳本

-- 1. 插入交易對
INSERT INTO symbols (symbol, base_currency, quote_currency)
VALUES 
    ('BTCUSDT', 'BTC', 'USDT'),
    ('ETHUSDT', 'ETH', 'USDT'),
    ('BNBUSDT', 'BNB', 'USDT'),
    ('SOLUSDT', 'SOL', 'USDT')
ON CONFLICT (symbol) DO NOTHING;

-- 2. 插入市場資料 (以 BTCUSDT 為例)
INSERT INTO market_data (symbol_id, timestamp, open, high, low, close, volume, timeframe)
SELECT 
    (SELECT id FROM symbols WHERE symbol = 'BTCUSDT'),
    NOW() - (n || ' minutes')::INTERVAL,
    95000 + (n % 20 - 10) * 50,
    95000 + (n % 20 - 10) * 50 + 200,
    95000 + (n % 20 - 10) * 50 - 200,
    95000 + (n % 20 - 10) * 50 + 100,
    100.5 + n * 2,
    '1m'
FROM generate_series(0, 99) AS n
ON CONFLICT (symbol_id, timestamp, timeframe) DO NOTHING;

-- 3. 插入訓練記錄
INSERT INTO training_runs (
    run_id, symbol_id, symbol, algorithm, 
    start_time, end_time, status, 
    config, best_reward, final_pnl, total_episodes
)
VALUES (
    'run_btc_' || extract(epoch from NOW())::bigint || '_pg_test',
    (SELECT id FROM symbols WHERE symbol = 'BTCUSDT'),
    'BTCUSDT',
    'PPO',
    NOW() - INTERVAL '2 hours',
    NOW(),
    'completed',
    '{"learning_rate": 0.0003, "n_steps": 2048}'::jsonb,
    1500.50,
    5200.75,
    500
);

-- 4. 插入 Episodes
INSERT INTO episodes (
    run_id, episode_num, timestamp, 
    episode_reward, episode_length, win_rate, 
    sharpe_ratio, max_drawdown, episode_pnl, total_trades
)
SELECT 
    'run_btc_' || extract(epoch from NOW())::bigint || '_pg_test',
    n,
    NOW() - (n * 10 || ' minutes')::INTERVAL,
    100.0 + n * 20,
    1000 + n * 50,
    0.50 + n * 0.03,
    1.5 + n * 0.1,
    -0.10 - n * 0.01,
    200.0 + n * 100,
    25 + n * 5
FROM generate_series(1, 10) AS n
ON CONFLICT (run_id, episode_num) DO NOTHING;

-- 5. 插入模型
INSERT INTO models (
    run_id, symbol_id, symbol, model_name, model_path,
    performance_metrics, is_best
)
VALUES (
    'run_btc_' || extract(epoch from NOW())::bigint || '_pg_test',
    (SELECT id FROM symbols WHERE symbol = 'BTCUSDT'),
    'BTCUSDT',
    'ppo_btc_' || extract(epoch from NOW())::bigint || '_pg',
    'models/ppo_btc_' || extract(epoch from NOW())::bigint || '_pg.zip',
    '{"final_pnl": 5200.75, "sharpe_ratio": 2.5, "max_drawdown": -0.18, "win_rate": 0.68}'::jsonb,
    true
);

-- 查詢結果
\echo '========================================='
\echo '📊 資料庫統計'
\echo '========================================='

SELECT 'symbols' as table_name, COUNT(*) as count FROM symbols
UNION ALL
SELECT 'market_data', COUNT(*) FROM market_data
UNION ALL
SELECT 'training_runs', COUNT(*) FROM training_runs
UNION ALL
SELECT 'episodes', COUNT(*) FROM episodes
UNION ALL
SELECT 'models', COUNT(*) FROM models
UNION ALL
SELECT 'backtest_runs', COUNT(*) FROM backtest_runs
UNION ALL
SELECT 'trades', COUNT(*) FROM trades
UNION ALL
SELECT 'system_logs', COUNT(*) FROM system_logs;

\echo ''
\echo '========================================='
\echo '📈 最新訓練記錄'
\echo '========================================='

SELECT run_id, symbol, algorithm, status, final_pnl
FROM v_latest_training_runs
WHERE symbol = 'BTCUSDT'
LIMIT 5;

\echo ''
\echo '========================================='
\echo '🏆 模型排行榜'
\echo '========================================='

SELECT model_name, symbol, training_pnl
FROM v_model_leaderboard
LIMIT 5;
