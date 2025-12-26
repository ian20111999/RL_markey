#!/usr/bin/env python3
"""
PostgreSQL 資料庫測試腳本 - 插入測試資料並驗證
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

# 加入專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 設定使用 PostgreSQL
os.environ['DB_TYPE'] = 'postgresql'
os.environ['POSTGRES_HOST'] = 'localhost'
os.environ['POSTGRES_PORT'] = '5432'
os.environ['POSTGRES_DB'] = 'rl_market'
os.environ['POSTGRES_USER'] = 'rl_user'
os.environ['POSTGRES_PASSWORD'] = 'rl_password'

from utils.database import get_database

def test_postgres_database():
    """測試 PostgreSQL 資料庫"""
    
    print("\n" + "=" * 70)
    print("🐘 PostgreSQL 資料庫測試")
    print("=" * 70)
    
    db = get_database()
    
    # 1. 插入交易對
    print("\n1️⃣  插入交易對...")
    symbols = [
        ('BTCUSDT', 'BTC', 'USDT'),
        ('ETHUSDT', 'ETH', 'USDT'),
        ('BNBUSDT', 'BNB', 'USDT'),
        ('SOLUSDT', 'SOL', 'USDT'),
    ]
    
    for symbol, base, quote in symbols:
        db.execute("""
            INSERT INTO symbols (symbol, base_currency, quote_currency)
            VALUES (%s, %s, %s)
            ON CONFLICT (symbol) DO NOTHING
        """, (symbol, base, quote))
    db.commit()
    
    symbol_count = db.fetchone("SELECT COUNT(*) as count FROM symbols")['count']
    print(f"   ✅ 交易對數量: {symbol_count}")
    
    # 2. 插入市場資料
    print("\n2️⃣  插入市場資料...")
    symbol_id = db.fetchone("SELECT id FROM symbols WHERE symbol = 'BTCUSDT'")['id']
    
    base_time = datetime.now()
    base_price = 95000.0
    
    for i in range(100):
        timestamp = base_time - timedelta(minutes=i)
        price_change = (i % 20 - 10) * 50  # 價格波動
        
        db.execute("""
            INSERT INTO market_data 
            (symbol_id, timestamp, open, high, low, close, volume, timeframe)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol_id, timestamp, timeframe) DO NOTHING
        """, (
            symbol_id,
            timestamp,
            base_price + price_change,
            base_price + price_change + 200,
            base_price + price_change - 200,
            base_price + price_change + 100,
            100.5 + i * 2,
            '1m'
        ))
    
    db.commit()
    market_count = db.fetchone("SELECT COUNT(*) as count FROM market_data")['count']
    print(f"   ✅ 市場資料數量: {market_count}")
    
    # 3. 插入訓練記錄
    print("\n3️⃣  插入訓練記錄...")
    run_id = f"run_btc_{int(datetime.now().timestamp())}_pg_test"
    
    db.execute("""
        INSERT INTO training_runs 
        (run_id, symbol_id, symbol, algorithm, start_time, end_time, 
         status, config, best_reward, final_pnl, total_episodes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        run_id,
        symbol_id,
        'BTCUSDT',
        'PPO',
        datetime.now() - timedelta(hours=2),
        datetime.now(),
        'completed',
        json.dumps({'learning_rate': 0.0003, 'n_steps': 2048}),
        1500.50,
        5200.75,
        500
    ))
    db.commit()
    print(f"   ✅ 訓練記錄: {run_id}")
    
    # 4. 插入 Episodes
    print("\n4️⃣  插入 Episodes...")
    for ep in range(1, 11):
        db.execute("""
            INSERT INTO episodes 
            (run_id, episode_num, timestamp, episode_reward, episode_length,
             win_rate, sharpe_ratio, max_drawdown, episode_pnl, total_trades)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, episode_num) DO NOTHING
        """, (
            run_id,
            ep,
            datetime.now() - timedelta(hours=2, minutes=ep*10),
            100.0 + ep * 20,
            1000 + ep * 50,
            0.50 + ep * 0.03,
            1.5 + ep * 0.1,
            -0.10 - ep * 0.01,
            200.0 + ep * 100,
            25 + ep * 5
        ))
    
    db.commit()
    episode_count = db.fetchone(f"SELECT COUNT(*) as count FROM episodes WHERE run_id = '{run_id}'")['count']
    print(f"   ✅ Episodes 數量: {episode_count}")
    
    # 5. 插入模型
    print("\n5️⃣  插入模型...")
    model_name = f"ppo_btc_{int(datetime.now().timestamp())}_pg"
    
    db.execute("""
        INSERT INTO models 
        (run_id, symbol_id, symbol, model_name, model_path, 
         performance_metrics, is_best)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        run_id,
        symbol_id,
        'BTCUSDT',
        model_name,
        f'models/{model_name}.zip',
        json.dumps({
            'final_pnl': 5200.75,
            'sharpe_ratio': 2.5,
            'max_drawdown': -0.18,
            'win_rate': 0.68
        }),
        True
    ))
    db.commit()
    print(f"   ✅ 模型: {model_name}")
    
    # 6. 查詢視圖
    print("\n6️⃣  查詢資料庫視圖...")
    
    # 最新訓練
    latest_runs = db.fetchall("""
        SELECT * FROM v_latest_training_runs 
        WHERE symbol = 'BTCUSDT'
        LIMIT 5
    """)
    print(f"   ✅ v_latest_training_runs: {len(latest_runs)} 筆")
    if latest_runs:
        print(f"      最新: {latest_runs[0]['run_id']} | PnL: {latest_runs[0]['final_pnl']}")
    
    # 幣種表現
    symbol_perf = db.fetchall("""
        SELECT * FROM v_symbol_performance 
        WHERE symbol = 'BTCUSDT'
    """)
    print(f"   ✅ v_symbol_performance: {len(symbol_perf)} 筆")
    if symbol_perf:
        print(f"      BTCUSDT: 訓練 {symbol_perf[0]['total_runs']} 次 | 平均 PnL: {symbol_perf[0]['avg_pnl']:.2f}")
    
    # 模型排行榜
    leaderboard = db.fetchall("""
        SELECT * FROM v_model_leaderboard 
        LIMIT 5
    """)
    print(f"   ✅ v_model_leaderboard: {len(leaderboard)} 筆")
    for i, model in enumerate(leaderboard, 1):
        print(f"      {i}. {model['model_name']}: {model['symbol']} | PnL: {model['training_pnl']:.2f}")
    
    # 7. 統計摘要
    print("\n" + "=" * 70)
    print("📊 資料庫統計摘要")
    print("=" * 70)
    
    stats = {
        'symbols': db.fetchone("SELECT COUNT(*) as count FROM symbols")['count'],
        'market_data': db.fetchone("SELECT COUNT(*) as count FROM market_data")['count'],
        'training_runs': db.fetchone("SELECT COUNT(*) as count FROM training_runs")['count'],
        'episodes': db.fetchone("SELECT COUNT(*) as count FROM episodes")['count'],
        'models': db.fetchone("SELECT COUNT(*) as count FROM models")['count'],
        'backtest_runs': db.fetchone("SELECT COUNT(*) as count FROM backtest_runs")['count'],
        'trades': db.fetchone("SELECT COUNT(*) as count FROM trades")['count'],
        'system_logs': db.fetchone("SELECT COUNT(*) as count FROM system_logs")['count'],
    }
    
    for table, count in stats.items():
        print(f"  {table:20s}: {count:5d} 筆")
    
    print("\n✅ PostgreSQL 資料庫測試完成！")
    print("=" * 70 + "\n")
    
    db.close()

if __name__ == "__main__":
    try:
        test_postgres_database()
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
