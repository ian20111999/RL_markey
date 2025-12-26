#!/usr/bin/env python3
"""
快速驗證訓練流程是否能正常寫入新資料庫結構
"""

import sys
import os
from pathlib import Path

# 加入專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.database import get_database
from datetime import datetime
import json

def test_training_workflow():
    """測試完整的訓練工作流程資料寫入"""
    
    db = get_database()
    print("\n" + "=" * 60)
    print("🧪 測試訓練工作流程資料庫寫入")
    print("=" * 60)
    
    # 0. 初始化資料庫結構
    print("\n0️⃣  初始化資料庫結構...")
    db.init_schema()
    print("   ✅ 資料庫結構初始化完成")
    
    # 1. 確保交易對存在
    print("\n1️⃣  檢查/建立交易對...")
    db.execute("""
        INSERT OR REPLACE INTO symbols (symbol, base_currency, quote_currency)
        VALUES ('BTCUSDT', 'BTC', 'USDT')
    """)
    db.commit()
    
    symbol_id = db.fetchone("SELECT id FROM symbols WHERE symbol = 'BTCUSDT'")['id']
    print(f"   ✅ BTCUSDT symbol_id: {symbol_id}")
    
    # 2. 建立訓練執行記錄
    print("\n2️⃣  建立訓練執行記錄...")
    run_id = f"run_btc_{int(datetime.now().timestamp())}_workflow_test"
    
    db.execute("""
        INSERT INTO training_runs 
        (run_id, symbol_id, symbol, algorithm, start_time, status, config, total_episodes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        symbol_id,
        'BTCUSDT',  # 加入 symbol
        'PPO',
        datetime.now(),
        'running',
        json.dumps({
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64
        }),
        100
    ))
    db.commit()
    print(f"   ✅ 訓練執行已建立: {run_id}")
    
    # 3. 模擬 5 個 episodes
    print("\n3️⃣  寫入訓練 Episodes...")
    for ep in range(1, 6):
        db.execute("""
            INSERT INTO episodes 
            (run_id, episode_num, timestamp, episode_reward, episode_length,
             win_rate, sharpe_ratio, max_drawdown, episode_pnl, total_trades)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            ep,
            datetime.now(),
            50.0 + ep * 10,  # 遞增獎勵
            1000 + ep * 100,
            0.5 + ep * 0.05,
            1.0 + ep * 0.2,
            -0.1 - ep * 0.02,
            100.0 + ep * 50,
            20 + ep * 5
        ))
    db.commit()
    print(f"   ✅ 已寫入 5 個 Episodes")
    
    # 4. 完成訓練並記錄模型
    print("\n4️⃣  完成訓練並記錄模型...")
    db.execute("""
        UPDATE training_runs 
        SET status = 'completed',
            end_time = ?,
            best_reward = 100.0,
            final_pnl = 350.0
        WHERE run_id = ?
    """, (datetime.now(), run_id))
    
    model_name = f"ppo_btc_{int(datetime.now().timestamp())}"
    db.execute("""
        INSERT INTO models 
        (run_id, symbol_id, model_name, model_path, algorithm, 
         performance_metrics, is_best)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        symbol_id,
        model_name,
        f'models/{model_name}.zip',
        'PPO',
        json.dumps({
            'final_pnl': 350.0,
            'sharpe_ratio': 2.0,
            'max_drawdown': -0.18,
            'win_rate': 0.65
        }),
        1  # 標記為最佳模型
    ))
    db.commit()
    print(f"   ✅ 模型已記錄: {model_name}")
    
    # 5. 驗證資料完整性
    print("\n5️⃣  驗證資料完整性...")
    
    # 檢查訓練記錄
    run = db.fetchone("""
        SELECT tr.*, s.symbol 
        FROM training_runs tr
        JOIN symbols s ON tr.symbol_id = s.id
        WHERE tr.run_id = ?
    """, (run_id,))
    
    assert run is not None, "訓練記錄不存在"
    assert run['symbol'] == 'BTCUSDT', f"交易對錯誤: {run['symbol']}"
    assert run['status'] == 'completed', f"狀態錯誤: {run['status']}"
    print(f"   ✅ 訓練記錄正確: {run['run_id']}")
    
    # 檢查 Episodes
    episodes = db.fetchall("""
        SELECT * FROM episodes 
        WHERE run_id = ? 
        ORDER BY episode_num
    """, (run_id,))
    
    assert len(episodes) == 5, f"Episodes 數量錯誤: {len(episodes)}"
    assert episodes[0]['episode_num'] == 1, "Episode 編號錯誤"
    assert episodes[4]['episode_num'] == 5, "Episode 編號錯誤"
    print(f"   ✅ Episodes 正確: 5 個")
    
    # 檢查模型
    model = db.fetchone("""
        SELECT m.*, s.symbol
        FROM models m
        JOIN symbols s ON m.symbol_id = s.id
        WHERE m.model_name = ?
    """, (model_name,))
    
    assert model is not None, "模型記錄不存在"
    assert model['is_best'] == 1, "最佳模型標記錯誤"
    assert model['symbol'] == 'BTCUSDT', f"模型交易對錯誤: {model['symbol']}"
    print(f"   ✅ 模型記錄正確: {model['model_name']}")
    
    # 6. 測試視圖查詢
    print("\n6️⃣  測試資料庫視圖...")
    
    # 最新訓練視圖
    latest_runs = db.fetchall("""
        SELECT * FROM v_latest_training_runs 
        WHERE symbol = 'BTCUSDT'
        LIMIT 3
    """)
    print(f"   ✅ v_latest_training_runs: {len(latest_runs)} 筆")
    
    # 幣種表現視圖
    symbol_perf = db.fetchall("""
        SELECT * FROM v_symbol_performance 
        WHERE symbol = 'BTCUSDT'
    """)
    assert len(symbol_perf) > 0, "幣種表現視圖無資料"
    print(f"   ✅ v_symbol_performance: 平均 PnL = {symbol_perf[0]['avg_pnl']:.2f}")
    
    # 模型排行榜視圖
    leaderboard = db.fetchall("""
        SELECT * FROM v_model_leaderboard 
        WHERE symbol = 'BTCUSDT'
        LIMIT 5
    """)
    assert len(leaderboard) > 0, "模型排行榜視圖無資料"
    print(f"   ✅ v_model_leaderboard: Top {len(leaderboard)} 模型")
    
    # 7. 測試更新觸發器
    print("\n7️⃣  測試自動更新觸發器...")
    old_updated_at = run['updated_at']
    
    import time
    time.sleep(1)  # 確保時間戳有變化
    
    db.execute("""
        UPDATE training_runs 
        SET final_pnl = 400.0
        WHERE run_id = ?
    """, (run_id,))
    db.commit()
    
    updated_run = db.fetchone("""
        SELECT updated_at FROM training_runs WHERE run_id = ?
    """, (run_id,))
    
    assert updated_run['updated_at'] != old_updated_at, "updated_at 觸發器未生效"
    print(f"   ✅ 自動更新觸發器正常")
    
    # 8. 完成
    print("\n" + "=" * 60)
    print("✅ 所有測試通過！資料庫結構運作正常")
    print("=" * 60)
    
    print("\n📊 測試摘要:")
    print(f"   - 訓練執行 ID: {run_id}")
    print(f"   - Episodes 數量: 5")
    print(f"   - 模型名稱: {model_name}")
    print(f"   - 最終 PnL: {run['final_pnl']}")
    print(f"   - 最佳獎勵: {run['best_reward']}")
    
    db.close()
    return True

if __name__ == "__main__":
    try:
        test_training_workflow()
    except AssertionError as e:
        print(f"\n❌ 測試失敗: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
