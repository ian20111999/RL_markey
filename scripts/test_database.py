#!/usr/bin/env python3
"""
資料庫測試與驗證腳本

功能：
1. 初始化資料庫結構
2. 插入測試資料
3. 驗證資料完整性
4. 測試查詢效能
"""

import os
import sys
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.database import get_database, get_connection_info
import logging

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseTester:
    """資料庫測試類"""
    
    def __init__(self):
        self.db = get_database()
        self.db_info = get_connection_info()
        logger.info(f"連接到資料庫: {self.db_info['type']}")
    
    def init_database(self):
        """初始化資料庫結構"""
        logger.info("=" * 60)
        logger.info("初始化資料庫結構...")
        logger.info("=" * 60)
        
        try:
            self.db.connect()
            self.db.init_schema()
            logger.info("✅ 資料庫結構初始化成功")
            return True
        except Exception as e:
            logger.error(f"❌ 資料庫初始化失敗: {e}")
            return False
    
    def insert_test_symbols(self):
        """插入測試交易對"""
        logger.info("\n插入測試交易對...")
        
        symbols_data = [
            ('BTCUSDT', 'BTC', 'USDT', 'binance', True),
            ('ETHUSDT', 'ETH', 'USDT', 'binance', True),
            ('BNBUSDT', 'BNB', 'USDT', 'binance', True),
            ('SOLUSDT', 'SOL', 'USDT', 'binance', True),
        ]
        
        try:
            for symbol, base, quote, exchange, active in symbols_data:
                self.db.execute("""
                    INSERT INTO symbols (symbol, base_currency, quote_currency, exchange, is_active)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (symbol) DO NOTHING
                """, (symbol, base, quote, exchange, active))
            
            self.db.commit()
            
            # 驗證
            result = self.db.fetchall("SELECT * FROM symbols")
            logger.info(f"✅ 成功插入/更新 {len(result)} 個交易對")
            for row in result:
                logger.info(f"  - {row['symbol']}: {row['base_currency']}/{row['quote_currency']}")
            
            return True
        except Exception as e:
            logger.error(f"❌ 插入交易對失敗: {e}")
            self.db.rollback()
            return False
    
    def insert_test_market_data(self):
        """插入測試市場資料"""
        logger.info("\n插入測試市場資料...")
        
        try:
            # 獲取 BTCUSDT 的 symbol_id
            btc_symbol = self.db.fetchone("SELECT id FROM symbols WHERE symbol = %s", ('BTCUSDT',))
            if not btc_symbol:
                logger.warning("找不到 BTCUSDT，跳過市場資料插入")
                return False
            
            symbol_id = btc_symbol['id']
            
            # 生成最近 100 根 K 線資料
            base_time = datetime.now() - timedelta(hours=100)
            base_price = 45000.0
            
            for i in range(100):
                timestamp = base_time + timedelta(hours=i)
                
                # 模擬價格波動
                open_price = base_price + random.uniform(-500, 500)
                high_price = open_price + random.uniform(0, 300)
                low_price = open_price - random.uniform(0, 300)
                close_price = open_price + random.uniform(-200, 200)
                volume = random.uniform(100, 1000)
                
                self.db.execute("""
                    INSERT INTO market_data 
                    (symbol_id, timestamp, timeframe, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol_id, timestamp, timeframe) DO NOTHING
                """, (symbol_id, timestamp, '1h', open_price, high_price, 
                      low_price, close_price, volume))
            
            self.db.commit()
            
            # 驗證
            result = self.db.fetchone("""
                SELECT COUNT(*) as count FROM market_data WHERE symbol_id = %s
            """, (symbol_id,))
            logger.info(f"✅ 成功插入 {result['count']} 條市場資料")
            
            return True
        except Exception as e:
            logger.error(f"❌ 插入市場資料失敗: {e}")
            self.db.rollback()
            return False
    
    def insert_test_training_run(self):
        """插入測試訓練執行"""
        logger.info("\n插入測試訓練執行...")
        
        try:
            # 獲取 BTCUSDT 的 symbol_id
            btc_symbol = self.db.fetchone("SELECT id FROM symbols WHERE symbol = %s", ('BTCUSDT',))
            symbol_id = btc_symbol['id'] if btc_symbol else None
            
            run_id = f"run_btc_{int(datetime.now().timestamp())}_test"
            start_time = datetime.now() - timedelta(hours=2)
            end_time = datetime.now()
            
            config = {
                'algorithm': 'PPO',
                'learning_rate': 0.0003,
                'n_steps': 2048,
                'batch_size': 64
            }
            
            hyperparams = {
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2
            }
            
            training_data_info = {
                'start_date': '2024-01-01',
                'end_date': '2024-12-31',
                'data_points': 8760,
                'timeframe': '1h'
            }
            
            self.db.execute("""
                INSERT INTO training_runs 
                (run_id, symbol_id, symbol, start_time, end_time, status, algorithm,
                 total_timesteps, total_episodes, best_reward, final_pnl, 
                 config, hyperparameters, training_data_info)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, symbol_id, 'BTCUSDT', start_time, end_time, 'completed',
                  'PPO', 1000000, 100, 1250.5, 3500.75,
                  json.dumps(config), json.dumps(hyperparams), json.dumps(training_data_info)))
            
            self.db.commit()
            logger.info(f"✅ 成功插入訓練執行: {run_id}")
            
            return run_id
        except Exception as e:
            logger.error(f"❌ 插入訓練執行失敗: {e}")
            self.db.rollback()
            return None
    
    def insert_test_episodes(self, run_id):
        """插入測試 Episodes"""
        logger.info("\n插入測試 Episodes...")
        
        try:
            base_time = datetime.now() - timedelta(hours=2)
            
            for i in range(10):
                timestamp = base_time + timedelta(minutes=i * 12)
                episode_reward = random.uniform(50, 150)
                episode_pnl = random.uniform(-100, 200)
                win_rate = random.uniform(0.4, 0.7)
                
                metrics = {
                    'inventory': random.uniform(-5, 5),
                    'position_value': random.uniform(0, 10000),
                    'trades_long': random.randint(5, 15),
                    'trades_short': random.randint(5, 15)
                }
                
                self.db.execute("""
                    INSERT INTO episodes 
                    (run_id, episode_num, timestamp, episode_reward, episode_pnl,
                     total_trades, win_rate, sharpe_ratio, metrics)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, episode_num) DO NOTHING
                """, (run_id, i + 1, timestamp, episode_reward, episode_pnl,
                      random.randint(20, 50), win_rate, random.uniform(0.5, 2.0),
                      json.dumps(metrics)))
            
            self.db.commit()
            
            # 驗證
            result = self.db.fetchone("""
                SELECT COUNT(*) as count FROM episodes WHERE run_id = %s
            """, (run_id,))
            logger.info(f"✅ 成功插入 {result['count']} 個 Episodes")
            
            return True
        except Exception as e:
            logger.error(f"❌ 插入 Episodes 失敗: {e}")
            self.db.rollback()
            return False
    
    def insert_test_model(self, run_id):
        """插入測試模型"""
        logger.info("\n插入測試模型...")
        
        try:
            # 獲取 BTCUSDT 的 symbol_id
            btc_symbol = self.db.fetchone("SELECT id FROM symbols WHERE symbol = %s", ('BTCUSDT',))
            symbol_id = btc_symbol['id'] if btc_symbol else None
            
            model_name = f"ppo_btc_{int(datetime.now().timestamp())}"
            model_path = f"/app/models/{model_name}.zip"
            
            performance_metrics = {
                'sharpe_ratio': 1.85,
                'win_rate': 0.62,
                'max_drawdown': -0.15,
                'total_pnl': 3500.75,
                'total_trades': 450
            }
            
            config = {
                'algorithm': 'PPO',
                'learning_rate': 0.0003,
                'trained_timesteps': 1000000
            }
            
            self.db.execute("""
                INSERT INTO models 
                (run_id, symbol_id, symbol, model_name, model_path, model_type,
                 performance_metrics, config, is_best)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, symbol_id, 'BTCUSDT', model_name, model_path, 'PPO',
                  json.dumps(performance_metrics), json.dumps(config), True))
            
            self.db.commit()
            logger.info(f"✅ 成功插入模型: {model_name}")
            
            return True
        except Exception as e:
            logger.error(f"❌ 插入模型失敗: {e}")
            self.db.rollback()
            return False
    
    def verify_data(self):
        """驗證資料完整性"""
        logger.info("\n" + "=" * 60)
        logger.info("驗證資料完整性...")
        logger.info("=" * 60)
        
        tests = [
            ("symbols", "SELECT COUNT(*) as count FROM symbols"),
            ("market_data", "SELECT COUNT(*) as count FROM market_data"),
            ("training_runs", "SELECT COUNT(*) as count FROM training_runs"),
            ("episodes", "SELECT COUNT(*) as count FROM episodes"),
            ("models", "SELECT COUNT(*) as count FROM models"),
        ]
        
        results = {}
        for table_name, query in tests:
            try:
                result = self.db.fetchone(query)
                count = result['count']
                results[table_name] = count
                logger.info(f"✅ {table_name}: {count} 筆記錄")
            except Exception as e:
                logger.error(f"❌ {table_name}: 查詢失敗 - {e}")
                results[table_name] = 0
        
        return results
    
    def test_views(self):
        """測試視圖"""
        logger.info("\n" + "=" * 60)
        logger.info("測試資料庫視圖...")
        logger.info("=" * 60)
        
        # 測試最新訓練執行視圖
        try:
            result = self.db.fetchall("SELECT * FROM v_latest_training_runs LIMIT 5")
            logger.info(f"\n📊 最新訓練執行 (共 {len(result)} 筆):")
            for row in result:
                logger.info(f"  - {row['run_id']}: {row['symbol']} | "
                          f"狀態: {row['status']} | PnL: {row['final_pnl']}")
        except Exception as e:
            logger.error(f"❌ v_latest_training_runs 查詢失敗: {e}")
        
        # 測試幣種表現視圖
        try:
            result = self.db.fetchall("SELECT * FROM v_symbol_performance")
            logger.info(f"\n📊 幣種表現統計 (共 {len(result)} 個幣種):")
            for row in result:
                avg_pnl_str = f"{row['avg_pnl']:.2f}" if row['avg_pnl'] else 'N/A'
                logger.info(f"  - {row['symbol']}: {row['total_runs']} 次訓練 | "
                          f"平均 PnL: {avg_pnl_str}")
        except Exception as e:
            logger.error(f"❌ v_symbol_performance 查詢失敗: {e}")
        
        # 測試模型排行榜視圖
        try:
            result = self.db.fetchall("SELECT * FROM v_model_leaderboard LIMIT 5")
            logger.info(f"\n📊 模型排行榜 (Top 5):")
            for i, row in enumerate(result, 1):
                training_pnl_str = f"{row['training_pnl']:.2f}" if row['training_pnl'] else 'N/A'
                logger.info(f"  {i}. {row['model_name']}: {row['symbol']} | "
                          f"訓練 PnL: {training_pnl_str}")
        except Exception as e:
            logger.error(f"❌ v_model_leaderboard 查詢失敗: {e}")
    
    def run_all_tests(self):
        """執行所有測試"""
        logger.info("\n" + "=" * 60)
        logger.info("🚀 開始資料庫完整測試")
        logger.info("=" * 60)
        
        # 1. 初始化資料庫
        if not self.init_database():
            logger.error("資料庫初始化失敗，中止測試")
            return False
        
        # 2. 插入測試資料
        self.insert_test_symbols()
        self.insert_test_market_data()
        
        run_id = self.insert_test_training_run()
        if run_id:
            self.insert_test_episodes(run_id)
            self.insert_test_model(run_id)
        
        # 3. 驗證資料
        results = self.verify_data()
        
        # 4. 測試視圖
        self.test_views()
        
        # 5. 總結
        logger.info("\n" + "=" * 60)
        logger.info("✅ 資料庫測試完成！")
        logger.info("=" * 60)
        logger.info(f"\n資料統計:")
        logger.info(f"  - 交易對: {results.get('symbols', 0)} 個")
        logger.info(f"  - 市場資料: {results.get('market_data', 0)} 條")
        logger.info(f"  - 訓練執行: {results.get('training_runs', 0)} 次")
        logger.info(f"  - Episodes: {results.get('episodes', 0)} 個")
        logger.info(f"  - 模型: {results.get('models', 0)} 個")
        
        return True


def main():
    """主函數"""
    tester = DatabaseTester()
    success = tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
