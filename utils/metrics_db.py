"""
Metrics Database for tracking all training runs

支援 SQLite 和 PostgreSQL 的統一介面
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

from utils.database import get_database, get_connection_info

logger = logging.getLogger(__name__)


class MetricsDatabase:
    """Store and query training metrics (支援 SQLite 和 PostgreSQL)"""
    
    def __init__(self, db_path: Path = None):
        """
        初始化 Metrics Database
        
        Args:
            db_path: SQLite 資料庫路徑（僅在 SQLite 模式下使用）
        """
        # 如果提供了 db_path 且不是使用 PostgreSQL，設定環境變數
        if db_path and os.getenv('DB_TYPE', 'sqlite').lower() == 'sqlite':
            os.environ['SQLITE_DB_PATH'] = str(db_path)
            if db_path.parent:
                db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 獲取資料庫實例
        self.db = get_database()
        self.db_info = get_connection_info()
        
        logger.info(f"使用資料庫類型: {self.db_info['type']}")
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        try:
            self.db.connect()
            self.db.init_schema()
            logger.info("資料庫初始化完成")
        except Exception as e:
            logger.error(f"資料庫初始化失敗: {e}")
            raise
    
    def add_training_run(self, run_info: Dict):
        """Add a new training run"""
        try:
            self.db.connect()
            
            # 準備資料
            config_json = json.dumps(run_info.get('config', {}))
            data_metrics_json = json.dumps(run_info.get('data_metrics', {}))
            
            # 使用統一的 SQL（參數佔位符會在各資料庫層處理）
            self.db.execute('''
                INSERT INTO training_runs 
                (run_id, symbol, start_time, status, config, total_episodes)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (
                run_info['run_id'],
                run_info['symbol'],
                run_info.get('timestamp', datetime.now().isoformat()),
                run_info.get('status', 'started'),
                config_json,
                run_info.get('attempt', 0)
            ))
            
            self.db.commit()
            logger.info(f"已添加訓練執行: {run_info['run_id']}")
            
        except Exception as e:
            logger.error(f"添加訓練執行失敗: {e}")
            self.db.rollback()
            raise
    
    def update_run_status(self, run_id: str, status: str, duration: float = None):
        """Update run status"""
        try:
            self.db.connect()
            
            if duration is not None:
                # 計算結束時間
                self.db.execute('''
                    UPDATE training_runs 
                    SET status = %s, end_time = %s
                    WHERE run_id = %s
                ''', (status, datetime.now().isoformat(), run_id))
            else:
                self.db.execute('''
                    UPDATE training_runs 
                    SET status = %s
                    WHERE run_id = %s
                ''', (status, run_id))
            
            self.db.commit()
            logger.info(f"已更新執行狀態: {run_id} -> {status}")
            
        except Exception as e:
            logger.error(f"更新執行狀態失敗: {e}")
            self.db.rollback()
            raise
    
    def add_results(self, run_id: str, results: Dict):
        """Add training results"""
        try:
            self.db.connect()
            
            # 更新 training_runs 表的結果欄位
            self.db.execute('''
                UPDATE training_runs 
                SET best_reward = %s, final_pnl = %s, end_time = %s
                WHERE run_id = %s
            ''', (
                results.get('composite_score', 0),
                results.get('mean_pnl', 0),
                datetime.now().isoformat(),
                run_id
            ))
            
            self.db.commit()
            logger.info(f"已添加訓練結果: {run_id}")
            
        except Exception as e:
            logger.error(f"添加訓練結果失敗: {e}")
            self.db.rollback()
            raise
    
    def update_best_model(self, symbol: str, run_id: str, model_path: str, score: float, pnl: float, sharpe: float = None):
        """Update best model for symbol"""
        try:
            self.db.connect()
            
            # 先將該 symbol 的所有模型設為非最佳
            self.db.execute('''
                UPDATE models 
                SET is_best = FALSE
                WHERE symbol = %s
            ''', (symbol,))
            
            # 插入或更新新的最佳模型
            performance_metrics = {
                'composite_score': score,
                'mean_pnl': pnl,
                'sharpe_ratio': sharpe
            }
            
            self.db.execute('''
                INSERT INTO models 
                (run_id, symbol, model_path, performance_metrics, is_best, created_at)
                VALUES (%s, %s, %s, %s, TRUE, %s)
            ''', (
                run_id,
                symbol,
                model_path,
                json.dumps(performance_metrics),
                datetime.now().isoformat()
            ))
            
            self.db.commit()
            logger.info(f"已更新最佳模型: {symbol} -> {run_id}")
            
        except Exception as e:
            logger.error(f"更新最佳模型失敗: {e}")
            self.db.rollback()
            raise
    
    def get_best_run_for_symbol(self, symbol: str) -> Optional[Dict]:
        """Get best training run for a symbol"""
        try:
            self.db.connect()
            
            result = self.db.fetchone('''
                SELECT run_id, best_reward, final_pnl, start_time
                FROM training_runs
                WHERE symbol = %s AND status = 'completed' AND final_pnl IS NOT NULL
                ORDER BY final_pnl DESC
                LIMIT 1
            ''', (symbol,))
            
            if result:
                return {
                    'run_id': result['run_id'],
                    'mean_pnl': result['final_pnl'],
                    'sharpe_ratio': None,
                    'composite_score': result['best_reward'],
                    'timestamp': result['start_time']
                }
            return None
            
        except Exception as e:
            logger.error(f"獲取最佳執行失敗: {e}")
            return None
    
    def get_recent_runs(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Get recent training runs"""
        try:
            self.db.connect()
            
            if symbol:
                results = self.db.fetchall('''
                    SELECT run_id, symbol, start_time, end_time, status, 
                           best_reward, final_pnl, total_episodes
                    FROM training_runs
                    WHERE symbol = %s
                    ORDER BY start_time DESC
                    LIMIT %s
                ''', (symbol, limit))
            else:
                results = self.db.fetchall('''
                    SELECT run_id, symbol, start_time, end_time, status,
                           best_reward, final_pnl, total_episodes
                    FROM training_runs
                    ORDER BY start_time DESC
                    LIMIT %s
                ''', (limit,))
            
            runs = []
            for row in results:
                # 計算持續時間
                duration = None
                if row.get('end_time') and row.get('start_time'):
                    try:
                        start = datetime.fromisoformat(str(row['start_time']))
                        end = datetime.fromisoformat(str(row['end_time']))
                        duration = (end - start).total_seconds()
                    except:
                        pass
                
                runs.append({
                    'run_id': row['run_id'],
                    'symbol': row['symbol'],
                    'timestamp': row['start_time'],
                    'status': row['status'],
                    'duration_seconds': duration,
                    'mean_pnl': row.get('final_pnl'),
                    'composite_score': row.get('best_reward'),
                    'is_acceptable': row.get('status') == 'completed'
                })
            
            return runs
            
        except Exception as e:
            logger.error(f"獲取最近執行失敗: {e}")
            return []
    
    def export_to_json(self, output_path: Path):
        """Export all data to JSON for frontend"""
        try:
            self.db.connect()
            
            # 獲取所有資料
            training_runs = self.db.fetchall("SELECT * FROM training_runs")
            models = self.db.fetchall("SELECT * FROM models")
            
            # 轉換日期時間為字串
            for run in training_runs:
                for key in ['start_time', 'end_time', 'created_at', 'updated_at']:
                    if key in run and run[key]:
                        run[key] = str(run[key])
                # 解析 JSON 欄位
                if 'config' in run and isinstance(run['config'], str):
                    try:
                        run['config'] = json.loads(run['config'])
                    except:
                        pass
            
            for model in models:
                for key in ['created_at']:
                    if key in model and model[key]:
                        model[key] = str(model[key])
                # 解析 JSON 欄位
                for json_key in ['performance_metrics', 'config']:
                    if json_key in model and isinstance(model[json_key], str):
                        try:
                            model[json_key] = json.loads(model[json_key])
                        except:
                            pass
            
            data = {
                'training_runs': training_runs,
                'models': models,
                'exported_at': datetime.now().isoformat(),
                'database_type': self.db_info['type']
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"已匯出指標到 {output_path}")
            
        except Exception as e:
            logger.error(f"匯出指標失敗: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        try:
            self.db.close()
        except Exception as e:
            logger.error(f"關閉資料庫連接失敗: {e}")
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        try:
            self.close()
        except:
            pass
