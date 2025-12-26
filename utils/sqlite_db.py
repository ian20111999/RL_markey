"""
SQLite 資料庫實作

提供 SQLite 資料庫的具體實作，用於本地開發和測試。
"""

import os
import sqlite3
from typing import Optional, List, Dict, Any
import logging
import json
from pathlib import Path

from utils.database import DatabaseInterface

logger = logging.getLogger(__name__)


class SQLiteDatabase(DatabaseInterface):
    """SQLite 資料庫實作"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        初始化 SQLite 連接
        
        Args:
            db_path: 資料庫檔案路徑，如果為 None 則從環境變數讀取
        """
        if db_path is None:
            db_path = os.getenv('SQLITE_DB_PATH', 'logs/metrics.db')
        
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        # 確保資料庫目錄存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def connect(self):
        """建立資料庫連接"""
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
                self.conn.row_factory = sqlite3.Row
                self.cursor = self.conn.cursor()
                logger.info(f"SQLite 連接已建立: {self.db_path}")
        except Exception as e:
            logger.error(f"連接 SQLite 失敗: {e}")
            raise
    
    def close(self):
        """關閉資料庫連接"""
        try:
            if self.cursor:
                self.cursor.close()
                self.cursor = None
            
            if self.conn:
                self.conn.close()
                self.conn = None
                logger.info("SQLite 連接已關閉")
        except Exception as e:
            logger.error(f"關閉連接時發生錯誤: {e}")
    
    def execute(self, query: str, params: tuple = None):
        """
        執行 SQL 查詢
        
        Args:
            query: SQL 查詢字串
            params: 查詢參數
        """
        try:
            if self.cursor is None:
                self.connect()
            
            # 將 PostgreSQL 的 %s 參數佔位符轉換為 SQLite 的 ?
            if '%s' in query:
                query = query.replace('%s', '?')
            
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            logger.debug(f"執行 SQL: {query[:100]}...")
        except Exception as e:
            logger.error(f"執行 SQL 失敗: {e}\nQuery: {query}")
            raise
    
    def fetchone(self, query: str, params: tuple = None) -> Optional[Dict[str, Any]]:
        """
        查詢並返回單一結果
        
        Args:
            query: SQL 查詢字串
            params: 查詢參數
        
        Returns:
            Optional[Dict[str, Any]]: 查詢結果字典
        """
        try:
            self.execute(query, params)
            result = self.cursor.fetchone()
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"fetchone 失敗: {e}")
            raise
    
    def fetchall(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        查詢並返回所有結果
        
        Args:
            query: SQL 查詢字串
            params: 查詢參數
        
        Returns:
            List[Dict[str, Any]]: 查詢結果列表
        """
        try:
            self.execute(query, params)
            results = self.cursor.fetchall()
            return [dict(row) for row in results] if results else []
        except Exception as e:
            logger.error(f"fetchall 失敗: {e}")
            raise
    
    def commit(self):
        """提交事務"""
        try:
            if self.conn:
                self.conn.commit()
                logger.debug("事務已提交")
        except Exception as e:
            logger.error(f"提交事務失敗: {e}")
            raise
    
    def rollback(self):
        """回滾事務"""
        try:
            if self.conn:
                self.conn.rollback()
                logger.warning("事務已回滾")
        except Exception as e:
            logger.error(f"回滾事務失敗: {e}")
            raise
    
    def init_schema(self):
        """初始化資料庫結構"""
        try:
            self.connect()
            
            # 檢查是否為舊結構（有 timestamp 欄位）
            old_schema = self._check_old_schema()
            
            if old_schema:
                logger.info("檢測到舊的資料庫結構，執行遷移...")
                self._migrate_old_schema()
                return
            
            # 使用 SQL 檔案初始化新結構
            sql_file = Path(__file__).parent.parent / 'scripts' / 'init_db_sqlite.sql'
            
            if sql_file.exists():
                logger.info(f"從 SQL 檔案初始化資料庫: {sql_file}")
                with open(sql_file, 'r', encoding='utf-8') as f:
                    sql_script = f.read()
                
                # 執行整個 SQL 腳本
                self.conn.executescript(sql_script)
                self.commit()
                logger.info("SQLite 資料庫結構初始化完成")
            else:
                logger.warning(f"找不到 SQL 檔案: {sql_file}，使用內建結構")
                self._init_schema_legacy()
            
        except Exception as e:
            logger.error(f"初始化資料庫結構失敗: {e}")
            self.rollback()
            raise
    
    def _init_schema_legacy(self):
        """舊版內建結構（作為備援）"""
        # 建立基本的 training_runs 表
        self.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                status TEXT NOT NULL,
                config TEXT,
                best_reward REAL,
                final_pnl REAL,
                total_episodes INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 建立基本索引
        self.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_symbol ON training_runs(symbol)")
        self.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status)")
        
        self.commit()
        logger.info("SQLite 基本資料庫結構初始化完成")
    
    def _check_old_schema(self) -> bool:
        """檢查是否為舊的資料庫結構"""
        try:
            # 檢查 training_runs 表是否存在 timestamp 欄位（舊結構）
            result = self.fetchone("""
                SELECT COUNT(*) as count 
                FROM pragma_table_info('training_runs') 
                WHERE name='timestamp'
            """)
            return result and result['count'] > 0
        except:
            return False
    
    def _migrate_old_schema(self):
        """將舊的資料庫結構遷移到新結構"""
        logger.info("開始遷移舊資料庫結構...")
        
        try:
            # 1. 重命名舊表
            self.execute("ALTER TABLE training_runs RENAME TO training_runs_old")
            
            # 2. 建立新表結構
            self.execute("""
                CREATE TABLE training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    status TEXT NOT NULL,
                    config TEXT,
                    best_reward REAL,
                    final_pnl REAL,
                    total_episodes INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 3. 複製資料（映射舊欄位到新欄位）
            self.execute("""
                INSERT INTO training_runs 
                (run_id, symbol, start_time, status, config, total_episodes, created_at)
                SELECT 
                    run_id, 
                    symbol, 
                    timestamp as start_time,
                    COALESCE(status, 'unknown') as status,
                    config_json as config,
                    COALESCE(attempt, 0) as total_episodes,
                    timestamp as created_at
                FROM training_runs_old
            """)
            
            # 4. 建立索引
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_training_runs_symbol ON training_runs(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status)",
                "CREATE INDEX IF NOT EXISTS idx_training_runs_start_time ON training_runs(start_time DESC)"
            ]
            for index_query in indexes:
                self.execute(index_query)
            
            # 5. 處理其他舊表（如果存在）
            # training_results -> 合併到 training_runs
            try:
                self.execute("""
                    UPDATE training_runs
                    SET 
                        best_reward = (SELECT composite_score FROM training_results WHERE training_results.run_id = training_runs.run_id),
                        final_pnl = (SELECT mean_pnl FROM training_results WHERE training_results.run_id = training_runs.run_id)
                    WHERE EXISTS (SELECT 1 FROM training_results WHERE training_results.run_id = training_runs.run_id)
                """)
                self.execute("DROP TABLE IF EXISTS training_results")
            except:
                pass  # training_results 可能不存在
            
            # best_models -> 新的 models 表
            try:
                self.execute("""
                    CREATE TABLE IF NOT EXISTS models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        model_path TEXT NOT NULL,
                        performance_metrics TEXT,
                        config TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_best INTEGER DEFAULT 1,
                        FOREIGN KEY (run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
                    )
                """)
                
                self.execute("""
                    INSERT INTO models (run_id, symbol, model_path, performance_metrics, created_at, is_best)
                    SELECT 
                        run_id,
                        symbol,
                        COALESCE(model_path, ''),
                        json_object(
                            'composite_score', composite_score,
                            'mean_pnl', mean_pnl,
                            'sharpe_ratio', sharpe_ratio
                        ),
                        COALESCE(saved_at, CURRENT_TIMESTAMP),
                        1
                    FROM best_models
                """)
                self.execute("DROP TABLE IF EXISTS best_models")
            except Exception as e:
                logger.warning(f"遷移 best_models 時出錯: {e}")
            
            # 6. 刪除舊表
            self.execute("DROP TABLE IF EXISTS training_runs_old")
            
            # 7. 建立 episodes 表（如果不存在）
            self.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    episode_num INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    episode_reward REAL,
                    episode_length INTEGER,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_pnl REAL,
                    total_trades INTEGER,
                    metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE,
                    UNIQUE(run_id, episode_num)
                )
            """)
            
            self.commit()
            logger.info("✅ 舊資料庫結構遷移完成")
            
        except Exception as e:
            logger.error(f"遷移失敗: {e}")
            self.rollback()
            raise
    
    def __del__(self):
        """析構函數，確保連接被關閉"""
        self.close()
