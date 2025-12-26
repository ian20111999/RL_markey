"""
PostgreSQL 資料庫實作

提供 PostgreSQL 資料庫的具體實作，包含連接池管理和查詢優化。
"""

import os
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor
from typing import Optional, List, Dict, Any, Tuple
import logging
from contextlib import contextmanager

from utils.database import DatabaseInterface

logger = logging.getLogger(__name__)


class PostgresDatabase(DatabaseInterface):
    """PostgreSQL 資料庫實作"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 PostgreSQL 連接
        
        Args:
            config: 資料庫配置字典，如果為 None 則從環境變數讀取
        """
        if config is None:
            config = {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': int(os.getenv('POSTGRES_PORT', '5432')),
                'database': os.getenv('POSTGRES_DB', 'rl_market'),
                'user': os.getenv('POSTGRES_USER', 'rl_user'),
                'password': os.getenv('POSTGRES_PASSWORD', ''),
            }
        
        self.config = config
        self.conn = None
        self.cursor = None
        self._connection_pool = None
        
        # 初始化連接池
        self._init_connection_pool()
    
    def _init_connection_pool(self):
        """初始化連接池"""
        try:
            min_conn = int(os.getenv('POSTGRES_MIN_CONN', '1'))
            max_conn = int(os.getenv('POSTGRES_MAX_CONN', '10'))
            
            self._connection_pool = psycopg2.pool.ThreadedConnectionPool(
                min_conn,
                max_conn,
                **self.config
            )
            logger.info(f"PostgreSQL 連接池已建立 (min={min_conn}, max={max_conn})")
        except Exception as e:
            logger.error(f"建立連接池失敗: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """
        從連接池獲取連接的上下文管理器
        
        Yields:
            psycopg2.connection: 資料庫連接
        """
        conn = None
        try:
            conn = self._connection_pool.getconn()
            yield conn
        finally:
            if conn:
                self._connection_pool.putconn(conn)
    
    def connect(self):
        """建立資料庫連接"""
        try:
            if self.conn is None or self.conn.closed:
                self.conn = self._connection_pool.getconn()
                self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
                logger.info("PostgreSQL 連接已建立")
        except Exception as e:
            logger.error(f"連接 PostgreSQL 失敗: {e}")
            raise
    
    def close(self):
        """關閉資料庫連接"""
        try:
            if self.cursor:
                self.cursor.close()
                self.cursor = None
            
            if self.conn:
                self._connection_pool.putconn(self.conn)
                self.conn = None
                logger.info("PostgreSQL 連接已關閉")
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
            
            # 建立 training_runs 表
            self.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(255) UNIQUE NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    status VARCHAR(50) NOT NULL,
                    config JSONB,
                    best_reward DOUBLE PRECISION,
                    final_pnl DOUBLE PRECISION,
                    total_episodes INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 建立 episodes 表
            self.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(255) NOT NULL,
                    episode_num INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    episode_reward DOUBLE PRECISION,
                    episode_length INTEGER,
                    win_rate DOUBLE PRECISION,
                    sharpe_ratio DOUBLE PRECISION,
                    max_drawdown DOUBLE PRECISION,
                    total_pnl DOUBLE PRECISION,
                    total_trades INTEGER,
                    metrics JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE,
                    UNIQUE(run_id, episode_num)
                )
            """)
            
            # 建立 models 表
            self.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(255) NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    model_path VARCHAR(500) NOT NULL,
                    performance_metrics JSONB,
                    config JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_best BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
                )
            """)
            
            # 建立索引以優化查詢
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_training_runs_symbol ON training_runs(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status)",
                "CREATE INDEX IF NOT EXISTS idx_training_runs_start_time ON training_runs(start_time DESC)",
                "CREATE INDEX IF NOT EXISTS idx_episodes_run_id ON episodes(run_id)",
                "CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_models_run_id ON models(run_id)",
                "CREATE INDEX IF NOT EXISTS idx_models_symbol ON models(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_models_is_best ON models(is_best) WHERE is_best = TRUE"
            ]
            
            for index_query in indexes:
                self.execute(index_query)
            
            # 建立更新 updated_at 的觸發器
            self.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """)
            
            self.execute("""
                DROP TRIGGER IF EXISTS update_training_runs_updated_at ON training_runs;
                CREATE TRIGGER update_training_runs_updated_at
                    BEFORE UPDATE ON training_runs
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            """)
            
            self.commit()
            logger.info("PostgreSQL 資料庫結構初始化完成")
            
        except Exception as e:
            logger.error(f"初始化資料庫結構失敗: {e}")
            self.rollback()
            raise
    
    def __del__(self):
        """析構函數，確保連接被關閉"""
        self.close()
        if self._connection_pool:
            self._connection_pool.closeall()
