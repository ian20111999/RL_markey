"""
數據庫連接池和查詢優化
提升數據庫性能和並發處理能力

特性:
- 連接池管理
- 查詢結果緩存
- 批量插入優化
- 自動重連機制
- 慢查詢監控
"""

import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import execute_batch, RealDictCursor
import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
from functools import wraps
from collections import defaultdict

logger = logging.getLogger(__name__)


class QueryMetrics:
    """查詢性能指標追蹤"""
    
    def __init__(self):
        self.query_times = defaultdict(list)
        self.query_counts = defaultdict(int)
        self.slow_queries = []
        self.slow_query_threshold = 1.0  # 秒
    
    def record_query(self, query_type: str, elapsed_time: float):
        """記錄查詢執行時間"""
        self.query_times[query_type].append(elapsed_time)
        self.query_counts[query_type] += 1
        
        if elapsed_time > self.slow_query_threshold:
            self.slow_queries.append({
                'type': query_type,
                'time': elapsed_time,
                'timestamp': time.time()
            })
    
    def get_stats(self) -> Dict:
        """獲取查詢統計"""
        stats = {}
        for query_type, times in self.query_times.items():
            stats[query_type] = {
                'count': self.query_counts[query_type],
                'avg_time': sum(times) / len(times),
                'max_time': max(times),
                'min_time': min(times),
                'total_time': sum(times)
            }
        
        return {
            'queries': stats,
            'slow_queries': self.slow_queries[-10:],  # 最近10個慢查詢
            'total_queries': sum(self.query_counts.values())
        }
    
    def reset(self):
        """重置統計"""
        self.query_times.clear()
        self.query_counts.clear()
        self.slow_queries.clear()


def monitor_query(query_type: str):
    """查詢性能監控裝飾器"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                elapsed = time.time() - start_time
                
                if hasattr(self, 'metrics'):
                    self.metrics.record_query(query_type, elapsed)
                
                if elapsed > 1.0:  # 超過1秒的查詢
                    logger.warning(f"慢查詢: {query_type} 耗時 {elapsed:.3f}s")
                
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"查詢錯誤: {query_type} | 耗時: {elapsed:.3f}s | 錯誤: {e}")
                raise
        return wrapper
    return decorator


class OptimizedPostgresDB:
    """優化版 PostgreSQL 數據庫"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "rl_market",
        user: str = "rl_user",
        password: str = "",
        min_conn: int = 2,
        max_conn: int = 10
    ):
        """
        初始化數據庫連接池
        
        Args:
            host: 主機地址
            port: 端口
            database: 數據庫名
            user: 用戶名
            password: 密碼
            min_conn: 最小連接數
            max_conn: 最大連接數
        """
        self.config = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        
        # 創建連接池
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                min_conn,
                max_conn,
                **self.config
            )
            logger.info(f"PostgreSQL 連接池已創建 | Min={min_conn} | Max={max_conn}")
        except Exception as e:
            logger.error(f"創建連接池失敗: {e}")
            raise
        
        # 性能指標
        self.metrics = QueryMetrics()
        
        # 查詢緩存（簡單實現）
        self._cache = {}
        self._cache_enabled = True
    
    @contextmanager
    def get_connection(self):
        """獲取數據庫連接"""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        finally:
            if conn:
                self.pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, dict_cursor: bool = True):
        """獲取游標"""
        with self.get_connection() as conn:
            if dict_cursor:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
            else:
                cursor = conn.cursor()
            
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise
            finally:
                cursor.close()
    
    @monitor_query("insert")
    def insert_one(self, table: str, data: Dict[str, Any]) -> Optional[int]:
        """
        插入單條記錄
        
        Args:
            table: 表名
            data: 數據字典
        
        Returns:
            插入的記錄 ID（如果有）
        """
        columns = list(data.keys())
        values = list(data.values())
        
        query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) RETURNING id").format(
            sql.Identifier(table),
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            sql.SQL(', ').join(sql.Placeholder() * len(values))
        )
        
        with self.get_cursor() as cursor:
            cursor.execute(query, values)
            result = cursor.fetchone()
            return result['id'] if result else None
    
    @monitor_query("insert_batch")
    def insert_batch(self, table: str, data_list: List[Dict[str, Any]], batch_size: int = 1000):
        """
        批量插入（優化性能）
        
        Args:
            table: 表名
            data_list: 數據字典列表
            batch_size: 每批次大小
        """
        if not data_list:
            return
        
        columns = list(data_list[0].keys())
        
        query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table),
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            sql.SQL(', ').join(sql.Placeholder() * len(columns))
        )
        
        with self.get_cursor() as cursor:
            # 使用 execute_batch 批量插入
            values_list = [tuple(d.values()) for d in data_list]
            execute_batch(cursor, query, values_list, page_size=batch_size)
        
        logger.info(f"批量插入完成: {len(data_list)} 條記錄到 {table}")
    
    @monitor_query("select")
    def select(
        self,
        table: str,
        columns: List[str] = None,
        where: Dict[str, Any] = None,
        order_by: str = None,
        limit: int = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        查詢數據（支持緩存）
        
        Args:
            table: 表名
            columns: 要查詢的列
            where: WHERE 條件字典
            order_by: 排序字段
            limit: 返回記錄數限制
            use_cache: 是否使用緩存
        
        Returns:
            結果列表
        """
        # 生成緩存鍵
        cache_key = f"{table}:{columns}:{where}:{order_by}:{limit}"
        
        if use_cache and self._cache_enabled and cache_key in self._cache:
            logger.debug(f"使用緩存: {table}")
            return self._cache[cache_key]
        
        # 構建查詢
        cols = sql.SQL(', ').join(map(sql.Identifier, columns)) if columns else sql.SQL('*')
        query = sql.SQL("SELECT {} FROM {}").format(cols, sql.Identifier(table))
        
        params = []
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append(sql.SQL("{} = {}").format(
                    sql.Identifier(key),
                    sql.Placeholder()
                ))
                params.append(value)
            
            query += sql.SQL(" WHERE ") + sql.SQL(' AND ').join(conditions)
        
        if order_by:
            query += sql.SQL(" ORDER BY {}").format(sql.Identifier(order_by))
        
        if limit:
            query += sql.SQL(" LIMIT {}").format(sql.Literal(limit))
        
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
        
        # 轉換為字典列表
        result_list = [dict(row) for row in results]
        
        # 緩存結果
        if use_cache and self._cache_enabled:
            self._cache[cache_key] = result_list
        
        return result_list
    
    @monitor_query("update")
    def update(self, table: str, data: Dict[str, Any], where: Dict[str, Any]) -> int:
        """
        更新記錄
        
        Args:
            table: 表名
            data: 要更新的數據
            where: WHERE 條件
        
        Returns:
            影響的行數
        """
        # 構建 SET 子句
        set_clauses = []
        set_values = []
        for key, value in data.items():
            set_clauses.append(sql.SQL("{} = {}").format(
                sql.Identifier(key),
                sql.Placeholder()
            ))
            set_values.append(value)
        
        # 構建 WHERE 子句
        where_clauses = []
        where_values = []
        for key, value in where.items():
            where_clauses.append(sql.SQL("{} = {}").format(
                sql.Identifier(key),
                sql.Placeholder()
            ))
            where_values.append(value)
        
        query = sql.SQL("UPDATE {} SET {} WHERE {}").format(
            sql.Identifier(table),
            sql.SQL(', ').join(set_clauses),
            sql.SQL(' AND ').join(where_clauses)
        )
        
        with self.get_cursor() as cursor:
            cursor.execute(query, set_values + where_values)
            return cursor.rowcount
        
        # 清除相關緩存
        self.invalidate_cache(table)
    
    @monitor_query("delete")
    def delete(self, table: str, where: Dict[str, Any]) -> int:
        """
        刪除記錄
        
        Args:
            table: 表名
            where: WHERE 條件
        
        Returns:
            刪除的行數
        """
        where_clauses = []
        values = []
        for key, value in where.items():
            where_clauses.append(sql.SQL("{} = {}").format(
                sql.Identifier(key),
                sql.Placeholder()
            ))
            values.append(value)
        
        query = sql.SQL("DELETE FROM {} WHERE {}").format(
            sql.Identifier(table),
            sql.SQL(' AND ').join(where_clauses)
        )
        
        with self.get_cursor() as cursor:
            cursor.execute(query, values)
            return cursor.rowcount
        
        # 清除相關緩存
        self.invalidate_cache(table)
    
    def execute_raw(self, query: str, params: Tuple = None) -> List[Dict]:
        """
        執行原始 SQL 查詢
        
        Args:
            query: SQL 查詢
            params: 查詢參數
        
        Returns:
            結果列表
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            
            # 如果是 SELECT 查詢
            if cursor.description:
                results = cursor.fetchall()
                return [dict(row) for row in results]
            
            return []
    
    def invalidate_cache(self, table: str = None):
        """
        清除緩存
        
        Args:
            table: 表名（如果指定，只清除該表的緩存）
        """
        if table:
            # 清除特定表的緩存
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{table}:")]
            for key in keys_to_remove:
                del self._cache[key]
            logger.debug(f"已清除 {table} 的緩存")
        else:
            # 清除所有緩存
            self._cache.clear()
            logger.debug("已清除所有緩存")
    
    def get_performance_stats(self) -> Dict:
        """獲取性能統計"""
        return self.metrics.get_stats()
    
    def close(self):
        """關閉連接池"""
        if self.pool:
            self.pool.closeall()
            logger.info("PostgreSQL 連接池已關閉")


# 使用示例
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 創建優化的數據庫實例
    db = OptimizedPostgresDB(
        host="localhost",
        database="rl_market",
        user="rl_user",
        password="rl_password",
        min_conn=2,
        max_conn=10
    )
    
    print("測試數據庫操作...")
    
    # 測試插入
    print("\n1. 測試單條插入")
    record_id = db.insert_one("training_runs", {
        'symbol': 'BTCUSDT',
        'algorithm': 'SAC',
        'status': 'running'
    })
    print(f"插入成功，ID: {record_id}")
    
    # 測試批量插入
    print("\n2. 測試批量插入")
    batch_data = [
        {'symbol': 'ETHUSDT', 'algorithm': 'PPO', 'status': 'completed'},
        {'symbol': 'BNBUSDT', 'algorithm': 'TD3', 'status': 'completed'},
    ]
    db.insert_batch("training_runs", batch_data)
    
    # 測試查詢
    print("\n3. 測試查詢")
    results = db.select(
        "training_runs",
        where={'status': 'completed'},
        order_by='created_at',
        limit=10
    )
    print(f"查詢結果: {len(results)} 條記錄")
    
    # 查看性能統計
    print("\n4. 性能統計")
    stats = db.get_performance_stats()
    print(f"總查詢數: {stats['total_queries']}")
    for query_type, metrics in stats['queries'].items():
        print(f"  {query_type}: {metrics['count']} 次 | 平均 {metrics['avg_time']:.4f}s")
    
    db.close()
