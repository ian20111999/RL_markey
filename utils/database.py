"""
統一資料庫介面，支援 SQLite 和 PostgreSQL

這個模組提供資料庫抽象層，允許應用程式在 SQLite 和 PostgreSQL 之間無縫切換。
"""

import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class DatabaseInterface(ABC):
    """資料庫操作的抽象基類"""
    
    @abstractmethod
    def connect(self):
        """建立資料庫連接"""
        pass
    
    @abstractmethod
    def close(self):
        """關閉資料庫連接"""
        pass
    
    @abstractmethod
    def execute(self, query: str, params: tuple = None):
        """執行 SQL 查詢"""
        pass
    
    @abstractmethod
    def fetchone(self, query: str, params: tuple = None):
        """查詢並返回單一結果"""
        pass
    
    @abstractmethod
    def fetchall(self, query: str, params: tuple = None):
        """查詢並返回所有結果"""
        pass
    
    @abstractmethod
    def commit(self):
        """提交事務"""
        pass
    
    @abstractmethod
    def rollback(self):
        """回滾事務"""
        pass
    
    @abstractmethod
    def init_schema(self):
        """初始化資料庫結構"""
        pass


def get_database(db_type: Optional[str] = None) -> DatabaseInterface:
    """
    根據環境變數或參數獲取資料庫實例
    
    Args:
        db_type: 資料庫類型 ('sqlite' 或 'postgresql')，
                 如果為 None，則從環境變數 DB_TYPE 讀取，預設為 'sqlite'
    
    Returns:
        DatabaseInterface: 資料庫實例
    
    環境變數:
        DB_TYPE: 資料庫類型 ('sqlite' 或 'postgresql')
        POSTGRES_HOST: PostgreSQL 主機位址
        POSTGRES_PORT: PostgreSQL 端口
        POSTGRES_DB: PostgreSQL 資料庫名稱
        POSTGRES_USER: PostgreSQL 使用者名稱
        POSTGRES_PASSWORD: PostgreSQL 密碼
        SQLITE_DB_PATH: SQLite 資料庫檔案路徑
    """
    if db_type is None:
        db_type = os.getenv('DB_TYPE', 'sqlite').lower()
    
    if db_type == 'postgresql':
        from utils.postgres_db import PostgresDatabase
        return PostgresDatabase()
    elif db_type == 'sqlite':
        from utils.sqlite_db import SQLiteDatabase
        return SQLiteDatabase()
    else:
        raise ValueError(f"不支援的資料庫類型: {db_type}")


def get_connection_info() -> Dict[str, Any]:
    """
    獲取當前資料庫連接資訊
    
    Returns:
        Dict[str, Any]: 包含資料庫類型和連接參數的字典
    """
    db_type = os.getenv('DB_TYPE', 'sqlite').lower()
    
    info = {
        'type': db_type
    }
    
    if db_type == 'postgresql':
        info.update({
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'rl_market'),
            'user': os.getenv('POSTGRES_USER', 'rl_user')
        })
    elif db_type == 'sqlite':
        info.update({
            'path': os.getenv('SQLITE_DB_PATH', 'logs/metrics.db')
        })
    
    return info
