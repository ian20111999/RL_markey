"""
數據緩存管理器
優化數據加載和訓練效率，減少重複的磁盤 I/O

特性:
- LRU 緩存機制
- 數據預加載
- 內存管理
- 緩存統計
"""

import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict, Callable
from functools import lru_cache, wraps
import pandas as pd
import numpy as np
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class LRUCache:
    """自定義 LRU 緩存，支援大型數據對象"""
    
    def __init__(self, maxsize: int = 128, maxmemory_mb: Optional[int] = None):
        """
        初始化緩存
        
        Args:
            maxsize: 最大緩存項目數
            maxmemory_mb: 最大內存使用量（MB），None 表示不限制
        """
        self.maxsize = maxsize
        self.maxmemory_bytes = maxmemory_mb * 1024 * 1024 if maxmemory_mb else None
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.current_memory = 0
    
    def _get_size(self, obj: Any) -> int:
        """估算對象大小（字節）"""
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            # 使用 pickle 序列化來估算
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    
    def get(self, key: str) -> Optional[Any]:
        """獲取緩存項"""
        if key in self.cache:
            self.hits += 1
            # 移到最後（最近使用）
            self.cache.move_to_end(key)
            return self.cache[key]['value']
        else:
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """存入緩存"""
        value_size = self._get_size(value)
        
        # 如果單個對象超過內存限制，不緩存
        if self.maxmemory_bytes and value_size > self.maxmemory_bytes:
            logger.warning(f"對象過大 ({value_size / 1024 / 1024:.1f}MB)，跳過緩存")
            return
        
        # 移除舊項以騰出空間
        while len(self.cache) >= self.maxsize or \
              (self.maxmemory_bytes and self.current_memory + value_size > self.maxmemory_bytes):
            if not self.cache:
                break
            oldest_key = next(iter(self.cache))
            oldest_item = self.cache.pop(oldest_key)
            self.current_memory -= oldest_item['size']
        
        # 添加新項
        self.cache[key] = {
            'value': value,
            'size': value_size
        }
        self.current_memory += value_size
    
    def clear(self):
        """清空緩存"""
        self.cache.clear()
        self.current_memory = 0
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict:
        """獲取緩存統計"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'memory_mb': self.current_memory / 1024 / 1024
        }


class DataCache:
    """數據緩存管理器"""
    
    def __init__(
        self,
        cache_dir: str = ".cache",
        memory_cache_size: int = 32,
        memory_limit_mb: int = 1024,
        enable_disk_cache: bool = True
    ):
        """
        初始化數據緩存
        
        Args:
            cache_dir: 磁盤緩存目錄
            memory_cache_size: 內存緩存項目數
            memory_limit_mb: 內存緩存大小限制（MB）
            enable_disk_cache: 是否啟用磁盤緩存
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_cache = LRUCache(
            maxsize=memory_cache_size,
            maxmemory_mb=memory_limit_mb
        )
        
        self.enable_disk_cache = enable_disk_cache
        
        logger.info(f"數據緩存已初始化 | 目錄={cache_dir} | 內存限制={memory_limit_mb}MB")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """生成緩存鍵"""
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_disk_path(self, key: str) -> Path:
        """獲取磁盤緩存文件路徑"""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """
        獲取緩存數據
        優先從內存獲取，然後嘗試磁盤
        """
        # 1. 嘗試內存緩存
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # 2. 嘗試磁盤緩存
        if self.enable_disk_cache:
            disk_path = self._get_disk_path(key)
            if disk_path.exists():
                try:
                    with open(disk_path, 'rb') as f:
                        value = pickle.load(f)
                    
                    # 加載到內存緩存
                    self.memory_cache.put(key, value)
                    logger.debug(f"從磁盤加載緩存: {key}")
                    return value
                except Exception as e:
                    logger.warning(f"加載磁盤緩存失敗: {e}")
        
        return None
    
    def put(self, key: str, value: Any, save_to_disk: bool = True):
        """
        存儲緩存數據
        
        Args:
            key: 緩存鍵
            value: 數據
            save_to_disk: 是否同時保存到磁盤
        """
        # 存入內存
        self.memory_cache.put(key, value)
        
        # 存入磁盤
        if self.enable_disk_cache and save_to_disk:
            disk_path = self._get_disk_path(key)
            try:
                with open(disk_path, 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.debug(f"數據已緩存到磁盤: {key}")
            except Exception as e:
                logger.warning(f"保存磁盤緩存失敗: {e}")
    
    def clear_memory(self):
        """清空內存緩存"""
        self.memory_cache.clear()
        logger.info("內存緩存已清空")
    
    def clear_disk(self):
        """清空磁盤緩存"""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("磁盤緩存已清空")
    
    def clear_all(self):
        """清空所有緩存"""
        self.clear_memory()
        self.clear_disk()
    
    def stats(self) -> Dict:
        """獲取緩存統計"""
        memory_stats = self.memory_cache.stats()
        
        disk_files = list(self.cache_dir.glob("*.pkl")) if self.enable_disk_cache else []
        disk_size_mb = sum(f.stat().st_size for f in disk_files) / 1024 / 1024
        
        return {
            'memory': memory_stats,
            'disk': {
                'files': len(disk_files),
                'size_mb': disk_size_mb
            }
        }


# 緩存裝飾器
def cached(cache: DataCache, ttl: Optional[int] = None, save_to_disk: bool = True):
    """
    函數結果緩存裝飾器
    
    Args:
        cache: DataCache 實例
        ttl: 緩存有效期（秒），None 表示永久
        save_to_disk: 是否保存到磁盤
    
    Usage:
        @cached(data_cache)
        def load_data(symbol: str):
            return pd.read_csv(f"{symbol}.csv")
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成緩存鍵
            key = cache._generate_key(func.__name__, *args, **kwargs)
            
            # 嘗試獲取緩存
            value = cache.get(key)
            if value is not None:
                logger.debug(f"使用緩存: {func.__name__}")
                return value
            
            # 執行函數
            logger.debug(f"執行函數: {func.__name__}")
            value = func(*args, **kwargs)
            
            # 存入緩存
            cache.put(key, value, save_to_disk=save_to_disk)
            
            return value
        
        return wrapper
    return decorator


# 全局緩存實例
_global_cache: Optional[DataCache] = None


def get_cache() -> DataCache:
    """獲取全局緩存實例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = DataCache()
    return _global_cache


# 快速緩存裝飾器（使用全局實例）
def cache_result(save_to_disk: bool = True):
    """
    快速緩存裝飾器
    
    Usage:
        @cache_result()
        def expensive_function(x):
            return x ** 2
    """
    return cached(get_cache(), save_to_disk=save_to_disk)


if __name__ == "__main__":
    # 測試緩存
    cache = DataCache(cache_dir=".test_cache", memory_limit_mb=100)
    
    # 測試 DataFrame 緩存
    @cached(cache)
    def load_large_data(size):
        print(f"生成 {size} 行數據...")
        return pd.DataFrame({
            'a': np.random.rand(size),
            'b': np.random.rand(size)
        })
    
    # 第一次調用（無緩存）
    df1 = load_large_data(10000)
    print(f"DataFrame shape: {df1.shape}")
    
    # 第二次調用（使用緩存）
    df2 = load_large_data(10000)
    print(f"DataFrame shape: {df2.shape}")
    
    # 查看統計
    stats = cache.stats()
    print(f"\n緩存統計:")
    print(f"  命中率: {stats['memory']['hit_rate']:.2%}")
    print(f"  內存使用: {stats['memory']['memory_mb']:.2f}MB")
    print(f"  磁盤文件: {stats['disk']['files']}")
    
    # 清理測試緩存
    cache.clear_all()
    Path(".test_cache").rmdir()
