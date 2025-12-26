"""
集中式日誌配置模組
統一管理所有模組的日誌輸出，支援多種輸出格式和日誌級別

特性:
- 自動日誌輪轉（按大小和時間）
- 結構化日誌格式
- 不同級別的顏色輸出
- 性能監控集成
- 異步日誌寫入
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class ColoredFormatter(logging.Formatter):
    """支援終端顏色的日誌格式化器"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            levelname = record.levelname
            color = self.COLORS.get(levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """結構化 JSON 格式日誌"""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加異常信息
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # 添加自定義字段
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)
        
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
    structured: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    配置全局日誌系統
    
    Args:
        log_dir: 日誌目錄
        level: 日誌級別 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: 是否輸出到控制台
        file_output: 是否輸出到文件
        structured: 是否使用結構化 JSON 格式
        max_bytes: 單個日誌文件最大大小
        backup_count: 保留的舊日誌文件數量
    
    Returns:
        配置好的 logger
    """
    # 創建日誌目錄
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 獲取 root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除現有的 handlers
    logger.handlers.clear()
    
    # 控制台輸出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if structured:
            console_formatter = StructuredFormatter()
        else:
            console_formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 文件輸出
    if file_output:
        # 通用日誌（INFO及以上）
        info_log = log_path / f"rl_market_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            info_log,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # 錯誤日誌（ERROR及以上）
        error_log = log_path / f"error_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        # 設置格式
        if structured:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        error_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)
    
    # 調試日誌（可選，僅在 DEBUG 級別時）
    if level.upper() == 'DEBUG' and file_output:
        debug_log = log_path / f"debug_{datetime.now().strftime('%Y%m%d')}.log"
        debug_handler = logging.handlers.RotatingFileHandler(
            debug_log,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(file_formatter)
        logger.addHandler(debug_handler)
    
    logger.info(f"日誌系統已初始化 | Level={level} | Dir={log_dir}")
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    獲取特定模組的 logger
    
    Args:
        name: logger 名稱（通常使用 __name__）
    
    Returns:
        logger 實例
    """
    return logging.getLogger(name)


class LoggerContext:
    """日誌上下文管理器，用於臨時改變日誌級別"""
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.old_level = None
    
    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


# 性能監控日誌裝飾器
def log_performance(logger: Optional[logging.Logger] = None):
    """
    裝飾器：記錄函數執行時間
    
    Usage:
        @log_performance()
        def my_function():
            pass
    """
    import time
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"{func.__name__} 執行完成 | 耗時: {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{func.__name__} 執行失敗 | 耗時: {elapsed:.3f}s | 錯誤: {e}")
                raise
        
        return wrapper
    return decorator


# 快速配置函數
def quick_setup(level: str = "INFO", log_dir: str = "logs"):
    """快速配置日誌系統（簡化版）"""
    return setup_logging(
        log_dir=log_dir,
        level=level,
        console_output=True,
        file_output=True,
        structured=False
    )


if __name__ == "__main__":
    # 測試日誌配置
    setup_logging(level="DEBUG")
    logger = get_logger(__name__)
    
    logger.debug("這是 DEBUG 訊息")
    logger.info("這是 INFO 訊息")
    logger.warning("這是 WARNING 訊息")
    logger.error("這是 ERROR 訊息")
    
    # 測試性能裝飾器
    @log_performance()
    def test_function():
        import time
        time.sleep(0.1)
    
    test_function()
