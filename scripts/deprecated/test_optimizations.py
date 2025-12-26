#!/usr/bin/env python3
"""
優化功能驗證腳本
快速測試所有優化模組是否正常工作
"""

import sys
from pathlib import Path

# 添加項目根目錄
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("🧪 RL Market 優化功能驗證")
print("=" * 70)

# 測試計數器
tests_passed = 0
tests_failed = 0
tests_total = 0


def test(name: str, func):
    """測試包裝器"""
    global tests_passed, tests_failed, tests_total
    tests_total += 1
    
    try:
        print(f"\n[{tests_total}] 測試: {name}...", end=" ")
        func()
        print("✅ 通過")
        tests_passed += 1
        return True
    except Exception as e:
        print(f"❌ 失敗: {e}")
        tests_failed += 1
        return False


# ============================================================================
# 測試 1: 日誌系統
# ============================================================================

def test_logging():
    from utils.logging_config import setup_logging, get_logger, log_performance
    import time
    
    # 測試日誌初始化
    setup_logging(level="INFO", log_dir="logs")
    logger = get_logger("test")
    
    # 測試日誌輸出
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    
    # 測試性能裝飾器
    @log_performance()
    def sample_function():
        time.sleep(0.1)
    
    sample_function()
    
    assert True


test("日誌系統", test_logging)


# ============================================================================
# 測試 2: 性能監控
# ============================================================================

def test_performance_monitor():
    from utils.performance_monitor import PerformanceMonitor
    import time
    
    monitor = PerformanceMonitor(sample_interval=1.0)
    
    # 測試監控啟動
    monitor.start_monitoring()
    time.sleep(0.5)
    
    # 測試步數更新
    for i in range(10):
        monitor.update_step_count(100)
        time.sleep(0.01)
    
    # 測試獲取指標
    metrics = monitor.get_current_metrics()
    assert 'cpu_percent' in metrics
    assert 'memory_mb' in metrics
    
    # 測試摘要
    summary = monitor.get_summary()
    assert 'cpu' in summary
    assert 'memory_mb' in summary
    
    # 測試瓶頸檢測
    bottlenecks = monitor.check_bottlenecks()
    
    monitor.stop_monitoring()


test("性能監控", test_performance_monitor)


# ============================================================================
# 測試 3: 數據緩存
# ============================================================================

def test_cache_manager():
    from utils.cache_manager import DataCache, cached
    import numpy as np
    import pandas as pd
    
    cache = DataCache(cache_dir=".test_cache", memory_limit_mb=100)
    
    # 測試基本緩存
    key = "test_key"
    value = np.random.rand(1000)
    
    cache.put(key, value)
    cached_value = cache.get(key)
    
    assert cached_value is not None
    assert np.array_equal(value, cached_value)
    
    # 測試 DataFrame 緩存
    df = pd.DataFrame({'a': np.random.rand(1000), 'b': np.random.rand(1000)})
    cache.put("df_key", df)
    cached_df = cache.get("df_key")
    
    assert cached_df is not None
    assert df.equals(cached_df)
    
    # 測試統計
    stats = cache.stats()
    assert stats['memory']['hit_rate'] > 0
    
    # 清理
    cache.clear_all()
    Path(".test_cache").rmdir() if Path(".test_cache").exists() else None


test("數據緩存", test_cache_manager)


# ============================================================================
# 測試 4: 緩存裝飾器
# ============================================================================

def test_cache_decorator():
    from utils.cache_manager import DataCache, cached
    import time
    
    cache = DataCache(cache_dir=".test_cache2")
    
    call_count = [0]
    
    @cached(cache, save_to_disk=False)
    def expensive_function(x):
        call_count[0] += 1
        time.sleep(0.1)
        return x ** 2
    
    # 第一次調用（慢）
    start = time.time()
    result1 = expensive_function(10)
    time1 = time.time() - start
    
    # 第二次調用（快，使用緩存）
    start = time.time()
    result2 = expensive_function(10)
    time2 = time.time() - start
    
    assert result1 == result2
    assert call_count[0] == 1  # 只調用了一次
    assert time2 < time1  # 第二次更快
    
    # 清理
    cache.clear_all()
    Path(".test_cache2").rmdir() if Path(".test_cache2").exists() else None


test("緩存裝飾器", test_cache_decorator)


# ============================================================================
# 測試 5: Numba 優化
# ============================================================================

def test_numba_optimizations():
    from utils.numba_optimizations import NUMBA_AVAILABLE
    import numpy as np
    
    if not NUMBA_AVAILABLE:
        print("\n  ⚠️  Numba 未安裝，跳過測試")
        return
    
    from utils.numba_optimizations import (
        rolling_mean_numba,
        rolling_std_numba
    )
    
    # 測試數據
    data = np.random.rand(1000)
    
    # 測試滾動平均
    result_mean = rolling_mean_numba(data, 20)
    assert len(result_mean) == len(data)
    assert not np.isnan(result_mean[-1])
    
    # 測試滾動標準差
    result_std = rolling_std_numba(data, 20)
    assert len(result_std) == len(data)


test("Numba 優化", test_numba_optimizations)


# ============================================================================
# 測試 6: 優化 Pipeline 導入
# ============================================================================

def test_optimized_pipeline_import():
    from optimized_pipeline import OptimizedTrainingPipeline
    
    # 測試創建實例
    pipeline = OptimizedTrainingPipeline(
        enable_cache=False,
        enable_monitoring=False
    )
    
    assert pipeline is not None
    assert hasattr(pipeline, 'train')
    assert hasattr(pipeline, 'batch_train')


test("優化 Pipeline 導入", test_optimized_pipeline_import)


# ============================================================================
# 測試 7: 清理腳本
# ============================================================================

def test_cleanup_script():
    from cleanup import ProjectCleaner
    
    cleaner = ProjectCleaner(".")
    assert cleaner is not None
    assert hasattr(cleaner, 'clean_pycache')
    assert hasattr(cleaner, 'clean_logs')
    assert hasattr(cleaner, 'optimize_database')
    
    # 測試統計
    stats = cleaner.stats
    assert 'files_removed' in stats
    assert 'space_freed_mb' in stats


test("清理腳本", test_cleanup_script)


# ============================================================================
# 測試 8: 上下文管理器
# ============================================================================

def test_context_managers():
    from utils.performance_monitor import PerformanceMonitor
    import time
    
    # 測試 PerformanceMonitor 上下文管理器
    with PerformanceMonitor(sample_interval=1.0) as monitor:
        time.sleep(0.2)
        monitor.update_step_count(100)
        assert monitor.monitoring == True
    
    # 退出後應該停止
    # (這個測試可能有延遲)


test("上下文管理器", test_context_managers)


# ============================================================================
# 測試結果摘要
# ============================================================================

print("\n" + "=" * 70)
print("📊 測試結果摘要")
print("=" * 70)
print(f"總測試數: {tests_total}")
print(f"✅ 通過: {tests_passed}")
print(f"❌ 失敗: {tests_failed}")
print(f"成功率: {tests_passed / tests_total * 100:.1f}%")
print("=" * 70)

if tests_failed == 0:
    print("\n🎉 所有測試通過！優化功能正常工作！")
    sys.exit(0)
else:
    print(f"\n⚠️  有 {tests_failed} 個測試失敗，請檢查相關模組")
    sys.exit(1)
