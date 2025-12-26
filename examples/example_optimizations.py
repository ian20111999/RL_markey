#!/usr/bin/env python3
"""
優化功能使用示例
展示如何在實際訓練中使用所有優化功能
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("🚀 RL Market 優化功能示例\n")

# ============================================================================
# 示例 1: 統一日誌系統
# ============================================================================

print("=" * 70)
print("示例 1: 統一日誌系統")
print("=" * 70)

from utils.logging_config import setup_logging, get_logger, log_performance
import time

# 初始化日誌（只需在程序開始調用一次）
setup_logging(level="INFO", log_dir="logs")

# 獲取 logger
logger = get_logger(__name__)

logger.info("這是 INFO 日誌")
logger.warning("這是 WARNING 日誌")
logger.debug("這是 DEBUG 日誌（不會顯示，因為級別是 INFO）")

# 使用性能監控裝飾器
@log_performance()
def simulate_training_step():
    """模擬訓練步驟"""
    time.sleep(0.1)
    return 42

result = simulate_training_step()
print(f"✓ 訓練步驟完成，結果: {result}\n")


# ============================================================================
# 示例 2: 性能監控
# ============================================================================

print("=" * 70)
print("示例 2: 性能監控")
print("=" * 70)

try:
    from utils.performance_monitor import PerformanceMonitor
    
    # 創建監控器
    monitor = PerformanceMonitor(sample_interval=1.0)
    
    # 使用上下文管理器（推薦）
    print("開始監控訓練性能...")
    with monitor:
        # 模擬訓練循環
        for step in range(5):
            time.sleep(0.2)
            monitor.update_step_count(100)  # 更新步數
            
            # 每 2 步打印一次
            if step % 2 == 0:
                metrics = monitor.get_current_metrics()
                print(f"步數 {step * 100}: CPU {metrics.get('cpu_percent', 0):.1f}%, "
                      f"內存 {metrics.get('memory_mb', 0):.0f}MB")
    
    print("\n✓ 性能監控完成，摘要已自動生成\n")

except ImportError as e:
    print(f"⚠️  跳過性能監控示例: {e}")
    print("提示: 運行 'pip install psutil' 來啟用性能監控\n")


# ============================================================================
# 示例 3: 數據緩存
# ============================================================================

print("=" * 70)
print("示例 3: 數據緩存")
print("=" * 70)

from utils.cache_manager import DataCache, cached
import numpy as np

# 創建緩存管理器
cache = DataCache(cache_dir=".example_cache", memory_limit_mb=100)

# 使用裝飾器緩存函數結果
@cached(cache, save_to_disk=False)
def expensive_computation(size):
    """模擬昂貴的計算"""
    print(f"  執行昂貴計算（生成 {size} 個隨機數）...")
    time.sleep(0.5)
    return np.random.rand(size)

# 第一次調用（慢）
print("第一次調用（無緩存）:")
start = time.time()
result1 = expensive_computation(10000)
time1 = time.time() - start
print(f"✓ 完成，耗時: {time1:.3f}s")

# 第二次調用（快，使用緩存）
print("\n第二次調用（使用緩存）:")
start = time.time()
result2 = expensive_computation(10000)
time2 = time.time() - start
print(f"✓ 完成，耗時: {time2:.3f}s")

print(f"\n加速比: {time1 / time2:.1f}x")

# 查看緩存統計
stats = cache.stats()
print(f"緩存命中率: {stats['memory']['hit_rate']:.2%}")
print(f"內存使用: {stats['memory']['memory_mb']:.2f}MB\n")

# 清理示例緩存
cache.clear_all()
if Path(".example_cache").exists():
    Path(".example_cache").rmdir()


# ============================================================================
# 示例 4: 完整訓練流程（偽代碼）
# ============================================================================

print("=" * 70)
print("示例 4: 完整訓練流程（偽代碼）")
print("=" * 70)

print("""
from utils.logging_config import setup_logging
from utils.performance_monitor import PerformanceMonitor
from utils.cache_manager import cache_result

# 1. 初始化日誌
setup_logging(level="INFO")
logger = get_logger(__name__)

# 2. 使用緩存裝飾器
@cache_result()
def load_market_data(symbol):
    logger.info(f"加載數據: {symbol}")
    return pd.read_csv(f"data/{symbol}.csv")

# 3. 訓練時啟用監控
with PerformanceMonitor() as monitor:
    # 加載數據（會被緩存）
    data = load_market_data("BTCUSDT")
    
    # 創建環境
    env = create_environment(data)
    
    # 訓練模型
    for episode in range(1000):
        state = env.reset()
        done = False
        
        while not done:
            action = model.predict(state)
            state, reward, done, _ = env.step(action)
            monitor.update_step_count(1)  # 更新步數
        
        # 每 100 個 episode 記錄一次
        if episode % 100 == 0:
            logger.info(f"Episode {episode} 完成")
    
    # 監控會自動生成性能報告

logger.info("訓練完成！")
""")

print("✓ 這是完整訓練流程的示例代碼\n")


# ============================================================================
# 總結
# ============================================================================

print("=" * 70)
print("✨ 示例完成")
print("=" * 70)
print("""
關鍵要點:
1. 使用 setup_logging() 初始化統一日誌系統
2. 使用 PerformanceMonitor 追蹤訓練性能
3. 使用 @cached 裝飾器緩存重複計算
4. 定期運行 cleanup.py 清理項目

下一步:
• 閱讀 OPTIMIZATION_SUMMARY.md 了解詳細用法
• 閱讀 OPTIMIZATION_REPORT.md 了解技術細節
• 運行 test_optimizations.py 驗證功能
• 使用 optimized_pipeline.py 進行實際訓練

如有問題，請查看文檔或提交 Issue！
""")

print("🎉 感謝使用 RL Market 優化功能！")
