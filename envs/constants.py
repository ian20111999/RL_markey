"""
環境常數配置

定義 MarketMakingEnvV2 中使用的所有 magic numbers，
提高程式碼可讀性和可維護性。
"""

# =============================================================================
# Reward 計算常數
# =============================================================================

# Reward Debug
REWARD_DEBUG_INTERVAL = 100  # 每 N 步輸出一次 reward 調試資訊
REWARD_WARNING_THRESHOLD = 100.0  # Scaled reward 超過此值時警告

# 逆選擇檢測
ADVERSE_SELECTION_THRESHOLD = 0.0001  # 價格變動超過 0.01% 視為逆選擇

# =============================================================================
# 特徵計算常數
# =============================================================================

# 預設窗口大小
DEFAULT_VOLATILITY_WINDOWS = [5, 15, 60]  # 5分鐘、15分鐘、1小時
DEFAULT_MOMENTUM_WINDOWS = [5, 15]
DEFAULT_TREND_WINDOWS = [60, 240, 1440]  # 1h, 4h, 1d
DEFAULT_VOLUME_MA_WINDOW = 20

# 進階特徵窗口
DEFAULT_ORDER_FLOW_WINDOW = 20
DEFAULT_VWAP_WINDOW = 60
DEFAULT_MTF_WINDOWS = [15, 60, 240]
DEFAULT_EWMA_SPAN = 20

# =============================================================================
# 指標計算常數
# =============================================================================

# Sharpe Ratio
MINUTES_PER_YEAR = 365 * 24 * 60  # 年化因子（假設每步 1 分鐘）
RISK_FREE_RATE = 0.0  # 無風險利率

# VaR 和 ES
VAR_PERCENTILE = 5  # 95% VaR 使用第 5 百分位

# 庫存上限警告
INVENTORY_WARNING_RATIO = 0.9  # 庫存達到最大值的 90% 時計入警告

# =============================================================================
# 記憶體管理常數
# =============================================================================

# MetricsTracker 緩衝區大小
DEFAULT_METRICS_BUFFER_SIZE = 10000  # 保留最近 10000 步的資料

# =============================================================================
# 填充模型常數
# =============================================================================

# 簡單填充模型機率
SIMPLE_BASE_FILL_PROB = 0.3

# 中等填充模型機率
MODERATE_BASE_FILL_PROB = 0.2

# 真實填充模型機率
REALISTIC_BASE_FILL_PROB = 0.15

# =============================================================================
# 訓練常數
# =============================================================================

# 學習率
DEFAULT_LEARNING_RATE = 3e-4
STABILIZATION_LEARNING_RATE = 3e-5

# Batch Size
DEFAULT_BATCH_SIZE = 256
LARGE_BATCH_SIZE = 512

# Episode Length
DEFAULT_EPISODE_LENGTH = 1000
LONG_EPISODE_LENGTH = 2000

# =============================================================================
# 數值穩定性常數
# =============================================================================

EPSILON = 1e-8  # 避免除以零
MIN_PRICE = 1e-6  # 最小價格（避免除以零）
MIN_STD = 1e-8  # 最小標準差（Sharpe Ratio）

# =============================================================================
# 驗證常數
# =============================================================================

# 學習率更新驗證
LR_UPDATE_TOLERANCE = 1e-6  # 學習率驗證容差

# Reward 範圍
EXPECTED_REWARD_RANGE = (-10, 10)  # 預期的 scaled reward 範圍

# =============================================================================
# 性能配置
# =============================================================================

# Numba
NUMBA_CACHE = True  # 是否啟用 Numba 緩存
NUMBA_PARALLEL = False  # 是否啟用 Numba 並行（需謹慎使用）

# =============================================================================
# 日誌與調試
# =============================================================================

# 日誌頻率
LOG_INTERVAL = 1000  # 每 N 步輸出一次訓練日誌
EVAL_INTERVAL = 10000  # 每 N 步執行一次評估
SAVE_INTERVAL = 50000  # 每 N 步保存一次模型

# TensorBoard 更新頻率
TB_LOG_INTERVAL = 100  # TensorBoard 日誌更新頻率

# =============================================================================
# 資料處理
# =============================================================================

# CSV 欄位名稱
CSV_REQUIRED_COLUMNS = ["close"]
CSV_OPTIONAL_COLUMNS = ["high", "low", "volume", "open", "timestamp", "datetime"]

# =============================================================================
# 實用函數
# =============================================================================

def get_constant_info():
    """獲取所有常數資訊（用於調試）"""
    import inspect
    import sys
    
    current_module = sys.modules[__name__]
    constants = {}
    
    for name, value in inspect.getmembers(current_module):
        if name.isupper() and not name.startswith('_'):
            constants[name] = value
    
    return constants


def print_constants():
    """打印所有常數（用於調試）"""
    constants = get_constant_info()
    
    print("="*60)
    print("Environment Constants")
    print("="*60)
    
    categories = {
        "Reward": [],
        "Feature": [],
        "Metrics": [],
        "Memory": [],
        "Training": [],
        "Numerical": [],
        "Validation": [],
        "Performance": [],
        "Logging": [],
        "Data": [],
    }
    
    for name, value in sorted(constants.items()):
        categorized = False
        for category in categories:
            if category.upper() in name or name.startswith(category.upper()):
                categories[category].append((name, value))
                categorized = True
                break
        
        if not categorized:
            if "DEFAULT" in name:
                categories["Feature"].append((name, value))
            elif "EPSILON" in name or "MIN_" in name:
                categories["Numerical"].append((name, value))
            elif "INTERVAL" in name or "LOG" in name:
                categories["Logging"].append((name, value))
    
    for category, items in categories.items():
        if items:
            print(f"\n{category} Constants:")
            for name, value in items:
                print(f"  {name:40s} = {value}")
    
    print("="*60)


if __name__ == "__main__":
    print_constants()
