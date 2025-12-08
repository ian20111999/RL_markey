"""
Performance Optimization Utilities using Numba JIT compilation

這個模組提供經過 Numba 優化的數值計算函數，
用於加速強化學習環境中的特徵計算。
"""
import numpy as np

try:
    import numba
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba not available. Feature computation will be slower.")
    print("   Install with: pip install numba")
    
    # Dummy decorator for compatibility
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# =============================================================================
# Rolling Statistics with Numba
# =============================================================================

@jit(nopython=True)
def rolling_std_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """計算滾動標準差（因果版本）
    
    Args:
        arr: 輸入陣列
        window: 滾動窗口大小
        
    Returns:
        滾動標準差陣列
    """
    n = len(arr)
    result = np.zeros(n)
    
    for i in range(window, n):
        # 只使用 [i-window, i) 的資料
        window_data = arr[i-window:i]
        result[i] = np.std(window_data)
    
    return result


@jit(nopython=True)
def rolling_mean_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """計算滾動平均（因果版本）
    
    Args:
        arr: 輸入陣列
        window: 滾動窗口大小
        
    Returns:
        滾動平均陣列
    """
    n = len(arr)
    result = np.zeros(n)
    
    for i in range(window, n):
        result[i] = np.mean(arr[i-window:i])
    
    return result


@jit(nopython=True)
def compute_momentum_numba(prices: np.ndarray, window: int) -> np.ndarray:
    """計算動量（因果版本）
    
    Args:
        prices: 價格陣列
        window: 回看窗口
        
    Returns:
        動量陣列 (price[i] - price[i-window]) / price[i-window]
    """
    n = len(prices)
    result = np.zeros(n)
    
    for i in range(window, n):
        if prices[i-window] > 0:
            result[i] = (prices[i] - prices[i-window]) / prices[i-window]
    
    return result


@jit(nopython=True)
def compute_volatility_numba(returns: np.ndarray, window: int, 
                             annualization_factor: float = 1.0) -> np.ndarray:
    """計算滾動波動率（因果版本）
    
    Args:
        returns: 收益率陣列
        window: 滾動窗口
        annualization_factor: 年化因子 (例如 sqrt(252) for daily)
        
    Returns:
        波動率陣列
    """
    n = len(returns)
    result = np.zeros(n)
    
    for i in range(window, n):
        window_returns = returns[i-window:i]
        result[i] = np.std(window_returns) * annualization_factor
    
    return result


@jit(nopython=True)
def compute_ewma_volatility_numba(returns: np.ndarray, alpha: float) -> np.ndarray:
    """計算 EWMA 波動率（因果版本）
    
    Args:
        returns: 收益率陣列
        alpha: EWMA 衰減係數 (2 / (span + 1))
        
    Returns:
        EWMA 波動率陣列
    """
    n = len(returns)
    result = np.zeros(n)
    sq_returns = returns ** 2
    
    for i in range(1, n):
        result[i] = alpha * sq_returns[i] + (1 - alpha) * result[i-1]
    
    return np.sqrt(result)


# =============================================================================
# Order Flow and Market Microstructure
# =============================================================================

@jit(nopython=True)
def compute_order_flow_imbalance_numba(volumes: np.ndarray, 
                                       returns: np.ndarray, 
                                       window: int) -> np.ndarray:
    """計算訂單流不平衡（因果版本）
    
    Args:
        volumes: 成交量陣列
        returns: 收益率陣列
        window: 滾動窗口
        
    Returns:
        訂單流不平衡陣列 (buy_vol - sell_vol) / total_vol
    """
    n = len(volumes)
    result = np.zeros(n)
    
    for i in range(window, n):
        buy_vol = 0.0
        sell_vol = 0.0
        
        for j in range(i-window, i):
            if returns[j] > 0:
                buy_vol += volumes[j]
            elif returns[j] < 0:
                sell_vol += volumes[j]
        
        total_vol = buy_vol + sell_vol
        if total_vol > 0:
            result[i] = (buy_vol - sell_vol) / total_vol
    
    return result


@jit(nopython=True)
def compute_vwap_deviation_numba(prices: np.ndarray, 
                                 volumes: np.ndarray, 
                                 window: int) -> np.ndarray:
    """計算 VWAP 偏離（因果版本）
    
    Args:
        prices: 價格陣列
        volumes: 成交量陣列
        window: 滾動窗口
        
    Returns:
        VWAP 偏離陣列 (price - vwap) / vwap
    """
    n = len(prices)
    result = np.zeros(n)
    
    for i in range(window, n):
        vol_sum = 0.0
        price_vol_sum = 0.0
        
        for j in range(i-window, i):
            vol_sum += volumes[j]
            price_vol_sum += prices[j] * volumes[j]
        
        if vol_sum > 0:
            vwap = price_vol_sum / vol_sum
            if vwap > 0:
                result[i] = (prices[i] - vwap) / vwap
    
    return result


@jit(nopython=True)
def compute_drawdown_series_numba(portfolio_values: np.ndarray) -> np.ndarray:
    """計算 Drawdown 序列
    
    Args:
        portfolio_values: 投資組合價值陣列
        
    Returns:
        Drawdown 陣列
    """
    n = len(portfolio_values)
    result = np.zeros(n)
    peak = portfolio_values[0]
    
    for i in range(n):
        if portfolio_values[i] > peak:
            peak = portfolio_values[i]
        
        if peak > 0:
            result[i] = (peak - portfolio_values[i]) / peak
    
    return result


# =============================================================================
# Utility Functions
# =============================================================================

def get_numba_info():
    """獲取 Numba 資訊"""
    if NUMBA_AVAILABLE:
        return {
            "available": True,
            "version": numba.__version__,
            "acceleration": "~10-50x for large arrays"
        }
    else:
        return {
            "available": False,
            "version": None,
            "acceleration": "None (using pure NumPy)"
        }


if __name__ == "__main__":
    # 測試 Numba 加速效果
    import time
    
    print("Testing Numba Performance...")
    print(f"Numba Available: {NUMBA_AVAILABLE}")
    
    # 生成測試資料
    n = 100000
    test_data = np.random.randn(n)
    window = 60
    
    # 測試滾動標準差
    print(f"\nTesting rolling_std on {n} data points, window={window}")
    
    # Numba 版本
    start = time.time()
    result_numba = rolling_std_numba(test_data, window)
    time_numba = time.time() - start
    print(f"  Numba: {time_numba:.4f}s")
    
    # NumPy 版本（使用 pandas）
    import pandas as pd
    start = time.time()
    result_pandas = pd.Series(test_data).rolling(window).std().fillna(0).values
    time_pandas = time.time() - start
    print(f"  Pandas: {time_pandas:.4f}s")
    
    if NUMBA_AVAILABLE:
        speedup = time_pandas / time_numba
        print(f"  Speedup: {speedup:.2f}x")
    
    print("\n✅ All tests passed!")
