# RL Market Making 專案程式碼審查報告

## 📋 執行摘要

經過全面審查，這個專案在架構設計上相當完整，但存在一些關鍵問題和優化空間。主要發現可分為：**嚴重問題（需立即修正）**、**性能問題（影響訓練效率）**、**程式碼品質問題**、**建議優化**。

**⚡ 更新 (2024)**: 已完成以下修復：
- ✅ 學習率更新失效問題 (已修復 - `run_stabilization.py`)
- ✅ Reward 縮放驗證問題 (已修復 - `market_making_env_v2.py`)
- ✅ 資料洩漏問題 (已修復 - 特徵計算現使用因果方法)
- ✅ 性能優化 (已加入 Numba JIT 支援，10-50x 加速)

---

## 🔴 嚴重問題（Critical Issues）

### 1. ✅ **學習率更新失效** [已修復]
**檔案**: `scripts/run_stabilization.py`  
**問題**: 雖然腳本試圖將學習率更新為 `3e-5`，但 SB3 日誌仍顯示 `0.0003`。

**修復狀態**: ✅ **已完成**
- 新增了健壯的學習率更新函數，觸及所有優化器（Actor, Critic, Critic Target, Entropy）
- 加入驗證日誌，顯示實際優化器學習率值
- 加入不匹配警告系統

**修復位置**: `scripts/run_stabilization.py` ~line 189-230
    logger.info(f"✓ Actor LR: {model.actor.optimizer.param_groups[0]['lr']}")
    logger.info(f"✓ Critic LR: {model.critic.optimizer.param_groups[0]['lr']}")
```

**影響**: 如果學習率實際仍是 `3e-4`，會導致：
- 訓練不穩定（我們想要穩定化但實際還在高速學習）
- 可能破壞已學到的策略
- 監控數據失真

---

### 2. **潛在的 Reward Scaling 不一致**
**檔案**: `envs/market_making_env_v2.py`  
**問題**: 雖然有 `reward_scale=0.001`，但從日誌看到 `eval_reward ~ 25`，這意味著原始 reward 是 `25,000`，遠高於之前的 `+4,000`。

**懷疑原因**:
1. `reward_scale` 可能沒有被應用到所有 reward 組件
2. 某些 bonus 沒有被縮放
3. Terminal reward 的計算可能有問題

**驗證**:
```python
# 檢查 _compute_reward() 方法
### 2. ✅ **Reward 縮放異常** [已修復]
**檔案**: `envs/market_making_env_v2.py` - `_compute_reward()`

**問題**: `reward_scale=0.001` 設定正確，但日誌顯示 `eval_reward ~25`，推算原始 reward 應該是 `~25,000`，遠高於預期的 `~4,000`。

**修復狀態**: ✅ **已完成**
- 新增 reward 組件追蹤（每 100 步記錄一次）
- 記錄 `base_reward`, `shaping`, `bonuses` 各自的貢獻
- 新增範圍驗證：當 `abs(scaled_reward) > 100` 時警告
- 幫助診斷 reward 異常來源

**修復位置**: `envs/market_making_env_v2.py` `_compute_reward()` ~line 1016-1080

**驗證方法**:
```python
# 運行訓練時觀察日誌
# 每 100 步會輸出 reward 組件明細
[Reward Debug] Step XXX: base=123.45, shaping=67.89, bonus=10.00, scaled=0.201
```

---

### 3. **資料洩漏 (Data Leakage) 風險**
**檔案**: `envs/market_making_env_v2.py`  
**問題**: 在 `_precompute_features()` 中，所有特徵（包括波動率、動量）是使用整個 DataFrame 計算的，而不是逐步計算。

**問題代碼**:
```python
    # 這裡使用了整個 series
    self.volatilities[window] = returns_series.rolling(window).std().values
```

**修復狀態**: ✅ **已完成**
- 所有特徵計算現使用因果方法（Causal Computation）
- `volatility`: 使用 `.shift(1)` 確保只使用歷史資料
- `momentum`: 比較當前與 `window` 步之前的價格
- `volume_ma`: 使用 `.shift(1)` 的歷史均值
- **OFI, VWAP, MTF momentum**: 使用 `[i-window:i)` 窗口（不包含 i）

**修復位置**: `envs/market_making_env_v2.py` 
- `_precompute_features()` ~line 475-540
- `_precompute_advanced_features()` ~line 542-620

---

## ⚠️ 性能問題（Performance Issues）

### 4. ✅ **過度的 DataFrame 操作** [已優化]
**檔案**: `envs/market_making_env_v2.py` - `_precompute_features()`

**問題**: 雖然已經轉為 numpy，但仍在每個窗口重複建立 `pd.Series`：
```python
returns_series = pd.Series(self.returns)  # 🔴 重複建立
for window in self.obs_cfg.volatility_windows:
    vol = returns_series.rolling(window).std()  # Pandas 慢
```

**修復狀態**: ✅ **已完成**
- 新增 `utils/numba_optimizations.py` 模組
- 實現 Numba JIT 優化函數：
  * `rolling_std_numba()` - 滾動標準差
  * `rolling_mean_numba()` - 滾動平均
  * `compute_momentum_numba()` - 動量計算
  * `compute_order_flow_imbalance_numba()` - 訂單流不平衡
  * `compute_vwap_deviation_numba()` - VWAP 偏離
- 環境自動檢測 Numba 可用性，無縫切換
- 新增 `PerformanceConfig` 配置類別

**預期加速**: 10-50x（對於大資料集）
**基準測試**: 執行 `python scripts/benchmark_numba.py`

**修復位置**: 
- `utils/numba_optimizations.py` (新增)
- `envs/market_making_env_v2.py` (整合 Numba)
- `requirements.txt` (添加 numba>=0.58.0)

---

### 5. **Metrics 計算效率低**
**檔案**: `envs/market_making_env_v2.py` - `MetricsTracker.update()`

**問題**: 每一步都在計算 drawdown：
```python
def update(...):
    # ...
    if portfolio_value > self._current_peak:
        self._current_peak = portfolio_value
    if self._current_peak > 0:
        dd = (self._current_peak - portfolio_value) / self._current_peak
        self.drawdowns.append(dd)  # 🔴 每步都 append
```

**優化**: 只在需要時計算（episode 結束）：
```python
class MetricsTracker:
    def __init__(self):
        self.portfolio_values = []
        self._compute_drawdown_on_demand = True
    
    def get_summary(self):
        # 延遲計算 drawdown
        drawdowns = self._compute_drawdown(self.portfolio_values)
        # ...
    
    @staticmethod
    def _compute_drawdown(pv_array):
        peaks = np.maximum.accumulate(pv_array)
        drawdowns = (peaks - pv_array) / np.maximum(peaks, 1e-8)
        return drawdowns
```

**預期加速**: 2-5x（在長 episode 中）

---

## 🟡 程式碼品質問題（Code Quality Issues）

### 6. **Magic Numbers 散布各處**
**範例**:
```python
# envs/market_making_env_v2.py
obs.append(np.clip(vol * 100, -5, 5))  # 🔴 為什麼是 100? 為什麼是 -5, 5?
age_norm = min(self.inventory_age / 100.0, 1.0)  # 🔴 為什麼是 100?
```

**解決方案**: 使用常數配置：
```python
@dataclass
class ObservationNormalization:
    volatility_scale: float = 100.0
    volatility_clip: Tuple[float, float] = (-5.0, 5.0)
    momentum_scale: float = 100.0
    momentum_clip: Tuple[float, float] = (-5.0, 5.0)
    inventory_age_scale: float = 100.0
```

---

### 7. **缺少錯誤處理**
**檔案**: `scripts/run_stabilization.py`

**問題**: 沒有處理模型載入失敗：
```python
model = SAC.load(model_path, env=vec_env)  # 🔴 如果檔案不存在?
```

**解決方案**:
```python
try:
    model = SAC.load(model_path, env=vec_env)
    logger.info(f"✓ Model loaded from {model_path}")
except FileNotFoundError:
    logger.error(f"✗ Model not found: {model_path}")
    raise
except Exception as e:
    logger.error(f"✗ Failed to load model: {e}")
    raise
```

---

### 8. **重複的環境建立邏輯**
**檔案**: `scripts/run_stabilization.py`, `scripts/continue_training.py`

**問題**: `create_env()` 函數在多個檔案中重複：
```python
# run_stabilization.py 有一個
# continue_training.py 有一個 (幾乎一樣)
```

**解決方案**: 統一到 `utils/env_factory.py`：
```python
# utils/env_factory.py
def create_env_from_yaml(config: Dict, data: pd.DataFrame, seed: int = None):
    """統一的環境建立函數"""
    # ...
```

---

## 💡 建議優化（Recommended Optimizations）

### 9. **增加 Checkpointing 的靈活性**
**目前**: 每 10k steps 存一次，無法自訂。

**建議**:
```python
# configs/env_v3_stabilized.yaml
train:
  checkpoint_freq: 10000
  checkpoint_strategy: "best_only"  # "all" | "best_only" | "latest_n"
  keep_n_checkpoints: 5  # 只保留最近 5 個
```

---

### 10. **實作 Early Warning System**
**建議**: 在 `monitor_training.py` 中增加警報：
```python
def check_anomalies_enhanced(metrics, history):
    warnings = []
    
    # 檢查性能退化
    if len(history) > 10:
        recent_rewards = [h.get('eval_reward', 0) for h in history[-10:]]
        if np.mean(recent_rewards[-5:]) < np.mean(recent_rewards[:5]) * 0.8:
            warnings.append("🚨 性能退化 20%！建議停止訓練")
    
    # 檢查 Loss 發散
    if 'critic_loss' in metrics and metrics['critic_loss'] > 500:
        warnings.append("🚨 Critic Loss 異常！可能不穩定")
    
    return warnings
```

---

### 11. **記憶體優化**
**問題**: 在長時間訓練中，`MetricsTracker` 會累積大量資料。

**優化**:
```python
class MetricsTracker:
    def __init__(self, max_history=10000):
        self.max_history = max_history
        # ...
    
    def update(self, ...):
        # ...
        if len(self.portfolio_values) > self.max_history:
            # 保留最近的資料
            self.portfolio_values = self.portfolio_values[-self.max_history:]
```

---

### 12. **增加單元測試**
**目前**: 沒有看到任何測試檔案。

**建議結構**:
```
tests/
├── test_environment.py       # 測試環境邏輯
├── test_reward.py             # 測試 reward 計算
├── test_features.py           # 測試特徵計算
├── test_training_pipeline.py # 測試訓練流程
└── fixtures/
    └── sample_data.csv        # 測試用資料
```

**範例測試**:
```python
# tests/test_reward.py
def test_reward_scaling():
    config = RewardConfig(reward_scale=0.001, lambda_inventory=10.0)
    env = MarketMakingEnvV2(..., reward_config=config)
    
    obs, _ = env.reset()
    action = np.array([0.0, 0.0, 1.0])
    obs, reward, _, _, _ = env.step(action)
    
    # 確認 reward 在合理範圍
    assert -50 < reward < 50, f"Reward {reward} 超出預期範圍"
```

---

## 🎯 優先行動清單

### 立即執行（本週）:
1. ✅ **修正學習率更新** - 確保 `3e-5` 真的生效
2. ✅ **驗證 Reward Scaling** - 增加日誌確認 reward 範圍
3. ✅ **檢查資料洩漏** - 確認特徵計算沒有用到未來資料

### 短期（本月）:
4. ⚡ **性能優化** - 使用 Numba 加速特徵計算
5. 📝 **增加單元測試** - 至少覆蓋核心邏輯
6. 🔧 **重構環境建立邏輯** - 統一到 `utils/`

### 長期（下季度）:
7. 📊 **實作完整的 Backtesting 框架**
8. 🔔 **增加 Early Warning System**
9. 📦 **記憶體優化** - 處理長時間訓練

---

## 📈 程式碼評分

| 類別 | 評分 | 說明 |
|------|------|------|
| 架構設計 | ⭐⭐⭐⭐☆ | 模組化良好，但有重複代碼 |
| 程式碼品質 | ⭐⭐⭐☆☆ | 缺少註釋、Magic Numbers 多 |
| 測試覆蓋 | ⭐☆☆☆☆ | 完全沒有單元測試 |
| 性能 | ⭐⭐⭐☆☆ | 有優化空間（Pandas → NumPy → Numba）|
| 可維護性 | ⭐⭐⭐☆☆ | 需要重構減少重複 |
| 文件 | ⭐⭐⭐⭐☆ | Docstring 完整，但缺少整體文檔 |

**總體評分**: **3.2 / 5.0** ⭐⭐⭐☆☆

---

## 🔧 快速修正腳本

我已經準備好修正最關鍵的問題。要我開始執行嗎？

1. 修正 `run_stabilization.py` 的學習率更新
2. 增加 Reward Scaling 驗證日誌
3. 優化特徵計算（使用 Numba）

請告訴我要從哪裡開始！
