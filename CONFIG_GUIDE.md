# 📝 配置文件說明

本專案包含多個配置文件，用於不同的場景和功能。以下是完整的配置文件結構說明。

## 📁 配置文件結構

```
configs/
├── default.yaml           # ⭐ 主要配置（pipeline.py 使用）
├── pipeline_config.yaml   # 進階 pipeline 配置
└── [其他配置文件]         # 舊版或實驗性配置
```

## ⭐ default.yaml - 主要配置文件

這是 `pipeline.py` 使用的核心配置文件。

### 完整配置說明

```yaml
# =============================================================================
# 環境配置
# =============================================================================
env:
  id: "MarketMakingEnv"
  data_file: "data/btc_usdt_1m_2023.csv"  # 會被 pipeline 自動覆蓋
  
  # 基本參數
  episode_length: 1440        # 每個 episode 的步數（1440 = 1天）
  initial_cash: 10000.0       # 初始資金（會被自動調整）
  max_inventory: 2.0          # 最大持倉數量
  random_start: true          # 隨機起始位置（增加數據多樣性）
  
  # 市場參數
  fee_rate: 0.0004           # 手續費率（0.04%，幣安現貨標準）
  base_spread: 60.0          # 基礎價差（會被自動調整）

# =============================================================================
# 數據增強
# =============================================================================
data_augmentation:
  enable_price_flip: true    # 價格翻轉增強（左右鏡像數據）

# =============================================================================
# 成交模型
# =============================================================================
fill_model:
  enabled: false             # 是否啟用真實成交模擬（簡化版設為 false）

# =============================================================================
# 觀察空間配置
# =============================================================================
observation:
  include_price: true        # 包含價格信息
  include_inventory: true    # 包含庫存信息
  include_time: true         # 包含時間特徵
  include_volatility: true   # 包含波動率
  include_momentum: true     # 包含動量指標
  include_volume: true       # 包含成交量
  include_inventory_age: true # 包含庫存持有時間
  include_trend: true        # 包含趨勢指標
  
  # 特徵窗口
  trend_windows: [60, 240, 1440]       # 1小時、4小時、1天
  volatility_windows: [5, 15, 60]      # 5分、15分、1小時
  momentum_windows: [5, 15]            # 5分、15分

# =============================================================================
# 獎勵函數配置（核心！）
# =============================================================================
reward:
  mode: "shaped"             # 獎勵模式：dense, sparse, shaped, hybrid
  
  # 獎勵縮放（非常重要！）
  reward_scale: 1.0e-6       # 全局獎勵縮放因子
  
  # 懲罰權重
  lambda_inventory: 20.0     # 庫存懲罰（防止過度持倉）
  lambda_turnover: 0.01      # 換手率懲罰（防止過度交易）
  lambda_inventory_age: 0.1  # 庫存老化懲罰（鼓勵快速平倉）
  
  # 其他參數
  gamma: 0.99                # 折扣因子
  sparse_scale: 0.01         # sparse 模式縮放
  terminal_bonus_weight: 0.3 # 終止獎勵權重
  
  # 獎勵加成
  spread_capture_bonus: 0.5  # 捕獲價差獎勵
  round_trip_bonus: 0.0      # 完整交易獎勵
  inventory_revert_bonus: 0.5 # 庫存回歸獎勵

# =============================================================================
# 動作空間配置
# =============================================================================
action:
  mode: "asymmetric"         # 對稱/非對稱報價
  allow_no_quote: true       # 允許不報價（觀望）
  max_spread_multiplier: 2.0 # 最大價差倍數
  min_spread_multiplier: 0.5 # 最小價差倍數

# =============================================================================
# 動態持倉限制
# =============================================================================
dynamic_position_limit:
  enabled: true              # 根據市況動態調整持倉限制

# =============================================================================
# 訓練參數（SAC 算法）
# =============================================================================
train:
  learning_rate: 0.00003     # 學習率（3e-5）
  batch_size: 256            # 批次大小
  buffer_size: 100000        # 經驗回放緩衝區大小
  gamma: 0.99                # 折扣因子
  tau: 0.02                  # Soft update 參數
  train_freq: 1              # 訓練頻率（每步）
  gradient_steps: 1          # 每次訓練的梯度步數
  ent_coef: "auto"           # 熵係數（自動調整）
  target_entropy: "auto"     # 目標熵（自動設定）
  learning_starts: 10000     # 開始訓練前的隨機探索步數
  net_arch: [256, 256]       # 神經網路架構
```

## 🎯 關鍵參數調整指南

### 1. 提高穩定性

如果訓練不穩定或損失爆炸：

```yaml
train:
  learning_rate: 0.00001     # 降低學習率
  batch_size: 512            # 增加批次大小

reward:
  lambda_inventory: 30.0     # 增加庫存懲罰
```

### 2. 加快學習速度

如果學習太慢：

```yaml
train:
  learning_rate: 0.0001      # 提高學習率
  learning_starts: 5000      # 減少初始探索

reward:
  lambda_inventory: 10.0     # 降低懲罰（允許更多探索）
```

### 3. 防止過度交易

如果 agent 交易過於頻繁：

```yaml
reward:
  lambda_turnover: 0.05      # 增加換手率懲罰
  lambda_inventory_age: 0.2  # 增加老化懲罰
```

### 4. 提高盈利能力

如果盈利不足：

```yaml
reward:
  spread_capture_bonus: 1.0  # 增加價差獎勵
  round_trip_bonus: 0.5      # 增加完整交易獎勵

env:
  max_inventory: 3.0         # 允許更大持倉
```

## 🔧 pipeline_config.yaml - 進階配置

用於 `integrated_pipeline.py` 和 `auto_pipeline.py`：

```yaml
# 質量檢查配置
quality_checks:
  enabled: true
  min_data_points: 100000
  max_missing_ratio: 0.01

# 重試策略
retry_strategy:
  max_retries: 3
  adaptive_tuning: true

# 生產就緒檢查
production_checks:
  min_sharpe_ratio: 0.5
  max_drawdown: 0.3
  min_win_rate: 0.45

# 指標追蹤
metrics:
  enabled: true
  db_path: "metrics.db"
```

## 💡 最佳實踐

### 針對不同幣種調整

**高波動幣種（如 MEME 幣）：**
```yaml
env:
  max_inventory: 1.0         # 減少持倉
  fee_rate: 0.001            # 提高手續費模擬

reward:
  lambda_inventory: 40.0     # 大幅增加庫存懲罰
```

**穩定幣種（如 BTC、ETH）：**
```yaml
env:
  max_inventory: 5.0         # 可以持有更多

reward:
  lambda_inventory: 10.0     # 適度懲罰
```

### 針對不同市況

**震盪市：**
```yaml
reward:
  spread_capture_bonus: 1.0  # 鼓勵賺取價差
```

**趨勢市：**
```yaml
reward:
  round_trip_bonus: 1.0      # 鼓勵順勢交易
  lambda_inventory: 10.0     # 允許持倉
```

## 🧪 實驗性配置

如果你想嘗試新的配置，建議：

1. **複製 default.yaml**：
   ```bash
   cp configs/default.yaml configs/my_experiment.yaml
   ```

2. **修改參數**

3. **使用新配置訓練**：
   ```bash
   # 需要修改 pipeline.py 讀取不同的配置文件
   # 或者直接編輯 default.yaml 後訓練
   python pipeline.py --symbol btc
   ```

## 📊 配置模板

### 保守型（風險厭惡）

```yaml
env:
  max_inventory: 1.0

reward:
  lambda_inventory: 50.0
  lambda_turnover: 0.1
```

### 激進型（追求利潤）

```yaml
env:
  max_inventory: 5.0

reward:
  lambda_inventory: 5.0
  spread_capture_bonus: 2.0
```

### 平衡型（推薦）

使用預設配置即可。

## 🔍 配置驗證

修改配置後，檢查是否有效：

```python
import yaml

with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print(config)
```

## ⚠️ 注意事項

1. **不要隨意修改 `reward_scale`**：這個參數由 pipeline 自動調整
2. **batch_size 受限於 RAM**：如果記憶體不足，降低此值
3. **buffer_size 影響訓練速度**：過大會拖慢訓練
4. **net_arch 影響計算量**：GPU 可以用 [512, 512]，CPU 用 [256, 256]

---

**需要幫助？** 查看 [README.md](README.md) 或 [QUICKSTART.md](QUICKSTART.md)
