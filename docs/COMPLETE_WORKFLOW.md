# 🎯 RL 做市商完整流程指南

## 從 0 到可賺錢的 7 個步驟

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: 資料準備 → Step 2: 環境驗證 → Step 3: 基線測試        │
│                           ↓                                      │
│  Step 4: 初步訓練 → Step 5: 超參數調優 → Step 6: 完整訓練       │
│                           ↓                                      │
│  Step 7: Out-of-Sample 測試 → 通過? → 實盤小額測試              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: 資料準備 ✅ (已完成)

```bash
# 下載歷史數據
python scripts/fetch_binance_ohlcv.py --symbol BTCUSDT --interval 1m --year 2023

# 檢查數據品質
python -c "
import pandas as pd
df = pd.read_csv('data/btc_usdt_1m_2023.csv')
print(f'數據筆數: {len(df)}')
print(f'時間範圍: {df.timestamp.min()} ~ {df.timestamp.max()}')
print(f'缺失值: {df.isnull().sum().sum()}')
"
```

**產出**: `data/btc_usdt_1m_2023.csv`

---

## Step 2: 環境驗證 ✅ (已完成)

確保環境能正常運作：

```bash
python -c "
from envs.market_making_env_v2 import MarketMakingEnvV2
import pandas as pd

df = pd.read_csv('data/btc_usdt_1m_2023.csv')
env = MarketMakingEnvV2(df=df, episode_length=1440)

obs, _ = env.reset()
print(f'觀察空間: {env.observation_space.shape}')
print(f'動作空間: {env.action_space.shape}')

# 跑一個簡單測試
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    if done:
        break
print('環境測試通過!')
"
```

---

## Step 3: 基線測試 ✅ (已完成)

測試簡單策略作為比較基準：

```bash
python scripts/evaluate_v2.py --config configs/env_v3.yaml --mode baseline
```

**預期結果**:
- Random: ~-500 到 +500 (隨機波動)
- Fixed Spread: ~+200 到 +800 (穩定正收益)
- RL 必須比這兩個都好才有意義

---

## Step 4: 初步訓練 (快速驗證) ⬅️ 你在這裡

**目的**: 用較少步數快速確認訓練是否正常

```bash
# 50k 步快速測試
python scripts/run_v3_pipeline.py \
    --config configs/env_v3_low_variance.yaml \
    --algorithm SAC \
    --total_timesteps 50000 \
    --mode standard
```

**檢查點**:
- [ ] 獎勵是否在增長？
- [ ] 沒有 NaN 或崩潰？
- [ ] 比 Random 好？

---

## Step 5: 超參數調優

**用 Optuna 搜尋最佳參數**:

```bash
python scripts/run_v3_pipeline.py \
    --config configs/env_v3_low_variance.yaml \
    --mode distributed \
    --n_hp_trials 30 \
    --total_timesteps 100000
```

**關鍵超參數**:
| 參數 | 搜尋範圍 | 影響 |
|------|----------|------|
| learning_rate | 1e-5 ~ 3e-4 | 太高不穩定，太低學太慢 |
| tau | 0.001 ~ 0.02 | 目標網路更新速度 |
| gamma | 0.95 ~ 0.999 | 長期 vs 短期收益 |
| risk_lambda | 0.05 ~ 0.3 | 風險懲罰強度 |

---

## Step 6: 完整訓練

使用最佳超參數進行完整訓練：

```bash
python scripts/run_v3_pipeline.py \
    --config configs/env_v3_low_variance.yaml \
    --algorithm SAC \
    --total_timesteps 300000 \
    --mode full
```

**成功標準**:
- 平均獎勵 > 500
- 標準差 < 平均值的 50%
- 正收益率 > 80%

---

## Step 7: Out-of-Sample 測試 (最重要！)

**用從未見過的數據測試**:

```bash
python scripts/run_validation.py \
    --model_path runs/YOUR_BEST_MODEL/best_model/best_model.zip \
    --config configs/env_v3_low_variance.yaml \
    --mode full
```

**必須通過**:
- [ ] Walk-Forward 測試: 在滾動窗口上都能賺錢
- [ ] Monte Carlo: 95% 信賴區間下界 > 0
- [ ] 穩健性測試: 手續費 2x 時仍盈利

---

## 📊 各步驟預期時間

| 步驟 | 時間 | 產出 |
|------|------|------|
| Step 1 | 10 分鐘 | 數據文件 |
| Step 2 | 1 分鐘 | 環境驗證通過 |
| Step 3 | 5 分鐘 | 基線分數 |
| Step 4 | 10 分鐘 | 初步訓練結果 |
| Step 5 | 1-2 小時 | 最佳超參數 |
| Step 6 | 30-60 分鐘 | 訓練好的模型 |
| Step 7 | 10 分鐘 | 驗證報告 |

**總計**: 約 2-3 小時

---

## 🚨 常見問題

### Q: 訓練不收斂？
- 降低 learning_rate
- 增加 buffer_size
- 檢查 reward 設計

### Q: 方差太大？
- 啟用 `risk_sensitive.enabled: true`
- 增加 `risk_lambda`
- 降低 `max_inventory`

### Q: Out-of-sample 表現差？
- 可能過擬合
- 啟用 Domain Randomization
- 增加訓練數據多樣性

---

## 📁 文件結構

```
RL_markey/
├── configs/
│   ├── env_v3.yaml              # 標準配置
│   └── env_v3_low_variance.yaml # 低方差配置 (推薦)
├── scripts/
│   ├── run_v3_pipeline.py       # 主訓練腳本
│   └── run_validation.py        # 驗證腳本
├── runs/
│   └── [timestamp]/             # 訓練結果
│       ├── best_model/
│       ├── eval_logs/
│       └── config.yaml
└── docs/
    └── COMPLETE_WORKFLOW.md     # 本文件
```
