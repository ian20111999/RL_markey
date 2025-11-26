# 實驗工作流程指南

本文件說明如何使用 V3 Pipeline 進行完整的強化學習做市策略實驗。

---

## 1. 實驗配置

### 配置檔說明

| 配置檔 | 用途 | 特點 |
|--------|------|------|
| `env_v3_full.yaml` | **推薦** - 完整配置 | 包含所有進階功能選項 |
| `env_v2.yaml` | 基礎 V2 配置 | 適合快速測試 |
| `env_baseline.yaml` | 對照組 | 最簡單的設定 |

### 主要參數

```yaml
env:
  fee_rate: 0.0004          # 手續費率
  max_inventory: 10.0       # 最大庫存
  reward_config:
    mode: "hybrid"          # 獎勵模式
    inventory_penalty: 0.0005

train:
  total_timesteps: 500000
  learning_rate: 0.0003
```

---

## 2. 訓練流程

### 方式一：標準訓練（推薦新手）

```bash
python scripts/run_v3_pipeline.py \
    --config configs/env_v3_full.yaml \
    --mode standard \
    --algorithm SAC \
    --total_timesteps 200000
```

### 方式二：課程學習

漸進式難度訓練，從簡單到困難：

```bash
python scripts/run_v3_pipeline.py \
    --mode curriculum \
    --total_timesteps 300000
```

### 方式三：分散式訓練

包含超參數搜尋和多種子驗證：

```bash
python scripts/run_v3_pipeline.py \
    --mode distributed \
    --n_hp_trials 30 \
    --validation_seeds 42,43,44,45,46
```

### 方式四：完整流程

訓練 + 回測 + 可解釋性分析 + 報告生成：

```bash
python scripts/run_v3_pipeline.py \
    --mode full \
    --generate_report
```

---

## 3. 演算法選擇

| 演算法 | 指令 | 適用場景 |
|--------|------|----------|
| SAC | `--algorithm SAC` | 預設，連續動作空間 |
| PPO | `--algorithm PPO` | 穩定性優先 |
| TD3 | `--algorithm TD3` | 減少過估計 |

```bash
# 使用 PPO
python scripts/run_v3_pipeline.py --algorithm PPO

# 使用 TD3
python scripts/run_v3_pipeline.py --algorithm TD3
```

---

## 4. 進階功能

### 風險敏感訓練

```bash
python scripts/run_v3_pipeline.py --use_risk_wrapper
```

在配置檔中設定：
```yaml
risk_sensitive:
  enabled: true
  risk_lambda: 0.1        # 風險厭惡係數
  risk_type: "variance"   # variance, cvar, downside_variance
```

### 回測分析

```bash
python scripts/run_v3_pipeline.py --run_backtest
```

包含：
- Walk-Forward Analysis
- Monte Carlo Simulation
- 交易成本敏感度分析

### 可解釋性分析

```bash
python scripts/run_v3_pipeline.py --run_explainability
```

產出：
- 特徵重要性排名
- 動作分佈分析
- 狀態-動作熱力圖

---

## 5. 批次實驗

### 多配置比較

```bash
for config in configs/env_v2.yaml configs/env_v3_full.yaml; do
    python scripts/run_v3_pipeline.py \
        --config $config \
        --mode standard \
        --total_timesteps 100000
done
```

### 多演算法比較

```bash
for algo in SAC PPO TD3; do
    python scripts/run_v3_pipeline.py \
        --algorithm $algo \
        --total_timesteps 100000 \
        --output_dir runs/algo_comparison_${algo}
done
```

---

## 6. 輸出結構

每次運行產生：

```
runs/v3_standard_SAC_20241126_120000/
├── config.yaml              # 使用的配置
├── training_results.json    # 訓練指標
├── best_model/              # 最佳模型
├── checkpoints/             # 檢查點
├── eval_logs/               # 評估日誌
├── backtest_results.json    # 回測結果（如啟用）
├── explainability_results.json  # 可解釋性（如啟用）
└── report.html              # HTML 報告（如啟用）
```

---

## 7. 監控與除錯

### 即時監控

```bash
# 查看訓練日誌
tail -f runs/*/train_log.csv

# 使用 TensorBoard
tensorboard --logdir runs/
```

### 常見問題

1. **記憶體不足**：減少 `buffer_size` 或 `n_envs`
2. **訓練不穩定**：降低 `learning_rate`，增加 `batch_size`
3. **獎勵稀疏**：改用 `reward_mode: "dense"` 或 `"shaped"`

---

## 8. 資源建議

| 設備 | 建議配置 |
|------|----------|
| CPU | 8+ 核心，用於環境並行 |
| RAM | 16GB+，視 `buffer_size` 而定 |
| GPU | 可選，MPS/CUDA 加速訓練 |

### Apple Silicon 優化

```bash
# 使用 MPS 加速
python scripts/run_v3_pipeline.py --device mps
```

---

## 9. 預期時間

| 模式 | 100k steps | 500k steps |
|------|------------|------------|
| standard | ~10 分鐘 | ~50 分鐘 |
| curriculum | ~15 分鐘 | ~75 分鐘 |
| distributed | ~2 小時 | ~6 小時 |

*以 M1 MacBook Pro 為基準*
