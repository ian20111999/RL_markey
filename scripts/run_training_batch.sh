#!/bin/bash
# 批次訓練腳本
# 依序訓練已完成 Tuning 的模型

set -e  # 若有錯誤則停止

# 確保在專案根目錄
cd "$(dirname "$0")/.."

# 啟動虛擬環境
source .venv/bin/activate

echo "🚀 開始批次訓練任務..."

# 1. Baseline
echo "----------------------------------------------------------------"
echo "▶️  Training: Baseline Strategy"
python train_mm_sac.py \
    --config configs/env_baseline.yaml \
    --params_path models/env_baseline_best_params.json \
    --total_timesteps 1000000 \
    --device mps \
    --run_name "baseline_tuned"

# 2. Conservative Inventory
echo "----------------------------------------------------------------"
echo "▶️  Training: Conservative Inventory Strategy"
python train_mm_sac.py \
    --config configs/env_conservative_inventory.yaml \
    --params_path models/env_conservative_inventory_best_params.json \
    --total_timesteps 1000000 \
    --device mps \
    --run_name "conservative_tuned"

# 3. Turnover Penalty
echo "----------------------------------------------------------------"
echo "▶️  Training: Turnover Penalty Strategy"
python train_mm_sac.py \
    --config configs/env_turnover_penalty.yaml \
    --params_path models/env_turnover_penalty_best_params.json \
    --total_timesteps 1000000 \
    --device mps \
    --run_name "turnover_tuned"

echo "----------------------------------------------------------------"
echo "✅ 所有訓練任務已完成！"
