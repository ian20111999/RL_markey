#!/bin/bash
# scripts/run_full_pipeline.sh
# 一鍵執行完整 RL 實驗流程：Sanity -> Tuning -> Final Training

set -e  # 遇到錯誤立即停止

# 設定 Python 執行檔路徑
PYTHON_EXEC="$(pwd)/.venv/bin/python"
export PYTHONPATH="$(pwd):$PYTHONPATH"

CONFIG_FILE="configs/env_v2.yaml"
BEST_PARAMS_FILE="models/best_sac_params.json"

echo "========================================================"
echo "🚀 開始執行完整 RL 實驗流程"
echo "📅 日期: $(date)"
echo "⚙️  Config: $CONFIG_FILE"
echo "========================================================"

# 1. Sanity Check Training
echo ""
echo "--------------------------------------------------------"
echo "Step 1: Sanity Check Training (確認 RL 有在學習)"
echo "--------------------------------------------------------"
# train_sanity_sac.py 會自動重試，若最終失敗會回傳 exit code 1，觸發 set -e 停止腳本
$PYTHON_EXEC scripts/train_sanity_sac.py --config $CONFIG_FILE

echo ""
echo "✅ Sanity Check 通過！準備開始 Tuning..."
sleep 3

# 2. Hyperparameter Tuning
echo ""
echo "--------------------------------------------------------"
echo "Step 2: Hyperparameter Tuning (Optuna)"
echo "--------------------------------------------------------"
# 這裡設定 n_trials=20, train_timesteps=50000 做示範，實際可調大
$PYTHON_EXEC tune_mm_sac.py \
    --config $CONFIG_FILE \
    --n_trials 20 \
    --train_timesteps 50000 \
    --eval_episodes 5 \
    --save_best_params \
    --best_params_path $BEST_PARAMS_FILE

echo ""
echo "✅ Tuning 完成。最佳參數已儲存至 $BEST_PARAMS_FILE"

# 3. Final Training
echo ""
echo "--------------------------------------------------------"
echo "Step 3: Final Training (使用最佳參數長訓)"
echo "--------------------------------------------------------"
# 這裡設定 total_timesteps=300000 做示範，實際可設 500k~1M
$PYTHON_EXEC scripts/train_final_sac.py \
    --config $CONFIG_FILE \
    --params $BEST_PARAMS_FILE

echo ""
echo "========================================================"
echo "🎉 所有流程執行完畢！"
echo "請查看 runs/final_env_v2_sac/final_eval_summary.csv 確認最終績效。"
echo "========================================================"
