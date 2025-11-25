#!/bin/bash
# scripts/run_evaluation_batch.sh
# 批次評估已訓練好的模型

set -e  # 若有指令失敗則立即停止

# 使用 venv 中的 python
PYTHON_EXEC="/Users/ian/Desktop/Project/RL_markey/.venv/bin/python"
export PYTHONPATH="/Users/ian/Desktop/Project/RL_markey:$PYTHONPATH"

echo "🚀 開始批次評估..."

# 1. Baseline Tuned
echo "----------------------------------------------------------------"
echo "📊 Evaluating Baseline Tuned Model..."
$PYTHON_EXEC scripts/evaluate_policy.py \
    --model_path runs/SAC/20251123_105752_baseline_tuned/model.zip \
    --config runs/SAC/20251123_105752_baseline_tuned/config.yaml \
    --output_dir runs/SAC/20251123_105752_baseline_tuned/evaluation \
    --episodes 5

# 2. Conservative Tuned
echo "----------------------------------------------------------------"
echo "📊 Evaluating Conservative Tuned Model..."
$PYTHON_EXEC scripts/evaluate_policy.py \
    --model_path runs/SAC/20251123_134408_conservative_tuned/model.zip \
    --config runs/SAC/20251123_134408_conservative_tuned/config.yaml \
    --output_dir runs/SAC/20251123_134408_conservative_tuned/evaluation \
    --episodes 5

# 3. Turnover Tuned
echo "----------------------------------------------------------------"
echo "📊 Evaluating Turnover Tuned Model..."
$PYTHON_EXEC scripts/evaluate_policy.py \
    --model_path runs/SAC/20251123_135900_turnover_tuned/model.zip \
    --config runs/SAC/20251123_135900_turnover_tuned/config.yaml \
    --output_dir runs/SAC/20251123_135900_turnover_tuned/evaluation \
    --episodes 5

echo "----------------------------------------------------------------"
echo "✅ 所有評估任務已完成！"
