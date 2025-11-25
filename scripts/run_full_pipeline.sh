#!/bin/bash
# scripts/run_full_pipeline.sh
# 一鍵執行完整 RL 實驗流程：Sanity -> Tuning -> Final Training
# 支援：統一輸出結構、Config 一致性檢查、智慧跳過 Tuning、多 Seed 穩健性測試

set -e  # 遇到錯誤立即停止

# 設定 Python 執行檔路徑
PYTHON_EXEC="$(pwd)/.venv/bin/python"
export PYTHONPATH="$(pwd):$PYTHONPATH"

CONFIG_FILE="configs/env_v2.yaml"

# 建立統一的實驗目錄（以時間戳命名）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="runs/exp_${TIMESTAMP}"
mkdir -p "$EXP_DIR"

# 計算 Config Hash 確保一致性
CONFIG_HASH=$(md5 -q "$CONFIG_FILE" 2>/dev/null || md5sum "$CONFIG_FILE" | cut -d' ' -f1)
echo "$CONFIG_HASH" > "$EXP_DIR/config_hash.txt"
cp "$CONFIG_FILE" "$EXP_DIR/config_used.yaml"

BEST_PARAMS_FILE="$EXP_DIR/tuning/best_params.json"

echo "========================================================"
echo "🚀 開始執行完整 RL 實驗流程"
echo "📅 日期: $(date)"
echo "📁 實驗目錄: $EXP_DIR"
echo "⚙️  Config: $CONFIG_FILE"
echo "🔐 Config Hash: $CONFIG_HASH"
echo "========================================================"

# =============================================================================
# Step 1: Sanity Check Training
# =============================================================================
echo ""
echo "--------------------------------------------------------"
echo "Step 1: Sanity Check Training (確認 RL 有在學習)"
echo "--------------------------------------------------------"

$PYTHON_EXEC scripts/train_sanity_sac.py \
    --config "$CONFIG_FILE" \
    --exp_dir "$EXP_DIR"

echo ""
echo "✅ Sanity Check 通過！"

# 檢查是否可以跳過 Tuning
SANITY_STATUS_FILE="$EXP_DIR/sanity/sanity_status.json"
SKIP_TUNING=false

if [ -f "$SANITY_STATUS_FILE" ]; then
    SKIP_TUNING=$($PYTHON_EXEC -c "import json; print(json.load(open('$SANITY_STATUS_FILE')).get('skip_tuning', False))")
fi

if [ "$SKIP_TUNING" = "True" ]; then
    echo ""
    echo "🎯 Sanity 模型已顯著超越 Baseline，跳過 Tuning 階段！"
    echo "   直接使用 Sanity 模型作為最終模型..."
    
    mkdir -p "$EXP_DIR/final"
    cp "$EXP_DIR/sanity/model.zip" "$EXP_DIR/final/model.zip"
    cp "$EXP_DIR/sanity/eval_summary.csv" "$EXP_DIR/final/final_eval_summary.csv"
    
else
    # =============================================================================
    # Step 2: Hyperparameter Tuning
    # =============================================================================
    echo ""
    echo "--------------------------------------------------------"
    echo "Step 2: Hyperparameter Tuning (Optuna)"
    echo "--------------------------------------------------------"
    
    mkdir -p "$EXP_DIR/tuning"
    
    $PYTHON_EXEC tune_mm_sac.py \
        --config "$CONFIG_FILE" \
        --exp_dir "$EXP_DIR" \
        --n_trials 20 \
        --train_timesteps 50000 \
        --eval_episodes 5 \
        --n_eval_runs 3 \
        --base_seed 42 \
        --save_best_params \
        --best_params_path "$BEST_PARAMS_FILE"

    echo ""
    echo "✅ Tuning 完成。最佳參數已儲存至 $BEST_PARAMS_FILE"

    # =============================================================================
    # Step 3: Final Training (多 Seed 穩健性測試)
    # =============================================================================
    echo ""
    echo "--------------------------------------------------------"
    echo "Step 3: Final Training (使用最佳參數長訓 + 多 Seed 驗證)"
    echo "--------------------------------------------------------"
    
    # 主要訓練（Seed 42）
    $PYTHON_EXEC scripts/train_final_sac.py \
        --config "$CONFIG_FILE" \
        --params "$BEST_PARAMS_FILE" \
        --output_dir "$EXP_DIR/final" \
        --seed 42

    echo ""
    echo "✅ 主要 Final Training 完成"

    # 額外 Seed 驗證（可選，增加穩健性）
    echo ""
    echo "🔄 執行額外 Seed 穩健性測試..."
    
    for SEED in 43 44; do
        echo "   Training with Seed $SEED..."
        $PYTHON_EXEC scripts/train_final_sac.py \
            --config "$CONFIG_FILE" \
            --params "$BEST_PARAMS_FILE" \
            --output_dir "$EXP_DIR/final_seed_$SEED" \
            --seed $SEED \
            --quiet
    done
fi

# =============================================================================
# Step 4: 彙總結果
# =============================================================================
echo ""
echo "--------------------------------------------------------"
echo "Step 4: 彙總實驗結果"
echo "--------------------------------------------------------"

# 彙總所有 Seed 的結果
$PYTHON_EXEC -c "
import json
import pandas as pd
from pathlib import Path

exp_dir = Path('$EXP_DIR')
results = []

# 收集所有 final 結果
for d in exp_dir.glob('final*'):
    summary_file = d / 'final_eval_summary.csv'
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        rl_row = df[df['agent'].str.contains('RL', case=False)]
        if not rl_row.empty:
            results.append({
                'run': d.name,
                'net_pnl': rl_row['net_pnl'].values[0],
                'sharpe': rl_row['sharpe'].values[0] if 'sharpe' in df.columns else None,
            })

if results:
    df = pd.DataFrame(results)
    print('\\n📊 Final Results Summary:')
    print(df.to_string(index=False))
    print(f'\\n平均 Net PnL: {df[\"net_pnl\"].mean():.2f} ± {df[\"net_pnl\"].std():.2f}')
    df.to_csv(exp_dir / 'experiment_summary.csv', index=False)
"

echo ""
echo "========================================================"
echo "🎉 所有流程執行完畢！"
echo ""
echo "📁 實驗結果目錄: $EXP_DIR"
echo "   ├── config_used.yaml      # 本次使用的設定檔"
echo "   ├── sanity/               # Sanity Check 結果"
echo "   ├── tuning/               # Optuna Tuning 結果"
echo "   ├── final/                # 最終模型 (Seed 42)"
echo "   ├── final_seed_43/        # 穩健性測試 (Seed 43)"
echo "   ├── final_seed_44/        # 穩健性測試 (Seed 44)"
echo "   └── experiment_summary.csv"
echo "========================================================"
