#!/bin/bash
# scripts/run_v2_pipeline.sh
# V2 環境的完整 RL 實驗流程
# 支援：Potential-based reward shaping, 擴展 Observation/Action, Domain Randomization

set -e

# 設定
PYTHON_EXEC="$(pwd)/.venv/bin/python"
export PYTHONPATH="$(pwd):$PYTHONPATH"

CONFIG_FILE="configs/env_v3.yaml"

# 建立實驗目錄
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="runs/exp_v2_${TIMESTAMP}"
mkdir -p "$EXP_DIR"

# 複製 Config
cp "$CONFIG_FILE" "$EXP_DIR/config_used.yaml"
CONFIG_HASH=$(md5 -q "$CONFIG_FILE" 2>/dev/null || md5sum "$CONFIG_FILE" | cut -d' ' -f1)
echo "$CONFIG_HASH" > "$EXP_DIR/config_hash.txt"

echo "========================================================"
echo "🚀 V2 Environment Training Pipeline"
echo "📅 Date: $(date)"
echo "📁 Experiment Dir: $EXP_DIR"
echo "⚙️  Config: $CONFIG_FILE"
echo "🔐 Config Hash: $CONFIG_HASH"
echo "========================================================"

# =============================================================================
# Step 1: Sanity Check
# =============================================================================
echo ""
echo "--------------------------------------------------------"
echo "Step 1: Sanity Check (V2 Environment)"
echo "--------------------------------------------------------"

$PYTHON_EXEC scripts/train_v2.py \
    --config "$CONFIG_FILE" \
    --output_dir "$EXP_DIR/sanity" \
    --mode sanity \
    --seed 42

# 檢查結果
SANITY_STATUS=$(cat "$EXP_DIR/sanity/sanity_status.json")
SANITY_PASSED=$($PYTHON_EXEC -c "import json; print(json.loads('$SANITY_STATUS')['status'])")

if [ "$SANITY_PASSED" != "success" ]; then
    echo "❌ Sanity Check Failed. Aborting pipeline."
    exit 1
fi

echo "✅ Sanity Check Passed!"

# 檢查是否跳過 Tuning
SKIP_TUNING=$($PYTHON_EXEC -c "import json; print(json.loads('$SANITY_STATUS').get('skip_tuning', False))")

if [ "$SKIP_TUNING" = "True" ]; then
    echo ""
    echo "🎯 RL already exceeds Baseline significantly, skipping Tuning."
    mkdir -p "$EXP_DIR/final"
    cp "$EXP_DIR/sanity/model.zip" "$EXP_DIR/final/model.zip"
    cp "$EXP_DIR/sanity/eval_summary.csv" "$EXP_DIR/final/eval_summary.csv"
else
    # =============================================================================
    # Step 2: Hyperparameter Tuning (Optional)
    # =============================================================================
    echo ""
    echo "--------------------------------------------------------"
    echo "Step 2: Hyperparameter Tuning"
    echo "--------------------------------------------------------"
    echo "⚠️  Tuning with V2 environment requires adaptation of tune_mm_sac.py"
    echo "    For now, using default parameters from config..."
    
    mkdir -p "$EXP_DIR/tuning"
    echo '{"note": "Using default config parameters, V2 tuning TBD"}' > "$EXP_DIR/tuning/status.json"
    
    # =============================================================================
    # Step 3: Final Training
    # =============================================================================
    echo ""
    echo "--------------------------------------------------------"
    echo "Step 3: Final Training (V2 Environment)"
    echo "--------------------------------------------------------"
    
    # 主要訓練
    $PYTHON_EXEC scripts/train_v2.py \
        --config "$CONFIG_FILE" \
        --output_dir "$EXP_DIR/final" \
        --mode train \
        --seed 42 \
        --timesteps 500000

    echo ""
    echo "✅ Final Training Completed"
    
    # 多 Seed 驗證
    echo ""
    echo "🔄 Running Multi-Seed Validation..."
    
    for SEED in 43 44; do
        echo "   Training with Seed $SEED..."
        $PYTHON_EXEC scripts/train_v2.py \
            --config "$CONFIG_FILE" \
            --output_dir "$EXP_DIR/final_seed_$SEED" \
            --mode train \
            --seed $SEED \
            --timesteps 500000 \
            --quiet
    done
fi

# =============================================================================
# Step 4: Summary
# =============================================================================
echo ""
echo "--------------------------------------------------------"
echo "Step 4: Aggregating Results"
echo "--------------------------------------------------------"

$PYTHON_EXEC -c "
import pandas as pd
from pathlib import Path
import json

exp_dir = Path('$EXP_DIR')
results = []

for d in exp_dir.glob('final*'):
    summary_file = d / 'eval_summary.csv'
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        rl_row = df[df['agent'] == 'RL']
        if not rl_row.empty:
            row = rl_row.iloc[0].to_dict()
            row['run'] = d.name
            results.append(row)

if results:
    df = pd.DataFrame(results)
    print()
    print('📊 Multi-Seed Results:')
    cols = ['run', 'net_pnl', 'sharpe', 'max_drawdown', 'fill_rate', 'adverse_selection_rate']
    existing = [c for c in cols if c in df.columns]
    print(df[existing].to_string(index=False))
    print()
    print(f'📈 Average Net PnL: {df[\"net_pnl\"].mean():.2f} ± {df[\"net_pnl\"].std():.2f}')
    if 'sharpe' in df.columns:
        print(f'📈 Average Sharpe: {df[\"sharpe\"].mean():.4f} ± {df[\"sharpe\"].std():.4f}')
    df.to_csv(exp_dir / 'experiment_summary.csv', index=False)
"

echo ""
echo "========================================================"
echo "🎉 V2 Pipeline Completed!"
echo ""
echo "📁 Results: $EXP_DIR"
echo "   ├── config_used.yaml"
echo "   ├── sanity/"
echo "   │   ├── model.zip"
echo "   │   ├── eval_summary.csv"
echo "   │   └── sanity_status.json"
echo "   ├── final/"
echo "   │   ├── model.zip"
echo "   │   └── eval_summary.csv"
echo "   └── experiment_summary.csv"
echo "========================================================"
