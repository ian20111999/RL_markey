#!/bin/bash
# run_training_extended.sh
# Continue training from the successful v3_fixed run for a longer duration

# Activate virtual environment
source .venv/bin/activate

# Configuration
CONFIG_NAME="env_v3_stabilized"
TIMESTEPS=1000000
BASE_MODEL="runs/v3_fixed_20251209_155947/best_model/best_model.zip"

echo "======================================"
echo "   Starting Extended Training Run"
echo "======================================"
echo "Base Model: $BASE_MODEL"
echo "Config: $CONFIG_NAME"
echo "Timesteps: $TIMESTEPS"
echo "======================================"

python scripts/run_stabilization.py \
    --config configs/${CONFIG_NAME}.yaml \
    --total_timesteps $TIMESTEPS \
    --model_path $BASE_MODEL \
    --output_dir "runs/v3_extended_1M_$(date +%Y%m%d_%H%M%S)"
