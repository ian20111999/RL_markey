#!/bin/bash
# 
# 重新啟動訓練 - 使用修復後的程式碼
# 
# 用法:
#   ./restart_training.sh             # 使用預設參數
#   ./restart_training.sh --quick     # 快速測試模式 (10k steps)
#

set -e  # 遇到錯誤立即退出

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 預設參數
DEFAULT_BASE_MODEL="runs/v3_nuclear_long_run_20251207_002757/best_model/best_model.zip"
BASE_MODEL="$DEFAULT_BASE_MODEL"
CONFIG="configs/env_v3_stabilized.yaml"
TOTAL_TIMESTEPS=300000
LEARNING_RATE=3e-5
BATCH_SIZE=512
QUICK_MODE=false

# 解析參數
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --quick) QUICK_MODE=true ;;
        --model) BASE_MODEL="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# 根據模式調整參數
if [ "$QUICK_MODE" = true ]; then
    TOTAL_TIMESTEPS=10000
    echo -e "${YELLOW}[Quick Test Mode]${NC} Training for only 10k steps"
fi

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}   重新啟動訓練 - 使用修復後的程式碼${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "Base Model: ${YELLOW}$BASE_MODEL${NC}"

# ==================== 檢查環境 ====================
echo -e "\n${YELLOW}[1/6] 檢查環境...${NC}"

# 嘗試激活虛擬環境
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 檢查 Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found${NC}"
    exit 1
fi

# 檢查必要檔案
if [ ! -f "$BASE_MODEL" ]; then
    echo -e "${RED}❌ Base model not found: $BASE_MODEL${NC}"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}❌ Config not found: $CONFIG${NC}"
    exit 1
fi

# ==================== 檢查 Numba ====================
# 靜默檢查 Numba
if ! python -c "import numba" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Installing Numba...${NC}"
    python -m pip install -q numba
fi

# ==================== 評估當前模型 ====================
# 簡化輸出
if [[ "$BASE_MODEL" == *"best_model.zip" ]]; then
    MODEL_DIR=$(dirname "$BASE_MODEL")
    if [ -f "$MODEL_DIR/progress.csv" ]; then
        LAST_REWARD=$(tail -n 1 "$MODEL_DIR/progress.csv" | cut -d',' -f2)
        echo -e "   Last Reward: ${GREEN}$LAST_REWARD${NC}"
    fi
fi

# ==================== 準備訓練目錄 ====================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="v3_fixed_$TIMESTAMP"
RUN_DIR="runs/$RUN_NAME"
mkdir -p "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/env_config.yaml"

# ==================== 啟動訓練 ====================
echo -e "${YELLOW}[Running]${NC} Output: $RUN_DIR"
echo -e "${BLUE}======================================${NC}"

# 建立訓練日誌檔案
TRAIN_LOG="$RUN_DIR/training.log"

# 執行訓練 (使用 grep 過濾掉 Reward 異常警告，但保留在 log 檔中)
python scripts/run_stabilization.py \
    --base_model "$BASE_MODEL" \
    --config "$CONFIG" \
    --total_timesteps $TOTAL_TIMESTEPS \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --output_dir "$RUN_DIR" \
    2>&1 | tee "$TRAIN_LOG" | grep --line-buffered -v "Reward 異常"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# ==================== 訓練完成 ====================
echo -e "\n${BLUE}======================================${NC}"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed!${NC}"
    echo -e "   Log: $TRAIN_LOG"
    echo -e "   Tensorboard: tensorboard --logdir $RUN_DIR"
else
    echo -e "${RED}❌ Training failed.${NC} Check $TRAIN_LOG"
fi
