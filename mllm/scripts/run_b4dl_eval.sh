#!/bin/bash
# B4DL Full Evaluation Pipeline
# Usage:
#   bash scripts/run_b4dl_eval.sh              # Full pipeline (split + stage2 eval)
#   bash scripts/run_b4dl_eval.sh --stage3     # Include stage3 eval (after stage3 training)
set -e

cd "$(dirname "$0")/.."

MODEL_BASE=./base_model/vicuna-v1-5-7b
MM_PROJECTOR=./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin
STAGE2_CKPT=./checkpoints/vtimellm-vicuna-v1-5-7b-stage2
STAGE3_CKPT=./checkpoints/vtimellm-vicuna-v1-5-7b-stage3
FEAT_FOLDER=../encoders/lidarclip/b4dl/stage2_features
EVAL_DIR=./eval_results

mkdir -p "$EVAL_DIR"

# Step 1: Create splits
echo "===== Step 1: Creating data splits ====="
python scripts/create_splits.py

# Step 2: Evaluate Stage2 on test set
echo ""
echo "===== Step 2: Stage2 evaluation (test set) ====="
python vtimellm/eval/b4dl_eval.py \
    --model_base "$MODEL_BASE" \
    --pretrain_mm_mlp_adapter "$MM_PROJECTOR" \
    --stage2 "$STAGE2_CKPT" \
    --data_path ./b4dl_dataset/stage2_test.json \
    --feat_folder "$FEAT_FOLDER" \
    --log_path "$EVAL_DIR/stage2_test_log.jsonl"

python vtimellm/eval/b4dl_metrics.py \
    --log_path "$EVAL_DIR/stage2_test_log.jsonl" \
    --task stage2 \
    --output "$EVAL_DIR/stage2_metrics.json"

INCLUDE_STAGE3=false
if [[ "$1" == "--stage3" ]]; then
    INCLUDE_STAGE3=true
fi

if $INCLUDE_STAGE3; then
    if [ ! -d "$STAGE3_CKPT" ]; then
        echo ""
        echo "[WARN] Stage3 checkpoint not found at $STAGE3_CKPT"
        echo "Run: bash scripts/stage3.sh"
        echo "Then re-run: bash scripts/run_b4dl_eval.sh --stage3"
        exit 1
    fi

    echo ""
    echo "===== Step 3: Stage3 evaluation (test set) ====="
    python vtimellm/eval/b4dl_eval.py \
        --model_base "$MODEL_BASE" \
        --pretrain_mm_mlp_adapter "$MM_PROJECTOR" \
        --stage2 "$STAGE2_CKPT" \
        --stage3 "$STAGE3_CKPT" \
        --data_path ./b4dl_dataset/stage3_test.json \
        --feat_folder "$FEAT_FOLDER" \
        --log_path "$EVAL_DIR/stage3_test_log.jsonl"

    python vtimellm/eval/b4dl_metrics.py \
        --log_path "$EVAL_DIR/stage3_test_log.jsonl" \
        --task stage3 \
        --output "$EVAL_DIR/stage3_metrics.json"
fi

echo ""
echo "===== Pipeline complete ====="
echo "Stage2 metrics: $EVAL_DIR/stage2_metrics.json"
echo "Stage2 log: $EVAL_DIR/stage2_test_log.jsonl"
if $INCLUDE_STAGE3; then
    echo "Stage3 metrics: $EVAL_DIR/stage3_metrics.json"
    echo "Stage3 log: $EVAL_DIR/stage3_test_log.jsonl"
fi
