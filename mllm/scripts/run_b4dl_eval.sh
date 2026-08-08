#!/bin/bash
# B4DL Full Evaluation Pipeline (paper-aligned, 6 tasks × 7 metrics)
# ============================================================================
# Uses evaluation/test_b4dl.py — the paper-aligned evaluator that:
#   - runs all 6 B4DL tasks (Existence / Binary QA / Time Grounding /
#     Description / Temporal Understanding / Comprehensive Reasoning)
#   - injects the Metatoken prefix (<4DLiDAR> + <video> + question + <meta>)
#     at inference time (paper Figure 6)
#   - computes Accuracy, mIoU, BLEU-4, METEOR, ROUGE-L, BERTScore, GPT-4o
#   - compares against paper Table 3 reference values
#
# Usage:
#   bash scripts/run_b4dl_eval.sh                  # Full eval with Metatoken
#   bash scripts/run_b4dl_eval.sh --no_meta        # Ablation: omit <meta>
#   bash scripts/run_b4dl_eval.sh --no_4dlidar     # Ablation: omit <4DLiDAR>
#   bash scripts/run_b4dl_eval.sh --stage3         # Also eval stage3
#   bash scripts/run_b4dl_eval.sh --use_gpt         # Add GPT-4o scoring (needs OPENAI_API_KEY)
# ============================================================================
set -e

cd "$(dirname "$0")/.."

# ── Defaults ──
MODEL_BASE=./base_model/vicuna-v1-5-7b
MM_PROJECTOR=./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin
STAGE2_CKPT=./checkpoints/vtimellm-vicuna-v1-5-7b-stage2
STAGE3_CKPT=./checkpoints/vtimellm-vicuna-v1-5-7b-stage3
FEAT_FOLDER=../encoders/lidarclip/b4dl/stage2_features
EGO_META=./b4dl_dataset/ego_metadata.json
TEST_DATA=./b4dl_dataset/test_qa.json
EVAL_DIR=./eval_results

mkdir -p "$EVAL_DIR"

# ── Parse args ──
INCLUDE_STAGE3=false
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage3)      INCLUDE_STAGE3=true; shift ;;
        --no_meta)     EXTRA_ARGS="$EXTRA_ARGS --no_meta"; shift ;;
        --no_4dlidar)  EXTRA_ARGS="$EXTRA_ARGS --no_4dlidar"; shift ;;
        --use_gpt)     EXTRA_ARGS="$EXTRA_ARGS --use_gpt"; shift ;;
        --max_samples) EXTRA_ARGS="$EXTRA_ARGS --max_samples $2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  --stage3       Also evaluate stage3 checkpoint"
            echo "  --no_meta      Ablation: omit <meta> block"
            echo "  --no_4dlidar   Ablation: omit <4DLiDAR>"
            echo "  --use_gpt      Add GPT-4o scoring (needs OPENAI_API_KEY)"
            echo "  --max_samples N  Cap samples per task (smoke test)"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── Step 1: Create data splits ──
echo "===== Step 1: Creating data splits ====="
python scripts/create_splits.py

# ── Step 2: Stage2 evaluation ──
echo ""
echo "===== Step 2: Stage2 evaluation (test set, 6 tasks) ====="
python evaluation/test_b4dl.py \
    --model_base "$MODEL_BASE" \
    --pretrain_mm_mlp_adapter "$MM_PROJECTOR" \
    --stage2 "$STAGE2_CKPT" \
    --feat_folder "$FEAT_FOLDER" \
    --test_data "$TEST_DATA" \
    --ego_meta "$EGO_META" \
    --output "$EVAL_DIR/stage2_predictions.json" \
    --metrics_output "$EVAL_DIR/stage2_metrics.json" \
    $EXTRA_ARGS

# ── Step 3 (optional): Stage3 evaluation ──
if $INCLUDE_STAGE3; then
    if [ ! -d "$STAGE3_CKPT" ]; then
        echo ""
        echo "[WARN] Stage3 checkpoint not found at $STAGE3_CKPT"
        echo "Run: bash scripts/stage3.sh"
        echo "Then re-run: bash scripts/run_b4dl_eval.sh --stage3"
        exit 1
    fi

    echo ""
    echo "===== Step 3: Stage3 evaluation (test set, 6 tasks) ====="
    python evaluation/test_b4dl.py \
        --model_base "$MODEL_BASE" \
        --pretrain_mm_mlp_adapter "$MM_PROJECTOR" \
        --stage2 "$STAGE2_CKPT" \
        --stage3 "$STAGE3_CKPT" \
        --feat_folder "$FEAT_FOLDER" \
        --test_data "$TEST_DATA" \
        --ego_meta "$EGO_META" \
        --output "$EVAL_DIR/stage3_predictions.json" \
        --metrics_output "$EVAL_DIR/stage3_metrics.json" \
        $EXTRA_ARGS
fi

# ── Done ──
echo ""
echo "===== Pipeline complete ====="
echo "Stage2 predictions: $EVAL_DIR/stage2_predictions.json"
echo "Stage2 metrics:     $EVAL_DIR/stage2_metrics.json"
if $INCLUDE_STAGE3; then
    echo "Stage3 predictions: $EVAL_DIR/stage3_predictions.json"
    echo "Stage3 metrics:     $EVAL_DIR/stage3_metrics.json"
fi
echo ""
echo "Paper reference (B4DL, Table 3):"
echo "  Accuracy=0.762  mIoU=0.311  B@4=0.095  ROUGE-L=0.322"
echo "  METEOR=0.275  BERTScore=0.897  GPT=59.513"
