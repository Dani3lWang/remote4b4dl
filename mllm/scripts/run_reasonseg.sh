#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${B4DL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SEG_ENV_PREFIX="${B4DL_SEG_ENV_PREFIX:-$(dirname "$PROJECT_ROOT")/.conda-stuff/envs/reasonseg}"
NUSCENES_ROOT="${B4DL_NUSCENES_ROOT:-/root/autodl-tmp/nuScenes}"
B3_CHECKPOINT="${B4DL_B3_CHECKPOINT:-$PROJECT_ROOT/mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3}"
SEG_DATA_DIR="${B4DL_SEG_DATA_DIR:-$PROJECT_ROOT/mllm/reasonseg_data}"
OUTPUT_DIR="${B4DL_SEG_OUTPUT_DIR:-$PROJECT_ROOT/mllm/checkpoints/reasonseg-b3-single-frame}"
SPATIAL_CHECKPOINT="${B4DL_SEG_SPATIAL_CHECKPOINT:-$PROJECT_ROOT/mllm/checkpoints/reasonseg-spatial/spatial-best}"

if [ ! -x "$SEG_ENV_PREFIX/bin/python" ] || [ ! -x "$SEG_ENV_PREFIX/bin/accelerate" ]; then
    echo "错误: ReasonSeg 环境不完整: $SEG_ENV_PREFIX" >&2
    exit 1
fi

cd "$PROJECT_ROOT/mllm"
"$SEG_ENV_PREFIX/bin/python" scripts/preflight_reasonseg.py \
    --project-root "$PROJECT_ROOT" \
    --dataroot "$NUSCENES_ROOT" \
    --train-manifest "$SEG_DATA_DIR/reasonseg_train.jsonl" \
    --val-manifest "$SEG_DATA_DIR/reasonseg_val.jsonl" \
    --b3-checkpoint "$B3_CHECKPOINT" \
    --spatial-checkpoint "$SPATIAL_CHECKPOINT" \
    --require-cuda

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

"$SEG_ENV_PREFIX/bin/accelerate" launch scripts/train_reasonseg.py \
    --model-base ./base_model/vicuna-v1-5-7b \
    --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --b3-checkpoint "$B3_CHECKPOINT" \
    --train-manifest "$SEG_DATA_DIR/reasonseg_train.jsonl" \
    --validation-manifest "$SEG_DATA_DIR/reasonseg_val.jsonl" \
    --dataroot "$NUSCENES_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --spatial-checkpoint "$SPATIAL_CHECKPOINT" \
    --num-train-epochs 20 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --mixed-precision bf16
