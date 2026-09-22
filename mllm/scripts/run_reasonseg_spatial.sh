#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${B4DL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SEG_ENV_PREFIX="${B4DL_SEG_ENV_PREFIX:-$(dirname "$PROJECT_ROOT")/.conda-stuff/envs/reasonseg}"
NUSCENES_ROOT="${B4DL_NUSCENES_ROOT:-/root/autodl-tmp/nuScenes}"
SEG_DATA_DIR="${B4DL_SEG_DATA_DIR:-$PROJECT_ROOT/mllm/reasonseg_data}"
OUTPUT_DIR="${B4DL_SEG_SPATIAL_OUTPUT_DIR:-$PROJECT_ROOT/mllm/checkpoints/reasonseg-spatial}"
SEED="${B4DL_SEG_SEED:-20260917}"
VALIDATION_FRACTION="${B4DL_SEG_VALIDATION_FRACTION:-0.1}"
LABEL_SOURCE="${B4DL_SEG_SPATIAL_LABEL_SOURCE:-auto}"

if [ ! -x "$SEG_ENV_PREFIX/bin/python" ] || [ ! -x "$SEG_ENV_PREFIX/bin/accelerate" ]; then
    echo "错误: ReasonSeg 环境不完整: $SEG_ENV_PREFIX" >&2
    exit 1
fi

cd "$PROJECT_ROOT/mllm"
"$SEG_ENV_PREFIX/bin/python" scripts/preflight_reasonseg.py \
    --stage spatial \
    --project-root "$PROJECT_ROOT" \
    --dataroot "$NUSCENES_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --exclude-scenes "$SEG_DATA_DIR/b4dl_test_scenes.json" \
    --validation-fraction "$VALIDATION_FRACTION" \
    --spatial-label-source "$LABEL_SOURCE" \
    --seed "$SEED" \
    --require-cuda

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

"$SEG_ENV_PREFIX/bin/accelerate" launch scripts/train_reasonseg_spatial.py \
    --dataroot "$NUSCENES_ROOT" \
    --exclude-scenes "$SEG_DATA_DIR/b4dl_test_scenes.json" \
    --output-dir "$OUTPUT_DIR" \
    --validation-fraction "$VALIDATION_FRACTION" \
    --label-source "$LABEL_SOURCE" \
    --num-train-epochs 20 \
    --seed "$SEED" \
    --mixed-precision bf16
