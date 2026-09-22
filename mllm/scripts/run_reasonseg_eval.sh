#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${B4DL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SEG_ENV_PREFIX="${B4DL_SEG_ENV_PREFIX:-$(dirname "$PROJECT_ROOT")/.conda-stuff/envs/reasonseg}"
NUSCENES_ROOT="${B4DL_NUSCENES_ROOT:-/root/autodl-tmp/nuScenes}"
B3_CHECKPOINT="${B4DL_B3_CHECKPOINT:-$PROJECT_ROOT/mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3}"
SEG_DATA_DIR="${B4DL_SEG_DATA_DIR:-$PROJECT_ROOT/mllm/reasonseg_data}"
SEG_CHECKPOINT="${B4DL_SEG_CHECKPOINT:-}"
OUTPUT_DIR="${B4DL_SEG_EVAL_OUTPUT_DIR:-$PROJECT_ROOT/mllm/eval_results/reasonseg-b3}"

if [ ! -x "$SEG_ENV_PREFIX/bin/python" ]; then
    echo "错误: ReasonSeg 环境不完整: $SEG_ENV_PREFIX" >&2
    exit 1
fi
if [ -z "$SEG_CHECKPOINT" ]; then
    echo "错误: 必须设置 B4DL_SEG_CHECKPOINT（建议指向 checkpoint-best）" >&2
    exit 1
fi

cd "$PROJECT_ROOT/mllm"
"$SEG_ENV_PREFIX/bin/python" scripts/preflight_reasonseg.py \
    --stage evaluate \
    --project-root "$PROJECT_ROOT" \
    --dataroot "$NUSCENES_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --manifest "$SEG_DATA_DIR/reasonseg_val.jsonl" \
    --exclude-scenes "$SEG_DATA_DIR/b4dl_test_scenes.json" \
    --b3-checkpoint "$B3_CHECKPOINT" \
    --seg-checkpoint "$SEG_CHECKPOINT" \
    --require-cuda

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$SEG_ENV_PREFIX/bin/python" evaluation/evaluate_reasonseg.py \
    --model-base ./base_model/vicuna-v1-5-7b \
    --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --b3-checkpoint "$B3_CHECKPOINT" \
    --seg-checkpoint "$SEG_CHECKPOINT" \
    --manifest "$SEG_DATA_DIR/reasonseg_val.jsonl" \
    --dataroot "$NUSCENES_ROOT" \
    --output-dir "$OUTPUT_DIR"
