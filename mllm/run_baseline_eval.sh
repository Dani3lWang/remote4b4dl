#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${B4DL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WQLC_PREFIX="${B4DL_ENV_PREFIX:-$(dirname "$PROJECT_ROOT")/.conda-stuff/envs/wqlc}"

if [ ! -x "$WQLC_PREFIX/bin/python" ]; then
    echo "Error: Python environment not found: $WQLC_PREFIX" >&2
    exit 1
fi

export PATH="$WQLC_PREFIX/bin:$PATH"
cd "$PROJECT_ROOT/mllm"
echo "===== BASELINE EVAL (no per_sequence) ====="
echo "Start: $(date)"
"$WQLC_PREFIX/bin/python" evaluation/test_b4dl.py     --model_base ./base_model/vicuna-v1-5-7b     --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin     --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full     --feat_folder ../encoders/lidarclip/b4dl/stage2_features     --test_data ./b4dl_dataset/test_qa.json     --ego_meta ./b4dl_dataset/ego_metadata.json     --output ./evaluation/predictions_baseline.json     --metrics_output ./evaluation/eval_baseline_metrics.json     2>&1 | tee ./evaluation/baseline_eval.log
echo "End: $(date)"
