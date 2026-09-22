#!/bin/bash
# 最新基线评测入口：B3 whole-scene + meta2，同 run_b3.sh 的阶段 2。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${B4DL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WQLC_PREFIX="${B4DL_ENV_PREFIX:-$(dirname "$PROJECT_ROOT")/.conda-stuff/envs/wqlc}"

if [ ! -x "$WQLC_PREFIX/bin/python" ]; then
    echo "错误: wqlc Python 环境不存在: $WQLC_PREFIX" >&2
    exit 1
fi

export PATH="$WQLC_PREFIX/bin:$PATH"
# 语料随仓库放在 $PROJECT_ROOT/nltk_data，不在 nltk 默认搜索路径内；缺它会触发
# nltk.download 连 raw.githubusercontent.com，在无外网机器上永久挂死。
export NLTK_DATA="${NLTK_DATA:-$PROJECT_ROOT/nltk_data}"
cd "$PROJECT_ROOT/mllm"

OUT=./eval_results/b3
mkdir -p "$OUT"

echo "===== B3 BASELINE EVAL ====="
echo "Start: $(date)"
"$WQLC_PREFIX/bin/python" -u evaluation/test_b4dl.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --test_data ./b4dl_dataset/test_qa.json \
    --ego_meta ./b4dl_dataset/ego_metadata.json \
    --frame_motion ./b4dl_dataset/ego_frame_motion.json \
    --whole_scene --per_sequence --answer_frames \
    --output "$OUT/predictions.json" \
    --metrics_output "$OUT/metrics.json" \
    2>&1 | tee "$OUT/eval_log.txt"
echo "End: $(date)"
