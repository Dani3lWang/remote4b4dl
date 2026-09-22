#!/bin/bash
# Phase 3: 用 B3 的规范口径（run_baseline_eval.sh 的标志集，含 --whole_scene）
# 重跑 B4a 六任务评测。历史 B4a/framepos 结果由链脚本家族产出，而该家族不传
# --whole_scene（改用 --sequence_metadata 走 per-sequence 切片），两者是互斥的
# 视觉输入口径。本跑用于判定历史负结果是否被口径混淆：
#   若与存档 B4a 接近 -> 口径影响可忽略，历史对比可继续引用
#   若明显移动（尤其 TG）-> 存档 B4a/framepos 的归因不成立，需按统一口径重做
set -u
cd /root/autodl-tmp/mmb4dl/mllm
W=/root/autodl-tmp/.conda-stuff/envs/wqlc
export HF_HUB_OFFLINE=1 HF_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 语料在仓库内而不在 nltk 默认搜索路径里；缺它会触发 nltk.download 去
# raw.githubusercontent.com 并在无外网机器上永久挂死。
export NLTK_DATA="${NLTK_DATA:-/root/autodl-tmp/mmb4dl/nltk_data}"
export PATH="$W/bin:$PATH"

CKPT=./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b4a
OUT=./eval_results/stage2_full_seqv3_mixed_b4a_wholescene
mkdir -p "$OUT"

echo "[$(date '+%F %T')] B4a whole-scene eval start"
timeout 18000 "$W/bin/python" -u evaluation/test_b4dl.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 "$CKPT" \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --test_data ./b4dl_dataset/test_qa.json \
    --ego_meta ./b4dl_dataset/ego_metadata.json \
    --frame_motion ./b4dl_dataset/ego_frame_motion.json \
    --whole_scene --per_sequence --answer_frames \
    --output "$OUT/predictions.json" \
    --metrics_output "$OUT/metrics.json" \
    2>&1 | tee "$OUT/eval_log.txt"
echo "[$(date '+%F %T')] B4a whole-scene eval end rc=${PIPESTATUS[0]}"
