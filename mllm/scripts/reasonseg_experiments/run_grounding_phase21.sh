#!/bin/bash
# Phase 2.1：推理侧两个免训练臂，同一份 1000 条 val_thin、同一 internal679 best 档。
#
#   arm A（prior on） ：复算 Phase 2.0 基线 + 相对阈值解码族 p >= alpha*max(p)
#   arm B（prior off）：--disable-loc-prior，检验 LOC 先验是否在污染 fine_memory
#
# Phase 2.0 已测出 oracle top-K 的硬上界是 recall@0.5 = 0.110（K 取 GT 点数），
# 所以 arm A 的相对阈值族必然落在该上界之下，跑它只为确认能稳定拿到那部分收益；
# arm B 改变前向本身，不受该上界约束，是这一阶段唯一可能出惊喜的臂。
#
# 判据：任一臂 recall@0.5 >= 0.15（现状 0.0861 的 1.7 倍）→ 采纳为默认解码规则并
#       重报全部自由生成数值；否则推理侧关闭，转 Phase 2.2。
set -uo pipefail
cd /root/autodl-tmp/mmb4dl/mllm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
CKPT=${1:-./checkpoints/reasonseg-internal679/checkpoint-best}
OUTROOT=${2:-./eval_results}
MANIFEST=./reasonseg_data_trainval/reasonseg_val_thin.jsonl
REL=0.05,0.1,0.2,0.3,0.5

diagnose () {
  local tag=$1; shift
  echo "[$(date '+%F %T')] === grounding arm: $tag ==="
  $E/bin/python scripts/reasonseg_experiments/diagnose_reasonseg_grounding.py \
    --model-base ./base_model/vicuna-v1-5-7b \
    --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
    --seg-checkpoint "$CKPT" \
    --manifest "$MANIFEST" \
    --dataroot /root/autodl-tmp/nuScenes \
    --output "$OUTROOT/_grounding_$tag/report.json" \
    --max-samples 1000 --relative-thresholds "$REL" \
    --dtype fp16 --encoder-dtype fp32 "$@"
  echo "[$(date '+%F %T')] arm $tag rc=$?"
}

diagnose "p21_prioron"
diagnose "p21_prioroff" --disable-loc-prior
echo "=== PHASE2.1 DONE ==="
