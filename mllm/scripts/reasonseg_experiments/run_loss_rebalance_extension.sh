#!/bin/bash
# Phase 2.2 PASS 后的扩跑：把胜出臂按 internal679 的 20-epoch 配方从头重训。
#
# 从头重训而不是从 4-epoch 档续跑，因为对照（internal679 20 epoch）是同初始化同
# 调度跑满 20 轮的；续跑会让 warmup 与学习率衰减 schedule 与对照不同构，比较就不干净。
#
# 用法：run_loss_rebalance_extension.sh <arm-name> <loss flags...>
#   例：run_loss_rebalance_extension.sh reasonseg-lossA3-bal-tversky \
#         --bce-mode balanced --region-loss tversky --tversky-alpha 0.3 --tversky-beta 0.7
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "用法: $0 <arm-name> <loss flags...>" >&2
  exit 2
fi
NAME=$1; shift
cd "$(dirname "$0")/../.." || exit 1
V=${REASONSEG_MANIFEST_DIR:?set REASONSEG_MANIFEST_DIR to an audited manifest directory}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
ASSETS=${REASONSEG_ASSET_ROOT:-/root/autodl-tmp/mmb4dl/mllm}
OUT=./checkpoints/${NAME}-ext20
[ ! -e "$OUT" ] || { echo "output already exists: $OUT" >&2; exit 2; }
L=./training_logs/phase2
mkdir -p "$L"

echo "[$(date '+%F %T')] === EXTEND 20ep: $NAME ($*) -> $OUT ==="
$E/bin/accelerate launch --mixed_precision bf16 scripts/train_reasonseg.py \
  --model-base "$ASSETS/base_model/vicuna-v1-5-7b" \
  --pretrain-mm-mlp-adapter "$ASSETS/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin" \
  --b3-checkpoint "$ASSETS/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3" \
  --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
  --validation-manifest "$V/reasonseg_val_internal_es.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output-dir "$OUT" \
  --spatial-checkpoint "$ASSETS/checkpoints/reasonseg-spatial-tv/spatial-best" \
  --num-train-epochs 20 --per-device-train-batch-size 1 --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 --early-stopping-patience 12 --early-stopping-min-delta 1e-4 \
  --save-steps 200 --keep-last 2 --validation-samples 0 \
  --mixed-precision bf16 "$@" > "$L/${NAME}-ext20.log" 2>&1
RC=$?
echo "[$(date '+%F %T')] extend $NAME rc=$RC"

# 扩跑完在最终测试口径（官方 val 的 20 场景 thin）复算一次 TF，并做接地诊断，
# 这样新基线能与 internal679 的 val_thin 0.1089 直接对齐比较。
if [ $RC -eq 0 ]; then
  echo "[$(date '+%F %T')] === 扩跑后 val_thin TF 复算 ==="
  $E/bin/accelerate launch --mixed_precision bf16 scripts/train_reasonseg.py \
    --model-base "$ASSETS/base_model/vicuna-v1-5-7b" \
    --pretrain-mm-mlp-adapter "$ASSETS/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin" \
    --b3-checkpoint "$ASSETS/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3" \
    --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
    --validation-manifest "$V/reasonseg_val_thin.jsonl" \
    --dataroot /root/autodl-tmp/nuScenes \
    --output-dir ./checkpoints/_tfval_tmp \
    --eval-checkpoint "$OUT/checkpoint-best" \
    --validate-only --validation-samples 0 --validate-dtype-variants as_is \
    --mixed-precision bf16 > "$L/${NAME}-ext20.valthin.log" 2>&1
  echo "[$(date '+%F %T')] val_thin rc=$?"
  $E/bin/python scripts/reasonseg_experiments/diagnose_reasonseg_grounding.py \
    --model-base "$ASSETS/base_model/vicuna-v1-5-7b" \
    --pretrain-mm-mlp-adapter "$ASSETS/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin" \
    --b3-checkpoint "$ASSETS/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3" \
    --seg-checkpoint "$OUT/checkpoint-best" \
    --manifest "$V/reasonseg_val_thin.jsonl" \
    --dataroot /root/autodl-tmp/nuScenes \
    --output "./eval_results/_grounding_${NAME}-ext20/report.json" \
    --max-samples 1000 --dtype fp16 --encoder-dtype fp32 > "$L/${NAME}-ext20.grounding.log" 2>&1
  echo "[$(date '+%F %T')] grounding rc=$?"
fi
echo "=== EXTENSION DONE: $NAME ==="
