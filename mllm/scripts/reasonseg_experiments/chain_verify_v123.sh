#!/bin/bash
# V1+V2+V3 验证链：过滤不可达目标、用修好的分母复算两侧、补 A2 的自由生成。
#
# V3 过滤改的是数据口径，过滤后的数与所有历史 val_thin 数不可比，所以对照
# (internal679) 必须在新 manifest 上一起重算——本链把两侧并排跑。
# V2 修的是 teacher_forcing 分母：原先 present = object_valid & union.gt(0) 含
# predicted，导致 GT 全越界的物体只在模型触发时才被计入（IoU 必为 0），触发越多
# 的模型背的不可能项越多。改成 scorable = object_valid & (target & valid).any()，
# 纯数据口径。S2/S3 就是拿未过滤 manifest 验证这个修复：预期两侧 objects 都等于
# 2297（= 2440 - 143），且 mean IoU 落在手算的 0.11021 / 0.11718 附近。
# V1 是 A2-ext20 的自由生成，参数与对照那次 (reasonseg-internal679_encfp32)
# 逐字一致，含 --max-samples 300，保证评的是同一批 300 条。
#
# 只用 set -u：任一阶段失败都记 rc 并继续，不要让前面的成果被后面一步吞掉。
set -uo pipefail
cd /root/autodl-tmp/mmb4dl/mllm || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
export NLTK_DATA="${NLTK_DATA:-/root/autodl-tmp/mmb4dl/nltk_data}"
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
V=./reasonseg_data_trainval
L=./training_logs/verify_v123
mkdir -p "$L"
DRIVER="$L/driver.log"

CTRL=./checkpoints/reasonseg-internal679/checkpoint-best
A2=./checkpoints/reasonseg-lossA2-plain-tversky-ext20/checkpoint-best
REACH=$V/reasonseg_val_thin_reachable.jsonl

log () { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER"; }

tfval () { # $1=label $2=manifest $3=checkpoint
  log "=== TF $1 ==="
  $E/bin/accelerate launch --mixed_precision bf16 scripts/train_reasonseg.py \
    --model-base ./base_model/vicuna-v1-5-7b \
    --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
    --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
    --validation-manifest "$2" \
    --dataroot /root/autodl-tmp/nuScenes \
    --output-dir ./checkpoints/_tfval_tmp \
    --eval-checkpoint "$3" \
    --validate-only --validation-samples 0 --validate-dtype-variants as_is \
    --mixed-precision bf16 > "$L/$1.log" 2>&1
  log "$1 rc=$?"
}

log "=== S1/7 (CPU): V3 过滤不可达目标 ==="
$E/bin/python scripts/reasonseg_experiments/filter_unreachable_targets.py \
  --manifest "$V/reasonseg_val_thin.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output "$REACH" \
  --report "$L/filter_report.json" > "$L/s1_filter.log" 2>&1
log "S1 rc=$?"
tail -20 "$L/s1_filter.log" | tee -a "$DRIVER"

tfval s2_ctrl_unfiltered_fixed  "$V/reasonseg_val_thin.jsonl" "$CTRL"
tfval s3_a2_unfiltered_fixed    "$V/reasonseg_val_thin.jsonl" "$A2"
tfval s4_ctrl_reachable         "$REACH"                      "$CTRL"
tfval s5_a2_reachable           "$REACH"                      "$A2"

log "=== S6/7 (GPU): V1 A2-ext20 自由生成 300 条 ==="
timeout 10800 $E/bin/python -u evaluation/evaluate_reasonseg.py \
  --model-base ./base_model/vicuna-v1-5-7b \
  --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
  --seg-checkpoint "$A2" \
  --manifest "$V/reasonseg_val_thin.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output-dir ./eval_results/reasonseg-a2ext20_encfp32 \
  --threshold 0.5 --max-samples 300 --dtype fp16 --encoder-dtype fp32 \
  > "$L/s6_freegen.log" 2>&1
log "S6 rc=$?"

log "=== S7/7 (CPU): 自由生成失败分解 ==="
$E/bin/python scripts/reasonseg_experiments/analyze_reasonseg_failures.py \
  --eval-dir ./eval_results/reasonseg-a2ext20_encfp32 \
  --config "$A2/reasonseg_config.json" \
  --output ./eval_results/reasonseg-a2ext20_encfp32/failure_analysis.json \
  > "$L/s7_failure_analysis.log" 2>&1
log "S7 rc=$?"
rm -rf ./checkpoints/_tfval_tmp
log "=== VERIFY CHAIN DONE ==="
