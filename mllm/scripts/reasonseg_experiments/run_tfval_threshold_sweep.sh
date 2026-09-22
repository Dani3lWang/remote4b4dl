#!/bin/bash
# Sweep the mask binarisation threshold for both best checkpoints in one pass
# each.  Phase 0 found predicted masks have a median of 3 points at the default
# 0.5 cut, so the operating point - not the model - may be the binding constraint.
set -uo pipefail
cd /root/autodl-tmp/mmb4dl/mllm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
mkdir -p ./checkpoints/_tfval_tmp ./training_logs/tfval_threshold

run() { # $1=label $2=ckpt $3=val-manifest
  local log="./training_logs/tfval_threshold/$1.log"
  echo "[$(date '+%F %T')] start $1"
  $E/bin/accelerate launch --mixed_precision bf16 scripts/train_reasonseg.py \
    --model-base ./base_model/vicuna-v1-5-7b \
    --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
    --train-manifest ./reasonseg_data_trainval/reasonseg_train_thin.jsonl \
    --validation-manifest "$3" \
    --dataroot /root/autodl-tmp/nuScenes \
    --output-dir ./checkpoints/_tfval_tmp \
    --eval-checkpoint "$2" \
    --validate-only --validation-samples 0 --validate-dtype-variants as_is \
    --validation-threshold 0.5 \
    --validation-thresholds 0.02,0.05,0.1,0.2,0.3,0.4,0.6,0.7 \
    --mixed-precision bf16 > "$log" 2>&1
  echo "[$(date '+%F %T')] done $1"
}

V=./reasonseg_data_trainval
run sweep_tvenc680_valthin ./checkpoints/reasonseg-tvenc680/checkpoint-best "$V/reasonseg_val_thin.jsonl"
run sweep_tvenc_valthin    ./checkpoints/reasonseg-tvenc/checkpoint-best    "$V/reasonseg_val_thin.jsonl"
echo "=== ALL DONE ==="
