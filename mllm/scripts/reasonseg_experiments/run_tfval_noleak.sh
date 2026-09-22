#!/bin/bash
# Leakage-free teacher-forcing baseline for tvenc / tvenc680 best checkpoints.
# Arm 0 re-validates the old (leaky) val_thin as a calibration run: it must
# reproduce the training-time best before the new manifests are trusted.
set -uo pipefail
cd /root/autodl-tmp/mmb4dl/mllm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
mkdir -p ./checkpoints/_tfval_tmp ./training_logs/tfval_noleak

run() { # $1=label $2=ckpt $3=val-manifest
  local log="./training_logs/tfval_noleak/$1.log"
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
    --mixed-precision bf16 > "$log" 2>&1
  echo "[$(date '+%F %T')] $1 -> $(tr -d '\r' < "$log" | grep -a teacher_forcing_mean_iou | tail -1)"
}

V=./reasonseg_data_trainval
run calib_tvenc680_oldval   ./checkpoints/reasonseg-tvenc680/checkpoint-best "$V/reasonseg_val_thin.jsonl"
run tvenc680_internal_fresh ./checkpoints/reasonseg-tvenc680/checkpoint-best "$V/reasonseg_val_internal_fresh.jsonl"
run tvenc_internal_fresh    ./checkpoints/reasonseg-tvenc/checkpoint-best    "$V/reasonseg_val_internal_fresh.jsonl"
run tvenc_oov500            ./checkpoints/reasonseg-tvenc/checkpoint-best    "$V/reasonseg_val_oov500.jsonl"
echo "=== ALL DONE ==="
