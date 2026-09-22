#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/mmb4dl/mllm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
$E/bin/accelerate launch --mixed_precision bf16 scripts/train_reasonseg.py \
  --model-base ./base_model/vicuna-v1-5-7b \
  --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
  --train-manifest ./reasonseg_data_trainval/reasonseg_train_680x11.jsonl \
  --validation-manifest ./reasonseg_data_trainval/reasonseg_val_thin.jsonl \
  --dataroot /root/autodl-tmp/nuScenes \
  --output-dir ./checkpoints/reasonseg-tvenc680 \
  --spatial-checkpoint ./checkpoints/reasonseg-spatial-tv/spatial-best \
  --num-train-epochs 20 --per-device-train-batch-size 1 --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 --early-stopping-patience 12 --early-stopping-min-delta 1e-4 \
  --save-steps 200 --keep-last 2 --no-resume-state --validation-samples 0 --mixed-precision bf16
echo TRAIN680_EXIT=0
