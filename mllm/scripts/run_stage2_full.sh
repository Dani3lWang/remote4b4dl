#!/bin/bash
# B4DL Stage2 全量训练（对标论文：合并 stage2 + stage3 数据）
# 数据: stage2_full_train.json (118,722 条, 34% 简单QA + 50% 复杂描述)
# 预计: ~2,784 steps, ~15小时 (RTX 5090)

set -e

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate wqlc

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./training_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/stage2_full_${TIMESTAMP}.log"

echo "============================================================" | tee "$LOG_FILE"
echo "B4DL Stage2 全量训练" | tee -a "$LOG_FILE"
echo "启动时间: $(date)" | tee -a "$LOG_FILE"
echo "数据: b4dl_dataset/stage2_full_train.json (118,722 条)" | tee -a "$LOG_FILE"
echo "日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "wandb: $WANDB_MODE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

export WANDB_MODE=offline
export WANDB_PROJECT=b4dl-stage2-full
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_VERSION=vicuna-v1-5-7b

deepspeed --include localhost:0 --master_port 29575 vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path ./base_model/vicuna-v1-5-7b \
    --version v1 \
    --data_path ./b4dl_dataset/stage2_full_train.json \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin \
    --output_dir ./checkpoints/vtimellm-$MODEL_VERSION-stage2-full \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --freeze_mm_mlp_adapter True \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "训练完成: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
