#!/bin/bash
# Resume Stage2 full training from checkpoint-2000

eval "$(conda shell.bash hook)"
conda activate wqlc

cd /root/autodl-tmp/wql/mmb4dl/mllm

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./training_logs/stage2_full_resume_${TIMESTAMP}.log"

echo "============================================================" | tee "$LOG_FILE"
echo "恢复训练 from checkpoint-2000" | tee -a "$LOG_FILE"
echo "启动: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

export WANDB_MODE=offline
export WANDB_PROJECT=b4dl-stage2-full
export PYTHONUNBUFFERED=1

MODEL_VERSION=vicuna-v1-5-7b

deepspeed --include localhost:0 --master_port 29576 vtimellm/train/train_mem.py \
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
echo "完成: $(date)" | tee -a "$LOG_FILE"
