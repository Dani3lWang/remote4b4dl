#!/bin/bash
cd /root/autodl-tmp/wql/mmb4dl/mllm
eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)"
conda activate wqlc

echo "===== PER-SEQUENCE V3 TRAINING (paper-aligned seqv3 data) ====="
echo "Start: $(date)"
echo "Data: b4dl_dataset/stage2_full_train_seqv3_148k.json (TG 按 GT 帧范围恢复序列归属 + feat_indices 精确采样帧, 148,271 条)"

export WANDB_MODE=offline
export WANDB_PROJECT=b4dl-stage2-full-seqv3
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_VERSION=vicuna-v1-5-7b

deepspeed --include localhost:0 --master_port 29578 vtimellm/train/train_mem.py     --deepspeed ./scripts/zero3.json     --lora_enable True     --model_name_or_path ./base_model/vicuna-v1-5-7b     --version v1     --data_path ./b4dl_dataset/stage2_full_train_seqv3_148k.json     --feat_folder ../encoders/lidarclip/b4dl/stage2_features     --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin     --output_dir ./checkpoints/vtimellm-$MODEL_VERSION-stage2-full-seqv3     --bf16 True     --num_train_epochs 3     --per_device_train_batch_size 8     --gradient_accumulation_steps 16     --evaluation_strategy no     --save_strategy steps     --save_steps 200     --save_total_limit 3     --learning_rate 1e-4     --freeze_mm_mlp_adapter True     --lora_r 64     --lora_alpha 128     --weight_decay 0.     --warmup_ratio 0.03     --lr_scheduler_type cosine     --logging_steps 1     --model_max_length 2048     --gradient_checkpointing True     --dataloader_num_workers 4     --lazy_preprocess True     --report_to wandb     2>&1 | tee ./training_logs/stage2_full_seqv3_$(date +%Y%m%d_%H%M%S).log

echo "End: $(date)"
