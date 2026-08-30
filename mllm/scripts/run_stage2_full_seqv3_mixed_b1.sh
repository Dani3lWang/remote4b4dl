#!/bin/bash
# B1 混合训练（待修改清单执行主线）：与 B0（run_stage2_full_seqv3_mixed.sh）完全同超参，
# 仅两处不同——① stage1 projector 换 162K 重训版（同一路径，重训后自动生效）；
# ② 独立 output_dir -mixed-b1，B0 的 checkpoint 与评测产物保持原封不动。
# 断点续训：mllm train.py 已按步数数值排序取最新（e8639e2），此处 sort -V 与其对齐。
cd /root/autodl-tmp/wql/mmb4dl/mllm
eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)"
conda activate wqlc

echo "===== B1 MIXED TRAINING (seqv3 data + 162K projector) ====="
echo "Start: $(date)"
echo "Data: b4dl_dataset/stage2_full_train_seqv3_148k.json"

export WANDB_MODE=offline
export WANDB_PROJECT=b4dl-stage2-full-seqv3-mixed
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_VERSION=vicuna-v1-5-7b
OUT=vtimellm-$MODEL_VERSION-stage2-full-seqv3-mixed-b1

RESUME=""
LATEST=$(ls -d ./checkpoints/$OUT/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -n "$LATEST" ] && RESUME="--resume_from_checkpoint $LATEST" && echo "Resume: $LATEST"

deepspeed --include localhost:0 --master_port 29581 vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path ./base_model/vicuna-v1-5-7b \
    --version v1 \
    --data_path ./b4dl_dataset/stage2_full_train_seqv3_148k.json \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin \
    --output_dir ./checkpoints/$OUT \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --freeze_mm_mlp_adapter True \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    $RESUME \
    2>&1 | tee ./training_logs/stage2_full_seqv3_mixed_b1_$(date +%Y%m%d_%H%M%S).log

echo "End: $(date)"
