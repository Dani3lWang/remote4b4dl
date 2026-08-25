#!/bin/bash
# B4DL 两阶段训练驱动器（论文/官方方法，seqv3 数据）
#   Phase A: stage2.json（简单任务：existence/binary/TG）先训，2 epochs lr 1e-4
#   Merge:   stage2 LoRA 合并进 base（官方 pipeline：stage3 在 merged 上训练）
#   Phase B: stage3.json（复杂任务：description/temporal/comprehensive）
#            在 stage2-merged 上训练新 LoRA，3 epochs lr 2e-5
# 幂等设计：Phase A 在 merged 已存在时跳过；两阶段都支持 checkpoint 断点续训，
# 便于守护 cron 在任何中断点安全重启。
cd /root/autodl-tmp/wql/mmb4dl/mllm
eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)"
conda activate wqlc

echo "===== B4DL 两阶段训练（论文/官方方法 + seqv3 数据）====="
echo "Start: $(date)"

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_VERSION=vicuna-v1-5-7b
MERGED=./checkpoints/vtimellm-$MODEL_VERSION-stage2-seqv3-merged

# ── Phase A: stage2 简单任务 ──
if [ ! -f "$MERGED/mm_projector.bin" ]; then
    echo "===== Phase A: stage2 简单任务训练（2 epochs, lr 1e-4）====="
    export WANDB_PROJECT=b4dl-stage2-seqv3
    A_RESUME=""
    LATEST_A=$(ls -d ./checkpoints/vtimellm-$MODEL_VERSION-stage2-seqv3/checkpoint-* 2>/dev/null | tail -1)
    [ -n "$LATEST_A" ] && A_RESUME="--resume_from_checkpoint $LATEST_A" && echo "Phase A resume: $LATEST_A"
    deepspeed --include localhost:0 --master_port 29578 vtimellm/train/train_mem.py     --deepspeed ./scripts/zero3.json     --lora_enable True     --model_name_or_path ./base_model/vicuna-v1-5-7b     --version v1     --data_path ./b4dl_dataset/stage2_train_seqv3.json     --feat_folder ../encoders/lidarclip/b4dl/stage2_features     --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin     --output_dir ./checkpoints/vtimellm-$MODEL_VERSION-stage2-seqv3     --bf16 True     --num_train_epochs 2     --per_device_train_batch_size 8     --gradient_accumulation_steps 16     --evaluation_strategy no     --save_strategy steps     --save_steps 200     --save_total_limit 3     --learning_rate 1e-4     --freeze_mm_mlp_adapter True     --lora_r 64     --lora_alpha 128     --weight_decay 0.     --warmup_ratio 0.03     --lr_scheduler_type cosine     --logging_steps 1     --tf32 True     --model_max_length 2048     --gradient_checkpointing True     --dataloader_num_workers 4     --lazy_preprocess True     --report_to wandb     $A_RESUME     2>&1 | tee ./training_logs/stage2_seqv3_$(date +%Y%m%d_%H%M%S).log

    echo "===== Merge stage2 LoRA into base ====="
    python scripts/merge_stage2.py     --model_base ./base_model/vicuna-v1-5-7b     --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin     --stage2 ./checkpoints/vtimellm-$MODEL_VERSION-stage2-seqv3     --output_dir $MERGED     2>&1 | tee ./training_logs/stage2_seqv3_merge_$(date +%Y%m%d_%H%M%S).log
    touch /tmp/stage2_seqv3_done
fi

# ── Phase B: stage3 复杂任务（在 merged 上训练新 LoRA）──
LOG_B=./training_logs/stage3_seqv3_$(date +%Y%m%d_%H%M%S).log
B_RESUME=""
LATEST_B=$(ls -d ./checkpoints/vtimellm-$MODEL_VERSION-stage3-seqv3/checkpoint-* 2>/dev/null | tail -1)
[ -n "$LATEST_B" ] && B_RESUME="--resume_from_checkpoint $LATEST_B" && echo "Phase B resume: $LATEST_B"
echo "===== Phase B: stage3 复杂任务训练（3 epochs, lr 2e-5）$B_RESUME ====="
export WANDB_PROJECT=b4dl-stage3-seqv3
deepspeed --include localhost:0 --master_port 29579 vtimellm/train/train_mem.py     --deepspeed ./scripts/zero3.json     --lora_enable True     --model_name_or_path $MERGED     --version v1     --data_path ./b4dl_dataset/stage3_train_seqv3.json     --feat_folder ../encoders/lidarclip/b4dl/stage2_features     --pretrain_mm_mlp_adapter $MERGED/mm_projector.bin     --output_dir ./checkpoints/vtimellm-$MODEL_VERSION-stage3-seqv3     --bf16 True     --num_train_epochs 3     --per_device_train_batch_size 8     --gradient_accumulation_steps 16     --evaluation_strategy no     --save_strategy steps     --save_steps 200     --save_total_limit 3     --learning_rate 2e-5     --freeze_mm_mlp_adapter True     --lora_r 64     --lora_alpha 128     --weight_decay 0.     --warmup_ratio 0.03     --lr_scheduler_type cosine     --logging_steps 1     --tf32 True     --model_max_length 2048     --gradient_checkpointing True     --dataloader_num_workers 4     --lazy_preprocess True     --report_to wandb     $B_RESUME     2>&1 | tee $LOG_B

echo "End: $(date)" | tee -a $LOG_B
