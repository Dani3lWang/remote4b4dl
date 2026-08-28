#!/bin/bash
# 混合训练（论文 §4.2 原文："The model is trained on both Simple Tasks and
# Complex Tasks"）——stage2+stage3 全量 148,271 条一起训，单 LoRA。
# 数据: stage2_full_train_seqv3_148k.json（TG 13,124 条全部 GT 序列归属 +
# 真实帧 metatoken + feat_indices 精确采样帧）
# projector: 95K nu-caption 重训版（stage1 2026-08-25）
# 超参对齐 seqv2-148k（acc 0.7647 的配置）：3 epochs lr 1e-4，无 tf32。
cd /root/autodl-tmp/wql/mmb4dl/mllm
eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)"
conda activate wqlc

echo "===== MIXED TRAINING (paper §4.2 simple+complex, seqv3 data) ====="
echo "Start: $(date)"
echo "Data: b4dl_dataset/stage2_full_train_seqv3_148k.json"

export WANDB_MODE=offline
export WANDB_PROJECT=b4dl-stage2-full-seqv3-mixed
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_VERSION=vicuna-v1-5-7b

RESUME=""
LATEST=$(ls -d ./checkpoints/vtimellm-$MODEL_VERSION-stage2-full-seqv3-mixed/checkpoint-* 2>/dev/null | tail -1)
[ -n "$LATEST" ] && RESUME="--resume_from_checkpoint $LATEST" && echo "Resume: $LATEST"

deepspeed --include localhost:0 --master_port 29580 vtimellm/train/train_mem.py     --deepspeed ./scripts/zero3.json     --lora_enable True     --model_name_or_path ./base_model/vicuna-v1-5-7b     --version v1     --data_path ./b4dl_dataset/stage2_full_train_seqv3_148k.json     --feat_folder ../encoders/lidarclip/b4dl/stage2_features     --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin     --output_dir ./checkpoints/vtimellm-$MODEL_VERSION-stage2-full-seqv3-mixed     --bf16 True     --num_train_epochs 3     --per_device_train_batch_size 8     --gradient_accumulation_steps 16     --evaluation_strategy no     --save_strategy steps     --save_steps 200     --save_total_limit 3     --learning_rate 1e-4     --freeze_mm_mlp_adapter True     --lora_r 64     --lora_alpha 128     --weight_decay 0.     --warmup_ratio 0.03     --lr_scheduler_type cosine     --logging_steps 1     --model_max_length 2048     --gradient_checkpointing True     --dataloader_num_workers 4     --lazy_preprocess True     --report_to wandb     $RESUME     2>&1 | tee ./training_logs/stage2_full_seqv3_mixed_$(date +%Y%m%d_%H%M%S).log

echo "End: $(date)"
