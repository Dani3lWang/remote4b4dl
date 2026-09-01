#!/bin/bash
# B2 混合训练（整场景输入，对齐官方 B4DL）：与 B1（run_stage2_full_seqv3_mixed_b1.sh）唯一差异
# 是 --whole_scene True——视觉输入不再按 QA 序列切片，而是喂整场景 (39/40/41, 768) 特征，
# 与官方实现（ccho4702/B4DL，dataset.py 对 np.load 结果直接用）一致。
# 动机：seqv3 的序列切片使 time_grounding 学会"序列局部帧编号"，评测按场景全局编号解析，
# mIoU 被系统性压低（B1 0.2653 vs 论文 0.311）；整场景输入下模型学到全局编号。
# 其余（数据 148k、162K projector、3ep、lr 1e-4、LoRA r64/α128、ZeRO-3）与 B1 完全一致。
# 断点续训：mllm train.py 已按步数数值排序取最新（e8639e2），此处 sort -V 与其对齐。
cd /root/autodl-tmp/wql/mmb4dl/mllm
eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)"
conda activate wqlc

echo "===== B2 MIXED TRAINING (whole-scene input, paper-aligned) ====="
echo "Start: $(date)"
echo "Data: b4dl_dataset/stage2_full_train_seqv3_148k.json"

export WANDB_MODE=offline
export WANDB_PROJECT=b4dl-stage2-full-seqv3-mixed-b2
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_VERSION=vicuna-v1-5-7b
OUT=vtimellm-$MODEL_VERSION-stage2-full-seqv3-mixed-b2
set -o pipefail
LOG=./training_logs/stage2_full_seqv3_mixed_b2_$(date +%Y%m%d_%H%M%S).log

RESUME=""
LATEST=$(ls -d ./checkpoints/$OUT/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -n "$LATEST" ] && RESUME="--resume_from_checkpoint $LATEST" && echo "Resume: $LATEST"

deepspeed --include localhost:0 --master_port 29582 vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path ./base_model/vicuna-v1-5-7b \
    --version v1 \
    --data_path ./b4dl_dataset/stage2_full_train_seqv3_148k.json \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin \
    --output_dir ./checkpoints/$OUT \
    --whole_scene True \
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
    2>&1 | tee "$LOG"

echo "End: $(date)" | tee -a "$LOG"
