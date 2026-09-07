#!/bin/bash
# B4a 混合训练（TG 高帧段过采样）：与 B3（run_stage2_full_seqv3_mixed_b3.sh）唯一差异
# 是 --data_path 换为过采样版——GT 起始帧 >=25 的 1,951 条 TG 样本 deepcopy 追加一份
# （频率 ×2，148,271 -> 150,222 条，由 oversample_tg_highframe.py 生成并断言校验）。
# 动机：B3 的 TG 预测整体偏早（mean −3.9 帧）且高帧段欠覆盖——GT start>=25 占 ~15%
# 但预测仅 4% 落在高帧段、30+ 帧为 0；测试集同分布，补齐先验后 mIoU 有直接上升空间。
# 其余（整场景输入、meta2、162K projector、3ep、lr 1e-4、LoRA r64/α128、ZeRO-3）与 B3 完全一致。
# 断点续训：mllm train.py 已按步数数值排序取最新（e8639e2），此处 sort -V 与其对齐。
cd /root/autodl-tmp/wql/mmb4dl/mllm
eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)"
conda activate wqlc

echo "===== B4a MIXED TRAINING (TG high-frame oversampled data) ====="
echo "Start: $(date)"
echo "Data: b4dl_dataset/stage2_full_train_seqv3_meta2_oversampled_150k.json"

export WANDB_MODE=offline
export WANDB_PROJECT=b4dl-stage2-full-seqv3-mixed-b4a
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_VERSION=vicuna-v1-5-7b
OUT=vtimellm-$MODEL_VERSION-stage2-full-seqv3-mixed-b4a
set -o pipefail
LOG=./training_logs/stage2_full_seqv3_mixed_b4a_$(date +%Y%m%d_%H%M%S).log

RESUME=""
LATEST=$(ls -d ./checkpoints/$OUT/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -n "$LATEST" ] && RESUME="--resume_from_checkpoint $LATEST" && echo "Resume: $LATEST"

deepspeed --include localhost:0 --master_port 29584 vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path ./base_model/vicuna-v1-5-7b \
    --version v1 \
    --data_path ./b4dl_dataset/stage2_full_train_seqv3_meta2_oversampled_150k.json \
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
