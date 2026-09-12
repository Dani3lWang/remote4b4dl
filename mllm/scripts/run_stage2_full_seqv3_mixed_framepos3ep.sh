#!/bin/bash
# B3 配方 + 唯一变量：零初始化的可学习绝对场景帧位置 embedding（路线图 A.2）。
#
# 与 run_stage2_full_seqv3_mixed_b3.sh 逐参数等价——B3 未显式写出的 lora_dropout(0.05)、
# train_meta_embed(True)、optim(adamw_torch)、frame_position_max(64) 都是 train.py 的默认值，
# 因此本脚本相对 B3 只多出 --use_frame_position_embedding True 一项。
# 对照组直接复用已训完并评过的 B3（acc 0.7526 / mIoU 0.3467，显著阈值 ΔmIoU>0.013），
# 不需要再跑一个同 epoch 的基线。
# 断点续训：checkpoint-* 按版本排序取最新，与 train.py 内部的 max(checkpoints) 一致。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${B4DL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WQLC_PREFIX="${B4DL_ENV_PREFIX:-$(dirname "$PROJECT_ROOT")/.conda-stuff/envs/wqlc}"

if [ ! -x "$WQLC_PREFIX/bin/python" ] || [ ! -x "$WQLC_PREFIX/bin/deepspeed" ]; then
    echo "错误: wqlc 环境不完整: $WQLC_PREFIX" >&2
    echo "可通过 B4DL_ENV_PREFIX 指定环境目录。" >&2
    exit 1
fi

export PATH="$WQLC_PREFIX/bin:$PATH"
cd "$PROJECT_ROOT/mllm" || exit 1
python scripts/preflight_b3.py --project-root "$PROJECT_ROOT" --require-cuda || exit 1

echo "===== B3 配方 + 绝对帧位置 embedding (3 epochs) ====="
echo "Start: $(date)"
echo "Data: b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json"

export WANDB_MODE=offline
export WANDB_PROJECT=b4dl-stage2-full-seqv3-mixed-framepos3ep
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_VERSION=vicuna-v1-5-7b
OUT=vtimellm-$MODEL_VERSION-stage2-full-seqv3-mixed-framepos3ep
set -o pipefail
LOG=./training_logs/stage2_full_seqv3_mixed_framepos3ep_$(date +%Y%m%d_%H%M%S).log

RESUME=""
LATEST=$(ls -d ./checkpoints/$OUT/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -n "$LATEST" ] && RESUME="--resume_from_checkpoint $LATEST" && echo "Resume: $LATEST"

deepspeed --include localhost:0 --master_port 29587 vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path ./base_model/vicuna-v1-5-7b \
    --version v1 \
    --data_path ./b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json \
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
    --use_frame_position_embedding True \
    --frame_position_max 64 \
    --report_to wandb \
    $RESUME \
    2>&1 | tee "$LOG"

echo "End: $(date)" | tee -a "$LOG"
