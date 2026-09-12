#!/bin/bash
# 训练侧冒烟（换新机器或改过帧位置分支后，先跑这个再开 18h 正式训练）。
#
# 只验四件事：deepspeed 能起进程、帧位置前向/反向能走通、顶层 adapter 与
# non_lora_trainables.bin 里确有 frame_position_embedding 权重、断点续训能命中最新 checkpoint。
# batch/accum 与正式训练一致（8×16）且用真数据子集，所以显存与特征读取路径一并验到。
# 耗时约 8-10 分钟，大头是 13GB 权重加载而不是 2 个优化步。SMOKE_KEEP=1 保留产物。
set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${B4DL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WQLC_PREFIX="${B4DL_ENV_PREFIX:-$(dirname "$PROJECT_ROOT")/.conda-stuff/envs/wqlc}"
PY="$WQLC_PREFIX/bin/python"
[ -x "$WQLC_PREFIX/bin/deepspeed" ] || { echo "错误: wqlc 环境不完整: $WQLC_PREFIX" >&2; exit 1; }
export PATH="$WQLC_PREFIX/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT/mllm" || exit 1
mkdir -p training_logs

SRC=./b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json
SUB=./b4dl_dataset/_smoke_framepos_512.json
OUT=./checkpoints/_smoke_framepos3ep
LOG=./training_logs/smoke_framepos_$(date +%Y%m%d_%H%M%S).log

[ -s "$SRC" ] || { echo "错误: 缺源数据 $SRC" >&2; exit 1; }
if [ ! -s "$SUB" ]; then
    echo "生成 512 条子集 $SUB"
    SRC_PATH="$SRC" SUB_PATH="$SUB" "$PY" -c "
import json, os
data = json.load(open(os.environ['SRC_PATH']))
os.makedirs(os.path.dirname(os.environ['SUB_PATH']), exist_ok=True)
json.dump(data[:512], open(os.environ['SUB_PATH'], 'w'))
print(f'  {len(data[:512])} 条 -> {os.environ[\"SUB_PATH\"]}')" || exit 1
fi

run_train() {   # $1=max_steps $2=resume_from(可空)
    local steps="$1" resume="${2:-}"
    deepspeed --include localhost:0 --master_port 29589 vtimellm/train/train_mem.py \
        --deepspeed ./scripts/zero3.json \
        --lora_enable True \
        --model_name_or_path ./base_model/vicuna-v1-5-7b \
        --version v1 \
        --data_path "$SUB" \
        --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
        --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
        --output_dir "$OUT" \
        --whole_scene True \
        --bf16 True \
        --num_train_epochs 3 \
        --max_steps "$steps" \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps "$steps" \
        --save_total_limit 5 \
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
        --report_to none \
        $resume \
        2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
}

echo "===== 冒烟 1/2: 从零训练 2 步 ($(date '+%F %T')) ====="
rm -rf "$OUT"
run_train 2 || { echo "冒烟失败: 首次训练未正常结束（见 $LOG）"; exit 1; }

echo "===== 冒烟 2/2: 断点续训到 4 步 ($(date '+%F %T')) ====="
LATEST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -n "$LATEST" ] || { echo "冒烟失败: 没有生成 checkpoint-*（$OUT）"; exit 1; }
echo "Resume: $LATEST"
run_train 4 "--resume_from_checkpoint $LATEST" || { echo "冒烟失败: 续训未正常结束（见 $LOG）"; exit 1; }

"$PY" -c "
import json, os, sys, torch
out = '$OUT'
for f in ('adapter_model.safetensors', 'non_lora_trainables.bin', 'adapter_config.json', 'trainer_state.json'):
    p = os.path.join(out, f)
    if not os.path.isfile(p) or os.path.getsize(p) == 0:
        sys.exit(f'缺少产物: {p}')
cfg = json.load(open(os.path.join(out, 'config.json')))
print(f\"  config: use_frame_position_embedding={cfg.get('use_frame_position_embedding')} frame_position_max={cfg.get('frame_position_max')}\")
if not cfg.get('use_frame_position_embedding'):
    sys.exit('config 未记录帧位置开关，评测时不会被恢复')
non_lora = torch.load(os.path.join(out, 'non_lora_trainables.bin'), map_location='cpu', weights_only=True)
keys = [k for k in non_lora if 'frame_position' in k]
if not keys:
    sys.exit(f'non_lora_trainables.bin 里没有帧位置权重，现有键: {list(non_lora)[:5]}')
for k in keys:
    print(f'  {k} shape={tuple(non_lora[k].shape)} dtype={non_lora[k].dtype} absmax={non_lora[k].abs().max():.3e}')
step = json.load(open(os.path.join(out, 'trainer_state.json')))['global_step']
if step < 4:
    sys.exit(f'续训未推进 global_step={step}')
print(f'  global_step={step}（续训生效）')" || exit 1

echo "===== 训练侧冒烟 PASS ($(date '+%F %T')) ====="
if [ "${SMOKE_KEEP:-0}" != "1" ]; then
    echo "清理冒烟产物 $OUT 与 $SUB"
    rm -rf "$OUT" "$SUB"
fi
