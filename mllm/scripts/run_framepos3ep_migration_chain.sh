#!/bin/bash
# 迁移验收 → 帧位置 3ep 实验链（设计成跑在一个 tmux 会话里，断链不影响）。
#
#   阶段0  B3 同口径评测，须复现旧机 mIoU 0.3467 / acc 0.7526。|ΔmIoU| 超过显著阈值
#          0.013 说明新机数据或评测口径有漂移，此时任何新实验的 Δ 都无法归因，链停止。
#          若评测已在别处跑着（本脚本重启、或人工先起了评测），只等它的 metrics.json，不重复占卡。
#   阶段1  训练侧冒烟 scripts/smoke_framepos_train.sh（新机从未跑过训练，先用 2+2 步验
#          deepspeed 起进程 / 帧位置存档 / 断点续训，再开 18h 正式训练）。
#   阶段2  framepos3ep = B3 配方 + 零初始化绝对帧位置 embedding（唯一变量）训练 + 同口径评测。
#          对照组直接用已训完的 B3，省掉一个同 epoch 基线的 12h。
#
# 判据：训练看 trainer_state.json 的 epoch>=2.99（3 次断点续训重试）；评测成功且
# metrics.json 非空（3 次重试）。每个阶段之间检查磁盘，不足 MIN_FREE_GB 就等待而不是写爆盘。
# 训练产物：顶层 adapter_model.safetensors + non_lora_trainables.bin 才是评测入口，
# 中间 checkpoint-* 只为续训存在，评测通过后即删除（KEEP_CKPT=1 保留）。
set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${B4DL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WQLC_PREFIX="${B4DL_ENV_PREFIX:-$(dirname "$PROJECT_ROOT")/.conda-stuff/envs/wqlc}"
PY="$WQLC_PREFIX/bin/python"
[ -x "$PY" ] || { echo "错误: wqlc 环境不存在: $WQLC_PREFIX" >&2; exit 1; }
export PATH="$WQLC_PREFIX/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT/mllm" || exit 1
mkdir -p training_logs eval_results

B3_DIR=checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3
FP_DIR=checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-framepos3ep
B3_MIOU=0.3467
B3_ACC=0.7526
SIGMA=0.013
MIN_FREE_GB=8
EVAL_WAIT_SECS=21600

avail_gb() { local a; a=$(df -BG --output=avail "$PROJECT_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9'); echo "${a:-999}"; }

run_eval() {   # $1=stage2 目录 $2=结果目录名
    local stage2="$1" tag="$2" out="./eval_results/$2" attempt rc
    mkdir -p "$out"
    for attempt in 1 2 3; do
        timeout 21600 "$PY" -u evaluation/test_b4dl.py \
            --model_base ./base_model/vicuna-v1-5-7b \
            --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
            --stage2 "$stage2" \
            --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
            --test_data ./b4dl_dataset/test_qa.json \
            --ego_meta ./b4dl_dataset/ego_metadata.json \
            --frame_motion ./b4dl_dataset/ego_frame_motion.json \
            --whole_scene --per_sequence --answer_frames \
            --output "$out/predictions.json" \
            --metrics_output "$out/metrics.json" \
            >> "$out/eval_log.txt" 2>&1
        rc=$?
        echo "[$(date '+%F %T')] $tag 评测尝试 #$attempt rc=$rc"
        [ $rc -eq 0 ] && [ -s "$out/metrics.json" ] && return 0
        sleep 60
    done
    return 1
}

report() {   # $1=结果目录名
    METRICS="./eval_results/$1/metrics.json" LABEL="$1" REF="$B3_MIOU" "$PY" -c "
import json, os
f = json.load(open(os.environ['METRICS']))['final_scores']
print(f\"  {os.environ['LABEL']:14s} acc={f['accuracy']:.4f} mIoU={f['miou']:.4f} \"
      f\"(Δvs B3 {f['miou']-float(os.environ['REF']):+.4f}) bleu4={f['bleu4']:.4f} \"
      f\"rouge_l={f['rouge_l']:.4f} meteor={f.get('meteor',0):.4f} bertscore={f.get('bertscore',0):.4f}\")"
}

gate_vs_b3() {   # $1=结果目录名 $2=展示标签；ΔmIoU 超过显著阈值 → 返回非 0
    METRICS="./eval_results/$1/metrics.json" REF="$B3_MIOU" SIGMA="$SIGMA" LABEL="${2:-$1}" "$PY" -c "
import json, os, sys
miou = json.load(open(os.environ['METRICS']))['final_scores']['miou']
d = miou - float(os.environ['REF'])
print(f\"  {os.environ['LABEL']}: mIoU={miou:.4f} Δ={d:+.4f}（显著阈值 {os.environ['SIGMA']}）\")
sys.exit(0 if abs(d) <= float(os.environ['SIGMA']) else 1)"
}

echo "########## 迁移验收 → 帧位置 3ep 链 开始 $(date '+%F %T') ##########"
echo "项目: $PROJECT_ROOT  环境: $WQLC_PREFIX  磁盘余量: $(avail_gb)G"

echo "===== 阶段0: B3 迁移验收 ($(date '+%F %T')) ====="
[ -f "$B3_DIR/adapter_model.safetensors" ] || { echo "缺 B3 adapter，无法验收"; exit 1; }
if [ -s ./eval_results/migration_b3/metrics.json ]; then
    echo "已有 migration_b3/metrics.json，直接复用"
elif pgrep -f "test_b4dl.py.*$(basename "$B3_DIR")" > /dev/null 2>&1; then
    echo "检测到 B3 评测正在别处运行，等待其 metrics.json（上限 $((EVAL_WAIT_SECS/3600))h）"
    WAITED=0
    while [ ! -s ./eval_results/migration_b3/metrics.json ]; do
        [ "$WAITED" -ge "$EVAL_WAIT_SECS" ] && { echo "等待超时，链停止"; exit 1; }
        pgrep -f "test_b4dl.py.*$(basename "$B3_DIR")" > /dev/null 2>&1 || {
            echo "在跑的评测已退出但没有产出 metrics.json，链停止（避免重复占卡）"; exit 1; }
        sleep 120; WAITED=$((WAITED+120))
    done
    echo "[$(date '+%F %T')] 等到 metrics.json（等待 ${WAITED}s）"
else
    run_eval "./$B3_DIR" migration_b3 || { echo "阶段0 评测失败，链停止"; exit 1; }
fi
report migration_b3
gate_vs_b3 migration_b3 "新机 B3 复现" || { echo "新机口径漂移超过显著阈值 ${SIGMA}，链停止——先查搬运差异"; exit 1; }
echo "阶段0 通过：新机评测口径与旧机一致"

echo "===== 阶段1: 训练侧冒烟 ($(date '+%F %T')) ====="
bash scripts/smoke_framepos_train.sh 2>&1 | tee -a training_logs/chain_framepos3ep_smoke.log
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "阶段1 冒烟失败，链停止（详见 training_logs/chain_framepos3ep_smoke.log）"; exit 1; }
echo "阶段1 通过：帧位置训练与续训路径可用"

echo "===== 阶段2: framepos3ep 训练 ($(date '+%F %T')) ====="
FREE=$(avail_gb)
[ "$FREE" -ge "$MIN_FREE_GB" ] || { echo "磁盘余量 ${FREE}G < ${MIN_FREE_GB}G，不开训（中间档约 2.5G/个）"; exit 1; }
TRAIN_OK=0
for attempt in 1 2 3; do
    echo "[$(date '+%F %T')] framepos3ep 训练尝试 #$attempt"
    bash scripts/run_stage2_full_seqv3_mixed_framepos3ep.sh >> training_logs/chain_framepos3ep_stdout.log 2>&1
    if [ -f "$FP_DIR/adapter_model.safetensors" ] && [ -f "$FP_DIR/trainer_state.json" ] && \
       TRAIN_STATE="$FP_DIR/trainer_state.json" "$PY" -c \
       "import json,os,sys; sys.exit(0 if json.load(open(os.environ['TRAIN_STATE'])).get('epoch',0) >= 2.99 else 1)"; then
        TRAIN_OK=1; echo "[$(date '+%F %T')] framepos3ep 训练完成（尝试 #$attempt）"; break
    fi
    echo "[$(date '+%F %T')] 尝试 #$attempt 未完成，10 分钟后断点续训重试"
    sleep 600
done
[ "$TRAIN_OK" -eq 1 ] || { echo "framepos3ep 训练三次尝试均失败，链停止"; exit 1; }

echo "===== 阶段2: framepos3ep 评测 ($(date '+%F %T')) ====="
run_eval "./$FP_DIR" framepos3ep || { echo "framepos3ep 评测失败，链停止"; exit 1; }
if [ "${KEEP_CKPT:-0}" != "1" ]; then
    echo "清理中间 checkpoint-*（保留顶层 adapter）"
    rm -rf "$FP_DIR"/checkpoint-*
fi

echo "########## 链完成 $(date '+%F %T') ##########"
echo "对照（B3 旧机 acc $B3_ACC / mIoU $B3_MIOU，显著阈值 ΔmIoU>$SIGMA）："
report migration_b3
report framepos3ep
if gate_vs_b3 framepos3ep "framepos3ep"; then
    echo "结论：ΔmIoU 未超过显著阈值 ${SIGMA}，帧位置与 B3 同档（需换 seed/加大变量再判）"
else
    DIR=$(METRICS=./eval_results/framepos3ep/metrics.json REF="$B3_MIOU" "$PY" -c "
import json, os
d = json.load(open(os.environ['METRICS']))['final_scores']['miou'] - float(os.environ['REF'])
print('显著回退' if d < 0 else '显著提升')")
    echo "结论：framepos3ep 相对 B3 ${DIR}（|Δ|>${SIGMA}），对照基准 B3 mIoU $B3_MIOU"
fi
