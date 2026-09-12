#!/bin/bash
# 迁移验收 → 2-epoch 帧位置对照链（设计成在单个 tmux 会话里跑完，断链不影响）。
#   阶段0  B3 同口径评测，须复现旧机 mIoU 0.3467 / acc 0.7526；
#          |ΔmIoU| 超过显著阈值 0.013 判定为新机口径漂移，链停止（否则后续 Δ 无法归因）。
#   阶段1  paper2ep   = B3 配方 + 2 epoch，无帧位置（同期基线）
#   阶段2  framepos2ep = 同上 + 零初始化绝对帧位置 embedding（唯一变量）
# 每个实验训练+评测成功后删除其自身中间 checkpoint-*（顶层 adapter 才是评测产物）：
# 本机数据盘仅 19G，ZeRO-3 单个中间档含优化器状态约 2.5G，不清理放不下第二个实验。KEEP_CKPT=1 关闭。
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

B3_MIOU=0.3467
B3_ACC=0.7526
SIGMA=0.013
MIN_FREE_GB=8

avail_gb() { df -BG --output=avail "$PROJECT_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9'; }

run_eval() {
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

run_train() {
    local suffix="$1" out attempt free
    out="./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-$suffix"
    [ "$suffix" = paper2ep ] && out="$out-baseline"
    for attempt in 1 2 3; do
        free=$(avail_gb)
        if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_GB" ]; then
            echo "[$(date '+%F %T')] 磁盘余量 ${free}G < ${MIN_FREE_GB}G，等待清理/人工处理"; sleep 600; continue
        fi
        echo "[$(date '+%F %T')] $suffix 训练尝试 #$attempt"
        bash "scripts/run_stage2_full_seqv3_mixed_$suffix.sh" >> "training_logs/chain_${suffix}_stdout.log" 2>&1
        if [ -f "$out/adapter_model.safetensors" ] && [ -f "$out/trainer_state.json" ] && \
           TRAIN_STATE="$out/trainer_state.json" "$PY" -c \
           "import json,os,sys; sys.exit(0 if json.load(open(os.environ['TRAIN_STATE'])).get('epoch',0) >= 1.99 else 1)"; then
            echo "[$(date '+%F %T')] $suffix 训练完成（尝试 #$attempt）"
            TRAIN_STATE="$out/trainer_state.json" "$PY" -c \
            "import json,os;d=json.load(open(os.environ['TRAIN_STATE']));print(f\"  final epoch={d['epoch']:.3f} step={d['global_step']}\")"
            return 0
        fi
        echo "[$(date '+%F %T')] $suffix 尝试 #$attempt 未完成，10 分钟后断点续训重试"
        sleep 600
    done
    return 1
}

report() {
    local tag="$1" m
    m="./eval_results/$tag/metrics.json"
    METRICS="$m" REF="$B3_MIOU" LABEL="$tag" "$PY" -c "
import json, os
f = json.load(open(os.environ['METRICS']))['final_scores']
ref = float(os.environ['REF'])
print(f\"  {os.environ['LABEL']:12s} acc={f['accuracy']:.4f} mIoU={f['miou']:.4f} \"
      f\"(Δvs B3 {f['miou']-ref:+.4f}) bleu4={f['bleu4']:.4f} rouge_l={f['rouge_l']:.4f} \"
      f\"meteor={f.get('meteor',0):.4f} bertscore={f.get('bertscore',0):.4f}\")"
}

echo "########## 迁移验收 → 2ep 帧位置链 开始 $(date '+%F %T') ##########"
echo "项目: $PROJECT_ROOT  环境: $WQLC_PREFIX  磁盘余量: $(avail_gb)G"

echo "===== 阶段0: B3 迁移验收 ($(date '+%F %T')) ====="
[ -f checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3/adapter_model.safetensors ] \
    || { echo "缺 B3 adapter，无法验收"; exit 1; }
run_eval ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 migration_b3 \
    || { echo "阶段0 评测失败，链停止"; exit 1; }
report migration_b3
MIGRATION_OK=$(METRICS=./eval_results/migration_b3/metrics.json REF="$B3_MIOU" SIGMA="$SIGMA" "$PY" -c "
import json, os, sys
miou = json.load(open(os.environ['METRICS']))['final_scores']['miou']
d = miou - float(os.environ['REF'])
print(f'  ΔmIoU={d:+.4f} vs 旧机 {os.environ[\"REF\"]}（显著阈值 {os.environ[\"SIGMA\"]}）', file=sys.stderr)
sys.stdout.write('1' if abs(d) <= float(os.environ['SIGMA']) else '0')")
[ "$MIGRATION_OK" = 1 ] || { echo "新机口径漂移超过 ${SIGMA}，链停止——先排查搬运差异再开新实验"; exit 1; }
echo "阶段0 通过：新机评测口径与旧机一致"

for S in paper2ep framepos2ep; do
    echo "===== $S ($(date '+%F %T')) ====="
    run_train "$S" || { echo "$S 训练三次尝试均失败，链停止"; exit 1; }
    OUT="checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-$S"
    [ "$S" = paper2ep ] && OUT="$OUT-baseline"
    run_eval "./$OUT" "$S" || { echo "$S 评测失败，链停止"; exit 1; }
    report "$S"
    if [ "${KEEP_CKPT:-0}" != "1" ]; then
        echo "清理 $S 中间 checkpoint-*（保留顶层 adapter）"
        rm -rf "$OUT"/checkpoint-*
    fi
    echo "===== $S 完成，磁盘余量 $(avail_gb)G ====="
done

echo "########## 链完成 $(date '+%F %T') ##########"
echo "对照（B3 旧机 acc $B3_ACC / mIoU $B3_MIOU，显著阈值 ΔmIoU>$SIGMA）："
for tag in migration_b3 paper2ep framepos2ep; do report "$tag"; done
