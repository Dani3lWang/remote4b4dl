#!/bin/bash
# B4a 两阶段链（2026-09-06）：B3 配方 + TG 高帧段过采样数据（唯一变量）。
#   阶段1: stage2 混合重训（run_stage2_full_seqv3_mixed_b4a.sh，150k oversampled × 3 epochs）
#          —— 针对高帧段（GT start>=25 占 ~15% vs B3 预测 4%）欠覆盖与 −3.9 帧偏早
#   阶段2: 同口径评测（--whole_scene --per_sequence --answer_frames，与 B3 完全一致）
# 显存门控窗口 72h（CoRViD/xmuda 共用 GPU）；断点续训（sort -V）兜底。
# tmux 会话 b4apipeline 运行。
set -u
START_STAGE=${1:-1}
case "$START_STAGE" in 1|2) ;; *) echo "用法: bash $0 [1|2]"; exit 1 ;; esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${B4DL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WQLC_PREFIX="${B4DL_ENV_PREFIX:-$(dirname "$PROJECT_ROOT")/.conda-stuff/envs/wqlc}"
[ -x "$WQLC_PREFIX/bin/python" ] || { echo "错误: wqlc 环境不存在: $WQLC_PREFIX" >&2; exit 1; }
export PATH="$WQLC_PREFIX/bin:$PATH"
cd "$PROJECT_ROOT/mllm" || exit 1
export HF_HUB_OFFLINE=1 HF_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline PYTHONUNBUFFERED=1

"$WQLC_PREFIX/bin/python" scripts/preflight_b3.py \
    --project-root "$PROJECT_ROOT" \
    --train-data stage2_full_train_seqv3_meta2_oversampled_150k.json \
    --expected-train-count 150222 \
    --run-label B4a || exit 1

if [ "$START_STAGE" -le 1 ]; then
echo "===== 阶段1: mixed-b4a 重训（B3 配方 + 高帧段过采样）($(date '+%F %T')) ====="
OUT=checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b4a
# 等显存（28GB 门控，72h 上限：432 次 × 10 分钟）
GATE_OK=0
for i in $(seq 1 432); do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ -n "$FREE_MB" ] && [ "$FREE_MB" -ge 28000 ]; then GATE_OK=1; break; fi
    echo "[$(date '+%F %T')] 显存不足 (${FREE_MB}MB < 28GB)，10 分钟后重试... (第 ${i} 次)"
    sleep 600
done
[ $GATE_OK -ne 1 ] && { echo "阶段1 显存门控 72h 未放行，链停止"; exit 1; }
MIXED_OK=0
for attempt in 1 2 3; do
    bash scripts/run_stage2_full_seqv3_mixed_b4a.sh > /dev/null 2>&1
    if [ -f "$OUT/adapter_model.safetensors" ] && [ -f "$OUT/trainer_state.json" ] \
       && "$WQLC_PREFIX/bin/python" -c "import json,sys; d=json.load(open('$OUT/trainer_state.json')); sys.exit(0 if d.get('epoch',0) >= 2.99 else 1)"; then
        MIXED_OK=1
        echo "阶段1 完成（尝试 #$attempt）"
        break
    fi
    echo "阶段1 尝试 #$attempt 失败，10 分钟后断点续训重试..."
    sleep 600
done
[ $MIXED_OK -ne 1 ] && { echo "阶段1 三次尝试均失败，链停止"; exit 1; }
fi

echo "===== 阶段2: 同口径评测 b4a ($(date '+%F %T')) ====="
EVAL_OUT=./eval_results/stage2_full_seqv3_mixed_b4a
mkdir -p "$EVAL_OUT"
EVAL_OK=0
for attempt in 1 2 3 4 5; do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ -n "$FREE_MB" ] && [ "$FREE_MB" -lt 14336 ]; then
        echo "[$(date '+%F %T')] 显存不足 (${FREE_MB}MB < 14GB)，10 分钟后重试..."
        sleep 600
        continue
    fi
    timeout 18000 "$WQLC_PREFIX/bin/python" -u evaluation/test_b4dl.py \
        --model_base ./base_model/vicuna-v1-5-7b \
        --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
        --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b4a \
        --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
        --test_data ./b4dl_dataset/test_qa.json \
        --ego_meta ./b4dl_dataset/ego_metadata.json \
        --frame_motion ./b4dl_dataset/ego_frame_motion.json \
        --whole_scene --per_sequence --answer_frames \
        --output "$EVAL_OUT/predictions.json" \
        --metrics_output "$EVAL_OUT/metrics.json" \
        >> "$EVAL_OUT/eval_log.txt" 2>&1
    rc=$?
    echo "阶段2 尝试 #$attempt rc=$rc"
    [ $rc -eq 0 ] && { EVAL_OK=1; break; }
    sleep 600
done
[ $EVAL_OK -ne 1 ] && { echo "阶段2 五次尝试均失败，链停止"; exit 1; }

echo "===== B4a 链完成 ($(date '+%F %T')) ====="
echo "B4a 评测结果（对照 B3 mIoU 0.3467/acc 0.7526；论文 mIoU 0.311/acc 0.762；显著阈值 ΔmIoU>0.013）："
"$WQLC_PREFIX/bin/python" -c "
import json
m = json.load(open('$EVAL_OUT/metrics.json'))
f = m['final_scores']
print(f\"accuracy {f['accuracy']:.4f} (B3 0.7526, paper 0.762)\")
print(f\"mIoU      {f['miou']:.4f} (B3 0.3467, Δvs B3 {f['miou']-0.3467:+.4f}, paper 0.311)\")
for k in ('bleu4','meteor','rouge_l','bertscore'):
    print(f'{k:9s} {f[k]:.4f}')
pt = m.get('per_task', {})
print('per_task:', {k: (round(v.get('miou', v.get('accuracy', v.get('bleu4', 0))), 4) if isinstance(v, dict) else v) for k, v in pt.items()})
" 2>/dev/null || cat "$EVAL_OUT/metrics.json"
