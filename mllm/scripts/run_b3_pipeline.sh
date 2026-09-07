#!/bin/bash
# B3 两阶段链（2026-09-02）：整场景输入 + 修复版 metatoken（relative-to-previous，meta2 数据）。
#   阶段1: stage2 混合重训（run_stage2_full_seqv3_mixed_b3.sh，148k meta2 × 3 epochs）
#          —— 论文完整配置（整场景视觉 + 论文语义 meta）的最忠实复现
#   阶段2: 同口径评测（--whole_scene --per_sequence --answer_frames，meta 走修复版渲染）
# 显存门控窗口 72h（CoRViD/xmuda 共用 GPU）；断点续训（sort -V）兜底。
# tmux 会话 b3pipeline 运行。
set -u
START_STAGE=${1:-1}
case "$START_STAGE" in 1|2) ;; *) echo "用法: bash $0 [1|2]"; exit 1 ;; esac
cd /root/autodl-tmp/wql/mmb4dl/mllm
eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)"
conda activate wqlc
export HF_HUB_OFFLINE=1 HF_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline PYTHONUNBUFFERED=1

if [ "$START_STAGE" -le 1 ]; then
echo "===== 阶段1: mixed-b3 重训（整场景 + meta2）($(date '+%F %T')) ====="
OUT=checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3
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
    bash scripts/run_stage2_full_seqv3_mixed_b3.sh > /dev/null 2>&1
    if [ -f "$OUT/adapter_model.safetensors" ] && [ -f "$OUT/trainer_state.json" ] \
       && python3 -c "import json,sys; d=json.load(open('$OUT/trainer_state.json')); sys.exit(0 if d.get('epoch',0) >= 2.99 else 1)"; then
        MIXED_OK=1
        echo "阶段1 完成（尝试 #$attempt）"
        break
    fi
    echo "阶段1 尝试 #$attempt 失败，10 分钟后断点续训重试..."
    sleep 600
done
[ $MIXED_OK -ne 1 ] && { echo "阶段1 三次尝试均失败，链停止"; exit 1; }
fi

echo "===== 阶段2: 同口径评测 b3 ($(date '+%F %T')) ====="
EVAL_OUT=./eval_results/stage2_full_seqv3_mixed_b3
mkdir -p "$EVAL_OUT"
EVAL_OK=0
for attempt in 1 2 3 4 5; do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ -n "$FREE_MB" ] && [ "$FREE_MB" -lt 14336 ]; then
        echo "[$(date '+%F %T')] 显存不足 (${FREE_MB}MB < 14GB)，10 分钟后重试..."
        sleep 600
        continue
    fi
    timeout 18000 /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -u evaluation/test_b4dl.py \
        --model_base ./base_model/vicuna-v1-5-7b \
        --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
        --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
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

echo "===== B3 链完成 ($(date '+%F %T')) ====="
echo "B3 评测结果（对照论文 mIoU 0.311 / acc 0.762；B2 0.1992/0.7649）："
python3 -c "
import json
m = json.load(open('$EVAL_OUT/metrics.json'))
f = m['final_scores']
print(f\"accuracy {f['accuracy']:.4f} (paper 0.762, B2 0.7649)\")
print(f\"mIoU      {f['miou']:.4f} (paper 0.311, B2 0.1992, Δvs paper {f['miou']-0.311:+.4f})\")
for k in ('bleu4','meteor','rouge_l','bertscore'):
    print(f'{k:9s} {f[k]:.4f}')
" 2>/dev/null || cat "$EVAL_OUT/metrics.json"
