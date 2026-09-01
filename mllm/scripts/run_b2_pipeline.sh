#!/bin/bash
# B2 两阶段链（2026-09-01）：整场景输入（--whole_scene，对齐官方 B4DL）训练 + 同口径评测。
#   阶段1: stage2 混合重训（run_stage2_full_seqv3_mixed_b2.sh，独立 output_dir，B0/B1 不动）
#          输入=整场景 39/40/41 帧（B1 为 QA 序列切片）——唯一变量，超参与 B1 逐项一致
#   阶段2: 同口径评测（test_b4dl.py --whole_scene --per_sequence --answer_frames，
#          视频输入恒整场景、meta 仍按 QA/GT 帧锚定）
# 任一阶段硬失败即停链（cron 报告，人工决策），不做盲自动重启。
# tmux 会话 b2pipeline 运行。
# 用法: bash run_b2_pipeline.sh [起始阶段]，默认 1；断点恢复示例: bash run_b2_pipeline.sh 2
set -u
START_STAGE=${1:-1}
case "$START_STAGE" in 1|2) ;; *) echo "用法: bash $0 [1|2]，默认 1 全链"; exit 1 ;; esac
cd /root/autodl-tmp/wql/mmb4dl/mllm
eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)"
conda activate wqlc
export HF_HUB_OFFLINE=1 HF_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline PYTHONUNBUFFERED=1

if [ "$START_STAGE" -le 1 ]; then
echo "===== 阶段1: mixed-b2 混合重训（整场景输入）($(date '+%F %T')) ====="
OUT=checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b2
MIXED_OK=0
for attempt in 1 2 3; do
    bash scripts/run_stage2_full_seqv3_mixed_b2.sh > /dev/null 2>&1
    # 判据与 b1 pipeline 修复版一致：以 trainer_state 的 epoch 跑满（≥2.99）为准，
    # 不依赖日志尾部 "End:"（旧判据 bug 根因：tee 管道后 echo 进 stdout 被 /dev/null 吞掉）。
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

echo "===== 阶段2: 同口径评测 b2 ($(date '+%F %T')) ====="
EVAL_OUT=./eval_results/stage2_full_seqv3_mixed_b2
mkdir -p "$EVAL_OUT"
EVAL_OK=0
for attempt in 1 2 3 4 5; do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ -n "$FREE_MB" ] && [ "$FREE_MB" -lt 14336 ]; then
        echo "[$(date '+%F %T')] 显存不足 (${FREE_MB}MB < 14GB)，10 分钟后重试..."
        sleep 600
        continue
    fi
    timeout 18000 python -u evaluation/test_b4dl.py \
        --model_base ./base_model/vicuna-v1-5-7b \
        --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
        --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b2 \
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

echo "===== B2 链完成 ($(date '+%F %T')) ====="
echo "B2 评测结果（对照论文 mIoU 0.311 / acc 0.762；B1 为 0.2653 / 0.7787 错位口径）："
python -c "
import json
m = json.load(open('$EVAL_OUT/metrics.json'))
f = m['final_scores']
print(f\"accuracy {f['accuracy']:.4f} (paper 0.762, B1 0.7787)\")
print(f\"mIoU      {f['miou']:.4f} (paper 0.311, B1 0.2653, Δvs paper {f['miou']-0.311:+.4f})\")
for k in ('bleu4','meteor','rouge_l','bertscore'):
    print(f'{k:9s} {f[k]:.4f}')
" 2>/dev/null || cat "$EVAL_OUT/metrics.json"
