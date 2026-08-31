#!/bin/bash
# B1 全流水线链式驱动（2026-08-30，按 docs/learn docs/B4DL_待修改清单_20260829.md 执行主线）
#   阶段1: 全量重提特征（退火定稿编码器 ckpt_anneal/lidarclip_mm/last.ckpt，eval 模式）
#          1a stage1 sample_token 特征（28,130） 1b stage2 per-scene 特征（850）
#   阶段2: stage1 162K projector 重训（stage1.sh，数据 stage1_train.json 161,629 条）
#   阶段3: mixed-b1 混合重训（独立 output_dir，B0 不动；失败断点续训最多 3 次）
#   阶段4: 同口径评测 → eval_results/stage2_full_seqv3_mixed_b1/（显存门控 + HF offline）
# 任一阶段硬失败即停链（cron 报告，人工决策），不做盲自动重启。
# tmux 会话 b1pipeline 运行。
# 用法: bash run_b1_pipeline.sh [起始阶段]，默认 1 全链；断点恢复示例: bash run_b1_pipeline.sh 2（特征已就绪，从阶段2 起）
set -u
START_STAGE=${1:-1}
case "$START_STAGE" in 1|2|3|4) ;; *) echo "用法: bash $0 [1|2|3|4]，默认 1 全链"; exit 1 ;; esac
cd /root/autodl-tmp/wql/mmb4dl/mllm
eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)"
conda activate wqlc
export HF_HUB_OFFLINE=1 HF_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline PYTHONUNBUFFERED=1

ENC=/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip
CKPT=$ENC/ckpt_anneal/lidarclip_mm/last.ckpt
[ -f "$CKPT" ] || { echo "错误: 编码器 $CKPT 不存在"; exit 1; }

if [ "$START_STAGE" -le 1 ]; then
echo "===== 阶段1a: stage1 sample_token 特征重提 ($(date '+%F %T')) ====="
cd "$ENC"
python extract_pc_features_sample_token.py \
    --checkpoint "$CKPT" \
    --scene-metadata /root/autodl-tmp/wql/mmb4dl/dataset/nuScenes-B4DL/metadata/scene_metadata.json \
    --sample-json /root/autodl-tmp/Datasets/nuScenes/v1.0-trainval/sample.json \
    --data-path /root/autodl-tmp/Datasets/nuScenes \
    --save-dir ./b4dl/stage1_features_sample \
    > logs/extract_stage1_sample_token_b1.log 2>&1
rc=$?
echo "阶段1a rc=$rc"
[ $rc -ne 0 ] && { echo "阶段1a 失败，链停止（见 $ENC/logs/extract_stage1_sample_token_b1.log）"; exit 1; }
n1=$(ls b4dl/stage1_features_sample | wc -l)
echo "stage1 特征数: $n1（预期 28,130）"
[ "$n1" -lt 28000 ] && { echo "阶段1a 特征数异常，链停止"; exit 1; }

echo "===== 阶段1b: stage2 per-scene 特征重提 ($(date '+%F %T')) ====="
python extract_pc_features.py \
    --checkpoint "$CKPT" \
    --dataset-name with_path \
    --data-path /root/autodl-tmp/Datasets/nuScenes \
    --scene-json-path ./annotations/scene_metadata.json \
    --frame-json-path ./annotations/sequence_metadata.json \
    --stage1-save-dir ./b4dl/stage1_features \
    --stage2-save-dir ./b4dl/stage2_features \
    > logs/extract_stage2_b1.log 2>&1
rc=$?
echo "阶段1b rc=$rc"
[ $rc -ne 0 ] && { echo "阶段1b 失败，链停止（见 $ENC/logs/extract_stage2_b1.log）"; exit 1; }
n2=$(ls b4dl/stage2_features | wc -l)
echo "stage2 特征数: $n2（预期 850）"
[ "$n2" -ne 850 ] && { echo "阶段1b 特征数异常，链停止"; exit 1; }
fi

if [ "$START_STAGE" -le 2 ]; then
cd /root/autodl-tmp/wql/mmb4dl/mllm
echo "===== 阶段2: stage1 162K projector 重训 ($(date '+%F %T')) ====="
GATE_OK=0
for i in $(seq 1 12); do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ -n "$FREE_MB" ] && [ "$FREE_MB" -ge 28000 ]; then GATE_OK=1; break; fi
    echo "[$(date '+%F %T')] 显存不足 (${FREE_MB}MB < 28GB)，10 分钟后重试..."
    sleep 600
done
[ $GATE_OK -ne 1 ] && { echo "阶段2 显存门控 2 小时未放行，链停止"; exit 1; }
bash scripts/stage1.sh > training_logs/stage1_162k_b1_$(date +%Y%m%d_%H%M%S).log 2>&1
rc=$?
echo "阶段2 rc=$rc"
[ $rc -ne 0 ] && { echo "阶段2 失败，链停止"; exit 1; }
[ -f checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin ] || { echo "阶段2 未产出 projector，链停止"; exit 1; }
fi

if [ "$START_STAGE" -le 3 ]; then
echo "===== 阶段3: mixed-b1 混合重训 ($(date '+%F %T')) ====="
OUT=checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b1
MIXED_OK=0
for attempt in 1 2 3; do
    bash scripts/run_stage2_full_seqv3_mixed_b1.sh > /dev/null 2>&1
    # 判据修复：以 trainer_state 的 epoch 跑满（≥2.99，即 3 epochs 完成）为准。
    # 旧判据 tail 日志 grep "End:" 不可靠——脚本的 echo "End:" 在 tee 管道后输出到
    # 脚本 stdout，被上面的 > /dev/null 吞掉，日志中永远不会出现 "End:"（2026-08-31 误报三次失败根因）。
    if [ -f "$OUT/adapter_model.safetensors" ] && [ -f "$OUT/trainer_state.json" ] \
       && python3 -c "import json,sys; d=json.load(open('$OUT/trainer_state.json')); sys.exit(0 if d.get('epoch',0) >= 2.99 else 1)"; then
        MIXED_OK=1
        echo "阶段3 完成（尝试 #$attempt）"
        break
    fi
    echo "阶段3 尝试 #$attempt 失败，10 分钟后断点续训重试..."
    sleep 600
done
[ $MIXED_OK -ne 1 ] && { echo "阶段3 三次尝试均失败，链停止"; exit 1; }
fi

echo "===== 阶段4: 同口径评测 b1 ($(date '+%F %T')) ====="
EVAL_OUT=./eval_results/stage2_full_seqv3_mixed_b1
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
        --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b1 \
        --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
        --test_data ./b4dl_dataset/test_qa.json \
        --ego_meta ./b4dl_dataset/ego_metadata.json \
        --frame_motion ./b4dl_dataset/ego_frame_motion.json \
        --sequence_metadata ../encoders/lidarclip/annotations/sequence_metadata.json \
        --per_sequence --answer_frames \
        --output "$EVAL_OUT/predictions.json" \
        --metrics_output "$EVAL_OUT/metrics.json" \
        >> "$EVAL_OUT/eval_log.txt" 2>&1
    rc=$?
    echo "阶段4 尝试 #$attempt rc=$rc"
    [ $rc -eq 0 ] && { EVAL_OK=1; break; }
    sleep 600
done
[ $EVAL_OK -ne 1 ] && { echo "阶段4 五次尝试均失败，链停止"; exit 1; }

echo "===== B1 链完成 ($(date '+%F %T')) ====="
echo "B1 评测结果（对照 B0：acc 0.7629 / mIoU 0.2696，ΔmIoU > +0.013 才显著）："
python -c "
import json
m = json.load(open('$EVAL_OUT/metrics.json'))
f = m['final_scores']
print(f\"accuracy {f['accuracy']:.4f} (B0 0.7629)\")
print(f\"mIoU      {f['miou']:.4f} (B0 0.2696, Δ{f['miou']-0.2696:+.4f})\")
for k in ('bleu4','meteor','rouge_l','bertscore'):
    print(f'{k:9s} {f[k]:.4f}')
" 2>/dev/null || cat "$EVAL_OUT/metrics.json"
