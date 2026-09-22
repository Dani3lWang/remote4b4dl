#!/bin/bash
# 过夜串行链（2026-09-22 夜）：Phase 3 口径重跑 -> Phase 1b 无泄漏重训 -> 训后三评。
# 各阶段独立记 rc，前一阶段失败不阻断后续（GPU 时间已排期，宁可拿到部分结果）。
set -u
cd /root/autodl-tmp/mmb4dl/mllm
L=./training_logs/chain_overnight
mkdir -p "$L" ./checkpoints/_tfval_tmp
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
CKPT=./checkpoints/reasonseg-internal679/checkpoint-best
V=./reasonseg_data_trainval

echo "[$(date '+%F %T')] === STAGE 1/5: B4a whole-scene 六任务评测 ==="
bash scripts/run_b4a_wholescene_eval.sh > "$L/stage1_b4a_wholescene.log" 2>&1
echo "[$(date '+%F %T')] stage1 rc=$?"

echo "[$(date '+%F %T')] === STAGE 2/5: 无泄漏协议重训 (679 场景/7436 条, 20 epoch) ==="
bash scripts/reasonseg_experiments/run_reasonseg_internal679.sh > "$L/stage2_train_internal679.log" 2>&1
echo "[$(date '+%F %T')] stage2 rc=$?"

if [ ! -f "$CKPT/reasonseg_modules.pt" ]; then
    echo "[$(date '+%F %T')] 错误: 未找到 $CKPT，跳过训后评测"
    exit 1
fi

tfval() { # $1=label $2=val-manifest
    echo "[$(date '+%F %T')] === TF 复算 $1 ==="
    $E/bin/accelerate launch --mixed_precision bf16 scripts/train_reasonseg.py \
        --model-base ./base_model/vicuna-v1-5-7b \
        --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
        --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
        --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
        --validation-manifest "$2" \
        --dataroot /root/autodl-tmp/nuScenes \
        --output-dir ./checkpoints/_tfval_tmp \
        --eval-checkpoint "$CKPT" \
        --validate-only --validation-samples 0 --validate-dtype-variants as_is \
        --mixed-precision bf16 > "$L/$1.log" 2>&1
    echo "[$(date '+%F %T')] $1 rc=$?"
}

echo "[$(date '+%F %T')] === STAGE 3/5: TF 全量 internal val (20424 条) ==="
tfval stage3_tfval_internal_full "$V/reasonseg_val_internal.jsonl"

echo "[$(date '+%F %T')] === STAGE 4/5: TF val_thin (2248 条, 官方 val = 最终测试口径) ==="
tfval stage4_tfval_valthin "$V/reasonseg_val_thin.jsonl"

echo "[$(date '+%F %T')] === STAGE 5/5: 自由生成 300 条 + 失败分解 ==="
timeout 10800 $E/bin/python -u evaluation/evaluate_reasonseg.py \
    --model-base ./base_model/vicuna-v1-5-7b \
    --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
    --seg-checkpoint "$CKPT" \
    --manifest "$V/reasonseg_val_thin.jsonl" \
    --dataroot /root/autodl-tmp/nuScenes \
    --output-dir ./eval_results/reasonseg-internal679_encfp32 \
    --threshold 0.5 --max-samples 300 --dtype fp16 --encoder-dtype fp32 \
    > "$L/stage5_freegen.log" 2>&1
echo "[$(date '+%F %T')] stage5 rc=$?"
$E/bin/python scripts/reasonseg_experiments/analyze_reasonseg_failures.py \
    --eval-dir ./eval_results/reasonseg-internal679_encfp32 \
    --config "$CKPT/reasonseg_config.json" \
    --output ./eval_results/reasonseg-internal679_encfp32/failure_analysis.json \
    > "$L/stage5_failure_analysis.log" 2>&1
echo "[$(date '+%F %T')] failure-analysis rc=$?"
echo "=== CHAIN ALL DONE ==="
