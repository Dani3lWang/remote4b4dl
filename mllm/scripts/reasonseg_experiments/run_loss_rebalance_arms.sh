#!/bin/bash
# Phase 2.2：损失再平衡三臂，各 4 epoch，配方与 internal679 逐字一致（唯一变量 = 掩码损失）。
#
# 机制（Phase 2.0 实测）：掩码正例占帧内点 3e-4，逐点平均的 BCE 把正例梯度稀释约
# 3000 倍。在 20 个同类候选之间无法分辨时，"全部不触发"与"在所有候选上对冲"的总损失
# 只差 0.004，梯度因此没有动力去分辨实例 —— 表现为 AUC 随难度不降反升
# （0.818→0.908），而 mean_prob_positive 塌 12 倍（0.208→0.018）。
# 训练日志佐证：mask_class 第 6 轮归零、text_loss 到 1e-5，唯独 mask_seg 卡在 0.57，
# 且 p75=0.941 正好是 11 点物体"全空预测"的 dice 地板 1-1/12=0.917。
#
# 三臂构成 2x2 析因的三格（缺 plain+dice = 已有的 internal679 对照）：
#   A1 balanced + dice     隔离 BCE 归一方式的作用
#   A2 plain    + tversky  隔离区域损失形状的作用
#   A3 balanced + tversky  两者叠加
#
# 对照免跑：直接读 internal679 已有曲线的 epoch 0-4（同初始化、同数据、同 seed
# 20260917），ep4 TF mean IoU = 0.1080。
#
# 判据（epoch 4，TF on reasonseg_val_internal_es 2248 条 + 复跑接地诊断）：
#   PASS    AUC >= 0.93 且 mean_prob_positive >= 0.35 → 该臂扩到 20 epoch 作新基线
#   PARTIAL AUC in [0.88,0.93) 或 prob in [0.25,0.35)，且 TF mIoU >= 0.115 → 只扩最优一臂
#   FAIL    三臂 AUC <= 0.88 且 prob <= 0.20 → 损失再平衡判死，转 Phase 2.3 oracle-query 探针
#   护栏    predicted_size_median > 3x target_size_median，或 AUC < 0.80（换来的是无差别触发）
#           → 记为假阳性，不采纳
set -uo pipefail
cd /root/autodl-tmp/mmb4dl/mllm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
EPOCHS=${EPOCHS:-4}
L=./training_logs/phase2
mkdir -p "$L"

train_arm () {
  local name=$1; shift
  echo "[$(date '+%F %T')] === TRAIN ARM $name ==="
  $E/bin/accelerate launch --mixed_precision bf16 scripts/train_reasonseg.py \
    --model-base ./base_model/vicuna-v1-5-7b \
    --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
    --train-manifest ./reasonseg_data_trainval/reasonseg_train_internal679x11.jsonl \
    --validation-manifest ./reasonseg_data_trainval/reasonseg_val_internal_es.jsonl \
    --dataroot /root/autodl-tmp/nuScenes \
    --output-dir "./checkpoints/$name" \
    --spatial-checkpoint ./checkpoints/reasonseg-spatial-tv/spatial-best \
    --num-train-epochs "$EPOCHS" --per-device-train-batch-size 1 --gradient-accumulation-steps 16 \
    --learning-rate 1e-4 --early-stopping-patience 12 --early-stopping-min-delta 1e-4 \
    --save-steps 200 --keep-last 2 --no-resume-state --validation-samples 0 \
    --mixed-precision bf16 "$@" > "$L/$name.log" 2>&1
  echo "[$(date '+%F %T')] train $name rc=$?"
}

diagnose_arm () {
  local name=$1
  echo "[$(date '+%F %T')] === DIAGNOSE ARM $name ==="
  $E/bin/python scripts/reasonseg_experiments/diagnose_reasonseg_grounding.py \
    --model-base ./base_model/vicuna-v1-5-7b \
    --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
    --seg-checkpoint "./checkpoints/$name/checkpoint-best" \
    --manifest ./reasonseg_data_trainval/reasonseg_val_thin.jsonl \
    --dataroot /root/autodl-tmp/nuScenes \
    --output "./eval_results/_grounding_$name/report.json" \
    --max-samples 1000 --dtype fp16 --encoder-dtype fp32 > "$L/$name.grounding.log" 2>&1
  echo "[$(date '+%F %T')] diagnose $name rc=$?"
}

A1=reasonseg-lossA1-bal-dice
A2=reasonseg-lossA2-plain-tversky
A3=reasonseg-lossA3-bal-tversky
TVERSKY="--region-loss tversky --tversky-alpha 0.3 --tversky-beta 0.7"

train_arm "$A1" --bce-mode balanced --region-loss dice
train_arm "$A2" --bce-mode plain $TVERSKY
train_arm "$A3" --bce-mode balanced $TVERSKY
diagnose_arm "$A1"
diagnose_arm "$A2"
diagnose_arm "$A3"
echo "=== PHASE2.2 ALL DONE ==="
