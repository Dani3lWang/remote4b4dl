#!/bin/bash
# Phase 2 过夜串行链：单元测试 → 冒烟 → 2.1 推理侧两臂 → 2.2 损失再平衡三臂 → 判据裁定。
#
# 只用 set -u，不用 set -e：任一阶段失败都要留下 rc 并让后续阶段继续，避免像
# 09-22 那样因为一个 nltk 下载把整条 5 小时链挂在 timeout 里。
# 冒烟放在最前面：它同时验证损失改造、config 新字段、loader 放宽后的加载路径和
# checkpoint 落盘，任何一处崩了都只损失几分钟而不是 13 小时。
L=/root/autodl-tmp/mmb4dl/mllm/training_logs/phase2
mkdir -p "$L"
DRIVER="$L/chain_driver.log"
cd /root/autodl-tmp/mmb4dl/mllm || exit 1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg

log () { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER"; }

log "=== STAGE 0/4: 掩码损失单元测试（CPU，该环境无 pytest，走 unittest）==="
$E/bin/python -m unittest tests.test_reasonseg.MaskLossTests tests.test_reasonseg.DecoderTests \
  > "$L/stage0_unittest.log" 2>&1
STAGE0_RC=$?
log "stage0 rc=$STAGE0_RC"
tail -5 "$L/stage0_unittest.log" | tee -a "$DRIVER"
if [ $STAGE0_RC -ne 0 ]; then
  log "!!! 单元测试失败，终止整条链：损失改造或解码器回归未通过。"
  exit 1
fi

log "=== STAGE 0b/4: 4-epoch 冒烟（60 条训练 / 32 条验证，验证全链路）==="
head -60 ./reasonseg_data_trainval/reasonseg_train_internal679x11.jsonl > "$L/_smoke_train.jsonl"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
$E/bin/accelerate launch --mixed_precision bf16 scripts/train_reasonseg.py \
  --model-base ./base_model/vicuna-v1-5-7b \
  --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
  --train-manifest "$L/_smoke_train.jsonl" \
  --validation-manifest ./reasonseg_data_trainval/reasonseg_val_internal_es.jsonl \
  --dataroot /root/autodl-tmp/nuScenes \
  --output-dir ./checkpoints/_smoke_loss_rebalance \
  --spatial-checkpoint ./checkpoints/reasonseg-spatial-tv/spatial-best \
  --bce-mode balanced --region-loss tversky --tversky-alpha 0.3 --tversky-beta 0.7 \
  --num-train-epochs 1 --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 --save-steps 1000 --keep-last 1 --no-resume-state \
  --validation-samples 32 --mixed-precision bf16 > "$L/stage0b_smoke.log" 2>&1
SMOKE_RC=$?
log "stage0b rc=$SMOKE_RC"
if [ $SMOKE_RC -ne 0 ]; then
  log "!!! 冒烟失败，终止整条链（不浪费 13 GPU-小时）。尾部日志："
  tr '\r' '\n' < "$L/stage0b_smoke.log" | tail -25 | tee -a "$DRIVER"
  exit 1
fi
grep -o '{"step":[^}]*}' "$L/stage0b_smoke.log" | tail -3 | tee -a "$DRIVER"
rm -rf ./checkpoints/_smoke_loss_rebalance

log "=== STAGE 1/4: Phase 2.1 推理侧两臂（免训练）==="
bash scripts/reasonseg_experiments/run_grounding_phase21.sh > "$L/stage1_phase21.log" 2>&1
log "stage1 rc=$?"

log "=== STAGE 2/4: Phase 2.2 损失再平衡三臂（各 4 epoch）+ 逐臂接地诊断 ==="
bash scripts/reasonseg_experiments/run_loss_rebalance_arms.sh > "$L/stage2_phase22.log" 2>&1
log "stage2 rc=$?"

log "=== STAGE 3/4: 预注册判据自动裁定 ==="
$E/bin/python scripts/reasonseg_experiments/summarize_phase22.py --epochs 4 > "$L/stage3_summary.log" 2>&1
log "stage3 rc=$?"
cat "$L/stage3_summary.log" | tee -a "$DRIVER"

log "=== CHAIN ALL DONE ==="
