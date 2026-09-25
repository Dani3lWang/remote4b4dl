#!/bin/bash
# Phase 2.3：把"query→实例接地"这个瓶颈一分为二。
#
# 背景：Phase 2.2 的终局是七条杠杆全判死。A2（plain+Tversky）把尺寸塌缩治好了
# （3 点 → 11 点 = 1.00× GT），空掩码从 140 降到 99，但自由生成的 headline
# instance recall@0.5 与对照**逐位相同（都是 24/330）**；失败分解显示那 41 个
# 空掩码只转化成 29 个 near-miss + 15 个 right_class_wrong_place，而后者升为第一大
# 失败模式 —— 模型知道类别、知道大致区域，就是不知道是哪一个实例。
#
# 两臂按"信息量"排序，先跑更能定方向的那个：
#
#   B0 oracle-query 探针（先跑，约 2.6 h）
#      绕过 7B LM，用 GT 中心（按 point_cloud_range 归一化）+ one-hot 类别构造
#      query，只训 OracleQueryNet + QueryMaskDecoder，编码器冻结。其余全部与真实
#      头同构（含逐物体 memory 展开）。它量的是**天花板**：
#        recall@0.5 >= 0.50 -> 点特征与解码器够用，瓶颈在 LM query 通路
#                              （投向 scene query 容量 / 实例对比监督 / CoT 查询）
#        recall@0.5 <= 0.20 -> 编码器或点特征是瓶颈（投向 voxel_size / 多尺度 / 分辨率）
#        介于两者之间       -> 两侧都有份，单改任何一侧都补不上
#      先跑它的理由：若结论是"特征不够"，B1 根本不用跑，直接省 4 小时。
#
#   B1 切除 LOC 先验 + Tversky（约 3.9 h）
#      Phase 2.1 实测先验置零后 AUC 0.858→0.917、prob_pos 升 62%，但尺寸从 3 点
#      跳到 63 点（GT 11）；A2 恰好证明了 Tversky 能把尺寸校准到 1.00×。两者互补，
#      合起来是第一次有机会让 hit 数动起来。
#      判据（对照 A2@4ep：AUC 0.9167 / 尺寸 1.27× / internal_es ep3 mIoU 0.1255）：
#        PASS AUC >= 0.93 且 尺寸比 in [0.7,1.5] 且 ep3 mIoU > 0.1255
#             且诊断 recall@0.5 > 0.0861（对照值，即 hit 数第一次真的动了）
#        FAIL AUC < 0.90 或 尺寸比超出 [0.5,2.0] -> 先验切除不与 Tversky 复合
#      注意 --no-loc-prior 是架构字段：validate-only 复算时也必须带，否则 loader 拒绝加载。
#
# 冒烟前置：探针是全新代码、从未运行过，B1 的 --no-loc-prior 通路也没跑过。
# 两个冒烟共约 6 分钟，任一失败即 exit 1，不浪费 6.5 GPU-小时。
set -uo pipefail
cd /root/autodl-tmp/mmb4dl/mllm || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
V=./reasonseg_data_trainval
L=./training_logs/phase23
mkdir -p "$L"
DRIVER="$L/driver.log"
SPATIAL=./checkpoints/reasonseg-spatial-tv/spatial-best
TVERSKY="--region-loss tversky --tversky-alpha 0.3 --tversky-beta 0.7"
B1=reasonseg-lossB1-noprior-tversky

log () { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER"; }

log "=== S0/5 冒烟：oracle 探针（40 条训练 / 20 条评测）==="
$E/bin/python scripts/reasonseg_experiments/probe_oracle_query.py \
  --spatial-checkpoint "$SPATIAL" \
  --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
  --dev-manifest "$V/reasonseg_val_internal_es.jsonl" \
  --test-manifest "$V/reasonseg_val_thin.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output "$L/_smoke_probe.json" \
  --epochs 1 --max-train-records 40 --eval-records 20 --log-every 10 \
  > "$L/s0_smoke_probe.log" 2>&1
S0=$?
log "S0 rc=$S0"
if [ $S0 -ne 0 ]; then
  log "!!! 探针冒烟失败，终止全链。尾部："
  tr '\r' '\n' < "$L/s0_smoke_probe.log" | tail -25 | tee -a "$DRIVER"
  exit 1
fi
tr '\r' '\n' < "$L/s0_smoke_probe.log" | grep -E "trainable params|recall_at_0.5" | tee -a "$DRIVER"

log "=== S0b/5 冒烟：B1 的 --no-loc-prior 通路（60 条 / 1 epoch）==="
head -60 "$V/reasonseg_train_internal679x11.jsonl" > "$L/_smoke_train.jsonl"
$E/bin/accelerate launch --mixed_precision bf16 scripts/train_reasonseg.py \
  --model-base ./base_model/vicuna-v1-5-7b \
  --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
  --train-manifest "$L/_smoke_train.jsonl" \
  --validation-manifest "$V/reasonseg_val_internal_es.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output-dir ./checkpoints/_smoke_b1 \
  --spatial-checkpoint "$SPATIAL" \
  --no-loc-prior --bce-mode plain $TVERSKY \
  --num-train-epochs 1 --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 --save-steps 1000 --keep-last 1 --no-resume-state \
  --validation-samples 32 --mixed-precision bf16 > "$L/s0b_smoke_b1.log" 2>&1
S0B=$?
log "S0b rc=$S0B"
if [ $S0B -ne 0 ]; then
  log "!!! B1 冒烟失败，终止全链。尾部："
  tr '\r' '\n' < "$L/s0b_smoke_b1.log" | tail -25 | tee -a "$DRIVER"
  exit 1
fi
tr '\r' '\n' < "$L/s0b_smoke_b1.log" | grep -oE '\{"step":[^}]*\}' | tail -2 | tee -a "$DRIVER"
# 顺带验证 --no-loc-prior 的档能被同旗标的 validate-only 重新加载（架构字段比对）
$E/bin/accelerate launch --mixed_precision bf16 scripts/train_reasonseg.py \
  --model-base ./base_model/vicuna-v1-5-7b \
  --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
  --train-manifest "$L/_smoke_train.jsonl" \
  --validation-manifest "$V/reasonseg_val_internal_es.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output-dir ./checkpoints/_smoke_b1 \
  --eval-checkpoint ./checkpoints/_smoke_b1/checkpoint-best \
  --no-loc-prior --bce-mode plain $TVERSKY \
  --validate-only --validation-samples 32 --validate-dtype-variants as_is \
  --mixed-precision bf16 > "$L/s0c_reload_b1.log" 2>&1
S0C=$?
log "S0c（重载校验）rc=$S0C"
if [ $S0C -ne 0 ]; then
  log "!!! --no-loc-prior 的档无法被 validate-only 重载，终止全链。尾部："
  tr '\r' '\n' < "$L/s0c_reload_b1.log" | tail -20 | tee -a "$DRIVER"
  exit 1
fi
rm -rf ./checkpoints/_smoke_b1

log "=== S1/5 B0 oracle-query 探针（4 epoch，约 2.6 h）==="
$E/bin/python scripts/reasonseg_experiments/probe_oracle_query.py \
  --spatial-checkpoint "$SPATIAL" \
  --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
  --dev-manifest "$V/reasonseg_val_internal_es.jsonl" \
  --test-manifest "$V/reasonseg_val_thin.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output ./eval_results/_oracle_probe/report.json \
  --epochs 4 --eval-records 600 --bce-mode plain $TVERSKY \
  > "$L/s1_oracle_probe.log" 2>&1
log "S1 rc=$?"

log "=== S2/5 B1 切除 LOC 先验 + Tversky（4 epoch，约 3.6 h）==="
$E/bin/accelerate launch --mixed_precision bf16 scripts/train_reasonseg.py \
  --model-base ./base_model/vicuna-v1-5-7b \
  --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
  --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
  --validation-manifest "$V/reasonseg_val_internal_es.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output-dir "./checkpoints/$B1" \
  --spatial-checkpoint "$SPATIAL" \
  --no-loc-prior --bce-mode plain $TVERSKY \
  --num-train-epochs 4 --per-device-train-batch-size 1 --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 --early-stopping-patience 12 --early-stopping-min-delta 1e-4 \
  --save-steps 200 --keep-last 2 --no-resume-state --validation-samples 0 \
  --mixed-precision bf16 > "$L/s2_train_b1.log" 2>&1
log "S2 rc=$?"

log "=== S3/5 B1 接地诊断（val_thin 1000 条，与 A2/对照同口径）==="
# 不加 --disable-loc-prior：该档的 config 里 use_loc_prior 已是 False，
# 诊断脚本走 reasonseg_config=None 会自动沿用档内配置。
$E/bin/python scripts/reasonseg_experiments/diagnose_reasonseg_grounding.py \
  --model-base ./base_model/vicuna-v1-5-7b \
  --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
  --seg-checkpoint "./checkpoints/$B1/checkpoint-best" \
  --manifest "$V/reasonseg_val_thin.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output "./eval_results/_grounding_$B1/report.json" \
  --max-samples 1000 --dtype fp16 --encoder-dtype fp32 > "$L/s3_grounding_b1.log" 2>&1
log "S3 rc=$?"

log "=== S4/5 B1 在过滤后 val_thin 上的 TF（与 s4/s5 同口径）==="
$E/bin/accelerate launch --mixed_precision bf16 scripts/train_reasonseg.py \
  --model-base ./base_model/vicuna-v1-5-7b \
  --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
  --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
  --validation-manifest "$V/reasonseg_val_thin_reachable.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output-dir ./checkpoints/_tfval_tmp \
  --eval-checkpoint "./checkpoints/$B1/checkpoint-best" \
  --no-loc-prior --bce-mode plain $TVERSKY \
  --validate-only --validation-samples 0 --validate-dtype-variants as_is \
  --mixed-precision bf16 > "$L/s4_tfval_b1.log" 2>&1
log "S4 rc=$?"
rm -rf ./checkpoints/_tfval_tmp
log "=== PHASE2.3 CHAIN DONE ==="
