#!/bin/bash
# 线性探针：量三个空间编码器的 lidarseg 语义质量，协议逐字相同。
#
# 为什么要量：oracle 探针 v3/v4 用**实例**目标微调了这个原本用**语义**目标预训练的
# 编码器。若语义质量被冲掉，把 v4 的编码器插回完整模型会连带损害别的能力。
#
# 为什么不能用档内记录的 0.2876 当基线（两个独立原因，都已查证）：
#   ① save_spatial_encoder_checkpoint 只存 point_encoder 的 52 个张量，预训练那个
#      nn.Linear(256,32) 分类头**从未保存** ⇒ 直接"加载后验证"会用随机头算出看着正常
#      实际无意义的数（同一份权重实测 0.19633 与 0.00706，取决于样本集）。
#   ② c51c644（09-22）把 split="val" 从 nuScenes 官方 val（6,019 条）改成官方 train
#      场景内的防泄漏划分（2,806 条），而该档是 09-21 训的。
# 所以三个编码器全部用**同一套线性探针协议**重量，只横比、不与历史数值比。
#
# 协议：编码器冻结且永久 eval（BN 用自身 running stats）；分类头从零训 2 轮，
# lr 1e-3、wd 0.01、AdamW；fp32（--mixed-precision no，与原预训练数值一致，且 spconv
# 在 fp16 下硬崩、bf16 出过 NaN）；同 seed 故同划分与同头初始化；train 25,324 / val 2,806。
#
# --dataloader-num-workers 0：首跑用 4 个 worker 时在 epoch 0 的 step 10000 之后死于
# multiprocessing resource_sharer 的 socket 消失（FileNotFoundError，/dev/shm 60G 全空、
# fd 上限 1M，都不是资源问题）。worker 数为 0 时不存在这条 IPC 路径。GPU 现在空闲，
# 慢一点无所谓，稳定优先。
set -uo pipefail
cd /root/autodl-tmp/mmb4dl/mllm || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
L=./training_logs/phase23
mkdir -p "$L"
DRIVER="$L/linear_probe_chain.log"

log () { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER"; }

run_one () {
  local tag="$1" ckpt="$2"
  log "=== $tag: $ckpt ==="
  $E/bin/accelerate launch --mixed_precision no scripts/train_reasonseg_spatial.py \
    --dataroot /root/autodl-tmp/nuScenes \
    --output-dir /tmp/_unused_linear_probe \
    --validate-only --spatial-checkpoint "$ckpt" \
    --probe-epochs 2 --learning-rate 1e-3 --mixed-precision no \
    --dataloader-num-workers 0 --logging-steps 5000 \
    --validate-output "$L/linear_probe_$tag.json" \
    > "$L/linear_probe_$tag.log" 2>&1
  log "$tag rc=$?"
}

BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$BUSY" -ne 0 ]; then
  log "!!! GPU 上仍有 $BUSY 个计算进程，不抢卡"
  exit 1
fi

run_one pretrained ./checkpoints/reasonseg-spatial-tv/spatial-best
run_one v3_unfrozen ./eval_results/_oracle_probe_v3_unfrozen/report_encoder.pt
run_one v4_unfrozen_bnfix ./eval_results/_oracle_probe_v4_unfrozen_bnfix/report_encoder.pt

log "=== 三个编码器的语义质量对比（同协议线性探针）==="
$E/bin/python - <<'PY' | tee -a "$DRIVER"
import json, pathlib
rows = []
for tag in ("pretrained", "v3_unfrozen", "v4_unfrozen_bnfix"):
    p = pathlib.Path(f"training_logs/phase23/linear_probe_{tag}.json")
    if not p.exists():
        print(f"{tag}: 缺失"); continue
    d = json.load(p.open())
    rows.append((tag, d))
if rows:
    base = rows[0][1]["final"]["miou"]
    print(f"{'编码器':<22}{'miou':>9}{'Δ vs 预训练':>13}{'point_acc':>11}{'轮数':>5}{'val条数':>9}")
    for tag, d in rows:
        f = d["final"]
        print(f"{tag:<22}{f['miou']:>9.5f}{f['miou']-base:>+13.5f}"
              f"{f['point_accuracy']:>11.5f}{d['probe']['epochs']:>5}{d['val_samples']:>9}")
    print("\n逐轮 miou:")
    for tag, d in rows:
        print(f"  {tag:<22}", [round(h["miou"], 5) for h in d["history"]])
PY
log "=== LINEAR PROBE CHAIN DONE ==="
