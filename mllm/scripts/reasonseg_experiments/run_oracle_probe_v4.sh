#!/bin/bash
# oracle 探针第四跑（v4）：**真正微调编码器** + BN 钉在 eval + 20 epoch。
#
# 为什么要再跑一次：v3（`--unfreeze-encoder --encoder-lr 1e-5`，6 epoch）test AUC
# 0.99510 / recall_top_k 0.18990，比 v2 冻结臂（0.99110 / 0.11058）每个轴都好，
# 但事后把选定轮的编码器权重与预训练档逐张量一比，发现
#
#     卷积权重只动了 3.52% ｜ BN weight/bias 5.38% ｜ BN running stats 18.71%
#
# 即 lr 1e-5 × 6 epoch 根本没让特征本身改变，**"点特征能否被改造成实例判别的"这个
# 问题在 v3 里没有真正被检验**。而且 BN 统计量漂了 18.7% 意味着 v2 vs v3 不是干净的
# 单变量对照：batch size = 1 时 BN 的 train 模式用逐帧统计量归一化，**即使权重一个都
# 不更新也会改变特征**，所以 v3 那 +72% 的 top_k 里有多少来自"学习"分不出来。
#
# v4 针对这两点各改一处，第三处是 v3 明确没跑够：
#   1. 编码器 lr 1e-5 → **5e-5**（冒烟实测 60 条就把卷积权重推动 0.462%，
#      而 v3 跑 44,616 步才 3.52%）
#   2. **`--freeze-encoder-bn`**：train() 之后把 8 个 BatchNorm1d 逐个钉回 eval，
#      归一化一律用预训练 running stats 且不再更新（冒烟实测 stats 变化 0.000%）。
#      这样"解冻"才真的只意味着"权重可学"。
#   3. epoch 6 → **20**：v3 的 dev AUC 六轮单调升（0.9839→0.9957）、train loss 仍在降、
#      且选定轮就是最后一轮 —— 是被预算截断的，不是收敛。
# 每轮日志会打印编码器权重的相对变化量（conv / bn / stats 分开），直接盯住"到底学没学"。
#
# Report measured per-object top-K IoU/recall; AUC is auxiliary, not a feasibility gate.
# This experiment cannot establish a universal limit of single-frame features.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
ASSETS=${REASONSEG_ASSET_ROOT:-/root/autodl-tmp/mmb4dl/mllm}
V=${REASONSEG_MANIFEST_DIR:?set REASONSEG_MANIFEST_DIR to an audited manifest directory}
L=./training_logs/phase23
REPORT=./eval_results/_oracle_probe_v4_unfrozen_bnfix/report.json
[ ! -e "$REPORT" ] || { echo "report already exists: $REPORT" >&2; exit 2; }
mkdir -p "$L"

log () { echo "[$(date '+%F %T')] $*" | tee -a "$L/run_probe_v4.log"; }

BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$BUSY" -ne 0 ]; then
  log "!!! GPU 上仍有 $BUSY 个计算进程，不抢卡"
  exit 1
fi

log "=== oracle 探针 v4：解冻编码器 lr 5e-5 + BN 钉 eval + 20 epoch（约 7.4 h）==="
$E/bin/python scripts/reasonseg_experiments/probe_oracle_query.py \
  --spatial-checkpoint "$ASSETS/checkpoints/reasonseg-spatial-tv/spatial-best" \
  --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
  --dev-manifest "$V/reasonseg_val_internal_es.jsonl" \
  --test-manifest "$V/reasonseg_val_thin.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output "$REPORT" \
  --epochs 20 --learning-rate 1e-4 --eval-records 400 \
  --unfreeze-encoder --encoder-lr 5e-5 --freeze-encoder-bn \
  --bce-mode plain --region-loss tversky --tversky-alpha 0.3 --tversky-beta 0.7 \
  > "$L/probe_v4.log" 2>&1
RC=$?
log "probe_v4 rc=$RC"
if [ $RC -eq 0 ]; then
  $E/bin/python - "$REPORT" <<'PY' | tee -a "$L/run_probe_v4.log"
import json, sys
d = json.load(open(sys.argv[1]))
assert d['config']['encoder_bn_pinned_eval'] > 0
assert d['config']['encoder_lr'] == 5e-5
assert d['config']['epochs'] == 20
print(f"selected epoch {d['selected_epoch']}; diagnostic only, no AUC feasibility gate")
for split in ("dev", "test"):
    s = d[split]
    print(f"{split}: AUC {s['auc_mean']:.5f} | R@0.5 {s['recall_at_0.5']:.4f} "
          f"topK recall {s['recall_top_k']:.4f} topK IoU {s['iou_top_k_mean']:.4f}")
last = d["history"][-1]
print("末轮编码器权重相对变化:",
      {k: f"{v['relative_delta']:.2%}" for k, v in last.get("encoder_delta", {}).items()})
print("dev AUC 逐轮:", [round(h["dev"]["auc_mean"], 5) for h in d["history"]])
print("dev topK 逐轮:", [round(h["dev"]["recall_top_k"], 4) for h in d["history"]])
PY
fi
log "=== v4 收尾 ==="
exit $RC
