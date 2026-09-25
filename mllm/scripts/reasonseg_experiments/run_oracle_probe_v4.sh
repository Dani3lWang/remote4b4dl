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
# 预注册判据（先验推导，与 v3 观测无关，故不构成事后拟合）：
#   top-K 解码要能命中需 AUC >= 1 - K/N，K = GT 尺寸中位 12、N ~= 32530 框内点
#   => **门槛 0.99963**，等价于"压在正点上的负点 <= 12"。
# 裁定分支（test，选定轮只评一次）：
#   AUC >= 0.99963            -> top-K 可行；点特征**不是**硬约束，v2/v3 的裁定被推翻，
#                                投向"让编码器适应实例目标"这条线
#   0.998 <= AUC < 0.99963    -> 大幅逼近但仍不够（负点 65~12）；看 ep19 是否仍在升，
#                                仍在升就续跑，已平则判"接近但不足"
#   AUC < 0.998 且已平，且卷积权重变化 >= 15%
#                             -> 特征确实被大幅改造过仍分不开实例 ⇒ **单帧点特征撑不起
#                                实例接地**成立，转去改输出目标（粗区域）/ 引入时序多帧 /
#                                如实报负结果
#   AUC < 0.998 但卷积权重变化 < 5%
#                             -> 编码器仍没被真正训练，判据无效，先解决 lr/可训练性再谈裁定
# 辅助轴：recall_top_k（免幅值）与尺寸比 pred/target（应落在 [0.7,1.5]）。
#
# 成本：v3 实测 22.3 min/epoch（0.178 s/步 × 7436 步），20 epoch ≈ **7.4 h**。
set -uo pipefail
cd /root/autodl-tmp/mmb4dl/mllm || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
V=./reasonseg_data_trainval
L=./training_logs/phase23
mkdir -p "$L"

log () { echo "[$(date '+%F %T')] $*" | tee -a "$L/run_probe_v4.log"; }

BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$BUSY" -ne 0 ]; then
  log "!!! GPU 上仍有 $BUSY 个计算进程，不抢卡"
  exit 1
fi

log "=== oracle 探针 v4：解冻编码器 lr 5e-5 + BN 钉 eval + 20 epoch（约 7.4 h）==="
$E/bin/python scripts/reasonseg_experiments/probe_oracle_query.py \
  --spatial-checkpoint ./checkpoints/reasonseg-spatial-tv/spatial-best \
  --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
  --dev-manifest "$V/reasonseg_val_internal_es.jsonl" \
  --test-manifest "$V/reasonseg_val_thin.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output ./eval_results/_oracle_probe_v4_unfrozen_bnfix/report.json \
  --epochs 20 --learning-rate 1e-4 --eval-records 400 \
  --unfreeze-encoder --encoder-lr 5e-5 --freeze-encoder-bn \
  --bce-mode plain --region-loss tversky --tversky-alpha 0.3 --tversky-beta 0.7 \
  > "$L/probe_v4.log" 2>&1
RC=$?
log "probe_v4 rc=$RC"
if [ $RC -eq 0 ]; then
  $E/bin/python - <<'PY' | tee -a "$L/run_probe_v4.log"
import json
d = json.load(open("eval_results/_oracle_probe_v4_unfrozen/report.json"))
N, K = 32530.0, 12.0
need = 1.0 - K / N
print(f"selected epoch {d['selected_epoch']} | encoder_lr {d['config']['encoder_lr']} "
      f"| BN pinned {d['config']['encoder_bn_pinned_eval']}")
for split in ("dev", "test"):
    s = d[split]
    neg = (1.0 - s["auc_mean"]) * N
    print(f"{split}: AUC {s['auc_mean']:.5f} (门槛 {need:.5f}) -> 正点上方负点 "
          f"{neg:.0f} 个，需 <= {K:.0f}，差 {neg / K:.1f}x | R@0.5 {s['recall_at_0.5']:.4f} "
          f"topK {s['recall_top_k']:.4f} IoU {s['iou_mean']:.4f} "
          f"尺寸 {s['predicted_size_median']:.0f}/{s['target_size_median']:.0f}")
last = d["history"][-1]
print("末轮编码器权重相对变化:",
      {k: f"{v['relative_delta']:.2%}" for k, v in last.get("encoder_delta", {}).items()})
print("dev AUC 逐轮:", [round(h["dev"]["auc_mean"], 5) for h in d["history"]])
print("dev topK 逐轮:", [round(h["dev"]["recall_top_k"], 4) for h in d["history"]])
PY
fi
log "=== v4 收尾 ==="
exit $RC
