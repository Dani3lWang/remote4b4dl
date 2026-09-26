#!/bin/bash
# 多任务微调编码器：语义 + 实例两个目标共存实验，含跑完自动复量语义质量。
#
# 背景：线性探针（同协议）量出 v4 的实例微调把编码器语义质量砍掉 28.3%
# （预训练 0.30785 → v3 0.30560 → v4 0.22073），且损失与 dW conv 严格同向。
# 完整模型里同一个编码器还要压 scene token 给 LM，所以 v4 的编码器不能直接用。
# 本实验回答：加上语义损失后，两个目标能否共存于同一份特征。
#
# 与 v4 的关系：**唯一变量是多了语义 CE 这一项**。其余逐字对齐 v4——同 oracle query、
# 同 manifest、同 seed（故同洗牌顺序）、同 20 epoch、编码器 lr 5e-5、实例头 lr 1e-4、
# BN 钉 eval、Tversky(0.3,0.7)+plain、同按 dev iou_mean 选轮、test 只评一次。
#
# ============================ 预注册判据 ============================
# 裁定数一律用**线性探针**（与 §6.8 逐字相同：冻结编码器 + 永久 eval、分类头从零训
# 2 轮、lr 1e-3、wd 0.01、AdamW、fp32、同 seed、train 25,324 / val 2,806、零 worker）。
# **不能用训练期那个 sem val miou 裁定** —— 它与编码器联合训过，偏乐观，只看趋势。
#
# 参照值：预训练 miou 0.30785；v4 miou 0.22073、test R@0.5 0.24760 / top_k 0.33890 /
# AUC 0.99038。
#
#   共存成功     miou >= 0.2900（相对预训练损失 <= 5.8%）且 test R@0.5 >= 0.2476
#                且 top_k >= 0.3389（不劣于 v4）
#                -> 两目标可共存，v4 式编码器变得可用；下一步把它插回完整模型联合微调
#   部分共存     0.2500 <= miou < 0.2900 且 instance 不劣于 v4
#                -> 用 ep3/7/11/15/19 快照的探针结果画权衡曲线，按可接受的语义下限选点，
#                   并考虑扫 --semantic-weight
#   语义守住但实例退化   miou >= 0.2900 但 test R@0.5 < 0.2000
#                -> 语义损失把实例目标压住了，扫权重（降 semantic-weight 或升 instance-weight）
#   本质冲突     miou < 0.2500（加了语义损失仍掉超过 19%）
#                -> 同一份特征里不可兼得，转双编码器方案或改输出目标（粗区域）
#   判据无效     dW conv < 5%  -> 编码器没被真正训练，不下任何结论（沿用 v4 那条防呆）
#
# 辅助观察：dW stats 必须全程 0.000%（BN 钉住的硬校验）；enc grad 必须持续非零；
# 两支损失分别记录，看谁主导。
# ====================================================================
#
# 成本：冒烟实测 0.25 s/步 × 7,436 步 ≈ 31 min/epoch，20 epoch ≈ 10.4 h；加逐轮
# 实例 dev（400 条）与语义 val（2,806 条）约 1.3 min/epoch ⇒ **训练约 10.8 h**，
# 之后 6 次线性探针（选定轮 + 5 个快照）约 11 min/次 ⇒ **合计约 12 h**。
set -uo pipefail
cd /root/autodl-tmp/mmb4dl/mllm || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
V=./reasonseg_data_trainval
L=./training_logs/phase23
OUT=./eval_results/_multitask_encoder
DRIVER="$L/multitask_driver.log"
mkdir -p "$L" "$OUT"

log () { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER"; }

BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$BUSY" -ne 0 ]; then
  log "!!! GPU 上仍有 $BUSY 个计算进程，不抢卡"
  exit 1
fi

log "=== S1/2 多任务微调（20 epoch，约 10.8 h）==="
$E/bin/python scripts/reasonseg_experiments/probe_multitask_encoder.py \
  --spatial-checkpoint ./checkpoints/reasonseg-spatial-tv/spatial-best \
  --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
  --dev-manifest "$V/reasonseg_val_internal_es.jsonl" \
  --test-manifest "$V/reasonseg_val_thin.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output-dir "$OUT" \
  --epochs 20 --learning-rate 1e-4 --encoder-lr 5e-5 --semantic-head-lr 1e-3 \
  --semantic-weight 1.0 --instance-weight 1.0 \
  --eval-records 400 --snapshot-every 4 \
  --bce-mode plain --region-loss tversky --tversky-alpha 0.3 --tversky-beta 0.7 \
  > "$L/multitask.log" 2>&1
RC=$?
log "S1 rc=$RC"
if [ $RC -ne 0 ]; then
  log "!!! 多任务微调失败，尾部："
  tr '\r' '\n' < "$L/multitask.log" | tail -25 | tee -a "$DRIVER"
  exit 1
fi

log "=== S2/2 用线性探针复量选定轮 + 5 个快照的语义质量（约 11 min/次）==="
for ckpt in "$OUT"/encoder_selected.pt "$OUT"/encoder_ep3.pt "$OUT"/encoder_ep7.pt \
            "$OUT"/encoder_ep11.pt "$OUT"/encoder_ep15.pt "$OUT"/encoder_ep19.pt; do
  [ -f "$ckpt" ] || { log "跳过（不存在）: $ckpt"; continue; }
  tag=$(basename "$ckpt" .pt)
  log "--- 线性探针 $tag ---"
  $E/bin/accelerate launch --mixed_precision no scripts/train_reasonseg_spatial.py \
    --dataroot /root/autodl-tmp/nuScenes --output-dir /tmp/_unused_mt_probe \
    --validate-only --spatial-checkpoint "$ckpt" \
    --probe-epochs 2 --learning-rate 1e-3 --mixed-precision no \
    --dataloader-num-workers 0 --logging-steps 5000 \
    --validate-output "$L/linear_probe_mt_$tag.json" \
    > "$L/linear_probe_mt_$tag.log" 2>&1
  log "$tag rc=$?"
done

log "=== 汇总：语义 vs 实例 权衡 ==="
$E/bin/python - <<'PY' | tee -a "$DRIVER"
import json, pathlib
PRETRAIN, V4 = 0.30785, 0.22073
rep = json.load(open("eval_results/_multitask_encoder/report.json"))
t = rep["test"]
print(f"实例侧 test（选定 ep{rep['selected_epoch']}）: AUC {t['auc_mean']:.5f} "
      f"R@0.5 {t['recall_at_0.5']:.4f} topK {t['recall_top_k']:.4f} "
      f"IoU {t['iou_mean']:.4f} 尺寸 {t['predicted_size_median']:.0f}/{t['target_size_median']:.0f}")
print(f"  参照 v4: AUC 0.99038 R@0.5 0.24760 topK 0.33890 IoU 0.28710")
conv = rep["history"][rep["selected_epoch"]]["encoder_delta"].get("conv_weight", {})
print(f"选定轮 dW conv = {conv.get('relative_delta', 0):.2%}（v4 是 19.97%）"
      f" | dW stats = "
      f"{rep['history'][rep['selected_epoch']]['encoder_delta'].get('bn_stats', {}).get('relative_delta', 0):.2%}")
print(f"\n语义侧（线性探针，同 §6.8 协议）: 预训练 {PRETRAIN} / v4 {V4}")
for tag in ("selected", "ep3", "ep7", "ep11", "ep15", "ep19"):
    p = pathlib.Path(f"training_logs/phase23/linear_probe_mt_encoder_{tag}.json")
    if not p.exists():
        continue
    f = json.load(p.open())["final"]
    print(f"  {tag:<9} miou {f['miou']:.5f}  Δ预训练 {f['miou']-PRETRAIN:+.5f} "
          f"({(f['miou']-PRETRAIN)/PRETRAIN:+.1%})  Δv4 {f['miou']-V4:+.5f}  "
          f"acc {f['point_accuracy']:.5f}")
print("\n训练期逐轮（趋势，联合头偏乐观，不作裁定）:")
for h in rep["history"]:
    print(f"  ep{h['epoch']:>2} sem_val_miou {h['semantic_val']['miou']:.5f} | "
          f"dev R@0.5 {h['dev'].get('recall_at_0.5', float('nan')):.4f} "
          f"topK {h['dev'].get('recall_top_k', float('nan')):.4f} "
          f"AUC {h['dev'].get('auc_mean', float('nan')):.4f} "
          f"IoU {h['dev'].get('iou_mean', float('nan')):.4f} | "
          f"loss sem {h['train_loss_semantic']:.4f} inst {h['train_loss_instance']:.4f}")
PY
log "=== MULTITASK CHAIN DONE ==="
