#!/bin/bash
# 选项 A 的单臂 runner（A1/A2/A3 三臂各自独立跑一次，预注册见
# docs/learn docs/B4DL_ReasonSeg多帧A_预注册_20260927.md）。
#
# 用法: run_temporal_arm.sh <a1_compensated|a2_repeat|a3_naive> [帧数=3] [轮数=20]
#
# 与 30ep 单帧跑的差异**只有实例侧的 --temporal-arm/--num-frames**。语义侧、manifest、
# seed、lr、BN 钉 eval、Tversky(0.3,0.7)、按 dev iou_mean 选轮、test 只评一次，全部逐字对齐。
#
# 门槛必须按臂实测：这里拒绝在缺冒烟门产物时启动，因为它拿 val_thin 上该臂的 K/N 算门槛。
set -uo pipefail
ARM="${1:?用法: run_temporal_arm.sh <arm> [frames] [epochs]}"
FRAMES="${2:-3}"
EPOCHS="${3:-20}"
cd /root/autodl-tmp/mmb4dl/mllm || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
V=./reasonseg_data_trainval
L=./training_logs/temporal
OUT=./eval_results/_temporal_${ARM}_f${FRAMES}
SMOKE=./eval_results/_multiframe_smoke_reasonseg_val_thin.json
DRIVER="$L/${ARM}_driver.log"
mkdir -p "$L" "$OUT"
log () { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER"; }

BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$BUSY" -ne 0 ]; then log "!!! GPU 上仍有 $BUSY 个计算进程，不抢卡"; exit 1; fi
if [ ! -f "$SMOKE" ]; then log "!!! 缺 $SMOKE，门槛无法按臂实测，拒绝启动"; exit 1; fi

log "=== $ARM  F=$FRAMES  epochs=$EPOCHS ==="
$E/bin/python scripts/reasonseg_experiments/probe_multitask_encoder.py \
  --spatial-checkpoint ./checkpoints/reasonseg-spatial-tv/spatial-best \
  --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
  --dev-manifest "$V/reasonseg_val_internal_es.jsonl" \
  --test-manifest "$V/reasonseg_val_thin.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes --output-dir "$OUT" \
  --epochs "$EPOCHS" --learning-rate 1e-4 --encoder-lr 5e-5 --semantic-head-lr 1e-3 \
  --semantic-weight 1.0 --instance-weight 1.0 --eval-records 400 --snapshot-every 4 \
  --bce-mode plain --region-loss tversky --tversky-alpha 0.3 --tversky-beta 0.7 \
  --temporal-arm "$ARM" --num-frames "$FRAMES" > "$L/${ARM}.log" 2>&1
RC=$?
log "train rc=$RC"
if [ $RC -ne 0 ]; then
  log "!!! 失败，尾部："
  tr '\r' '\n' < "$L/${ARM}.log" | tail -25 | tee -a "$DRIVER"
  exit 1
fi

log "=== 汇总：按本臂实测门槛换算距离 ==="
$E/bin/python - "$OUT" "$SMOKE" "$ARM" <<'PY' | tee -a "$DRIVER"
import json, sys
out, smoke_path, arm = sys.argv[1], sys.argv[2], sys.argv[3]
rep = json.load(open(out + "/report.json"))
smoke = json.load(open(smoke_path))
sel, h, t = rep["selected_epoch"], rep["history"], rep["test"]
key = "threshold_single_frame_reference" if arm == "a2_repeat_control_check" else "threshold_" + arm
thr = smoke[key]["threshold_1_minus_K_over_N"]
single_thr = smoke["threshold_single_frame_reference"]["threshold_1_minus_K_over_N"]
SINGLE30, SINGLE20 = 0.99377, 0.99575
def gap(auc, base): return (1.0 - auc) / max(1.0 - base, 1e-12)
print(f"{arm} 选定 ep{sel} | dW conv "
      f"{h[sel]['encoder_delta']['conv_weight']['relative_delta']:.2%}")
print(f"本臂门槛(实测 K/N 于 val_thin) {thr:.6f} | 单帧门槛 {single_thr:.6f}")
print(f"test AUC {t['auc_mean']:.5f} -> 距本臂门槛 {gap(t['auc_mean'], thr):.1f}x"
      f"   参照：单帧 30ep {gap(SINGLE30, single_thr):.1f}x / 20ep {gap(SINGLE20, single_thr):.1f}x")
print(f"      R@0.5 {t['recall_at_0.5']:.4f} topK {t['recall_top_k']:.4f} IoU {t['iou_mean']:.4f}"
      f" 尺寸 {t['predicted_size_median']:.1f}/{t['target_size_median']:.1f}"
      f" objects {t['objects']} skipped_unreachable {t['skipped_unreachable']}")
dev = [x["dev"]["auc_mean"] for x in h]
tail = dev[max(0, len(dev) - 8):]
print(f"dev AUC 末 8 轮 {[round(v,5) for v in tail]} 极差 {max(tail)-min(tail):.5f}")
if arm == "a2_repeat":
    lo, hi = 0.99038, 0.99760
    ok = lo <= t["auc_mean"] <= hi
    print(f"失效门预言：a2 的 test AUC 应落在单帧各跑覆盖的带 [{lo}, {hi}] -> "
          f"{'成立，可以信 A1/A3 的读数' if ok else '不成立 ⇒ 拼接/复制改变了度量，A 全部读数作废'}")
PY
log "=== TEMPORAL $ARM CHAIN DONE ==="
