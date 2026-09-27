#!/bin/bash
# 选项 C：把多任务微调延长到 30 epoch，检验"AUC 平台"是真的还是被 20 轮预算截断。
#
# 为什么必须整轮重跑而不是从 ep19 续：首跑只落了 `encoder_ep*.pt`（纯编码器），两个头
# 的权重没保存，所以精确续跑做不到。本次已给脚本补上 `state_ep*.pt`（四个模块全量权重），
# 以后的跑可以热启动（注意仍非精确续跑——AdamW 的二阶矩没存）。
#
# 前 20 轮与首跑逐字同配置（同 seed 故同洗牌顺序），所以 ep0–19 应当复现首跑，这本身
# 就是一个一致性检查；新增信息在 ep20–29。
#
# 与首跑的唯一差异 = epoch 数 20 → 30。其余（oracle query、manifest、编码器 lr 5e-5、
# 实例头 1e-4、语义头 1e-3、两个权重各 1.0、BN 钉 eval、Tversky+plain、按 dev
# iou_mean 选轮、test 只评一次）全部不变 ⇒ 仍是"编码器能改造到什么程度"这条线上的延长。
#
# ============================ 预注册判据 ============================
# **主轴 = test AUC（排序质量）**，因为平台判定与 top-K 可行性都挂在它上面；
# **次轴 = recall@0.5 / iou_mean / 尺寸比（幅值轴）**，容差更大、且首跑已证明它会与主轴分离。
# 明确分出"优于最好预期"那一档——上次落空就是因为五档全假设 miou 会下降、没给正向意外留出口。
#
# 参照值：首跑（20 epoch）test AUC 0.99575 / R@0.5 0.2644 / top_k 0.3197 / IoU 0.2991；
#         其 dev AUC 在 ep9–19 稳在 0.9963–0.9976；线性探针语义 miou 0.33109。
#         top-K 可用门槛 AUC >= 1-K/N ~= 0.99963（K=12、N~=32530，先验推导）。
#
#   A' 推翻天花板      test AUC >= 0.99963
#                      -> top-K 可行，"点特征撑不起实例接地"被推翻，编码器改造升为主攻
#   B' 优于预期但未达标 test AUC >= 0.998（负点 <= 65）但 < 0.99963
#                      -> 比首跑的 0.99575 明显好；延长预算确实有效但差最后一个量级，
#                         如实报"预算内未达"，并据此决定是否再加长/加 lr
#   C' 仍在改善         test AUC < 0.998 且 ep22–29 dev AUC 存在新高
#                      -> 未收敛，20/30 轮都不够；报"平台尚未确立"，不据此下裁定
#   D' 平台定案         test AUC < 0.998 且 ep22–29 dev AUC 极差 <= 0.0015
#                         且 (AUC_ep29 - AUC_ep22)/7 <= 0.0002/轮
#                      -> 平台在 30 轮尺度成立，"单帧点特征撑不起实例接地"定案，
#                         下一步只能在 A（时序多帧，唯一直接改变 K/N 的杠杆）与 D（写负结果）间选
#   E' 共存结论失效     任一新快照（ep23/ep27/选定轮）线性探针 miou < 0.2900
#                      -> "语义与实例可共存"只在 20 轮内成立，需按轮数改写结论
#   分离情况（需如实记录，不算失败）排序轴平台但幅值轴仍在涨 -> 两个轴分离，
#                      报告时分开陈述，不许用幅值轴的改善去解释排序轴的停滞
#   判据无效           dW conv < 5%（不可能，首跑已 21.81%，仍列出以守住这条纪律）
# ====================================================================
#
# 成本：实测约 26 min/epoch ⇒ 30 epoch 约 13 h；之后 3–4 次线性探针约 11 min/次。
set -uo pipefail
cd /root/autodl-tmp/mmb4dl/mllm || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
V=./reasonseg_data_trainval
L=./training_logs/phase23
OUT=./eval_results/_multitask_encoder_ext
DRIVER="$L/multitask_ext_driver.log"
mkdir -p "$L" "$OUT"

log () { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER"; }

BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$BUSY" -ne 0 ]; then
  log "!!! GPU 上仍有 $BUSY 个计算进程，不抢卡"
  exit 1
fi

log "=== S1/2 多任务微调延长到 30 epoch（约 13 h）==="
$E/bin/python scripts/reasonseg_experiments/probe_multitask_encoder.py \
  --spatial-checkpoint ./checkpoints/reasonseg-spatial-tv/spatial-best \
  --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
  --dev-manifest "$V/reasonseg_val_internal_es.jsonl" \
  --test-manifest "$V/reasonseg_val_thin.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output-dir "$OUT" \
  --epochs 30 --learning-rate 1e-4 --encoder-lr 5e-5 --semantic-head-lr 1e-3 \
  --semantic-weight 1.0 --instance-weight 1.0 \
  --eval-records 400 --snapshot-every 4 \
  --bce-mode plain --region-loss tversky --tversky-alpha 0.3 --tversky-beta 0.7 \
  > "$L/multitask_ext.log" 2>&1
RC=$?
log "S1 rc=$RC"
if [ $RC -ne 0 ]; then
  log "!!! 延长跑失败，尾部："
  tr '\r' '\n' < "$L/multitask_ext.log" | tail -25 | tee -a "$DRIVER"
  exit 1
fi

log "=== S2/2 线性探针复量新快照（ep23 / ep27 / 选定轮）==="
for ckpt in "$OUT"/encoder_ep23.pt "$OUT"/encoder_ep27.pt "$OUT"/encoder_selected.pt; do
  [ -f "$ckpt" ] || { log "跳过（不存在）: $ckpt"; continue; }
  tag=$(basename "$ckpt" .pt)
  log "--- 线性探针 $tag ---"
  $E/bin/accelerate launch --mixed_precision no scripts/train_reasonseg_spatial.py \
    --dataroot /root/autodl-tmp/nuScenes --output-dir /tmp/_unused_ext_probe \
    --validate-only --spatial-checkpoint "$ckpt" \
    --probe-epochs 2 --learning-rate 1e-3 --mixed-precision no \
    --dataloader-num-workers 0 --logging-steps 5000 \
    --validate-output "$L/linear_probe_ext_$tag.json" \
    > "$L/linear_probe_ext_$tag.log" 2>&1
  log "$tag rc=$?"
done

log "=== 汇总与预注册裁定 ==="
$E/bin/python - <<'PY' | tee -a "$DRIVER"
import json, pathlib
N, K = 32530.0, 12.0
NEED = 1.0 - K / N
PRETRAIN_MIou, FIRST_MT_MIou, V4_MIou = 0.30785, 0.33109, 0.22073  # 名字必须带来源，曾因 BASE_MIou 拿到 v4 的数把 Δ 标签印反
rep = json.load(open("eval_results/_multitask_encoder_ext/report.json"))
h = rep["history"]
sel = rep["selected_epoch"]
t = rep["test"]
neg = (1 - t["auc_mean"]) * N
print(f"选定 ep{sel} | dW conv "
      f"{h[sel]['encoder_delta'].get('conv_weight', {}).get('relative_delta', 0):.2%}")
print(f"test: AUC {t['auc_mean']:.5f} (门槛 {NEED:.5f}) -> 负点 {neg:.0f} vs 需 <=12, "
      f"差 {neg/12:.1f}x | R@0.5 {t['recall_at_0.5']:.4f} topK {t['recall_top_k']:.4f} "
      f"IoU {t['iou_mean']:.4f} 尺寸 {t['predicted_size_median']:.0f}/{t['target_size_median']:.0f}")
print(f"  首跑(20ep) 参照: AUC 0.99575 R@0.5 0.2644 topK 0.3197 IoU 0.2991 负点 ~138")

tail = [x["dev"]["auc_mean"] for x in h[22:]]
print(f"\ndev AUC ep22-{len(h)-1}: {[round(v, 5) for v in tail]}")
if len(tail) >= 2:
    span = max(tail) - min(tail)
    slope = (tail[-1] - tail[0]) / (len(tail) - 1)
    print(f"极差 {span:.5f} | 每轮斜率 {slope:+.6f}")
    flat = span <= 0.0015 and slope <= 0.0002
else:
    span, slope, flat = float("nan"), float("nan"), False
still_rising = any(v > 0.9976 + 0.0005 for v in tail)

sem = {}
for tag in ("encoder_ep23", "encoder_ep27", "encoder_selected"):
    p = pathlib.Path(f"training_logs/phase23/linear_probe_ext_{tag}.json")
    if p.exists():
        sem[tag] = json.load(p.open())["final"]["miou"]
print("\n新快照线性探针语义 miou（基线 0.30785 / 首跑选定轮 0.33109 / v4 0.22073）:")
for tag, v in sem.items():
print(f"  {tag:<18} {v:.5f}  Δ预训练 {v-PRETRAIN_MIou:+.5f} ({(v-PRETRAIN_MIou)/PRETRAIN_MIou:+.1%})  Δv4 {v-V4_MIou:+.5f}  Δ首跑 {v-FIRST_MT_MIou:+.5f}")

auc = t["auc_mean"]
if auc >= NEED:
    verdict = "A' 推翻天花板：top-K 可行，'点特征撑不起实例接地'被推翻"
elif auc >= 0.998:
    verdict = "B' 优于预期但未达标：延长有效，但仍差最后一个量级"
elif not flat:
    verdict = "C' 仍在改善：平台尚未确立，不据此下裁定"
else:
    verdict = "D' 平台定案：单帧点特征撑不起实例接地（30 轮尺度）"
extra = ""
if min(sem.values(), default=1.0) < 0.2900:
    extra = " | 另触发 E'：共存结论只在 20 轮内成立"
if still_rising and not flat:
    extra += " | 注意 dev AUC 出现过新高"
print(f"\n裁定: {verdict}{extra}")
print(f"幅值轴是否仍在涨（与排序轴分离则分开陈述）: R@0.5 ep20+ 最高 "
      f"{max(x['dev']['recall_at_0.5'] for x in h[20:] if 'recall_at_0.5' in x['dev']):.4f} "
      f"vs 首跑 ep19 0.2887")
PY
log "=== MULTITASK EXT CHAIN DONE ==="
