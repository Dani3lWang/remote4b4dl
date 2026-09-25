#!/bin/bash
# 等 Phase 2.3 主链收尾后，用修正版探针重跑 oracle-query 天花板测量。
#
# 为什么要重跑：首跑（14:30 完成）有两个缺陷。一是末轮失稳——逐步 loss 从 ep2 的
# 0.779 跳到 ep3 中途 1.332、收尾 1.029，而评的正是这个最差末轮，且没有逐轮评测
# 与存档，epoch 2 的指标已无法取回；固定遍历顺序 + lr 3e-4 是失稳嫌疑。二是判据
# 选错轴：只用 recall@0.5 分档，实测 0.045 落在"编码器瓶颈"档，但同一份数据的
# AUC 0.978 直接反驳该裁定。
#
# 修正：每轮洗牌、lr 降到 1e-4、逐轮 dev 评测并按 dev iou_mean 选轮次、test 只评
# 一次、并补 recall_top_k（免幅值，把排序与尺寸校准分开）。判据换成先验推导的
# AUC 阈值——top-K=GT 尺寸要成立需 AUC >= 1-K/N ~= 0.99966（K~11、N~32500），
# 这个数与任何观测无关，所以不构成事后拟合。
#
# 独立会话，只读主链 driver 日志，不与主链共享文件句柄。
set -uo pipefail
cd /root/autodl-tmp/mmb4dl/mllm || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
L=./training_logs/phase23
D=$L/driver.log
W=$L/watcher_probe2.log
mkdir -p "$L"

log () { echo "[$(date '+%F %T')] $*" | tee -a "$W"; }

log "=== 等待 Phase 2.3 主链收尾 ==="
i=0
while [ $i -lt 480 ]; do
  if grep -q "PHASE2.3 CHAIN DONE" "$D" 2>/dev/null; then
    log "主链已完成（轮次 $i）"
    break
  fi
  if ! tmux has-session -t p23 2>/dev/null; then
    if grep -q "PHASE2.3 CHAIN DONE" "$D" 2>/dev/null; then
      log "主链已完成且会话已退出（轮次 $i）"
      break
    fi
    log "!!! 主链会话消失但未见 CHAIN DONE，判为异常终止，不抢卡"
    echo "NOGO chain_died" > "$L/probe2_decision.txt"
    exit 1
  fi
  i=$((i + 1))
  sleep 60
done
if [ $i -ge 480 ]; then
  log "!!! 等待超时（8 小时），不启动"
  echo "NOGO wait_timeout" > "$L/probe2_decision.txt"
  exit 1
fi

BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$BUSY" -ne 0 ]; then
  log "!!! GPU 上仍有 $BUSY 个计算进程，不抢卡"
  echo "NOGO gpu_busy_$BUSY" > "$L/probe2_decision.txt"
  exit 1
fi

log "=== 启动修正版 oracle 探针（6 epoch，lr 1e-4，逐轮 dev 评测）==="
echo "GO probe_rerun" > "$L/probe2_decision.txt"
$E/bin/python scripts/reasonseg_experiments/probe_oracle_query.py \
  --spatial-checkpoint ./checkpoints/reasonseg-spatial-tv/spatial-best \
  --train-manifest ./reasonseg_data_trainval/reasonseg_train_internal679x11.jsonl \
  --dev-manifest ./reasonseg_data_trainval/reasonseg_val_internal_es.jsonl \
  --test-manifest ./reasonseg_data_trainval/reasonseg_val_thin.jsonl \
  --dataroot /root/autodl-tmp/nuScenes \
  --output ./eval_results/_oracle_probe_v2/report.json \
  --epochs 6 --learning-rate 1e-4 --eval-records 400 \
  --bce-mode plain --region-loss tversky --tversky-alpha 0.3 --tversky-beta 0.7 \
  > "$L/probe_v2.log" 2>&1
log "probe_v2 rc=$?"
log "=== watcher 退出 ==="
