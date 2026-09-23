#!/bin/bash
# 独立 watcher：等 Phase 2 主链跑完，按预注册判据决定是否自动扩跑 20 epoch。
#
# 为什么单独起一个会话而不是加进主链：主链的 bash 进程正在按字节偏移读取自己的脚本，
# 改动它会让它执行到错乱位置。watcher 只读主链的 driver 日志，两者不共享文件句柄。
#
# 自动化边界（刻意保守）：只有 decide_phase22_extension.py 输出 GO 才扩跑，即
# 恰好一臂满足 AUC>=0.93 且 mean_prob_positive>=0.35、不触发任何护栏、且 epoch-4
# TF mIoU 超过对照。出现平局、护栏与机制判据冲突、或数据缺失一律 NOGO 停下等人——
# 尺寸护栏在 epoch 4 有可能误杀（"敢触发"先于"学会停止触发"到位），这种情况必须人判。
set -uo pipefail
cd /root/autodl-tmp/mmb4dl/mllm || exit 1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
L=./training_logs/phase2
D=$L/chain_driver.log
W=$L/watcher.log
mkdir -p "$L"

log () { echo "[$(date '+%F %T')] $*" | tee -a "$W"; }

log "=== watcher 启动，等待主链 CHAIN ALL DONE ==="
i=0
while [ $i -lt 1200 ]; do
  if grep -q "CHAIN ALL DONE" "$D" 2>/dev/null; then
    log "主链已完成（轮次 $i）"
    break
  fi
  if ! tmux has-session -t phase2 2>/dev/null; then
    if grep -q "CHAIN ALL DONE" "$D" 2>/dev/null; then
      log "主链已完成且会话已退出（轮次 $i）"
      break
    fi
    log "!!! 主链会话消失但未见 CHAIN ALL DONE，判为异常终止，不自动扩跑"
    echo "NOGO chain_died" > "$L/extension_decision.txt"
    exit 1
  fi
  i=$((i + 1))
  sleep 60
done
if [ $i -ge 1200 ]; then
  log "!!! 等待超时（20 小时），不自动扩跑"
  echo "NOGO wait_timeout" > "$L/extension_decision.txt"
  exit 1
fi

# 主链结束后 GPU 应当空出来；若仍有占用进程说明上一阶段没干净退出，不抢卡。
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$BUSY" -ne 0 ]; then
  log "!!! GPU 上仍有 $BUSY 个计算进程，不自动扩跑"
  echo "NOGO gpu_busy_$BUSY" > "$L/extension_decision.txt"
  exit 1
fi

DECISION=$($E/bin/python scripts/reasonseg_experiments/decide_phase22_extension.py \
  --summary "$L/phase22_summary.json" 2>&1 | tail -1)
log "判据裁定: $DECISION"
echo "$DECISION" > "$L/extension_decision.txt"

case "$DECISION" in
  GO\ *)
    ARM=$(echo "$DECISION" | awk '{print $2}')
    # flags= 固定在行尾且含空格，故只按该标记切，前面的 key=value 一并丢掉。
    FLAGS=${DECISION#*flags=}
    log "=== 自动扩跑 20 epoch: $ARM ==="
    log "损失旗标: $FLAGS"
    # shellcheck disable=SC2086
    bash scripts/reasonseg_experiments/run_loss_rebalance_extension.sh "$ARM" $FLAGS 2>&1 \
      | tee -a "$W"
    log "=== 扩跑结束 ==="
    ;;
  *)
    log "=== 不自动扩跑，等待人工裁定 ==="
    ;;
esac
log "=== watcher 退出 ==="
