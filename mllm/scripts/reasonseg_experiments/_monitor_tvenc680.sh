#!/bin/bash
# tvenc680 训练监测: 进度 + 磁盘 + ckpt 保护(轮转 keep N), 训练退出后汇总 val 指标再退出
LIVE=/root/autodl-tmp/mmb4dl/mllm/checkpoints/reasonseg-tvenc680
PROT=/root/autodl-tmp/mmb4dl/mllm/checkpoints/_protected_tvenc680
LOGF=/root/autodl-tmp/mmb4dl/mllm/training_logs/monitor_tvenc680.log
TRAINLOG=/root/autodl-tmp/mmb4dl/mllm/training_logs/tvenc680.log
KEEP=8
INTERVAL=1200
AVAIL_STOP_COPY_GB=15
MAX_LOOPS=200
mkdir -p "$PROT"
SEEN_RUNNING=0
loops=0

echo "===== monitor (tvenc680) started $(date '+%F %T') pid=$$ keep=$KEEP =====" >> "$LOGF"

while true; do
  loops=$((loops+1))
  ts=$(date '+%F %T')
  last=$(grep '"step"' "$TRAINLOG" 2>/dev/null | tail -1)
  step=$(echo "$last" | grep -oP '"step":\s*\K[0-9]+' | tail -1)
  epoch=$(echo "$last" | grep -oP '"epoch":\s*\K[0-9]+' | tail -1)
  loss=$(echo "$last" | grep -oP '"loss":\s*\K[0-9.]+' | tail -1)
  mseg=$(echo "$last" | grep -oP '"mask_seg":\s*\K[0-9.]+' | tail -1)
  running=$(pgrep -f 'scripts/train_reasonseg.py' >/dev/null && echo YES || echo NO)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null)
  avail=$(df -BG --output=avail /root/autodl-tmp 2>/dev/null | tail -1 | tr -dc '0-9')
  usep=$(df -h /root/autodl-tmp 2>/dev/null | tail -1 | awk '{print $5}')
  valrec=$(grep 'teacher_forcing_mean_iou' "$TRAINLOG" 2>/dev/null | tail -1)
  lastmiou=$(echo "$valrec" | grep -oP '"teacher_forcing_mean_iou":\s*\K[0-9.]+')
  lastgiou=$(echo "$valrec" | grep -oP '"teacher_forcing_global_iou":\s*\K[0-9.]+')
  nval=$(grep -c 'teacher_forcing_mean_iou' "$TRAINLOG" 2>/dev/null)
  live_n=$(ls -1 "$LIVE" 2>/dev/null | grep -cE '^checkpoint-[0-9]+$')
  prot_n=$(ls -1 "$PROT" 2>/dev/null | grep -cE '^checkpoint-[0-9]+$')
  echo "[$ts] running=$running step=${step:-?} epoch=${epoch:-?} loss=${loss:-?} mask_seg=${mseg:-?} mIoU=${lastmiou:-?} gIoU=${lastgiou:-?} nval=${nval:-0} gpu=${gpu:-?} disk_avail=${avail:-?}G use=${usep:-?} live_ckpt=$live_n prot_ckpt=$prot_n" >> "$LOGF"
  [ "$running" = "YES" ] && SEEN_RUNNING=1

  if [ "${avail:-0}" -ge "$AVAIL_STOP_COPY_GB" ]; then
    for d in "$LIVE"/checkpoint-*; do
      [ -d "$d" ] || continue
      b=$(basename "$d")
      if [ -f "$d/trainer_state.json" ] && [ ! -d "$PROT/$b" ]; then
        if rsync -a --quiet "$d" "$PROT/" 2>>"$LOGF"; then
          echo "[$ts] copied $b -> protected" >> "$LOGF"
        else
          echo "[$ts] COPY_FAIL $b" >> "$LOGF"
        fi
      fi
    done
  else
    echo "[$ts] skip copy: disk avail ${avail:-?}G < ${AVAIL_STOP_COPY_GB}G" >> "$LOGF"
  fi

  n=$(ls -1 "$PROT" 2>/dev/null | grep -cE '^checkpoint-[0-9]+$')
  if [ "$n" -gt "$KEEP" ]; then
    ls -1 "$PROT" | grep -E '^checkpoint-[0-9]+$' | sort -t- -k2 -n | head -n $((n - KEEP)) | while read -r old; do
      rm -rf "$PROT/$old" && echo "[$ts] pruned protected $old" >> "$LOGF"
    done
  fi

  if [ "$running" = "NO" ] && [ "$SEEN_RUNNING" = "1" ]; then
    echo "===== monitor (tvenc680) training exited $(date '+%F %T') =====" >> "$LOGF"
    echo "--- all val records ---" >> "$LOGF"
    grep 'teacher_forcing_mean_iou' "$TRAINLOG" >> "$LOGF" 2>&1
    best=$(grep 'teacher_forcing_mean_iou' "$TRAINLOG" 2>/dev/null | grep -oP '"teacher_forcing_mean_iou":\s*\K[0-9.]+' | sort -g | tail -1)
    echo "BEST_mean_iou=${best:-?}" >> "$LOGF"
    echo "--- trainlog tail ---" >> "$LOGF"
    tail -5 "$TRAINLOG" >> "$LOGF" 2>&1
    break
  fi
  if [ "$loops" -ge "$MAX_LOOPS" ]; then
    echo "===== monitor (tvenc680) hit MAX_LOOPS=$MAX_LOOPS, exiting $(date '+%F %T') =====" >> "$LOGF"
    break
  fi
  sleep "$INTERVAL"
done
