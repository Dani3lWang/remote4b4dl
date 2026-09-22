#!/bin/bash
# 等 tvenc680 主训练退出后自动跑自由生成评测。
# dtype 组合固定 fp16 + 编码器 fp32: stage A 已证实整体 fp16 会让 spconv 报
# "can't find suitable algorithm"，而编码器留 fp32 时 2248 个 batch 零 non-finite。
set -u
export HF_HUB_OFFLINE=1 HF_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=/root/autodl-tmp/mmb4dl/mllm
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
PY=$E/bin/python3.10
CKPT=$REPO/checkpoints/reasonseg-tvenc680/checkpoint-best
MANIFEST=$REPO/reasonseg_data_trainval/reasonseg_val_thin.jsonl
OUT=$REPO/eval_results/reasonseg-tvenc680_encfp32
LOG=$REPO/training_logs/eval_tvenc680.log
MAX_SAMPLES=300          # 子集先拿信号; 要全量改 0
THRESHOLD=0.5
FREE_MB_GATE=14336       # fp16 + 上下文约需 12.5G
EVAL_TIMEOUT=10800       # 单次上限 3h
MAX_ATTEMPTS=20

echo "[$(date '+%F %T')] === tvenc680 auto-eval watcher started (pid=$$) ===" >> "$LOG"

# 1) 先确认训练进程确实在跑, 避免在启动间隙误判为已结束
for _ in $(seq 1 60); do
  pgrep -f 'scripts/train_reasonseg.py' >/dev/null 2>&1 && break
  sleep 30
done
while pgrep -f 'scripts/train_reasonseg.py' >/dev/null 2>&1; do sleep 60; done
echo "[$(date '+%F %T')] 训练进程已退出" >> "$LOG"

# 2) 等 best 落盘
ok=0
for _ in $(seq 1 60); do
  if [ -f "$CKPT/trainer_state.json" ]; then ok=1; break; fi
  sleep 15
done
if [ "$ok" != "1" ]; then echo "[$(date '+%F %T')] [fail] 未找到 $CKPT" >> "$LOG"; exit 1; fi
echo "[$(date '+%F %T')] best ckpt 就绪:" >> "$LOG"
ls -la "$CKPT" >> "$LOG" 2>&1

mkdir -p "$OUT"
run_eval() {
  timeout "$EVAL_TIMEOUT" "$PY" -u "$REPO/evaluation/evaluate_reasonseg.py" \
    --model-base "$REPO/base_model/vicuna-v1-5-7b" \
    --pretrain-mm-mlp-adapter "$REPO/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin" \
    --b3-checkpoint "$REPO/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3" \
    --seg-checkpoint "$CKPT" \
    --manifest "$MANIFEST" \
    --dataroot /root/autodl-tmp/nuScenes \
    --output-dir "$OUT" \
    --threshold "$THRESHOLD" \
    --max-samples "$MAX_SAMPLES" \
    --dtype fp16 --encoder-dtype fp32
}

att=0
while [ $att -lt $MAX_ATTEMPTS ]; do
  att=$((att+1))
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$free" ] && [ "$free" -ge "$FREE_MB_GATE" ] && break
    echo "[$(date '+%F %T')] 显存不足 (free=${free:-?}MB < ${FREE_MB_GATE}MB), 10min 后重试" >> "$LOG"
    sleep 600
  done
  echo "[$(date '+%F %T')] 评测尝试 #$att 开始 (max=$MAX_SAMPLES thr=$THRESHOLD dtype=fp16 enc=fp32)" >> "$LOG"
  run_eval >> "$LOG" 2>&1; rc=$?
  if [ $rc -eq 0 ] && [ -f "$OUT/metrics.json" ]; then
    echo "[$(date '+%F %T')] 评测成功 #$att" >> "$LOG"
    break
  fi
  echo "[$(date '+%F %T')] 评测失败 rc=$rc #$att, 10min 后重试" >> "$LOG"
  sleep 600
done

echo "===== 最终 metrics.json =====" >> "$LOG"
cat "$OUT/metrics.json" >> "$LOG" 2>&1
# token_failure_rate > 0 就是 NaN 污染的识别标志 (09-21 曾因此拿到偏乐观的假结果)
tfr=$(grep -oP '"token_failure_rate":\s*\K[0-9.]+' "$OUT/metrics.json" 2>/dev/null | head -1)
if awk -v t="${tfr:-0}" 'BEGIN{exit !(t>0)}'; then
  echo "[warn] token_failure_rate=$tfr > 0 —— 结果可能被 NaN 污染, 不可直接采用" >> "$LOG"
fi
echo "[$(date '+%F %T')] done, 结果: $OUT/metrics.json  masks: $OUT/masks/" >> "$LOG"
