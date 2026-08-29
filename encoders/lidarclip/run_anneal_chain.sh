#!/bin/bash
# 编码器退火链（2026-08-29，用户批准方案）
#   背景：高 LR 续训（并行会话 16:36 启动，延续原 OneCycle 105,500 步日程，lr~9.7e-4）
#   由 early_stop_monitor_v2 在 step>=31,550 硬停（或更早触发平台规则）——届时 LR 仍未
#   实质退火。本链在其结束后自动执行：
#     1) 基线 val MSE probe（15 个 val 场景确定性子集，452 批）
#     2) 短程退火：从 last.ckpt 只载权重（--load-only-model），OneCycle max_lr=1e-4→0，
#        3 epoch（15,825 步），seed=0
#     3) 对退火产物逐个 val probe → 以 val MSE 最低者为最终编码器候选
#   tmux 会话 b4dlanneal 内运行。只读等待当前训练，绝不杀进程。
set -u
cd /root/autodl-tmp/wql/mmb4dl/encoders/lidarclip
PY=/root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python
export WANDB_MODE=offline

PAT="train.py --name lidarclip_nuscenes"
WAIT_LIMIT_S=$((12*3600))
waited=0
echo "[$(date '+%F %T')] 等待高 LR 训练退出（上限 12h）..."
while pgrep -f "$PAT" > /dev/null; do
    sleep 300
    waited=$((waited+300))
    if [ $waited -ge $WAIT_LIMIT_S ]; then
        echo "[$(date '+%F %T')] 等待超 12h 仍在训练（可能被人工续期），放弃链式退火，请人工确认后重跑本脚本。"
        exit 2
    fi
done
echo "[$(date '+%F %T')] 高 LR 训练已退出，等待 ckpt 落盘..."
sleep 120

BASE_CKPT=ckpt_nuscenes/lidarclip_mm/last.ckpt
if [ ! -f "$BASE_CKPT" ]; then
    echo "错误: 未找到 $BASE_CKPT"
    exit 1
fi
ls -la "$BASE_CKPT"

echo "[$(date '+%F %T')] 基线 val MSE probe..."
$PY val_mse_probe.py --checkpoint "$BASE_CKPT" --tag baseline \
    >> logs/val_mse_probe.log 2>&1 || echo "WARN: 基线 probe 失败，继续退火"

# train_loss.csv 换名归档，退火阶段另起干净 CSV（步数从 0 起，避免混淆）
mv logs/train_loss.csv "logs/train_loss_highlr_$(date +%m%d_%H%M).csv" 2>/dev/null || true

echo "[$(date '+%F %T')] 启动短程退火: 3 epoch / OneCycle max_lr=1e-4→0 / seed=0 / load-only-model"
ANNEAL_LOG="logs/train_anneal_$(date +%m%d_%H%M).log"
$PY train.py --name lidarclip_anneal --checkpoint-save-dir ./ckpt_anneal \
    --batch-size 32 --workers 16 --data-dir /root/autodl-tmp/Datasets/nuScenes \
    --clip-model ViT-L/14 --dataset-name nuscenes \
    --checkpoint "$BASE_CKPT" --load-only-model \
    --max-epochs 3 --scheduler-max-lr 1e-4 --seed 0 \
    > "$ANNEAL_LOG" 2>&1
echo "[$(date '+%F %T')] 退火训练退出 rc=$? 日志: $ANNEAL_LOG"

echo "[$(date '+%F %T')] 对退火产物逐个 probe..."
for ck in $(ls ckpt_anneal/*/*.ckpt 2>/dev/null | sort); do
    echo "[$(date '+%F %T')] probe $ck"
    $PY val_mse_probe.py --checkpoint "$ck" \
        --tag "anneal/$(basename "$(dirname "$ck")")/$(basename "$ck" .ckpt)" \
        >> logs/val_mse_probe.log 2>&1 || echo "WARN: probe 失败 $ck"
done

echo "[$(date '+%F %T')] 链完成。val MSE 汇总（CSV 全部行）："
cat logs/val_mse_probe.csv 2>/dev/null
