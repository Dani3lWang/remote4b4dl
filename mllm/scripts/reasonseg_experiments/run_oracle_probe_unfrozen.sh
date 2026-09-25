#!/bin/bash
# oracle 探针第三跑：**解冻编码器**臂。
#
# 为什么要跑：v2（冻结编码器、6 epoch、lr 1e-4、选定 ep4）拿到 test AUC 0.99110 /
# recall_top_k 0.11058，按先验门槛（top-K 需 AUC >= 1-K/N ~= 0.99963）换算，压在
# 每个正点之上的负点约 289 个、而需要 <= 12，**仍差约 24 倍**。但 v2 的编码器是
# 冻结的，所以它量到的是"现有语义预训练特征 + 完美 query"的天花板，**不是"点特征
# 本身"的天花板** —— 那个编码器只在 lidarseg（语义）上训过，特征编码"这是车的点"，
# 从未被任何监督要求过"这是哪一辆车"。
#
# 本臂是唯一变量 = 编码器解冻（配独立的更低 lr），其余与 v2 逐字相同：同 manifest、
# 同 seed（故同洗牌顺序）、同 6 epoch、同 lr 1e-4、同 400 条评测、同 Tversky(0.3/0.7)
# + plain BCE、test 仍只在 dev iou_mean 选定的那一轮评一次。
#
# 判据（与 v2 同一个，先验推导、与观测无关）：
#   test AUC >= 0.99963  -> top-K 可用，点特征够，瓶颈在别处
#   test AUC 停在 0.99 量级 -> 单帧点特征确实撑不起实例接地；剩下的诚实选项是
#                             换输出目标（粗区域）、引入时序多帧、或如实报负结果
# 注意 (1-AUC) 逐轮已在 0.006-0.007 平台（v2），达标需 0.00037，故"再训久一点就能到"
# 这条基本可排除；本臂要回答的是"让特征自己适应实例目标"能不能到。
#
# 与 v2 的两处必然差异（不是 bug，是微调的固有属性）：
#   1. 解冻后编码器在训练期进 train 模式，dropout 生效（评测期仍 eval）。
#   2. 选轮快照必须连编码器权重一起存，否则 test 评的会是"末轮编码器 + 选定轮的
#      头"这个从未存在过的组合；脚本已处理，并把选定轮的编码器权重落盘，供下一步
#      （量 lidarseg 语义质量是否被实例目标冲掉、插回完整模型）使用。
set -uo pipefail
cd /root/autodl-tmp/mmb4dl/mllm || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
V=./reasonseg_data_trainval
L=./training_logs/phase23
mkdir -p "$L"

BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$BUSY" -ne 0 ]; then
  echo "!!! GPU 上仍有 $BUSY 个计算进程，不抢卡" | tee -a "$L/run_probe_v3.log"
  exit 1
fi

echo "[$(date '+%F %T')] === oracle 探针 v3：解冻编码器（6 epoch，约 2.8 h）===" | tee -a "$L/run_probe_v3.log"
$E/bin/python scripts/reasonseg_experiments/probe_oracle_query.py \
  --spatial-checkpoint ./checkpoints/reasonseg-spatial-tv/spatial-best \
  --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
  --dev-manifest "$V/reasonseg_val_internal_es.jsonl" \
  --test-manifest "$V/reasonseg_val_thin.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output ./eval_results/_oracle_probe_v3_unfrozen/report.json \
  --epochs 6 --learning-rate 1e-4 --eval-records 400 \
  --unfreeze-encoder --encoder-lr 1e-5 \
  --bce-mode plain --region-loss tversky --tversky-alpha 0.3 --tversky-beta 0.7 \
  > "$L/probe_v3.log" 2>&1
RC=$?
echo "[$(date '+%F %T')] probe_v3 rc=$RC" | tee -a "$L/run_probe_v3.log"
exit $RC
