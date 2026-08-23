#!/bin/bash
# after_seqctx_train.sh — seqctx 训练完成后的自动后续处理
# 等待 stage2-full-seqctx 训练进程退出 → 校验最终 adapter → merge → 全量评测(--per_sequence)
cd /root/autodl-tmp/wql/mmb4dl/mllm
eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)"
conda activate wqlc

LOG=./training_logs/after_seqctx_watch.log
echo "[watch] start: $(date)" | tee -a $LOG

# 1. 等待训练进程退出（匹配 train_mem.py 且 output_dir 为 seqctx 的进程）
while pgrep -f "train_mem.py.*stage2-full-seqctx" > /dev/null; do
  sleep 300
done
echo "[watch] training process exited: $(date)" | tee -a $LOG

# 2. 校验最终 adapter 是否生成
CKPT=./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqctx
if [ ! -f "$CKPT/adapter_model.safetensors" ]; then
  echo "[watch] ERROR: 最终 adapter_model.safetensors 不存在，训练可能异常中断" | tee -a $LOG
  ls -la "$CKPT" | tee -a $LOG
  exit 1
fi
echo "[watch] final adapter OK" | tee -a $LOG

# 3. 合并 LoRA 权重为完整模型（供后续 stage3 / 部署使用）
echo "[watch] merging: $(date)" | tee -a $LOG
python scripts/merge_stage2.py \
  --model_base ./base_model/vicuna-v1-5-7b \
  --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --stage2 "$CKPT" \
  --output_dir ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqctx-merged 2>&1 | tail -5 | tee -a $LOG

# 4. 全量评测（30,145 条，--per_sequence：per-seq ego + 特征切片 + 帧范围前缀 prompt）
mkdir -p ./eval_results/stage2_full_seqctx
echo "[watch] full eval start: $(date)" | tee -a $LOG
python evaluation/test_b4dl.py \
  --model_base ./base_model/vicuna-v1-5-7b \
  --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --stage2 "$CKPT" \
  --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
  --test_data ./b4dl_dataset/test_qa.json \
  --ego_meta ./b4dl_dataset/ego_metadata.json \
  --per_sequence \
  --output ./eval_results/stage2_full_seqctx/predictions.json \
  --metrics_output ./eval_results/stage2_full_seqctx/metrics.json \
  2>&1 | tee ./eval_results/stage2_full_seqctx/eval_log.txt | tail -40 | tee -a $LOG

echo "[watch] all done: $(date)" | tee -a $LOG
