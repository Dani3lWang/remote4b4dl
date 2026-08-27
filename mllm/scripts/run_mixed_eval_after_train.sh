#!/bin/bash
# 等待 stage2-full-seqv3-mixed 训练结束后自动跑论文对齐评测（6 任务 × 7 指标）
# 与 stage23_seqv3 同口径：--per_sequence --frame_motion --sequence_metadata --answer_frames
# 用法: bash scripts/run_mixed_eval_after_train.sh  （nohup/后台运行）
set -u

cd "$(dirname "$0")/.."

TRAIN_PID=885593
CKPT_DIR=./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed
EVAL_OUT=./eval_results/stage2_full_seqv3_mixed
EVAL_LOG=./training_logs/eval_stage2_full_seqv3_mixed.log

echo "[$(date '+%F %T')] 等待训练进程 $TRAIN_PID 退出..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] 训练进程已退出"

# 等待最终 adapter 落盘（train.py 末尾 model.save_pretrained 到输出目录根）
for i in $(seq 1 60); do
    if [ -f "$CKPT_DIR/adapter_model.safetensors" ]; then break; fi
    sleep 30
done
if [ ! -f "$CKPT_DIR/adapter_model.safetensors" ]; then
    echo "[$(date '+%F %T')] 错误: 未找到 $CKPT_DIR/adapter_model.safetensors"
    exit 1
fi
echo "[$(date '+%F %T')] 最终 adapter 已就绪:"
ls -la "$CKPT_DIR"/adapter_model.safetensors "$CKPT_DIR"/adapter_config.json

# 训练日志尾部（确认正常收尾）
tail -3 ./training_logs/stage2_full_seqv3_mixed_20260826_131938.log

mkdir -p "$EVAL_OUT"

# 评测主命令（与 stage23_seqv3 同参数，仅单 --stage2；GPU 若被 CoRViD 占满则每 30 分钟重试）
PY=python3.10
if [ -x /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python3.10 ]; then
    PY=/root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python3.10
fi

# 关键：HF Hub 网络在本机被卡死（curl huggingface.co 无响应），
# transformers/peft 加载模型时的 hub 版本检查会让进程永久挂起。
# 全部走本地文件，禁止任何外网访问。
export HF_HUB_OFFLINE=1
export HF_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 单次尝试上限 5 小时（正常评测约 3h15m）；超时=挂起，直接失败进入重试
EVAL_TIMEOUT=18000

run_eval() {
    timeout "$EVAL_TIMEOUT" "$PY" -u evaluation/test_b4dl.py \
        --model_base ./base_model/vicuna-v1-5-7b \
        --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
        --stage2 "$CKPT_DIR" \
        --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
        --test_data ./b4dl_dataset/test_qa.json \
        --ego_meta ./b4dl_dataset/ego_metadata.json \
        --frame_motion ./b4dl_dataset/ego_frame_motion.json \
        --sequence_metadata ../encoders/lidarclip/annotations/sequence_metadata.json \
        --per_sequence \
        --answer_frames \
        --output "$EVAL_OUT/predictions.json" \
        --metrics_output "$EVAL_OUT/metrics.json"
}

ATTEMPT=0
MAX_ATTEMPTS=30
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT+1))

    # 显存门控：评测需 ~12.5GB（模型 fp16 + CUDA 上下文），CoRViD 占 19GB 时
    # 只有 ~12.6GB 空闲，必定 OOM。空闲 <14GB 时休眠 10 分钟再试，
    # 避免无谓的模型加载（每次 ~15min 全核占用，还会拖慢 CoRViD）。
    while :; do
        FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$FREE_MB" ] && [ "$FREE_MB" -ge 14336 ]; then
            break
        fi
        echo "[$(date '+%F %T')] 显存不足 (空闲 ${FREE_MB:-?}MB < 14GB)，10 分钟后重试..."
        sleep 600
    done

    echo "[$(date '+%F %T')] 评测尝试 #$ATTEMPT 开始..."
    run_eval >> "$EVAL_LOG" 2>&1
    RC=$?
    if [ $RC -eq 0 ]; then
        echo "[$(date '+%F %T')] 评测完成 (尝试 #$ATTEMPT)"
        break
    fi
    echo "[$(date '+%F %T')] 评测失败 (rc=$RC, 尝试 #$ATTEMPT)，10 分钟后重试..."
    tail -5 "$EVAL_LOG"
    sleep 600
done

echo ""
echo "===== 评测结果 ====="
cat "$EVAL_OUT/metrics.json" 2>/dev/null
echo ""
echo "日志: $EVAL_LOG"
