#!/bin/bash
# ============================================================
# B4DL 训练结果验证脚本
#
# 用法:
#   bash scripts/verify.sh                          # Stage2 验证
#   bash scripts/verify.sh --stage3                 # Stage2 + Stage3 验证
#   bash scripts/verify.sh --quick                  # 快速验证（仅 100 条样本）
#   bash scripts/verify.sh --stage3 --quick         # Stage3 快速验证
# ============================================================
set -e

cd "$(dirname "$0")/.."

# ---------- 配置 ----------
MODEL_BASE="${MODEL_BASE:-./base_model/vicuna-v1-5-7b}"
MM_PROJECTOR="${MM_PROJECTOR:-./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin}"
STAGE2_CKPT="${STAGE2_CKPT:-./checkpoints/vtimellm-vicuna-v1-5-7b-stage2}"
STAGE3_CKPT="${STAGE3_CKPT:-./checkpoints/vtimellm-vicuna-v1-5-7b-stage3}"
FEAT_FOLDER="${FEAT_FOLDER:-../encoders/lidarclip/b4dl/stage2_features}"
EVAL_DIR="${EVAL_DIR:-./eval_results}"
QUICK_MODE=false
INCLUDE_STAGE3=false
WANDB_ENABLED=false

# ---------- 解析参数 ----------
for arg in "$@"; do
    case $arg in
        --stage3) INCLUDE_STAGE3=true ;;
        --quick) QUICK_MODE=true ;;
        --wandb) WANDB_ENABLED=true ;;
    esac
done

# ---------- 前置检查 ----------
echo "============================================"
echo "  B4DL 训练验证"
echo "============================================"

check_file() {
    if [ ! -e "$1" ]; then
        echo "[ERROR] 缺少文件: $1"
        echo "  提示: 请先完成对应阶段的训练"
        exit 1
    fi
}

echo ""
echo "[1/4] 检查模型文件..."
check_file "$MODEL_BASE"
echo "  ✓ base model: $MODEL_BASE"
check_file "$MM_PROJECTOR"
echo "  ✓ mm_projector: $MM_PROJECTOR"
check_file "$STAGE2_CKPT"
echo "  ✓ stage2 checkpoint: $STAGE2_CKPT"
if $INCLUDE_STAGE3; then
    check_file "$STAGE3_CKPT"
    echo "  ✓ stage3 checkpoint: $STAGE3_CKPT"
fi

echo ""
echo "[2/4] 检查特征文件..."
feat_count=$(ls "$FEAT_FOLDER"/*.npy 2>/dev/null | wc -l)
echo "  特征文件数: $feat_count"
if [ "$feat_count" -eq 0 ]; then
    echo "[ERROR] 特征文件为空，请先运行 LiDAR-CLIP 特征提取"
    exit 1
fi

echo ""
echo "[3/4] 检查测试数据..."
for f in b4dl_dataset/stage2_test.json b4dl_dataset/stage2_val.json; do
    if [ -f "$f" ]; then
        count=$(python3 -c "import json; print(len(json.load(open('$f'))))")
        echo "  ✓ $f ($count 条)"
    fi
done
if $INCLUDE_STAGE3; then
    for f in b4dl_dataset/stage3_test.json b4dl_dataset/stage3_val.json; do
        if [ -f "$f" ]; then
            count=$(python3 -c "import json; print(len(json.load(open('$f'))))")
            echo "  ✓ $f ($count 条)"
        fi
    done
fi

# ---------- 评估 ----------
echo ""
echo "[4/4] 开始评估..."
mkdir -p "$EVAL_DIR"

EXTRA_ARGS=""
if $QUICK_MODE; then
    EXTRA_ARGS="--max_samples 100"
    echo "  (快速模式: 仅评估 100 条)"
fi

# Stage2 Test 评估
STAGE2_LOG="$EVAL_DIR/stage2_test_log.jsonl"
echo ""
echo "--- Stage2 Test 评估 ---"
conda run -n wqlc python vtimellm/eval/b4dl_eval.py \
    --model_base "$MODEL_BASE" \
    --pretrain_mm_mlp_adapter "$MM_PROJECTOR" \
    --stage2 "$STAGE2_CKPT" \
    --data_path ./b4dl_dataset/stage2_test.json \
    --feat_folder "$FEAT_FOLDER" \
    --log_path "$STAGE2_LOG" \
    $EXTRA_ARGS

echo ""
echo "--- 计算 Stage2 指标 ---"
conda run -n wqlc python vtimellm/eval/b4dl_metrics.py \
    --log_path "$STAGE2_LOG" \
    --task stage2 \
    --output "$EVAL_DIR/stage2_metrics.json"

# 打印结果摘要
echo ""
echo "============================================"
echo "  Stage2 评估结果"
echo "============================================"
python3 -c "
import json
try:
    m = json.load(open('$EVAL_DIR/stage2_metrics.json'))
    for k, v in m.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.2f}')
        else:
            print(f'  {k}: {v}')
except Exception as e:
    print(f'  无法读取指标: {e}')
"

# Stage3 评估（可选）
if $INCLUDE_STAGE3; then
    STAGE3_LOG="$EVAL_DIR/stage3_test_log.jsonl"
    echo ""
    echo "--- Stage3 Test 评估 ---"
    conda run -n wqlc python vtimellm/eval/b4dl_eval.py \
        --model_base "$MODEL_BASE" \
        --pretrain_mm_mlp_adapter "$MM_PROJECTOR" \
        --stage2 "$STAGE2_CKPT" \
        --stage3 "$STAGE3_CKPT" \
        --data_path ./b4dl_dataset/stage3_test.json \
        --feat_folder "$FEAT_FOLDER" \
        --log_path "$STAGE3_LOG" \
        $EXTRA_ARGS

    echo ""
    echo "--- 计算 Stage3 指标 ---"
    conda run -n wqlc python vtimellm/eval/b4dl_metrics.py \
        --log_path "$STAGE3_LOG" \
        --task stage3 \
        --output "$EVAL_DIR/stage3_metrics.json"

    echo ""
    echo "============================================"
    echo "  Stage3 评估结果"
    echo "============================================"
    python3 -c "
import json
try:
    m = json.load(open('$EVAL_DIR/stage3_metrics.json'))
    for k, v in m.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.2f}')
        else:
            print(f'  {k}: {v}')
except Exception as e:
    print(f'  无法读取指标: {e}')
"
fi

echo ""
echo "============================================"
echo "  验证完成"
echo "============================================"
echo "Stage2 日志: $STAGE2_LOG"
echo "Stage2 指标: $EVAL_DIR/stage2_metrics.json"
if $INCLUDE_STAGE3; then
    echo "Stage3 日志: $STAGE3_LOG"
    echo "Stage3 指标: $EVAL_DIR/stage3_metrics.json"
fi

# 上传到 wandb（可选）
if $WANDB_ENABLED; then
    echo ""
    echo "--- 上传指标到 wandb ---"
    conda run -n wqlc python -c "
import json, wandb
wandb.init(project='B4DL', name='eval-$(date +%m%d-%H%M)')
m = json.load(open('$EVAL_DIR/stage2_metrics.json'))
wandb.log({f'eval/{k}': v for k, v in m.items() if isinstance(v, (int, float))})
if $INCLUDE_STAGE3:
    m3 = json.load(open('$EVAL_DIR/stage3_metrics.json'))
    wandb.log({f'eval_stage3/{k}': v for k, v in m3.items() if isinstance(v, (int, float))})
wandb.finish()
print('  已上传到 wandb')
"
fi
