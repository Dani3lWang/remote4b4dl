#!/bin/bash
# ============================================================================
# run_metatoken.sh — B4DL Metatoken 数据准备与训练/评测一键脚本
# ============================================================================
# 论文 §4.1 / Appendix C / Figure 6：在 question 前后拼接 Metatoken 前缀
# 格式（Figure 6）：
#   <4DLiDAR>
#   <video>
#   <question>
#   <meta> The metadata of the first frame is '...' and the metadata of the last frame is '...'
#
# 用法：
#   bash scripts/run_metatoken.sh \
#       --nuscenes_root /path/to/nuScenes \
#       --stage2_data ./b4dl_dataset/stage2_train.json \
#       --test_data ./b4dl_dataset/test_qa.json
#
# 消融实验：
#   # 仅有 <4DLiDAR> 无 <meta>
#   bash scripts/run_metatoken.sh ... --no_meta
#
#   # 仅有 <meta> 无 <4DLiDAR>
#   bash scripts/run_metatoken.sh ... --no_4dlidar
#
#   # 都无 (= baseline, 直接用原始数据)
#   bash scripts/run_metatoken.sh ... --no_4dlidar --no_meta
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLLM_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$MLLM_DIR")"

# ── 默认值 ──
NUSCENES_ROOT=""
STAGE2_DATA="${MLLM_DIR}/b4dl_dataset/stage2_train.json"
TEST_DATA="${MLLM_DIR}/b4dl_dataset/test_qa.json"
SCENE_META="${REPO_ROOT}/encoders/lidarclip/annotations/scene_metadata.json"
SEQ_META="${REPO_ROOT}/encoders/lidarclip/annotations/sequence_metadata.json"
EGO_META="${MLLM_DIR}/b4dl_dataset/ego_metadata.json"
STAGE2_OUT="${MLLM_DIR}/b4dl_dataset/stage2_train_meta.json"
TEST_OUT="${MLLM_DIR}/b4dl_dataset/test_qa_meta.json"
NO_4DLIDAR=false
NO_META=false
SKIP_EGO=false
DRY_RUN=false

# ── 解析参数 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --nuscenes_root) NUSCENES_ROOT="$2"; shift 2 ;;
        --stage2_data)   STAGE2_DATA="$2"; shift 2 ;;
        --test_data)     TEST_DATA="$2"; shift 2 ;;
        --scene_meta)    SCENE_META="$2"; shift 2 ;;
        --seq_meta)      SEQ_META="$2"; shift 2 ;;
        --ego_meta)      EGO_META="$2"; shift 2 ;;
        --stage2_out)    STAGE2_OUT="$2"; shift 2 ;;
        --test_out)      TEST_OUT="$2"; shift 2 ;;
        --no_4dlidar)    NO_4DLIDAR=true; shift ;;
        --no_meta)       NO_META=true; shift ;;
        --skip_ego)      SKIP_EGO=true; shift ;;
        --dry_run)       DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 --nuscenes_root <path> [options]"
            echo ""
            echo "Options:"
            echo "  --nuscenes_root <path>   nuScenes dataset root (required unless --skip_ego)"
            echo "  --stage2_data <path>     Stage2 training data JSON"
            echo "  --test_data <path>       Test QA data JSON"
            echo "  --no_4dlidar             Ablate: omit <4DLiDAR>"
            echo "  --no_meta                Ablate: omit <meta> + ego text"
            echo "  --skip_ego               Skip ego metadata generation (use existing)"
            echo "  --dry_run                Print commands without executing"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── 验证 ──
if [ "$SKIP_EGO" = false ] && [ -z "$NUSCENES_ROOT" ]; then
    echo "Error: --nuscenes_root required (or use --skip_ego if ego_metadata.json exists)"
    exit 1
fi

ABLATION_TAG="full"
if [ "$NO_4DLIDAR" = true ] && [ "$NO_META" = true ]; then
    ABLATION_TAG="baseline"
elif [ "$NO_4DLIDAR" = true ]; then
    ABLATION_TAG="meta_only"
elif [ "$NO_META" = true ]; then
    ABLATION_TAG="4dlidar_only"
fi
echo "Metatoken config: ${ABLATION_TAG}"

# ── Step 1: 生成 ego motion 元数据 ──
if [ "$SKIP_EGO" = false ]; then
    echo ""
    echo "=== Step 1/3: Generate ego motion metadata ==="
    CMD="python ${SCRIPT_DIR}/generate_ego_metadata.py \
        --nuscenes_root ${NUSCENES_ROOT} \
        --scene_metadata ${SCENE_META} \
        --sequence_metadata ${SEQ_META} \
        --output ${EGO_META}"
    echo "  $CMD"
    if [ "$DRY_RUN" = false ]; then
        eval "$CMD"
    fi
else
    echo "=== Step 1/3: Skip (using existing ${EGO_META}) ==="
    if [ ! -f "$EGO_META" ]; then
        echo "Error: ${EGO_META} not found. Remove --skip_ego to generate it."
        exit 1
    fi
fi

# ── Step 2: 注入 Metatoken 到训练数据 ──
echo ""
echo "=== Step 2/3: Inject Metatoken into training data ==="
INJECT_ARGS=""
if [ "$NO_4DLIDAR" = true ]; then INJECT_ARGS="$INJECT_ARGS --no_4dlidar"; fi
if [ "$NO_META" = true ]; then INJECT_ARGS="$INJECT_ARGS --no_meta"; fi

if [ -n "${STAGE2_DATA:-}" ] && [ -f "${STAGE2_DATA:-}" ]; then
    CMD="python ${SCRIPT_DIR}/inject_metatoken.py \
        --input ${STAGE2_DATA} \
        --ego_meta ${EGO_META} \
        --output ${STAGE2_OUT} ${INJECT_ARGS}"
    echo "  $CMD"
    if [ "$DRY_RUN" = false ]; then
        eval "$CMD"
    fi
    echo "  → Training data with Metatoken: ${STAGE2_OUT}"
else
    echo "  Skip: no stage2 training data at ${STAGE2_DATA:-}"
fi

# ── Step 3: 注入 Metatoken 到测试数据 ──
echo ""
echo "=== Step 3/3: Inject Metatoken into test data ==="
if [ -n "${TEST_DATA:-}" ] && [ -f "${TEST_DATA:-}" ]; then
    CMD="python ${SCRIPT_DIR}/inject_metatoken.py \
        --input ${TEST_DATA} \
        --ego_meta ${EGO_META} \
        --output ${TEST_OUT} ${INJECT_ARGS}"
    echo "  $CMD"
    if [ "$DRY_RUN" = false ]; then
        eval "$CMD"
    fi
    echo "  → Test data with Metatoken: ${TEST_OUT}"
else
    echo "  Skip: no test data at ${TEST_DATA:-}"
fi

# ── 完成 ──
echo ""
echo "============================================================"
echo "Metatoken pipeline complete! (config: ${ABLATION_TAG})"
echo "============================================================"
echo ""
echo "Format (paper Figure 6):"
echo "  <4DLiDAR>"
echo "  <video>"
echo "  <question>"
echo "  <meta> The metadata of the first frame is '...' and the metadata of the last frame is '...'"
echo ""
echo "Next steps:"
echo ""
echo "  1. Train Stage2 with Metatoken:"
echo "     bash scripts/stage2.sh --data_path ${STAGE2_OUT} ..."
echo ""
echo "  2. Infer + evaluate with Metatoken:"
echo "     python evaluation/test_b4dl.py \\"
echo "         --test_data ${TEST_OUT} \\"
echo "         ... (other args)"
echo ""
echo "  3. Or use --ego_meta at inference time:"
echo "     python evaluation/test_b4dl.py \\"
echo "         --test_data ${TEST_DATA} \\"
echo "         --ego_meta ${EGO_META} \\"
echo "         ... (other args)"
echo ""
echo "  4. Ablation: re-run with --no_4dlidar / --no_meta"
