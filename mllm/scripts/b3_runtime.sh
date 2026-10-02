#!/bin/bash
# Shared single-GPU B3 profiles; source from train_b3.sh / run_b3.sh.
b3_configure_runtime() {
    B3_PROFILE=${B4DL_TRAIN_PROFILE:-baseline}
    B3_GPU_ID=${B4DL_GPU_ID:-0}
    case "$B3_PROFILE" in
        baseline)
            B3_MICRO_BATCH=8
            B3_GRAD_ACCUM=16
            B3_ZERO_CONFIG=./scripts/zero3.json
            B3_MIN_FREE_MB=28000
            B3_OUTPUT_SUFFIX=""
            ;;
        rtx4090)
            B3_MICRO_BATCH=1
            B3_GRAD_ACCUM=128
            B3_ZERO_CONFIG=./scripts/zero3_offload.json
            B3_MIN_FREE_MB=20000
            B3_OUTPUT_SUFFIX=-rtx4090
            ;;
        *) echo "错误: B4DL_TRAIN_PROFILE 只支持 baseline / rtx4090" >&2; return 1 ;;
    esac
    B3_MIN_FREE_MB=${B4DL_TRAIN_MIN_FREE_MB:-$B3_MIN_FREE_MB}
    if ! [[ "$B3_GPU_ID" =~ ^[0-9]+$ ]] || ! [[ "$B3_MIN_FREE_MB" =~ ^[1-9][0-9]*$ ]]; then
        echo "错误: B4DL_GPU_ID 必须非负，B4DL_TRAIN_MIN_FREE_MB 必须为正整数" >&2
        return 1
    fi
}

b3_gpu_memory() {
    local kind=$1 value
    value=$(nvidia-smi --id="$B3_GPU_ID" --query-gpu="memory.$kind" --format=csv,noheader,nounits) || return 1
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "错误: 无法读取 GPU $B3_GPU_ID 的 memory.$kind: $value" >&2
        return 1
    fi
    echo "$value"
}

b3_check_gpu_capacity() {
    local required=$1 total
    total=$(b3_gpu_memory total) || return 1
    if [ "$required" -gt "$total" ]; then
        echo "错误: GPU $B3_GPU_ID 总显存 ${total}MB 小于门槛 ${required}MB，不能等待满足。4090 请设置 B4DL_TRAIN_PROFILE=rtx4090。" >&2
        return 1
    fi
}
