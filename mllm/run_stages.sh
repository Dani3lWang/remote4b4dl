#!/bin/bash

# Exit immediately if any command fails
set -e

# --- Auto-detect CUDA_HOME ---
# Even if CUDA_HOME is set, verify nvcc actually exists there
_cuda_found=false

_find_nvcc() {
    # Priority 1: nvcc in PATH
    if command -v nvcc &> /dev/null; then
        export CUDA_HOME=$(dirname $(dirname $(command -v nvcc)))
        _cuda_found=true
        return
    fi
    # Priority 2: CUDA_HOME already set AND valid
    if [ -n "$CUDA_HOME" ] && [ -f "$CUDA_HOME/bin/nvcc" ]; then
        _cuda_found=true
        return
    fi
    # Priority 3: conda env's cuda packages
    if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/nvcc" ]; then
        export CUDA_HOME="$CONDA_PREFIX"
        _cuda_found=true
        return
    fi
    # Priority 4: common install paths
    for _ver in 12.8 12 12.6 12.5 12.4 12.3 12.2 12.1 11.8; do
        if [ -f "/usr/local/cuda-$_ver/bin/nvcc" ]; then
            export CUDA_HOME="/usr/local/cuda-$_ver"
            _cuda_found=true
            return
        fi
    done
    # Priority 5: default /usr/local/cuda (only if valid)
    if [ -f "/usr/local/cuda/bin/nvcc" ]; then
        export CUDA_HOME=/usr/local/cuda
        _cuda_found=true
        return
    fi
}

_find_nvcc

if [ "$_cuda_found" = false ]; then
    echo "[WARN] nvcc not found. Installing cuda-nvcc via conda..." >&2
    conda install -c nvidia cuda-nvcc -y
    if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/nvcc" ]; then
        export CUDA_HOME="$CONDA_PREFIX"
        _cuda_found=true
    fi
fi

if [ "$_cuda_found" = false ]; then
    echo "[ERROR] Cannot find or install nvcc." >&2
    echo "Please set CUDA_HOME to your CUDA installation path, e.g.:" >&2
    echo "  export CUDA_HOME=/usr/local/cuda-12.4" >&2
    exit 1
fi

echo "Using CUDA_HOME=$CUDA_HOME"

# --- Initialize Variables ---
COMMON_ARGS=() # Arguments applied to ALL stages (e.g., learning_rate, num_epochs)
S1_ARGS=()     # Arguments specific to Stage 1
S2_ARGS=()     # Arguments specific to Stage 2
S3_ARGS=()     # Arguments specific to Stage 3

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        # [Stage 1 Specific Options]
        --s1_data)
            S1_ARGS+=("--data_path" "$2") # Maps to --data_path
            shift 2
            ;;
        --s1_feat)
            S1_ARGS+=("--feat_folder" "$2") # Maps to --feat_folder
            shift 2
            ;;
        --s1_model)
            S1_ARGS+=("--model_name_or_path" "$2") # Maps to --model_name_or_path
            shift 2
            ;;

        # [Stage 2 Specific Options]
        --s2_data)
            S2_ARGS+=("--data_path" "$2")
            shift 2
            ;;
        --s2_feat)
            S2_ARGS+=("--feat_folder" "$2")
            shift 2
            ;;
        --s2_model)
            S2_ARGS+=("--model_name_or_path" "$2")
            shift 2
            ;;

        # [Common Options] 
        # Any flags not matched above (e.g., --learning_rate, --model_name_or_path)
        # are added here and passed to ALL active stages.
        *)
            COMMON_ARGS+=("$1")
            shift
            ;;
    esac
done

# --- Execution Block ---

# [Stage 1]
echo "Running stage 1..."
bash scripts/stage1.sh "${COMMON_ARGS[@]}" "${S1_ARGS[@]}"

# [Stage 2]
echo "Running stage 2..."
# Combine Stage 2 specific args with common args.
# Common args come last to allow overriding defaults if necessary.
bash scripts/stage2.sh "${COMMON_ARGS[@]}" "${S2_ARGS[@]}"


echo "All stages completed successfully!"


################## run example ##################
# bash runner.sh \
#     --s1_data ./lidarllm_only_dataset/stage1_lidarllm_mm.json \
#     --s1_feat ./lidarclip/stage1_features \
#     --s2_data ./b4dl_dataset/stage2.json \
#     --s2_feat ./b4dl/stage2_features \
#     --model_name_or_path ./base_model/vicuna-v1-5-7b
################## run example ##################
