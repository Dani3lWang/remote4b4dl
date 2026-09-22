#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/mmb4dl/mllm
export PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
$E/bin/python scripts/build_reasonseg_scene_subset.py \
  --dataroot /root/autodl-tmp/nuScenes --version v1.0-trainval \
  --output ./reasonseg_data_trainval/reasonseg_train_680x11.jsonl \
  --frames-per-scene 11 --per-frame-queries 1 \
  --exclude-scenes ./reasonseg_trainval/scenes_reasonseg_val.json \
  --seed 20260921
