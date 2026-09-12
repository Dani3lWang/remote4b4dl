#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${B4DL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
NUSCENES_ROOT="${B4DL_NUSCENES_ROOT:-$(dirname "$PROJECT_ROOT")/nuScenes}"
cd "$PROJECT_ROOT/datageneration" || exit 1

python3 generate_description.py \
    --start_index 10 \
    --end_index 20 \
    --api_key {your openai api key} \
    --nuscenes_root "$NUSCENES_ROOT" \
    --dataroot "$PROJECT_ROOT/datageneration/data"
