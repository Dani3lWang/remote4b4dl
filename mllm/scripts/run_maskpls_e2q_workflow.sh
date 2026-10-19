#!/usr/bin/env bash
# Run only after deploying a byte-preserving, SHA256-verified source snapshot.
set -Eeuo pipefail
: "${E2Q_ROOT:?set the independent deployment root}"
: "${E2Q_PARENT_ROOT:=/root/autodl-tmp/mmb4dl-maskpls-improved-20261016}"
: "${E2Q_RUNTIME:=/root/autodl-tmp/mmb4dl-maskpls-20261016/runtime}"
: "${E2Q_DATA:=/root/autodl-tmp/nuScenes}"
: "${E2Q_OUT:=$E2Q_ROOT/training_logs/e2q-001}"
: "${E2Q_REFERENCE:=$E2Q_ROOT/reference}"
PYTHON="$E2Q_RUNTIME/wqlc/bin/python"
cd "$E2Q_ROOT"
test -f SOURCE_COMMIT.txt
test -f deployment_manifest.json
test ! -e "$E2Q_OUT"
mkdir -p workflow
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$E2Q_RUNTIME/wqlc/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$E2Q_ROOT/mllm" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export KEOPS_CACHE_FOLDER="$E2Q_RUNTIME/keops-cache"
trap 'rc=$?; printf "%s\n" "$rc" > workflow/exit_code.txt; date -Is > workflow/end_time.txt' EXIT
date -Is > workflow/start_time.txt
"$PYTHON" -c 'import hashlib,json,pathlib; r=pathlib.Path("."); m=json.loads((r/"deployment_manifest.json").read_text()); assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==s for p,s in m["files"].items()); print("DEPLOYMENT_VERIFIED",len(m["files"]))' > workflow/deployment.log 2>&1
printf 'cpu_contracts\n' > workflow/stage.txt
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m unittest discover -s mllm/maskpls_b4dl/tests -p 'test_*.py' -v > workflow/cpu-tests.log 2>&1
ARGS=(--parent-root "$E2Q_PARENT_ROOT" --out "$E2Q_OUT" --reference "$E2Q_REFERENCE" --data "$E2Q_DATA" --source-commit "$(cat SOURCE_COMMIT.txt)")
printf 'prepare\n' > workflow/stage.txt
CUDA_VISIBLE_DEVICES='' "$PYTHON" mllm/scripts/run_maskpls_e2q.py prepare "${ARGS[@]}" > workflow/prepare.log 2>&1
# GPU phases never start while no device is attached or another job is using it.
nvidia-smi --query-gpu=name --format=csv,noheader > workflow/gpu.txt
test -s workflow/gpu.txt
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
for stage in smoke run audit finalize; do
  printf '%s\n' "$stage" > workflow/stage.txt
  "$PYTHON" mllm/scripts/run_maskpls_e2q.py "$stage" "${ARGS[@]}" > "workflow/$stage.log" 2>&1
done
printf 'completed\n' > workflow/stage.txt
