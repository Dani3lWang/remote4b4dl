#!/bin/bash
# 等 oracle 探针 v3（tmux p24）收尾后，用**修好口径的**接地诊断复算三个档。
#
# 为什么要等：诊断要载 7B（fp16 约 19G），探针占 2.7G，同时跑会顶到 24G 上限、
# 有 OOM 打死探针的风险，而探针已经跑了几个小时。
#
# 为什么现在才改口径：diagnose_reasonseg_grounding.py 原先把"框内点数"当"前缀长度"
# 用（[:point_count]），实测越界点是散布的、尾部集中度只有 0.0585，所以前缀里混着约
# 2035 个概率恒 0 的越界点（抬高 AUC）、又丢掉尾部同样多的框内点，还把 2.42% 物体的
# GT 截成空并被 `if target_size == 0: continue` 静默吞掉。Phase 2.3 的预注册判据是按
# 旧口径定的，裁定前换口径等于移动球门，所以一直压着（待办 #20）。Phase 2.3 已于
# 2026-09-25 18:21 裁定完，现在可以改了。
#
# 旧报告一律保留、新报告写到 *_inrange/，这样"同一批档、两种口径"可以并排比。
# 失真对所有 run 完全相同，所以 run 之间的比较本来就成立；这次要拿的是**无偏的绝对值**。
set -uo pipefail
cd /root/autodl-tmp/mmb4dl/mllm || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
V=./reasonseg_data_trainval
L=./training_logs/phase23
W=$L/watcher_rediagnose.log
mkdir -p "$L"

log () { echo "[$(date '+%F %T')] $*" | tee -a "$W"; }

BASE="--model-base ./base_model/vicuna-v1-5-7b
--pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin
--b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3
--manifest $V/reasonseg_val_thin.jsonl
--dataroot /root/autodl-tmp/nuScenes
--max-samples 1000 --dtype fp16 --encoder-dtype fp32"

log "=== 等 oracle 探针 v3（p24）收尾 ==="
i=0
while [ $i -lt 300 ]; do
  if grep -q "probe_v3 rc=" "$L/run_probe_v3.log" 2>/dev/null; then
    log "探针已结束：$(grep 'probe_v3 rc=' "$L/run_probe_v3.log" | tail -1)"
    break
  fi
  if ! tmux has-session -t p24 2>/dev/null; then
    if grep -q "probe_v3 rc=" "$L/run_probe_v3.log" 2>/dev/null; then
      log "探针已结束且会话已退出"
      break
    fi
    log "!!! p24 会话消失但未见 rc，判为异常终止，不抢卡"
    echo "NOGO probe_died" > "$L/rediagnose_decision.txt"
    exit 1
  fi
  i=$((i + 1))
  sleep 60
done
if [ $i -ge 300 ]; then
  log "!!! 等待超时（5 小时），不启动"
  echo "NOGO wait_timeout" > "$L/rediagnose_decision.txt"
  exit 1
fi

BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$BUSY" -ne 0 ]; then
  log "!!! GPU 上仍有 $BUSY 个计算进程，不抢卡"
  echo "NOGO gpu_busy_$BUSY" > "$L/rediagnose_decision.txt"
  exit 1
fi
echo "GO rediagnose" > "$L/rediagnose_decision.txt"

# 冒烟中止门：新口径没在真模型上跑过。5 条足够验证三件事——不崩、config 里出现
# skipped_unreachable、每物体的 point_count 等于框内点数（而不是前缀长度）。
log "=== S0/4 冒烟：修好口径的诊断（5 条）==="
$E/bin/python scripts/reasonseg_experiments/diagnose_reasonseg_grounding.py \
  $BASE --seg-checkpoint ./checkpoints/reasonseg-internal679/checkpoint-best \
  --output "$L/_smoke_rediagnose/report.json" --max-samples 5 \
  > "$L/s0_smoke_rediagnose.log" 2>&1
S0=$?
log "S0 rc=$S0"
if [ $S0 -ne 0 ]; then
  log "!!! 冒烟失败，终止。尾部："
  tr '\r' '\n' < "$L/s0_smoke_rediagnose.log" | tail -20 | tee -a "$W"
  exit 1
fi
if ! grep -q "skipped_unreachable" "$L/_smoke_rediagnose/report.json"; then
  log "!!! 报告里没有 skipped_unreachable，新口径没生效，终止"
  exit 1
fi
$E/bin/python - <<'PY' | tee -a "$W"
import json
rep = json.load(open("training_logs/phase23/_smoke_rediagnose/report.json"))
rows = [json.loads(l) for l in open("training_logs/phase23/_smoke_rediagnose/report.per_object.jsonl")]
print(f"冒烟：objects={rep['config']['objects']} "
      f"skipped_unreachable={rep['config']['skipped_unreachable']}")
print(f"逐物体 point_count 取值集合={sorted({r['point_count'] for r in rows})}")
print(f"target_size 最小值={min(r['target_size'] for r in rows)}（应 >0，为 0 的已被计数跳过）")
PY

for name in internal679 reasonseg-lossA2-plain-tversky-ext20 reasonseg-lossB1-noprior-tversky; do
  log "=== 复算 $name（新口径，val_thin 1000 条）==="
  $E/bin/python scripts/reasonseg_experiments/diagnose_reasonseg_grounding.py \
    $BASE --seg-checkpoint "./checkpoints/$name/checkpoint-best" \
    --output "./eval_results/_grounding_${name}_inrange/report.json" \
    > "$L/rediagnose_${name}.log" 2>&1
  log "$name rc=$?"
done

log "=== REDIAGNOSE DONE ==="
$E/bin/python - <<'PY' | tee -a "$W"
import json, pathlib
print(f"{'run':<40} {'口径':<10} {'n':>5} {'AUC':>8} {'R@0.5':>7} {'topK':>7} {'尺寸':>8}")
for name in ("internal679", "reasonseg-lossA2-plain-tversky-ext20",
             "reasonseg-lossB1-noprior-tversky"):
    for tag, path in (("旧/前缀", f"eval_results/_grounding_{name}/report.json"),
                      ("新/in-range", f"eval_results/_grounding_{name}_inrange/report.json")):
        p = pathlib.Path(path)
        if not p.exists():
            continue
        o = json.load(p.open())["overall"]
        print(f"{name:<40} {tag:<10} {o['n']:>5} {o['auc_mean']:>8.4f} "
              f"{o['recall_thresholded']:>7.4f} {o['recall_top_k']:>7.4f} "
              f"{o['predicted_size_median']:>3.0f}/{o['target_size_median']:.0f}")
PY
