#!/bin/bash
# GRPO 时间定位训练链（单进程单卡，设计成跑在一个 tmux 会话里，断链不影响）。
# 基底固定为 B3（当前最优 acc 0.7526 / mIoU 0.3467）；RL-LoRA 叠加在「已合并 B3」之上，
# 参考策略 = 关掉 RL-LoRA 的同一模型（model.disable_adapter()），不再常驻第二份 14G 权重。
#
#   阶段 m31   训练侧冒烟：1 步 P=4/G=4。门控=backward 流通(grad_norm>0)、ratio==1
#              (ratio_dev<1e-3，即 G2.4c 在 trainer 内的实证)、fp32 master 真的更新
#              (lora_delta>0；纯 bf16 在 lr=1e-5 下会被 ULP 抹成 0)。
#   阶段 m32   短训 100 步 P=16/G=8。门控=跑满 100 步且 loss 无 NaN；同时打印奖励首末、
#              退化组率、中心 std 供人工看是否坍缩。
#   阶段 full  1 epoch P=32/G=8 正式训练（--resume 断点续训，save-total-limit=2 滚动删档）。
#   阶段 eval  M3.3 同口径评测：test_b4dl 支持 --stage3，在「已合并 B3(stage2)」之上再合并
#              RL-LoRA(stage3)，复用冻结评测环（--whole_scene --per_sequence --answer_frames、
#              fp16，与 B3 的 mIoU 0.3467 逐位同口径）。已用 m31 单步档实证双合并可加载并出
#              metrics.json，故 M3.3 无需新写评测入口。报数须带先验地板 0.2998 + 黑客面 +
#              中心熵/众数 + --answer_frames 的 oracle 输入选择声明。
#
# 硬约束：奖励只调 evaluation.evaluate_model（不重写）；只用训练集 rl_tg_train.jsonl；
# logp_old/logp_ref/logp_actor 全走 path B(forward_sequence)，绝不用 generate 的 scores。
# 每阶段前查磁盘，余量 < MIN_FREE_GB 不开训（RL-LoRA 单档 fp32 约 0.61G——save_embedding_layers
# =False 已剔掉冻结 embed/lm_head 的 0.55G；save-total-limit=2 滚动，留足评测/中间产物）。
set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${B4DL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WQLC_PREFIX="${B4DL_ENV_PREFIX:-$(dirname "$PROJECT_ROOT")/.conda-stuff/envs/wqlc}"
PY="$WQLC_PREFIX/bin/python"
[ -x "$PY" ] || { echo "错误: wqlc 环境不存在: $PY" >&2; exit 1; }
export PATH="$WQLC_PREFIX/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline PYTHONUNBUFFERED=1
# evaluate_model 导入时会 _ensure_nltk_data()，缺语料会触发 nltk.download 联网卡死约 7 分钟；
# 指向项目内已暂存的 nltk_data 即可秒过（见 training_logs 里 M2 冒烟的教训）。
export NLTK_DATA="$PROJECT_ROOT/nltk_data"
cd "$PROJECT_ROOT/mllm" || exit 1
mkdir -p training_logs

B3_DIR=checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3
RL_DATA=b4dl_dataset/rl_tg_train.jsonl
SMOKE_JSON=training_logs/smoke_m2.json
MIN_FREE_GB=8
STAGE="${1:-m31}"

avail_gb() { local a; a=$(df -BG --output=avail "$PROJECT_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9'); echo "${a:-999}"; }

run_trainer() {   # $@ 透传给 grpo_trainer
    "$PY" -u -m vtimellm.rl.grpo_trainer --stage2 "./$B3_DIR" --data "$RL_DATA" "$@"
}

echo "########## GRPO-TG 链 阶段=$STAGE 开始 $(date '+%F %T') ##########"
echo "项目: $PROJECT_ROOT  环境: $WQLC_PREFIX  磁盘余量: $(avail_gb)G"

# ── 前置门：B3 / RL 数据 / M2 冒烟全绿 / 磁盘 ──
[ -f "$B3_DIR/adapter_model.safetensors" ] || { echo "缺 B3 adapter: $B3_DIR"; exit 1; }
[ -s "$RL_DATA" ] || { echo "缺 RL 数据 $RL_DATA（先跑 python -m vtimellm.rl.build_rl_data）"; exit 1; }
SMOKE_OK=$(SMOKE="$SMOKE_JSON" "$PY" -c "import json,os,sys; sys.exit(0 if os.path.exists(os.environ['SMOKE']) and json.load(open(os.environ['SMOKE'])).get('all_pass') else 1)" && echo 1 || echo 0)
[ "$SMOKE_OK" = "1" ] || { echo "M2 冒烟未全绿（$SMOKE_JSON all_pass!=true），不跑 trainer——这是方案的硬门"; exit 1; }
FREE=$(avail_gb); [ "$FREE" -ge "$MIN_FREE_GB" ] || { echo "磁盘余量 ${FREE}G < ${MIN_FREE_GB}G，不开训"; exit 1; }
echo "前置门通过：B3 / RL 数据 / M2 全绿 / 磁盘 ${FREE}G"

case "$STAGE" in
  m31)
    OUT=checkpoints/grpo_tg_m31
    rm -rf "$OUT"; mkdir -p "$OUT"
    echo "===== M3.1 训练侧冒烟 1 步 (P=4/G=4) $(date '+%F %T') ====="
    run_trainer --out-dir "$OUT" --max-steps 1 --prompts-per-step 4 --group-size 4 \
        --micro-batch 4 --save-steps 1 --log-steps 1 2>&1 | tee training_logs/grpo_m31.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "M3.1 trainer 非 0 退出，停止"; exit 1; }
    "$PY" -m vtimellm.rl.check_gate "$OUT/grpo_log.jsonl" m31 \
        || { echo "M3.1 门控未过，停止（不写/不跑 m32/full）"; exit 1; }
    echo "M3.1 通过：trainer 机械正确，可进 M3.2"
    ;;
  m32)
    OUT=checkpoints/grpo_tg_m32
    rm -rf "$OUT"; mkdir -p "$OUT"
    echo "===== M3.2 短训 100 步 (P=16/G=8) $(date '+%F %T') ====="
    run_trainer --out-dir "$OUT" --max-steps 100 --prompts-per-step 16 --group-size 8 \
        --micro-batch 4 --save-steps 50 --log-steps 5 2>&1 | tee training_logs/grpo_m32.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "M3.2 trainer 非 0 退出，停止"; exit 1; }
    "$PY" -m vtimellm.rl.check_gate "$OUT/grpo_log.jsonl" m32 \
        || { echo "M3.2 门控未过，停止"; exit 1; }
    echo "M3.2 通过：100 步无 NaN，趋势可继续；下一步 full 1 epoch"
    ;;
  full)
    OUT=checkpoints/grpo_tg
    mkdir -p "$OUT"
    echo "===== GRPO 正式训练 1 epoch (P=32/G=8, --resume) $(date '+%F %T') ====="
    run_trainer --out-dir "$OUT" --epochs 1 --prompts-per-step 32 --group-size 8 \
        --micro-batch 4 --save-steps 50 --save-total-limit 2 --log-steps 5 --resume \
        2>&1 | tee -a training_logs/grpo_full.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "full 训练非 0 退出（可重跑本阶段断点续训）"; exit 1; }
    "$PY" -m vtimellm.rl.check_gate "$OUT/grpo_log.jsonl" full
    echo "训练结束 -> $OUT；下一步 bash scripts/run_grpo_tg.sh eval 跑 M3.3 同口径评测"
    ;;
  eval)
    OUT=checkpoints/grpo_tg
    EVAL_OUT=eval_results/grpo_tg
    [ -f "$OUT/trainer_state.json" ] || { echo "缺 $OUT/trainer_state.json，full 未跑完"; exit 1; }
    STEP=$("$PY" -c "import json;print(json.load(open('$OUT/trainer_state.json'))['global_step'])")
    RL="$OUT/rl_lora_step$STEP"
    [ -f "$RL/adapter_model.safetensors" ] || { echo "缺 RL adapter: $RL"; exit 1; }
    mkdir -p "$EVAL_OUT"
    echo "===== M3.3 同口径评测 --stage2 B3 --stage3 rl_lora_step$STEP $(date '+%F %T') ====="
    timeout 21600 "$PY" -u evaluation/test_b4dl.py \
        --model_base ./base_model/vicuna-v1-5-7b \
        --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
        --stage2 "./$B3_DIR" --stage3 "$RL" \
        --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
        --test_data ./b4dl_dataset/test_qa.json \
        --ego_meta ./b4dl_dataset/ego_metadata.json \
        --frame_motion ./b4dl_dataset/ego_frame_motion.json \
        --whole_scene --per_sequence --answer_frames \
        --output "$EVAL_OUT/predictions.json" \
        --metrics_output "$EVAL_OUT/metrics.json" \
        2>&1 | tee "$EVAL_OUT/eval_log.txt"
    [ "${PIPESTATUS[0]}" -eq 0 ] && [ -s "$EVAL_OUT/metrics.json" ] \
        || { echo "M3.3 评测失败（非 0 退出或 metrics.json 空）"; exit 1; }
    echo "M3.3 完成 -> $EVAL_OUT/metrics.json；对照 B3 mIoU 0.3467 / acc 0.7526（SIGMA 0.013）。"
    echo "报数须带：先验地板 0.2998、奖励黑客面、中心熵/众数占比，并声明 --answer_frames 是 oracle 输入选择。"
    ;;
  *)
    echo "用法: bash scripts/run_grpo_tg.sh {m31|m32|full|eval}"; exit 2 ;;
esac

echo "########## 阶段=$STAGE 结束 $(date '+%F %T') ##########"
