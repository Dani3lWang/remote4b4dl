# 这是一个只读勘察任务，不要修改任何文件。项目根目录 /root/autodl-tmp/wql/mmb4dl（B4D...

| 项 | 值 |
|---|---|
| 会话 ID | `sess_subagent_agent_6876dee9-d242-4ead-b4d5-359a61b3fb94` |
| 工作目录 | `/root/autodl-tmp/wql/mmb4dl` |
| 父会话 | `sess_4814c410-0a8c-4f8a-bace-414dfacba965` |
| 时间 | 2026-09-07 21:10 → 2026-09-07 21:12 |
| 模型 | deepseek-v4-flash |
| 消息数 | 12（文本块 7，工具调用 26） |

---

### ASSISTANT  ·  `deepseek-v4-flash`

> 🔀 模型切换：None → 2801678c-7b5c-456f-ad57-1e9cb4a76bba/deepseek-v4-flash


### USER

这是一个只读勘察任务，不要修改任何文件。项目根目录 /root/autodl-tmp/wql/mmb4dl（B4DL 复现项目，评测在 mllm/evaluation/）。我要设计"用 NLTK METEOR 后端评测新训练结果"的方案，需要你收集以下事实，全部给出 file:line 引用和关键原文片段：

1. mllm/evaluation/evaluate_model.py 的完整结构：类与 CLI 入口（argparse 参数有哪些）、compute_meteor / compute_bleu 的当前实现与后端优先级（pycocoevalcap vs NLTK fallback）、metric_backend 字段如何写入结果 JSON、save_results 输出结构（per_task_metrics / final_scores / metric_backend 长什么样）、main() 如何被调用（是库还是脚本）、还有哪些 compute_* 函数（bertscore/rouge/gpt 等）与其依赖项。

2. mllm/evaluation/test_b4dl.py 的评测主流程：main() 的参数（--metrics_output、--whole_scene、--per_sequence、--answer_frames、--max_samples 等）、它如何逐任务调用 evaluator、predictions.json 的保存结构（每任务 {predictions, ground_truths, questions}？）、断点续跑机制、最终 metrics.json 由谁写。

3. 现有评测运行脚本：在 mllm/ 或仓库里找 run_*b3*、run_*b4a*、*pipeline*.sh 等脚本中"评测步骤"的具体命令行实例（含全部参数），以及它们评测完成后对比哪些 JSON。

4. 口径规则文档原文：docs/learn docs/B4DL_基线锁定与对比规则_20260829.md 与 docs/learn docs/B4DL_工作进度总结_20260906.md 中关于"同后端对比""metric_backend""METEOR 口径"的规则性表述（引用原句）；docs/B4DL_复现方案.md 或 CLAUDE.md 若有评测口径段落也摘录。

5. NLTK 可用性：wqlc 环境 python（/root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python）能否 import nltk 与 nltk.tokenize（不执行重活，只 import 检查）；nltk 版本。

6. 历史评测产物中 NLTK 与 pycocoevalcap 数值共存的情况：mllm/eval_results/stage2_full_seqv3_mixed/ 下 metrics.json（旧 NLTK 0.3344）与 metrics_recomputed_20260829.json（新 jar 0.1729）的文件内容结构差异（metric_backend 字段等）。

请输出结构化摘要（不要贴整文件），重点给出：CLI 参数清单、compute_meteor 的行号范围与后端逻辑原文、metric_backend 写入代码位置、test_b4dl.py 调用 evaluator 的位置与参数传递方式、评测脚本命令原文、规则文档原句。广度 medium。


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Let me start by exploring the project structure to understand what we're working with.

I'll do multiple parallel operations to gather facts efficiently:
1. Read mllm/evaluation/evaluate_model.py
2. Read mllm/evaluation/test_b4dl.py
3. Find run scripts
4. Read docs
5. Check nltk importability
6. Look at eval results

Let me start.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls /root/autodl-tmp/wql/mmb4dl && ls /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/ && ls /root/autodl-tmp/wql/mmb4dl/docs 2>/dev/null
description: List project root, evaluation dir, docs dir
```

</details>

<details><summary>输出摘要</summary>

```
CLAUDE.md
Claude_record
README.md
assets
backups
datageneration
dataset
docs
encoders
evaluation_results.json
mllm
models
requirements.txt
smoke_test.log
training_logs
wandb
wiki
README.md
__pycache__
analyze_behavior.py
baseline_eval.log
baseline_eval_nohup.log
build_test_split.py
eval_baseline_metrics.json
eval_full_metrics.json
eval_smoke_metrics.json
eval_stage2_full_metrics.json
evaluate_model.py
evaluation_results.json
full_eval_20260810_134614.log
predictions_baseline.json
predictions_full.json
predictions_smoke.json
predictions_stage2_full.json
smoke_stage2_full.log
split_dataset.py
test_b4dl.py
B4DL_复现方案.md
learn docs
mmb4dl-md
mmb4dl.pdf
参考
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: wc -l /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/evaluate_model.py /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/test_b4dl.py 2>/dev/null; ls /root/autodl-tmp/wql/mmb4dl/mllm/ | head -50
description: Get line counts and mllm dir listing
```

</details>

<details><summary>输出摘要</summary>

```
  761 /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/evaluate_model.py
  686 /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/test_b4dl.py
 1447 total
README.md
b4dl_dataset
base_model
checkpoints
data
docs
eval_results
evaluation
requirements.txt
run_baseline_eval.sh
run_stages.sh
scripts
training_logs
vtimellm
wandb
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/evaluate_model.py
```

</details>

<details><summary>输出摘要</summary>

```
1	"""
2	B4DL Model Evaluation - metrics library
3	=======================================
4	Implements the evaluation metrics described in the B4DL paper (Section 3.1 /
5	5.1). Usable both as a library (import B4DLEvaluator) and as a CLI:
6	
7	    from evaluation.evaluate_model import B4DLEvaluator
8	    ev = B4DLEvaluator()
9	    metrics = ev.evaluate_all(results)
10	
11	Metrics (paper Table 3):
12	  Simple Tasks
13	    - Existence / Binary QA : Accuracy (exact answer matching)
14	    - Time Grounding        : mIoU (overlap of predicted vs GT temporal segments)
15	  Complex Tasks (Description / Temporal / Comprehensive Reasoning)
16	    - BLEU-4, METEOR, ROUGE-L, BERTScore, (optional) GPT-4o Score (0-100)
17	
18	Final aggregation (paper Section 5.1):
19	  Each metric is computed per task,
…（已截断，原文 38273 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/test_b4dl.py
```

</details>

<details><summary>输出摘要</summary>

```
1	"""
2	test_b4dl.py - B4DL end-to-end test / evaluation harness
3	========================================================
4	Loads the trained B4DL (VTimeLLM-based) model and its precomputed LiDAR
5	features, runs autoregressive inference over the B4DL *test* split across all
6	six tasks, writes per-task predictions, and finally computes the paper metrics
7	(paper Table 3): Accuracy, mIoU, BLEU-4, METEOR, ROUGE-L, BERTScore and,
8	optionally, the reference-free GPT-4o score.
9	
10	This is the "verify model results" test code. It mirrors the inference path
11	used by mllm/vtimellm/inference.py, but targets the B4DL 6-task benchmark
12	rather than VTimeLLM's video grounding/captioning benchmark.
13	
14	Input test data format
15	----------------------
16	A single JSON file (--test_data) whic
…（已截断，原文 33761 字符）
```

</details>

<details><summary>思考过程</summary>

Let me read the two main files, and explore docs/learn docs directory.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls "/root/autodl-tmp/wql/mmb4dl/docs/learn docs" 2>/dev/null; ls /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/ 2>/dev/null; ls /root/autodl-tmp/wql/mmb4dl/mllm/scripts/ 2>/dev/null
description: List learn docs, eval_results, scripts dirs
```

</details>

<details><summary>输出摘要</summary>

```
B4DL_B3整场景meta修复mIoU超论文_20260906.md
B4DL_LiDARCLIP权重核对_20260828.md
B4DL_seqv2评测对比与坍缩分析_20260825.md
B4DL_seqv3混合训练评测分析_20260827.md
B4DL_vs_LiDAR-LLM_comparison.md
B4DL_两阶段训练评测与混合法决策_20260826.md
B4DL_全仓库论文对齐审查_20260829.md
B4DL_基线版本训练方法对比_20260830.md
B4DL_基线锁定与对比规则_20260829.md
B4DL_工作进度总结_20260906.md
B4DL_待修改清单_20260829.md
B4DL_训练方法对比_论文vs复现_20260828.md
B4DL_论文对齐审计_20260824.md
B4DL_评测数值对比与问题清单_20260828.md
VTimeLLM_analysis.md
_meteor_sweep
eval_log.txt
official_ckpt_nometa
official_model_stage2
seq_predictions.json.ckpt
smoke_2000_log.txt
smoke_2000_metrics.json
smoke_2000_predictions.json
smoke_meta_metrics.json
smoke_meta_predictions.json
smoke_predictions.json.ckpt
smoke_seq_log.txt
smoke_seq_metrics.json
smoke_seq_predictions.json
stage23_seqv3
stage2_full_seq
stage2_full_seqv2
stage2_ful
…（已截断，原文 1742 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Now let me look at the run scripts with eval steps, the docs, and NLTK availability in parallel.

</details>

Now let me look at the run scripts' eval steps, the rule docs, and NLTK availability in parallel.

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -n "evaluation\|test_b4dl\|metrics\|evaluate_model\|predictions" /root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b1_pipeline.sh /root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b2_pipeline.sh /root/auto
…（已截断，原文 391 字符）
description: Find eval steps in pipeline scripts
```

</details>

<details><summary>输出摘要</summary>

```
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b4a_pipeline.sh:56:    timeout 18000 /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -u evaluation/test_b4dl.py \
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b4a_pipeline.sh:65:        --output "$EVAL_OUT/predictions.json" \
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b4a_pipeline.sh:66:        --metrics_output "$EVAL_OUT/metrics.json" \
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b4a_pipeline.sh:79:m = json.load(open('$EVAL_OUT/metrics.json'))
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b4a_pipeline.sh:87:" 2>/dev/null || cat "$EVAL_OUT/metrics.json"
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b1_pipeline.sh:110:    timeout 18000 python -u evaluation/test_b4dl.py \
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b1_pipeline.sh:120:        
…（已截断，原文 4338 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -c "import nltk; print('nltk OK', nltk.__version__)" 2>&1; /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -c "from nltk.tokenize import word_toke
…（已截断，原文 282 字符）
description: Check nltk import and version in wqlc env
```

</details>

<details><summary>输出摘要</summary>

```
nltk OK 3.10.0
tokenize+meteor import OK
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed/ && ls /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b1/ /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/
…（已截断，原文 324 字符）
description: List eval_results subdirectories
```

</details>

<details><summary>输出摘要</summary>

```
metrics.json
metrics_recomputed_20260829.json
predictions.json
/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b1/:
eval_log.txt
metrics.json
predictions.json

/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b2/:
eval_log.txt
metrics.json
predictions.json

/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b3/:
eval_log.txt
metrics.json
predictions.json
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

NLTK 3.10.0 available in wqlc env. Now let me read the full pipeline script eval commands (the exact full command lines with all args) and the metric backend parts of the eval results JSON files. Also let me read the docs sections about the comparison rules.

</details>

NLTK 3.10.0 imports fine in the wqlc env. Now the full eval command blocks from pipeline scripts, and the two metrics JSONs.

**🔧 Bash**

<details><summary>输入</summary>

```
command: sed -n '50,95p' /root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b2_pipeline.sh; echo ===B3===; sed -n '50,90p' /root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b3_pipeline.sh
description: Read eval steps of run_b2 and run_b3 pipeline scripts
```

</details>

<details><summary>输出摘要</summary>

```
EVAL_OUT=./eval_results/stage2_full_seqv3_mixed_b2
mkdir -p "$EVAL_OUT"
EVAL_OK=0
for attempt in 1 2 3 4 5; do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ -n "$FREE_MB" ] && [ "$FREE_MB" -lt 14336 ]; then
        echo "[$(date '+%F %T')] 显存不足 (${FREE_MB}MB < 14GB)，10 分钟后重试..."
        sleep 600
        continue
    fi
    timeout 18000 python -u evaluation/test_b4dl.py \
        --model_base ./base_model/vicuna-v1-5-7b \
        --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
        --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b2 \
        --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
        --test_data ./b4dl_dataset/test_qa.json \
        --ego_meta ./
…（已截断，原文 3447 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: sed -n '50,90p' /root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b4a_pipeline.sh; echo ===B1===; sed -n '100,145p' /root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b1_pipeline.sh
description: Read eval steps of run_b4a and run_b1 pipeline scripts
```

</details>

<details><summary>输出摘要</summary>

```
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ -n "$FREE_MB" ] && [ "$FREE_MB" -lt 14336 ]; then
        echo "[$(date '+%F %T')] 显存不足 (${FREE_MB}MB < 14GB)，10 分钟后重试..."
        sleep 600
        continue
    fi
    timeout 18000 /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -u evaluation/test_b4dl.py \
        --model_base ./base_model/vicuna-v1-5-7b \
        --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
        --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b4a \
        --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
        --test_data ./b4dl_dataset/test_qa.json \
        --ego_meta ./b4dl_dataset/ego_metadata.json \
        --frame_motion ./b4dl_dat
…（已截断，原文 3677 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: head -c 3000 /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed/metrics.json; echo; echo ====RECOMPUTED====; head -c 3000 /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_s
…（已截断，原文 243 字符）
description: Show heads of both metrics JSONs
```

</details>

<details><summary>输出摘要</summary>

```
{
  "per_task_metrics": {
    "existence": {
      "accuracy": 0.7071618037135279
    },
    "binary_qa": {
      "accuracy": 0.8187375415282392
    },
    "time_grounding": {
      "miou": 0.2696124556495854
    },
    "description": {
      "bleu4": 0.12199418783956004,
      "meteor": 0.34148856781000575,
      "rouge_l": 0.31549536432684416,
      "bertscore": 0.9685461521148682,
      "gpt_score": 0.0
    },
    "temporal_understanding": {
      "bleu4": 0.0972006442120154,
      "meteor": 0.3319029448247337,
      "rouge_l": 0.3251877957504161,
      "bertscore": 0.9786718487739563,
      "gpt_score": 0.0
    },
    "comprehensive_reasoning": {
      "bleu4": 0.10466348144249252,
      "meteor": 0.3296669090366687,
      "rouge_l": 0.3325654411045476,
      "bertscore": 0.97312372922
…（已截断，原文 2380 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Interesting. The old metrics.json (with NLTK meteor 0.3344) doesn't have metric_backend field at the end (it may have been truncated by head -c 3000). Let me check the tail of metrics.json to see if it contains metric_backend. Also note in the old file there's gpt_score 0.0 and berstscore 0.9734 (computed with local path model defaulting... interesting, no metric_backend printed in first 3000 chars). Let me check the tail.

Also I need:
- The tail of metrics.json (check metric_backend)
- The docs: B4DL_基线锁定与对比规则_20260829.md and B4DL_工作进度总结_20260906.md for rules on same-backend comparison, metric_backend, METEOR caliber
- docs/B4DL_复现方案.md and CLAUDE.md eval-related sections
- run_mixed_eval_after_train.sh (the "eval after train" script for new training results) - actually that might be exactly relevant. Let me check.

Let me do these in parallel.

</details>

The old metrics.json shows NLTK-ish METEOR 0.3344. Let me check its tail for metric_backend, read the rule docs, and check remaining scripts.

**🔧 Bash**

<details><summary>输入</summary>

```
command: tail -20 /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed/metrics.json; echo ===; python3 -c "
import json
m=json.load(open('/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_f
…（已截断，原文 313 字符）
description: Check tail and keys of old metrics.json
```

</details>

<details><summary>输出摘要</summary>

```
      "gpt_score": 0.0
    },
    "comprehensive_reasoning": {
      "bleu4": 0.10466348144249252,
      "meteor": 0.3296669090366687,
      "rouge_l": 0.3325654411045476,
      "bertscore": 0.9731237292289734,
      "gpt_score": 0.0
    }
  },
  "final_scores": {
    "accuracy": 0.7629496726208835,
    "miou": 0.2696124556495854,
    "bleu4": 0.1079527711646893,
    "meteor": 0.3343528072238027,
    "rouge_l": 0.3244162003939359,
    "bertscore": 0.9734472433725992,
    "gpt_score": 0.0
  }
}===
keys: ['per_task_metrics', 'final_scores']
metric_backend: None
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -n "NLTK\|nltk\|metric_backend\|同后端\|同口径\|METEOR\|meteor\|后端\|pycocoevalcap\|对比规则\|可比" "/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_基线锁定与对比规则_20260829.md" | head -60
description: Grep rules doc for backend rules
```

</details>

<details><summary>输出摘要</summary>

```
1:# B4DL 复现基线锁定与对比规则（B0 = seqv3-mixed，2026-08-29）
4:> 后续所有训练改进（stage1 补数据、编码器更换、超参调整等）一律与 B0 同口径对比。
30:| 特征目录 | `encoders/lidarclip/b4dl/stage2_features/` | 850 个 .npy（目录级固定；**重新提特征 = 破坏可比性**，须换基线代号） | — |
65:| METEOR | **0.3344** | 0.275 | +0.059 | — |
77:| description | 3,770 | B@4 / METEOR / ROUGE-L / BS | 0.1220 / 0.3415 / 0.3155 / 0.9685 |
85:- **文本指标**（BLEU/METEOR/ROUGE/BS）：未做逐样本 bootstrap，按跨版本经验波动取保守阈值 **≥0.01** 视为有意义。
91:## 五、后续对比规则
93:1. **同口径原则**：新模型评测必须用 §2.2 命令原样执行（仅换 `--stage2` checkpoint 路径与输出目录）；数据文件 MD5 与 §2.1 一致才可与 B0 对比。
96:   - 更换 LiDAR-CLIP 编码器或重新提取 `stage2_features`（特征语义变化，跨代不可比）；
106:2. **BLEU-4 为 NLTK 实现**（method1 smoothing）：与 pycocoevalcap 实现存在已知虚高差异（见 `B4DL_评测方法学陷阱` 记忆/文档）；论文实现未披露，跨论文对比时以"同实现纵向对比"为准。
107:3. **BERTScore**：本地 roberta-large（避免 BERTScore OOM 与 HF 网络）；与
…（已截断，原文 936 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -rn "NLTK\|nltk\|metric_backend\|同后端\|METEOR\|pycocoevalcap\|口径\|0.3344\|0.1729\|0.334\|0.173" "/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_工作进度总结_20260906.md" | head -50
description: Grep progress doc for backend and METEOR rules
```

</details>

<details><summary>输出摘要</summary>

```
/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_工作进度总结_20260906.md:8:- **复现阶段闭环**：B3（整场景输入 + 论文语义 metatoken）mIoU **0.3467 > 论文 0.311**（+0.036），六任务分项齐全，达到并超过论文水平；官方仓库被实证**从未发布**论文 Table 3 的完整模型（发布 checkpoint 无 meta，mIoU 0.1737 ≈ 论文 Table 4"无 Metatoken"行）。
/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_工作进度总结_20260906.md:11:- **遗留口径问题**：METEOR 0.175 vs 论文 0.275（B0-B3 一致，唯一未对齐指标）、GPT-4o 指标待 API key、基线 Wiki 文档待从 B0 切到 B3。
/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_工作进度总结_20260906.md:21:| meteor | 0.275 | 0.1729 | 0.1750 | 0.1747 | 0.1747 |
/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_工作进度总结_20260906.md:32:5. **官方模型评测（09-02，决定性）**：官方 checkpoint + 官方特征 → acc 0.7542 / mIoU 0.1737；官方训练数据无 meta、无 `<4DLiDAR>` → **论文完整模型从未发布**；对照论文 Table 4"无 Metatoken"行（0.763/0.161）吻合。
/root/autodl-tmp/w
…（已截断，原文 2148 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Good. Now I need:
1. Read the B4DL_基线锁定与对比规则_20260829.md full key sections (§2.2 command original and rule quote) — get original sentences in full.
2. Read key sections of the progress summary doc for the METEOR-caliber original text (more complete).
3. docs/B4DL_复现方案.md and CLAUDE.md eval-caliber paragraphs.
4. Look at run_mixed_eval_after_train.sh since it's the "eval after train" script.
5. Also the recon docs mention "B4DL_评测方法学陷阱" — maybe stored as memory elsewhere. Also check docs/learn docs/B4DL_seqv2评测对比与坍缩分析_20260825.md or _meteor_sweep folder. The user asked medium breadth; the key rule docs are 基线锁定与对比规则 and 工作进度总结. Also B4DL_复现方案.md and CLAUDE.md.

Let me get the full text of the rules doc sections around the key areas (§2.2 evaluation command, §五 后续对比规则), and grep the progress doc for original sentences about metric_backend/同后端.

</details>

Now let me pull the exact rule-doc passages and check remaining pieces.

**🔧 Bash**

<details><summary>输入</summary>

```
command: sed -n '1,60p' "/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_基线锁定与对比规则_20260829.md"
description: Read rules doc section 1-2.2
```

</details>

<details><summary>输出摘要</summary>

```
# B4DL 复现基线锁定与对比规则（B0 = seqv3-mixed，2026-08-29）

> **结论：正式将 2026-08-27 的 seqv3-mixed 全量评测锁定为复现基线，代号 B0。**
> 后续所有训练改进（stage1 补数据、编码器更换、超参调整等）一律与 B0 同口径对比。
> 数值可靠性验证与问题分析见配套文档：`B4DL_评测数值对比与问题清单_20260828.md`、`B4DL_训练方法对比_论文vs复现_20260828.md`。

## 一、B0 资格验证（为什么可以当基线）

| 验证项 | 结果 |
|---|---|
| 独立复算 | 从 `predictions.json` 用纯标准库重写解析+闭区间 IoU（不复用项目代码），得 **0.2696124556**，与 `metrics.json` 一致（误差 <1e-12） |
| 口径等价性 | 帧集合口径复算同样得 0.2696124556（闭区间数学等价） |
| 完整性 | 6 任务合计 **30,145 条** = 测试集全量；TG 2,783 条；**0 条跳过**；`eval_results/stage2_full_seqv3_mixed/` 下无 `.ckpt` 断点残留（排除断点续跑混样） |
| 抽样核验 | 预测文本 → 帧区间解析逐条正确（如 "from frame 000 to frame 008."） |
| 代码一致性 | `evaluation/` 自 8b85cd2（08-25，评测前最后实质改动）至今**内容零变化**（merge 24a3ab9 仅把 test_b4dl.py 行尾改为 CRLF，`--ignore-cr-at-eol` diff 为空）；`evaluate_model.py` 的 `compute_miou`（`ev
…（已截断，原文 3054 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: sed -n '60,108p' "/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_基线锁定与对比规则_20260829.md"
description: Read rules doc sections 2.2-五
```

</details>

<details><summary>输出摘要</summary>

```
| 指标 | B0 | 论文 | 差值 | 95% CI（B0 自身） |
|---|---|---|---|---|
| accuracy | **0.7629** | 0.762 | +0.001 | [0.7545, 0.7714]（±0.0085） |
| mIoU | **0.2696** | 0.311 | −0.041 | [0.2564, 0.2828]（±0.0132） |
| BLEU-4 | **0.1080** | 0.095 | +0.013 | — |
| METEOR | **0.3344** | 0.275 | +0.059 | — |
| ROUGE-L | **0.3244** | 0.322 | +0.002 | — |
| BERTScore | **0.9734** | 0.897 | +0.076 | — |
| GPT-4o | —（无 API key） | 59.513 | — | 可离线后补（见 §六） |

### 3.2 分任务

| 任务 | 样本数 | 指标 | B0 |
|---|---|---|---|
| existence | 3,770 | accuracy | 0.7072 |
| binary_qa | 7,525 | accuracy | 0.8187 |
| time_grounding | 2,783 | mIoU | 0.2696 |
| description | 3,770 | B@4 / METEOR / ROUGE-L / BS | 0.1220 / 0.3415 / 0.3155 / 0.9685 |
| temporal_understanding | 4,757 | 同上 | 0.0972 / 0.3319 / 0.3252 / 0.9787 |
| comprehensive_r
…（已截断，原文 2429 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -n "NLTK\|nltk\|metric_backend\|同后端\|METEOR\|meteor\|pycocoevalcap\|口径\|后端\|评测" /root/autodl-tmp/wql/mmb4dl/docs/B4DL_复现方案.md | head -30; echo ===CLAUDE===; grep -n "NLTK\|nltk\|metric_backend\|M
…（已截断，原文 290 字符）
description: Grep reproduction plan and CLAUDE.md for eval-caliber text
```

</details>

<details><summary>输出摘要</summary>

```
5:> 修改版仓库：`D:\tmp\remote4b4dl`（GitHub: Dani3lWang/remote4b4dl）——修复了多项 bug，添加了内置评测代码、Stage 3 训练脚本、Gradio Demo，并采用单环境方案
32:| Accuracy↑ | mIoU↑ | B@4↑ | ROUGE-L↑ | METEOR↑ | BERTScore↑ | GPT Score↑ |
63:                    └─ 推理 + 六任务评测 → Table 3 / Table 4
165:**测试集构建**：HF 数据集的测试集按任务分文件存放（6 个 JSON），评测时需要合并为一个统一的 `test_qa.json`。【remote4b4dl】仓库的 `mllm/evaluation/build_test_split.py` 可以完成此工作：
189:| OpenAI API（GPT-4o） | ①数据生成（若不直接下载 HF 数据集）；②GPT Score 评测 | 数据生成：5100 序列 × (2 次描述 + 5 次任务 QA) ≈ 3.6 万次调用，含多图输入，估计 **$150–400**；GPT Score 评测：30k 测试样本 × 每条约 $0.01–0.03 ≈ $300–900（可用 gpt-4o-mini 降本做冒烟） |
222:# 评测依赖
223:pip install bert-score==0.3.13 pycocoevalcap==1.2 nltk rouge-score openai
279:# 评测依赖
280:pip install bert-score==0.3.13 pycocoevalcap==1.2 nltk rouge-score openai
459:**问题**：提取特征时需要全部 850 个 
…（已截断，原文 4464 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

I have a lot of info. Remaining items to gather:
1. CLAUDE.md eval-caliber sections around lines 184-260 to get more context on rules of METEOR backend comparisons. Also look for "metric_backend"/"同后端"/NLTK in CLAUDE.md.
2. The 工作进度总结 doc's METEOR section (more complete original sentences around lines 49-60, 77).
3. run_mixed_eval_after_train.sh full eval command, and check whether run_mixed_eval_after_train.sh compares JSONs after eval.
4. The doc "B4DL_评测数值对比与问题清单_20260828.md" may contain NLTK-vs-jar discussions (0.3344 vs 0.1729). Also "_meteor_sweep" folder in docs/learn docs. But scope is medium. The user's question 4 focuses on two specific docs plus 复现方案/CLAUDE.md. I have most of it.

Also confirm how metric_backend is used when recomputed: check the recompute script referenced (metrics_recomputed_20260829.json). Find where recomputation was done, e.g. scripts or docs mention "recompute". Let me search for "recomputed" in scripts/docs. Medium breadth—quick greps.

Also check git log for evaluate_model.py to understand metric_backend history? Not needed.

Let me do the remaining grep queries in parallel:
- CLAUDE.md lines 184-260 for eval section + any NLTK mention
- 工作进度 doc lines 49-80 verbatim (the whole METEOR口径 part)
- find recompute script
- run_mixed_eval_after_train.sh (full eval block and comparisons)

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: sed -n '180,215p' /root/autodl-tmp/wql/mmb4dl/CLAUDE.md; echo ====; grep -n "meteor\|METEOR\|nltk\|NLTK\|metric_backend" /root/autodl-tmp/wql/mmb4dl/CLAUDE.md | head -20
description: Read CLAUDE.md eval section and meteor mentions
```

</details>

<details><summary>输出摘要</summary>

```
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    --video_path /path/to/video.mp4

# B4DL 六任务评测（论文 Table 3 对齐，含 metatoken 注入）
conda run -n wqlc python evaluation/test_b4dl.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --test_data ./b4dl_dataset/test_qa.json \
    --ego_meta ./b4dl_dataset/ego_metadata.json \
    --output ./evaluation/predictions.json \
    --metrics_output ./evaluation/evaluation_results.json

# 加 --per_sequence 启用 per-sequence 模式（论文 Appendix C
…（已截断，原文 1868 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: sed -n '45,80p' "/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_工作进度总结_20260906.md"
description: Read progress doc eval methodology section
```

</details>

<details><summary>输出摘要</summary>

```
- LiDAR-CLIP 编码器：nuScenes 自训 + 退火定稿（`ckpt_anneal/lidarclip_mm/last.ckpt`，seed 0 可复现）；
- 特征：`encoders/lidarclip/b4dl/stage2_features` 850 场景 + stage1 28,130 帧，全部出自退火编码器（ONCE 旧特征备份 `*_once`）；
- val MSE 探针存档：`encoders/lidarclip/logs/val_mse_probe.csv`（baseline 0.1061 → anneal 0.0992）。

## 五、评测方法学与指标口径现状

| 项 | 状态 |
|---|---|
| mIoU（闭区间）/ accuracy | ✅ 与论文口径一致 |
| BERTScore | ✅ 0.8973（roberta-large 17 层，修复层号 bug 后命中论文 0.897） |
| BLEU-4 | ✅ pycocoevalcap 语料级（0.0965 vs 论文 0.095） |
| ROUGE-L | ✅ 0.3234 vs 论文 0.322 |
| METEOR | ⚠️ 0.1747 vs 论文 0.275，B0-B3 一致偏低，Meteor-1.5 jar 口径存疑（唯一未对齐指标，待考证论文所用变体） |
| GPT-4o 指标 | ⏳ predictions.json 已落盘，配 API key 后离线补算（无需重跑推理） |
| 评测口径声明 | 所有报告持续携带 `--answer_frames`（GT 仅用于选输入帧/meta 锚定）oracle 声明 |
| 环境约束 | 评测需 HF offline + 显存门控（CoRViD/xmuda 共用 GPU）；断点续训按步数排序 |

## 六
…（已截断，原文 1864 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -rln "metrics_recomputed\|recompute" /root/autodl-tmp/wql/mmb4dl/mllm/scripts/ /root/autodl-tmp/wql/mmb4dl/docs/ "/root/autodl-tmp/wql/mmb4dl/docs/learn docs/" /root/autodl-tmp/wql/mmb4dl/mllm/ev
…（已截断，原文 316 字符）
description: Find recompute references and read run_mixed_eval_after_train.sh
```

</details>

<details><summary>输出摘要</summary>

```
/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_工作进度总结_20260906.md
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/generate_ego_metadata.py
/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_工作进度总结_20260906.md
===
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

# 等
…（已截断，原文 3572 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Now I have nearly everything. A few last details to verify quickly:

1. Whether the wqlc python actually runs evaluate_model.py with HAS_COCO_METEOR=True — does pycocoevalcap import + java exist in that env? Useful for the user's plan (NLTK backend design). Check import pycocoevalcap + java presence. Read-only checks are fine (importing pycocoevalcap imports but doesn't launch java; launching java would be heavier—avoid). Let me check pycocoevalcap import and `which java` / java -version. java -version is fine.

2. Which run script was the last used for "new training results" — run_mixed_eval_after_train.sh is the frozen command of §2.2.

3. In the progress doc line 21 and 106, the numbers of the table (line 21: meteor 0.275 | 0.1729 | 0.1750 | ...) — I should capture the doc's METEOR comparative table original rows. Let me get lines 14-30 of that doc.

4. Note that the metric_backend doc statement: The progress doc says "BLEU-4: pycocoevalcap 语料级". No explicit "metric_backend" mention in docs? Let me grep "metric_backend" in the whole repo docs to confirm whether any doc quotes it.

5. Also check where metric_backend is written: evaluate_model.py:219 init, 240-248 set, 548-553 save_results. Recomputed file includes it; old metrics.json does not have it (metric_backend key absent) — old file predates the field. Let me check when metric_backend was added via git log/blame quickly — useful to understand which runs have it. Quick check: git log -S "metric_backend".

6. The question about how compute functions vs compute_*: bertscore/rouge/gpt dependency guards — lines 42-147, done.

7. Also confirm b4a metrics.json has metric_backend (newer run). Check quickly.

Let me also get table at line 14-30 of the progress summary doc for exact numbers, and grep for "metric_backend" across docs + git log for when introduced.

</details>

Almost complete. A few verification greps.

**🔧 Bash**

<details><summary>输入</summary>

```
command: sed -n '12,30p' "/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_工作进度总结_20260906.md"; echo ===TABLE105-115===; sed -n '100,115p' "/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_工作进度总结_20260906.md"
description: Read progress doc tables with METEOR numbers
```

</details>

<details><summary>输出摘要</summary>

```
## 二、复现闭环：B0 → B3 全链

| 指标 | 论文 | B0 (seqv3-mixed) | B1 (切片+新编码器) | B2 (整场景) | **B3 (整场景+meta2)** |
|---|---|---|---|---|---|
| accuracy | 0.762 | 0.7629 | 0.7787 | 0.7649 | **0.7526** |
| mIoU | 0.311 | 0.2696 | 0.2653 | 0.1992 | **0.3467** |
| bleu4 | 0.095 | 0.0973 | 0.0994 | 0.0959 | 0.0965 |
| rouge_l | 0.322 | 0.3244 | 0.3272 | 0.3229 | 0.3234 |
| meteor | 0.275 | 0.1729 | 0.1750 | 0.1747 | 0.1747 |
| bertscore | 0.897 | 0.8973 | 0.8979 | 0.8967 | 0.8967 |

B3 per-task：existence 0.6761 / binary_qa 0.8291 / TG mIoU 0.3467 / description bleu4 0.1055 / temporal bleu4 0.0853 / comprehensive bleu4 0.0987。

时间线（2026-08 ~ 09）：

1. **B0 锁定（08-27，08-29 规则化）**：seqv3 两阶段法失败（acc 0.0001）后改单阶段混合训练（同数据同 projector，混合法消解格式漂移与 TG 坍缩），acc 0.7629 / mIoU 0.2696 四位一体锁定为基线；显著性规则 ΔmIoU>0.013。
2. **编码器定稿（08-
…（已截断，原文 2307 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -rn "metric_backend" /root/autodl-tmp/wql/mmb4dl/docs/ /root/autodl-tmp/wql/mmb4dl/CLAUDE.md /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/*/metrics.json 2>/dev/null | grep -v Binary | head -15; 
…（已截断，原文 334 字符）
description: Grep metric_backend across docs and eval results, git history
```

</details>

<details><summary>输出摘要</summary>

```
/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b3/metrics.json:39:  "metric_backend": {
/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_全仓库论文对齐审查_20260829.md:150:**重算结论（修正 §五 前的预期）**：预期 "BLEU ~0.095 / BERTScore ~0.895" 双双命中（0.0973 / 0.8973）；METEOR 的 jar 实测值（0.1729）低于论文 0.275，超出预期——修复后 B0 与论文的可比指标为 acc持平、BLEU/BERTScore/ROUGE-L 三项命中、mIoU 与 METEOR 两项未达（mIoU 属模型能力，METEOR 属口径待考证）。修正口径已冻结：结果 JSON 自带 `metric_backend` 记录，后续 B0/B1 对比必须同后端（pycocoevalcap 语料级 BLEU + Meteor-1.5 jar + roberta-large L17 BERTScore）。
/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b2/metrics.json:39:  "metric_backend": {
/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b1/metrics.json:39:  "metric_backend": {
/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/of
…（已截断，原文 1287 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -c "import pycocoevalcap; from pycocoevalcap.bleu.bleu import Bleu; from pycocoevalcap.meteor.meteor import Meteor; print('pycocoevalcap OK')" 2>&1 |
…（已截断，原文 383 字符）
description: Check pycocoevalcap import and java presence in wqlc env
```

</details>

<details><summary>输出摘要</summary>

```
pycocoevalcap OK
None
/usr/bin/java
/usr/bin/java
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Interesting — pycocoevalcap import OK but the __file__ print came out "None"? Wait, the second python command printed "None" — hmm, `print(pycocoevalcap.__file__)` should print a file path... it printed None. That may mean pycocoevalcap is a namespace package? Actually, import pycocoevalcap succeeded and importing Meteor succeeded. The __file__ None is odd but likely a namespace package situation (pycocoevalcap as dir without __init__? or a meta path finder). Wait, actually maybe pycocoevalcap is installed under a path... Actually if from pycocoevalcap.meteor.meteor import Meteor succeeded, then modules exist. Let me not over-verify — key result: import OK, java present at /usr/bin/java.

Let me quickly verify whether HAS_COCO_METEOR would be true in the wqlc env, i.e. does evaluate_model.py find pycocoevalcap AND would Meteor jar init work? Init launches java; I shouldn't run heavy things but checking java availability is done. Good enough: import works and java exists → HAS_COCO_METEOR would be True.

Let me double check pycocoevalcap's __file__ anomaly - maybe the envs uses a .pth or pycocoevalcap has no __init__.py in old versions (pycocoevalcap 1.2 does have __init__.py). The print order: first command output "pycocoevalcap OK" then second printed "None". Possibly pycocoevalcap is being imported from a zip or as namespace package. Let me quickly check `pip show pycocoevalcap` in wqlc env. Not critical though. Quick check for the report on where pycocoevalcap lives.

Also check b3 metrics.json metric_backend and b1/b2 metric_backend values to confirm jar field names, plus check the B4DL_全仓库论文对齐审查_20260829.md section around line 150 for the full quote, which is a rule doc-like statement on "同后端". The user asked for docs rules from the two learn-docs files + 复现方案 + CLAUDE.md. The audit doc is another doc in docs/learn docs. The user asked specifically for those listed files but this additional quote on 同后端 is quite relevant since it explicitly uses the phrase "同后端" — a strong find. Let me include it, quoting with the file name.

Let me now view line ~140-160 of that audit doc to quote the full rule context, and check b1/b2/b3 metrics.json metric_backend blocks. Also get per-task/per metrics values in recomputed file tail: the recomputed file also includes metric_backend — shown earlier. And old metrics.json has no metric_backend, no per-task gpt_score null... Actually old one has gpt_score 0.0 and bertscore 0.97 (broken layer 24), and final meteor 0.3344 (NLTK), matching rules doc §3.1 B0 numbers (METEOR 0.3344, bertscore 0.9734). The recomputed file shows meteor 0.1729 (final) with backend pycocoevalcap-meteor-1.5 and bertscore 0.8973.

Note discrepancy: 工作进度总结 table says B0 meteor 0.1729; the recomputed file shows 0.1729. Good.

One more detail: the question asks about predictions.json structure per task {predictions, ground_truths, questions} — confirmed at test_b4dl.py:645-651 (clean_results without _task). Also note that predictions.json written also includes per-task arrays only.

Also, checkpoint resume details: checkpoint file <output>.ckpt with {"results": results}; completed_ids computed but only used... lines 532-545: resume loads results, counts done per task; completed_ids built but never used to skip (interesting: it's built but unused for skipping since start_idx from len(results[task])). Fine—describe mechanism as: per-task length-based resume.

Question 2 also asks "how it calls evaluator per task": lines 659-671 — after inference, creates B4DLEvaluator(use_gpt...), evaluate_all(results) over all tasks at once (not per task). Also evaluate_task computes metrics per-task internally (evaluate_all loops over ALL_TASKS and calls evaluate_task per task).

Now, which compute_* functions exist: compute_accuracy (310), compute_miou (317), compute_bleu4 (343), compute_meteor (373), compute_rouge_l (403), compute_bertscore (410), compute_gpt_score (440). Dependencies: nltk (42-84 incl. punkt/wordnet), rouge_score (86-91), bert_score (93-98), pycocoevalcap bleu (104-110), pycocoevalcap meteor + _DrainedMeteor (112-140), openai (142-147).

Also main() invocation in evaluate_model.py: both library and CLI (module docstring says both). __main__ guard at 761-762.

CLI parameters evaluate_model.py main(): --predictions, --ground_truth, --output (default evaluation_results.json), --use_gpt, --gpt_api_key (default env OPENAI_API_KEY), --gpt_model (default gpt-4o), --demo, --bert_model. Lines 683-694.

test_b4dl.py main() CLI args full list (389-465): model_base (required), pretrain_mm_mlp_adapter, stage2, stage3, feat_folder (required), test_data, predictions_dir, scene_metadata, nuscenes_root, output (default predictions.json), metrics_output (default evaluation_results.json), max_samples (-1), checkpoint_interval (50), dtype (float16/bfloat16), use_gpt, gpt_api_key, gpt_model, ego_meta, frame_motion, sequence_metadata, answer_frames, no_4dlidar, no_meta, per_sequence, whole_scene.

Wait — the user's question said "--whole_scene、--per_sequence、--answer_frames、--max_samples 等" yes all included.

Now check the audit doc section § around line 150 for quote and check b1 metrics.json backend values quickly. Also the doc at line 150 is "B4DL_全仓库论文对齐审查_20260829.md".

Also let me confirm predictions.json structure at stage2_full_seqv3_mixed_b3/predictions.json: keys are task names; each has predictions/ground_truths/questions. Quick head.

And check eval log from a recent run to see printed backend warnings? Not necessary.

Let me run the remaining greps in parallel and finalize. Also capture the full quote of the frozen eval command from run_mixed_eval_after_train.sh — got already.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: sed -n '140,160p' "/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_全仓库论文对齐审查_20260829.md"
description: Read audit doc recompute rule passage
```

</details>

<details><summary>输出摘要</summary>

```
| 指标 | 论文 Table 3 | B0 原报 | 修正口径实测（0829 晚重算） | 说明 |
|---|---|---|---|---|
| Accuracy | 0.762 | 0.7629 | **0.7629（逐位不变）** | exact match 协议对齐，持平 |
| mIoU | 0.311 | 0.2696 | **0.2696（逐位不变）** | 唯一显著未达标项（Δ-0.041 > 阈值 0.013），根因链指向 stage1 数据量 + TG 帧级定位 |
| BLEU-4 | 0.095 | 0.1080 | **0.0973 ✅ ≈论文** | pycocoevalcap 语料级；分项 0.1024/0.0854/0.1042 |
| ROUGE-L | 0.322 | 0.3244 | **0.3244（逐位不变）** | 唯一口径完全一致的文本指标 |
| METEOR | 0.275 | 0.3344 | **0.1729 ⚠️ 低于论文** | Meteor-1.5 jar 标准后端；换库后反而显著低于论文，说明论文所用 METEOR 变体仍未知（NLTK 0.334 虚高、jar 0.173 偏低，两头不靠），唯一遗留的口径考证项 |
| BERTScore | 0.897 | 0.9734 | **0.8973 ✅ 精确命中** | 层号 24→17 一行修复；分项 0.8973/0.8975/0.8970 高度稳定 |
| GPT Score | 59.513 | 缺失（0.0） | 待离线补算（现记 null） | prompt 已逐字就绪 |

**重算结论（修正 §五 前的预期）**：预期 "BLEU ~0.095 / BERTScore ~0.895" 双双命中（0.0973 / 0.8973）；METEOR 的 
…（已截断，原文 1560 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: for d in b1 b2 b3; do echo "== $d =="; python3 -c "
import json
m=json.load(open('/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_$d/metrics.json'))
print('keys:', list(m.keys())
…（已截断，原文 328 字符）
description: Check metric_backend and finals in b1 b2 b3 metrics.json
```

</details>

<details><summary>输出摘要</summary>

```
== b1 ==
keys: ['per_task_metrics', 'final_scores', 'metric_backend']
backend: {'bertscore': {'model_type': '/root/autodl-tmp/wql/mmb4dl/models/roberta-large', 'num_layers': 17, 'rescale_with_baseline': False}, 'bleu4': 'pycocoevalcap-corpus', 'meteor': 'pycocoevalcap-meteor-1.5'}
final: {'accuracy': 0.7787, 'miou': 0.2653, 'bleu4': 0.0994, 'meteor': 0.175, 'rouge_l': 0.3272, 'bertscore': 0.8979}
== b2 ==
keys: ['per_task_metrics', 'final_scores', 'metric_backend']
backend: {'bertscore': {'model_type': '/root/autodl-tmp/wql/mmb4dl/models/roberta-large', 'num_layers': 17, 'rescale_with_baseline': False}, 'bleu4': 'pycocoevalcap-corpus', 'meteor': 'pycocoevalcap-meteor-1.5'}
final: {'accuracy': 0.7649, 'miou': 0.1992, 'bleu4': 0.0959, 'meteor': 0.1747, 'rouge_l': 0.3229, 'bertscore': 0.8967}
…（已截断，原文 1201 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: python3 -c "
import json
m=json.load(open('/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b3/predictions.json'))
print('top keys:', list(m.keys()))
for k in list(m)[:2]:
    pri
…（已截断，原文 318 字符）
description: Inspect predictions.json structure
```

</details>

<details><summary>输出摘要</summary>

```
top keys: ['existence', 'binary_qa', 'time_grounding', 'description', 'temporal_understanding', 'comprehensive_reasoning']
existence -> ['predictions', 'ground_truths', 'questions'] lens: 3770 3770 3770
binary_qa -> ['predictions', 'ground_truths', 'questions'] lens: 7525 7525 7525
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

I have all the facts. Let me compile the structured report now. One last small confirmation — where the evaluate_task / evaluator call is made in test_b4dl.py (lines 659-671) and checkpoint/resume mechanism details. All done.

Let me also double check the wqlc pycocoevalcap location since import succeeded (probably installed in that env) — a quick find to report the path would be useful for the user's design, but import works, that's enough. Actually the question 5 was about nltk only — done (3.10.0, tokenize import OK).

Let me now write the structured final report, in Chinese given the user's request language, with file:line references and original snippets. Keep concise but complete per the six questions.

</details>

All facts gathered. Here is the structured report.

# 勘察报告：NLTK METEOR 后端评测方案所需事实

## 1. mllm/evaluation/evaluate_model.py（761 行）

**文件性质**：双用途——库（`from evaluation.evaluate_model import B4DLEvaluator`）+ CLI（`__main__` 守卫在 :761-762 调 `main()`）。模块 docstring :1-26 明言 "Usable both as a library ... and as a CLI"。

**唯一类 `B4DLEvaluator`**（:187-553）：
- 常量：`SIMPLE_TASKS`（:190）、`COMPLEX_TASKS`（:191）、`ALL_TASKS`（:192）、`TASK_ALIASES`（:195-204）。
- `__init__(self, use_gpt=False, gpt_api_key=None, gpt_model="gpt-4o", bert_model_path=None)`（:210-248）：初始化 `self.metric_backend: Dict[str, Any] = {}`（:219，注释 "serialized alongside the metrics so every results file records which implementations produced its numbers"）。
- **metric_backend 写入位置**（:240-248）：
  ```python
  self.metric_backend['bertscore'] = {'model_type': ..., 'num_layers': ..., 'rescale_with_baseline': False}
  self.metric_backend['bleu4'] = ('pycocoevalcap-corpus' if HAS_COCO_BLEU else 'nltk-sentence-smoothed')
  self.metric_backend['meteor'] = ('pycocoevalcap-meteor-1.5' if HAS_COCO_METEOR else 'nltk-meteor-1.0')
  ```
  （bertscore 具体值在 :240-244 由 `_resolve_bertscore_model` 决定，:250-274。）

**compute_* 函数清单**：`compute_accuracy` :310-315；`compute_miou` :317-340；`compute_bleu4` :343-371；`compute_meteor` :373-401；`compute_rouge_l` :403-408；`compute_bertscore` :410-438；`compute_gpt_score` :440-471。

**compute_meteor 后端优先级原文**（:373-401）：
```python
def compute_meteor(self, predictions, ground_truths):
    """METEOR via the pycocoevalcap Meteor-1.5 jar (needs java).
    Fallback: NLTK METEOR-1.0-style (no paraphrase table, different
    parameters) — systematically higher and NOT comparable to the paper.
    """
    ...
    if HAS_COCO_METEOR:
        meteor = None
        try:
            meteor = _DrainedMeteor()          # 封装 Meteor jar（:115-134，排空 stderr 防死锁）
            gts = {i: [gt] for i, gt in enumerate(ground_truths)}
            res = {i: [p] for i, p in enumerate(predictions)}
            score, _ = meteor.compute_score(gts, res)
            return float(score)
        except Exception as e:
            print(f"pycocoevalcap METEOR failed ({e}); falling back to NLTK.")
        ...
    if not HAS_NLTK:
        return 0.0
    scores = []
    for pred, gt in zip(predictions, ground_truths):
        pred_tokens = word_tokenize(pred.lower())
        gt_tokens = word_tokenize(gt.lower())
        scores.append(meteor_score([gt_tokens], pred_tokens))
    return float(np.mean(scores)) if scores else 0.0
```
即：jar 优先（corpus 级）；任何异常/NLTK 缺失时逐样本 NLTK `meteor_score` 平均。compute_bleu4 同构（:343-371）：pycocoevalcap corpus BLEU 优先，fallback NLTK sentence_bleu method1。

**依赖守卫**（模块顶层，:42-147）：NLTK（:42-84，含 `_ensure_nltk_data()` :55-80 自动下载 punkt_tab/punkt/wordnet/omw-1.4；METEOR 兼容旧版 `single_meteor_score` :47-52）；rouge_score :86-91；bert_score :93-98；pycocoevalcap Bleu :104-110；pycocoevalcap Meteor + `_DrainedMeteor` :112-140；openai :142-147。头部注释 :100-103 明确声明 pycocoevalcap "are the fact-standard backends behind the paper's BLEU-4 / METEOR numbers; the NLTK implementations below are documented fallbacks that score systematically higher and are NOT comparable to the paper."

**save_results**（:548-553）：
```python
json.dump({'per_task_metrics': all_metrics, 'final_scores': final_scores,
           'metric_backend': self.metric_backend}, f, indent=2)
```

**CLI argparse 参数**（main() :683-694）：`--predictions`、`--ground_truth`、`--output`（default `evaluation_results.json`）、`--use_gpt`、`--gpt_api_key`（default env OPENAI_API_KEY）、`--gpt_model`（default gpt-4o）、`--demo`、`--bert_model`。evaluate_all :501-519 按 `ALL_TASKS` 逐任务调 `evaluate_task`（:474-499），后者对 complex 任务算 bleu4/meteor/rouge_l/bertscore(+gpt_score)，简单任务算 accuracy/miou。

## 2. mllm/evaluation/test_b4dl.py（686 行）

**CLI 参数全清单**（main() argparse :389-465）：`--model_base`（required）、`--pretrain_mm_mlp_adapter`、`--stage2`、`--stage3`、`--feat_folder`（required）、`--test_data`、`--predictions_dir`、`--scene_metadata`、`--nuscenes_root`、`--output`（default predictions.json）、`--metrics_output`（default evaluation_results.json）、`--max_samples`（-1=all）、`--checkpoint_interval`（50）、`--dtype`（float16/bfloat16）、`--use_gpt`/`--gpt_api_key`/`--gpt_model`、`--ego_meta`、`--frame_motion`、`--sequence_metadata`、`--answer_frames`、`--no_4dlidar`、`--no_meta`、`--per_sequence`、`--whole_scene`。

**评测主流程**：逐任务推理（`iter_items_by_task` :260-270，每任务 `for i, it in enumerate(pbar)` :570）；特征切片选择 :591-599（`whole_scene` → 全场景；`per_sequence` → `resolve_feat_slice` :594-597）；query 由 `build_query` :601-612 构造（含 `<4DLiDAR>/<video>/<meta>` metatoken 注入、`answer_frames` GT 帧回退解析 :328-332）。推理用 `vtimellm.inference.inference`（:382-384），模型加载 `load_pretrained_model(args, stage2, stage3)` :522-523。

**evaluator 调用位置与方式**（:659-671）——推理全部完成**之后一次性**对 `results` 调 `evaluator.evaluate_all(results)`（不是逐任务）：
```python
evaluator = B4DLEvaluator(use_gpt=args.use_gpt, gpt_api_key=args.gpt_api_key, gpt_model=args.gpt_model)
per_task = evaluator.evaluate_all(results)
final = evaluator.compute_final_score(per_task)
...
evaluator.save_results(per_task, final, args.metrics_output)
```

**predictions.json 保存结构**（:644-652）：6 个 canonical task key（existence/binary_qa/time_grounding/description/temporal_understanding/comprehensive_reasoning），每个 `{predictions, ground_truths, questions}`（`_task` 元数据被剥离，:645-648）。实测 b3 产物各数组等长（existence 3770 / binary_qa 7525 等，共 30,145 条）。

**断点续跑**：每 `checkpoint_interval` 样本 + 每任务完成时原子写 `<output>.ckpt`（`_save_checkpoint` :374-379，内容 `{"results": results}`；:626-639）。启动时若有 .ckpt 则加载并按 `len(results[task].get('predictions',[]))` 决定该任务起点（:532-563）；干净退出后删除 .ckpt（:654-657）。注意 `completed_ids`（:543）被构建但实际跳过逻辑按长度计数。

**metrics.json 由谁写**：test_b4dl.py 自身（:671 `save_results`）——不经 evaluate_model.py 的 main()。

## 3. 现有评测运行脚本命令实例

**冻结命令** `mllm/scripts/run_mixed_eval_after_train.sh`（= B0 锁定的 §2.2 逐字命令；脚本头部注释 "与 stage23_seqv3 同口径：--per_sequence --frame_motion --sequence_metadata --answer_frames"）。其 `run_eval()` 实参（:77-92，wrapper 行号）：
```bash
timeout 18000 python3.10 -u evaluation/test_b4dl.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 "$CKPT_DIR" \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --test_data ./b4dl_dataset/test_qa.json \
    --ego_meta ./b4dl_dataset/ego_metadata.json \
    --frame_motion ./b4dl_dataset/ego_frame_motion.json \
    --sequence_metadata ../encoders/lidarclip/annotations/sequence_metadata.json \
    --per_sequence --answer_frames \
    --output "$EVAL_OUT/predictions.json" --metrics_output "$EVAL_OUT/metrics.json"
```
前置环境（:63-66）：`export HF_HUB_OFFLINE=1 HF_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；显存门控 <14GB 时 sleep 600 重试；评测后仅 `cat "$EVAL_OUT/metrics.json"`（:120-121），不与其他 JSON 比对。

**B1 变体** `mllm/scripts/run_b1_pipeline.sh:110-129`：同参数但换 `--stage2 ...-stage2-full-seqv3-mixed-b1` 并带 `--sequence_metadata ../encoders/lidarclip/annotations/sequence_metadata.json`、`--per_sequence --answer_frames`（**无 `--whole_scene`**，切片口径）；评测后 :134-139 python 打印 final_scores 并注释对照 "B0：acc 0.7629 / mIoU 0.2696，ΔmIoU > +0.013 才显著"。

**B2/B3/B4a 变体**：`run_b2_pipeline.sh:60-75`（B2 用系统 python）、`run_b3_pipeline.sh:56-71`、`run_b4a_pipeline.sh:56-71`（后两者显式 `/root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -u evaluation/test_b4dl.py`）：同参数加 **`--whole_scene --per_sequence --answer_frames`**，仅 `--stage2` checkpoint 名与 `$EVAL_OUT`（`./eval_results/stage2_full_seqv3_mixed_b{2,3,4a}`）不同。评测后各自打印 final_scores 与上一代号对照（B3 脚本 :83-84 打印 "B3 评测结果（对照论文 mIoU 0.311/acc 0.762；B2 0.1992/0.7649）"；B4a 脚本 :80-87 对照 "B3 mIoU 0.3467/acc 0.7526"）。

**对比对象**：这些脚本只打数字，跨版本对比靠 `docs/learn docs/B4DL_工作进度总结_20260906.md` 第二节总表（论文 vs B0-B3 的 accuracy/mIoU/bleu4/rouge_l/meteor/bertscore）完成，数据源为各 `eval_results/*/metrics.json` 的 `final_scores`。

## 4. 规则文档原句

**`docs/learn docs/B4DL_基线锁定与对比规则_20260829.md`**（B0 = seqv3-mixed 基线锁定规则）：
- 引言（:4）："后续所有训练改进（stage1 补数据、编码器更换、超参调整等）一律与 B0 同口径对比。"
- 2.2 冻结命令说明（:44-45）："口径要点：`--per_sequence --frame_motion --sequence_metadata --answer_frames`（oracle 输入选择，GT 仅用于选输入帧/metatoken，不进 prompt）；`--dtype` 默认 float16；greedy 解码。"
- 五.1 同口径原则（:93-94）："**同口径原则**：新模型评测必须用 §2.2 命令原样执行（仅换 `--stage2` checkpoint 路径与输出目录）；数据文件 MD5 与 §2.1 一致才可与 B0 对比。"
- 五.3 基线重置条件（:96-99）："修改评测脚本 mIoU/指标计算逻辑或 `--answer_frames` 口径"、"更换 LiDAR-CLIP 编码器或重新提取 `stage2_features`（特征语义变化，跨代不可比）" 即开新代号。
- 六.2（:106）："**BLEU-4 为 NLTK 实现**（method1 smoothing）：与 pycocoevalcap 实现存在已知虚高差异（见 `B4DL_评测方法学陷阱` 记忆/文档）；论文实现未披露，跨论文对比时以"同实现纵向对比"为准。"（注：此条写于 08-29 晚修正前；§3.1 表里 B0 报 NLTK METEOR 0.3344 / BERTScore 0.9734。）

**`docs/learn docs/B4DL_全仓库论文对齐审查_20260829.md`**（修正口径冻结处，:150-151）：
"修正口径已冻结：结果 JSON 自带 `metric_backend` 记录，后续 B0/B1 对比必须同后端（pycocoevalcap 语料级 BLEU + Meteor-1.5 jar + roberta-large L17 BERTScore）。"
其表（:140-149）METEOR 行："METEOR | 0.275 | 0.3344 | **0.1729 ⚠️ 低于论文** | Meteor-1.5 jar 标准后端；换库后反而显著低于论文，说明论文所用 METEOR 变体仍未知（NLTK 0.334 虚高、jar 0.173 偏低，两头不靠），唯一遗留的口径考证项"。

**`docs/learn docs/B4DL_工作进度总结_20260906.md`**（当前最新口径）：
- 五、评测方法学表（:55-57）："BLEU-4 | ✅ pycocoevalcap 语料级（0.0965 vs 论文 0.095）"；"METEOR | ⚠️ 0.1747 vs 论文 0.275，B0-B3 一致偏低，Meteor-1.5 jar 口径存疑（唯一未对齐指标，待考证论文所用变体）"。
- 七.4（:77）："**METEOR 口径**：0.175 vs 0.275，唯一未对齐指标；"
- 二表（:21）meteor 行：`0.275 | 0.1729 | 0.1750 | 0.1747 | 0.1747`（B0-B3 全为 jar 值，B0 已取 0.1729 而非旧 NLTK 0.3344）。

**`docs/B4DL_复现方案.md`**：无口径规则性表述；仅 §四 Step 8 方案 A（:583-594）给出 remote4b4dl 评测用法与 `evaluation/` 两文件职责，:640 汇总表注明文本指标用 "`pycocoevalcap`（Bleu/Meteor/Rouge）+ `rouge-score` + `bert-score`（默认模型 F1）"。`requirements.txt` 依赖行 :222/:279：`pip install bert-score==0.3.13 pycocoevalcap==1.2 nltk rouge-score openai`。

**CLAUDE.md**：:19 "评测的 METEOR 走 pycocoevalcap Meteor-1.5 jar，需系统 java：`apt-get install -y --no-install-recommends default-jre-headless`"。评测示例命令在 :184-201（`conda run -n wqlc python evaluation/test_b4dl.py ...`，per_sequence 需 `--frame_motion + --sequence_metadata`）；:251 "⚠️ 训练和评测必须用相同代际的格式：seqv2 模型评测时加 `--per_sequence --frame_motion --sequence_metadata`；seq 模型只加 `--per_sequence`；旧模型都不加。"

## 5. NLTK 可用性（wqlc 环境）

- `/root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -c "import nltk"` → OK，版本 **nltk 3.10.0**。
- `from nltk.tokenize import word_tokenize` + `import nltk.translate.meteor_score` → 均 OK（纯 import，未触发下载/计算）。
- 同环境 pycocoevalcap 可 import（`pycocoevalcap.bleu.bleu.Bleu`、`pycocoevalcap.meteor.meteor.Meteor` 均成功），且 `/usr/bin/java` 存在 → evaluate_model.py 中 `HAS_COCO_METEOR` 将为 True，现默认走 jar 而非 NLTK（**若要做 NLTK 后端评测，需强制改 `metric_backend`/`compute_meteor` 逻辑或临时令 `HAS_COCO_METEOR=False`，否则 NLTK 分支不会被执行**）。
- 注意：NLTK 3.10 下 `nltk.tokenize.word_tokenize` 需要 punkt_tab/punkt 数据（evaluate_model.py :55-80 有自动下载逻辑，仅在 LookupError 时静默下载）。

## 6. 历史产物 NLTK vs pycocoevalcap 数值共存

`mllm/eval_results/stage2_full_seqv3_mixed/`（B0 目录，即 docs 中 "b0 可用 metrics_recomputed_20260829.json"）两文件为同一份 predictions.json 的两种后端算值：

- **metrics.json（旧，NLTK 时代，git commit 3d2e9db 之前产出）**：结构仅 `{'per_task_metrics', 'final_scores'}` 两个顶层 key，**无 metric_backend 字段**；complex 任务含 `gpt_score: 0.0`（现代码已改为记 null）；数值为虚高口径——final `meteor: 0.3343528072238027`、`bleu4: 0.1079527711646893`、`bertscore: 0.9734472433725992`（层号 24 bug）。
- **metrics_recomputed_20260829.json（新，jar 口径）**：结构 `{'per_task_metrics', 'final_scores', 'metric_backend'}`，含 backend 块：`"meteor": "pycocoevalcap-meteor-1.5"`, `"bleu4": "pycocoevalcap-corpus"`, `"bertscore": {"model_type": "/root/autodl-tmp/wql/mmb4dl/models/roberta-large", "num_layers": 17, "rescale_with_baseline": false}`；final `meteor: 0.17287597997052098`、`bleu4: 0.09734052836342329`、`bertscore: 0.8972703417142233`；accuracy/miou/rouge_l 逐位不变（0.7629/0.2696/0.3244）。差异由 commit `3d2e9db`（"fix: 评测口径对齐论文（审查 A1-A5）… METEOR 切 Meteor-1.5 jar（0.3344→0.1729）… 结果 JSON 随附 metric_backend"）引入。
- 后续产物 b1/b2/b3（`eval_results/stage2_full_seqv3_mixed_b{1,2,3}/metrics.json`）均为新结构（三 key + metric_backend 三字段），final meteor 0.1750/0.1747/0.1747、bleu4 0.0994/0.0959/0.0965、bertscore 0.8979/0.8967/0.8967。

## 对"NLTK METEOR 后端评测"方案的关键提示（供设计参考）

1. 当前代码 NLTK METEOR 只是 fallback（evaluate_model.py:373-401），pycocoevalcap jar 可用时不会触发；NLTK 3.10 的 `meteor_score` 需要预分词（代码 :398-400 已做 word_tokenize）。
2. 若要"同后端对比"结论，最省事的合规路线是沿用 test_b4dl.py:671 的 save_results（自动带 metric_backend）；若改用 evaluate_model.py CLI 重算 predictions.json，则 CLI 参数为 --predictions/--output（evaluate_model.py:683-694），输出仍含 metric_backend。
3. 规则文档口径已从 NLTK（0.3344，08-29 前）冻结切换为 jar（0.1729-0.1747，08-29 后，"同后端 pycocoevalcap 语料级 BLEU + Meteor-1.5 jar + roberta-large L17"）；任何 NLTK 后端结果须自带 metric_backend 标记并在文档声明，否则与 09-06 进度表的 B0-B3 jar 值不可比。

