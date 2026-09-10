# 探索 B4DL 复现仓库 /root/autodl-tmp/wql/mmb4dl 的评测侧代码与产物，目的是评估"...

| 项 | 值 |
|---|---|
| 会话 ID | `sess_subagent_agent_724947e0-28fe-4b7b-be5a-fee329ea822a` |
| 工作目录 | `/root/autodl-tmp/wql/mmb4dl` |
| 父会话 | `sess_cbeefbda-5c90-4e6b-b689-01f850ea4973` |
| 时间 | 2026-09-06 15:16 → 2026-09-06 15:19 |
| 模型 | GLM-5.3-Flash |
| 消息数 | 9（文本块 7，工具调用 24） |

---

### ASSISTANT  ·  `GLM-5.3-Flash`

> 🔀 模型切换：None → builtin:bigmodel-start-plan/GLM-5.3-Flash


### USER

探索 B4DL 复现仓库 /root/autodl-tmp/wql/mmb4dl 的评测侧代码与产物，目的是评估"TG 预测 vs GT 可视化分析"的实施成本。搜索广度：medium。只读操作，不要修改任何文件。

需要回答的问题：

1. **predictions.json 结构**：用 wqlc 环境 python（/root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python）加载 /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b3/predictions.json，打印：总条数、前 2 条完整记录的 keys 和内容（截断长文本）、task 类型字段取值分布、TG 样本里预测区间和 GT 区间分别存在哪些字段（原始文本还是解析后的 start/end）。同样看一眼 metrics.json 全文。

2. **评测代码**：读 /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/test_b4dl.py（重点：TG 预测解析逻辑、mIoU 计算、--answer_frames 的 oracle 语义、predictions.json 落盘字段、metric 计算在哪个文件）。报告关键函数名和行号。

3. **现有分析脚本**：B3 分析文档里提到过"TG 预测形态分析"（start 分布直方图、−3.9 帧偏早、高帧段覆盖统计）——在 /root/autodl-tmp/wql/mmb4dl/scripts/ 和 mllm/ 下找生成这些统计的脚本（可能叫 analysis/tg_*/bias 之类，也可 git log --oneline -30 里找线索）。如果找不到脚本，说明可能是临时 one-off 分析。

4. **可视化依赖与素材**：
   - wqlc 环境 python 是否有 matplotlib（python -c "import matplotlib; print(matplotlib.__version__)"）；
   - 仓库里是否有可用的点云/标注渲染素材：找 nuScenes 原始数据路径配置（grep -r "nuscenes" scripts/ mllm/ 里的路径配置，或 datageneration 里的数据根目录），确认 .pcd.bin 点云文件、nuScenes 标注（sample_annotation）是否在服务器上可访问（只 ls 目录确认存在即可，不要加载大文件）。

5. **eval_results 其他目录**：/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/ 下 b0(baseline stage2_full_seqv3_mixed)、b1、b2 的 predictions.json 是否都在（文件大小即可），供后续跨版本对比可视化。

输出：每个问题给出简明结论 + 关键文件:行号引用 + predictions.json 的实际字段结构示例。


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

The user wants me to explore the B4DL repo's evaluation-side code and artifacts. Let me break down the tasks:

1. Load predictions.json with wqlc python, print stats
2. Read test_b4dl.py
3. Find existing analysis scripts
4. Check matplotlib, nuScenes data paths
5. Check eval_results other dirs

Let me start with parallel operations. First, let me do several independent calls in parallel:
- Run python to analyze predictions.json
- Read test_b4dl.py
- List eval_results directory
- Search for analysis scripts

Let me be efficient with parallel tool calls.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -c "
import json
p='/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b3/predictions.json'
d=json.load(open(p))
print('TYPE:',typ
…（已截断，原文 411 字符）
description: Inspect top-level structure of predictions.json
```

</details>

<details><summary>输出摘要</summary>

```
TYPE: <class 'dict'>
TOP KEYS: ['existence', 'binary_qa', 'time_grounding', 'description', 'temporal_understanding', 'comprehensive_reasoning']
existence <class 'dict'> 3
binary_qa <class 'dict'> 3
time_grounding <class 'dict'> 3
description <class 'dict'> 3
temporal_understanding <class 'dict'> 3
comprehensive_reasoning <class 'dict'> 3
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls -la /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/ 2>/dev/null && echo "---" && ls -la /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/*/ 2>/dev/null | head -60
description: List eval_results directories with file sizes
```

</details>

<details><summary>输出摘要</summary>

```
total 1424
drwxr-xr-x 11 root root   4096 Sep  5 13:59 .
drwxr-xr-x 13 root root   4096 Aug 29 21:20 ..
-rw-r--r--  1 root root 479487 Aug 22 21:07 eval_log.txt
drwxr-xr-x  2 root root     86 Sep  2 16:34 official_ckpt_nometa
drwxr-xr-x  2 root root    113 Sep  2 16:05 official_model_stage2
-rw-r--r--  1 root root 692905 Aug 22 21:07 seq_predictions.json.ckpt
-rw-r--r--  1 root root  30254 Aug 23 13:21 smoke_2000_log.txt
-rw-r--r--  1 root root    991 Aug 23 13:21 smoke_2000_metrics.json
-rw-r--r--  1 root root  84975 Aug 23 13:21 smoke_2000_predictions.json
-rw-r--r--  1 root root    947 Aug  9 01:01 smoke_meta_metrics.json
-rw-r--r--  1 root root   5127 Aug  9 00:53 smoke_meta_predictions.json
-rw-r--r--  1 root root  12467 Aug 22 21:09 smoke_predictions.json.ckpt
-rw-r--r--  1 root root
…（已截断，原文 4346 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls /root/autodl-tmp/wql/mmb4dl/scripts/ 2>/dev/null; echo "===MMLL==="; ls /root/autodl-tmp/wql/mmb4dl/mllm/ 2>/dev/null
description: List scripts and mllm directories
```

</details>

<details><summary>输出摘要</summary>

```
===MMLL===
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

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && git log --oneline -30 2>/dev/null
description: Show last 30 git commits
```

</details>

<details><summary>输出摘要</summary>

```
ff7293b docs: B3 整场景+meta 修复评测分析归档——mIoU 0.3467 超论文 0.311 全链复盘（B1 错位分析→B2 整场景反证→官方模型评测 0.174 证实论文完整模型未发布→ego_text relative-to-previous 修复→B3 闭环）；附 TG 预测形态验证与残余问题清单
448ea7a fix: b3 训练脚本 LOG 文件名误写为 b2（sed 漏替换下划线命名）；清除 b3 空壳 checkpoint-400（保存中断残留，阻塞 sort -V 续训）
252c478 feat: B3 pipeline——整场景+修复版 metatoken（meta2）重训与评测链，显存门控 72h 窗口（与 CoRViD 共用 GPU），评测同口径 --whole_scene --per_sequence --answer_frames
adb0d8f feat: B3 训练脚本（整场景 + 修复版 meta2 数据，148k 3ep）——对齐论文完整配置（整场景输入 + relative-to-previous metatoken）
4413b6b fix: ego_text.py metatoken 渲染对齐论文 relative-to-previous 语义——原 first frame 位置恒 'at the starting position'、运动用前向(next)差分；论文 §4.1/Figure 6 定义为每帧相对前一帧的方向/位置/速度/转向/加速度（样例 'slightly ahead and to the right'）。first>0 时改用 spd_prev/yaw_prev/acc_prev + 相对前一帧位移方向，帧 0 保留 fallback；新增 re_render_meta.py 对已注入 J
…（已截断，原文 5922 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -c "import matplotlib; print('matplotlib', matplotlib.__version__)" 2>&1
description: Check matplotlib in wqlc env
```

</details>

<details><summary>输出摘要</summary>

```
matplotlib 3.5.2
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Good progress. The predictions.json is a dict of task_type -> dict with 3 keys. Let me dig deeper into structure. Also need to:
- Read test_b4dl.py
- Find analysis scripts (scripts/ dir listing returned nothing visible? Actually the first part printed nothing for scripts/ — the output shows "===MMLL===" immediately, meaning /root/autodl-tmp/wql/mmb4dl/scripts/ doesn't exist or is empty). Let me check where scripts are — mllm/scripts exists.
- Find TG analysis scripts.
- Check nuScenes data paths.
- metrics.json for b3.

Let me run parallel calls.

</details>

初步结构清楚了。predictions.json 是 dict: task_type -> 子 dict。继续深挖 b3 的 TG 结构、metrics.json、评测代码和分析脚本。

**🔧 Bash**

<details><summary>输入</summary>

```
command: /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -c "
import json
p='/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b3/predictions.json'
d=json.load(open(p))
print('=== task t
…（已截断，原文 348 字符）
description: Show inner structure of each task type in predictions.json
```

</details>

<details><summary>输出摘要</summary>

```
=== task types and inner structure ===
existence <class 'dict'> ['predictions', 'ground_truths', 'questions']
binary_qa <class 'dict'> ['predictions', 'ground_truths', 'questions']
time_grounding <class 'dict'> ['predictions', 'ground_truths', 'questions']
description <class 'dict'> ['predictions', 'ground_truths', 'questions']
temporal_understanding <class 'dict'> ['predictions', 'ground_truths', 'questions']
comprehensive_reasoning <class 'dict'> ['predictions', 'ground_truths', 'questions']
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls -la /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b3/ /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b2/ /root/autodl-tmp/wql/mmb4dl/mllm/eval_results
…（已截断，原文 304 字符）
description: List b0/b1/b2/b3 eval_results directories with sizes
```

</details>

<details><summary>输出摘要</summary>

```
/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed/:
total 8164
drwxr-xr-x  2 root root     106 Aug 29 21:26 .
drwxr-xr-x 11 root root    4096 Sep  5 13:59 ..
-rw-r--r--  1 root root    1094 Aug 27 19:28 metrics.json
-rw-r--r--  1 root root    1266 Aug 29 21:26 metrics_recomputed_20260829.json
-rw-r--r--  1 root root 8345202 Aug 27 19:27 predictions.json

/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b1/:
total 14044
drwxr-xr-x  2 root root      86 Sep  1 04:08 .
drwxr-xr-x 11 root root    4096 Sep  5 13:59 ..
-rw-r--r--  1 root root 6030096 Sep  1 04:10 eval_log.txt
-rw-r--r--  1 root root    1268 Sep  1 04:10 metrics.json
-rw-r--r--  1 root root 8332277 Sep  1 04:08 predictions.json

/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv
…（已截断，原文 1458 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls -R /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/ /root/autodl-tmp/wql/mmb4dl/mllm/scripts/ 2>/dev/null | head -80; echo "===root scripts?==="; ls /root/autodl-tmp/wql/mmb4dl/ 2>/dev/null
description: List evaluation and mllm/scripts directories
```

</details>

<details><summary>输出摘要</summary>

```
/root/autodl-tmp/wql/mmb4dl/mllm/evaluation/:
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

/root/autodl-tmp/wql/mmb4dl/mllm/evaluation/__pycache__:
build_test_split.cpython-310.pyc
build_test_split.cpython-311.pyc
evaluate_model.cpython-310.pyc
evaluate_model.cpython-311.pyc
evaluate_model.cpython-313.pyc
split_dataset.cpython-310.pyc
test_b4dl.cpython-310.pyc
test_b4dl.cpython-311.pyc
test_b4dl.cpython-313.p
…（已截断，原文 2008 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -rn "tg_\|tg\b\|time_grounding" /root/autodl-tmp/wql/mmb4dl/mllm/scripts/*.py 2>/dev/null | grep -i "analysis\|hist\|bias\|dist\|start" | head -20; echo "===find analysis scripts anywhere==="; fi
…（已截断，原文 378 字符）
description: Search for TG analysis scripts
```

</details>

<details><summary>输出摘要</summary>

```
===find analysis scripts anywhere===
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: wc -l /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/test_b4dl.py 2>/dev/null; ls /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/ 2>/dev/null
description: Check test_b4dl.py size and evaluation dir
```

</details>

<details><summary>输出摘要</summary>

```
686 /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/test_b4dl.py
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
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

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

No dedicated TG analysis script found in the quick find. There's `analyze_behavior.py` in evaluation dir. Let me check it, and also search docs for the "TG 预测形态分析" mentioned in B3 analysis doc (ff7293b commit). Let me check what the commit changed, and look for scripts in docs or backups.

Also need:
- predictions.json TG samples detail
- metrics.json full
- test_b4dl.py reading (686 lines)
- nuScenes data paths

Parallel batch:
1. python: print TG first 2 records + field distributions + counts per task
2. cat metrics.json b3
3. Read test_b4dl.py (686 lines, read fully)
4. git show ff7293b --stat to see files touched
5. grep for nuscenes paths

</details>

找到关键线索：`analyze_behavior.py`，以及最新 commit ff7293b 是 B3 分析文档归档。继续并行深挖。

**🔧 Bash**

<details><summary>输入</summary>

```
command: /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -c "
import json
p='/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b3/predictions.json'
d=json.load(open(p))
for task, sub in 
…（已截断，原文 534 字符）
description: Print counts per task and first 2 TG records
```

</details>

<details><summary>输出摘要</summary>

```
existence n = 3770 | gt = 3770 | q = 3770
binary_qa n = 7525 | gt = 7525 | q = 7525
time_grounding n = 2783 | gt = 2783 | q = 2783
description n = 3770 | gt = 3770 | q = 3770
temporal_understanding n = 4757 | gt = 4757 | q = 4757
comprehensive_reasoning n = 7540 | gt = 7540 | q = 7540

=== TG sample 0 ===
  [predictions] (str) from frame 018 to frame 026.
  [ground_truths] (str) from frame 006 to frame 014.
  [questions] (str) When do the pedestrians appear in the scene?
=== TG sample 1 ===
  [predictions] (str) from frame 012 to frame 020.
  [ground_truths] (str) from frame 006 to frame 014.
  [questions] (str) When does the car behind start approaching closer to the ego vehicle?
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cat /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b3/metrics.json
description: Show b3 metrics.json full content
```

</details>

<details><summary>输出摘要</summary>

```
{
  "per_task_metrics": {
    "existence": {
      "accuracy": 0.6761273209549071
    },
    "binary_qa": {
      "accuracy": 0.8291029900332226
    },
    "time_grounding": {
      "miou": 0.3467090076609961
    },
    "description": {
      "bleu4": 0.10546335683789328,
      "meteor": 0.19132778664934172,
      "rouge_l": 0.3170865313456506,
      "bertscore": 0.8966626524925232
    },
    "temporal_understanding": {
      "bleu4": 0.08527908131439146,
      "meteor": 0.1650673953348514,
      "rouge_l": 0.32646092078428257,
      "bertscore": 0.897441565990448
    },
    "comprehensive_reasoning": {
      "bleu4": 0.09866227107834157,
      "meteor": 0.1678388003198059,
      "rouge_l": 0.32654780256381316,
      "bertscore": 0.8959501385688782
    }
  },
  "final_scores": {
    "accur
…（已截断，原文 1268 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && git show ff7293b --stat | head -40
description: Show files changed in B3 analysis commit
```

</details>

<details><summary>输出摘要</summary>

```
commit ff7293b59b6c498d0ea4af8ad8843b35c7fb0425
Author: kelly <lilchen1@163.com>
Date:   Sun Sep 6 10:08:19 2026 +0800

    docs: B3 整场景+meta 修复评测分析归档——mIoU 0.3467 超论文 0.311 全链复盘（B1 错位分析→B2 整场景反证→官方模型评测 0.174 证实论文完整模型未发布→ego_text relative-to-previous 修复→B3 闭环）；附 TG 预测形态验证与残余问题清单

 ...50\266\205\350\256\272\346\226\207_20260906.md" | 92 ++++++++++++++++++++++
 1 file changed, 92 insertions(+)
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -rn "nuscenes" /root/autodl-tmp/wql/mmb4dl/mllm/scripts/*.sh /root/autodl-tmp/wql/mmb4dl/mllm/scripts/*.py 2>/dev/null | grep -iv "nuscenes_key\|#" | head -15; echo "===datageneration roots==="; 
…（已截断，原文 334 字符）
description: Find nuScenes data root path configuration
```

</details>

<details><summary>输出摘要</summary>

```
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_metatoken.sh:52:        --nuscenes_root) NUSCENES_ROOT="$2"; shift 2 ;;
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_metatoken.sh:65:            echo "Usage: $0 --nuscenes_root <path> [options]"
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_metatoken.sh:68:            echo "  --nuscenes_root <path>   nuScenes dataset root (required unless --skip_ego)"
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_metatoken.sh:83:    echo "Error: --nuscenes_root required (or use --skip_ego if ego_metadata.json exists)"
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_metatoken.sh:102:        --nuscenes_root ${NUSCENES_ROOT} \
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/convert_lidarllm_to_stage1.py:21:        --nuscenes_root /root/autodl-tmp/Datasets/nuScenes \
/root/auto
…（已截断，原文 3806 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Now I need:
1. evaluate_model.py — where mIoU is computed (parse TG prediction, IoU). Key functions + line numbers.
2. Check if TG analysis scripts exist anywhere — search more broadly (docs mentioned "TG 预测形态验证"). The B3 analysis doc is in docs/. Let me find the doc file from that commit. Also search for "histogram"/"直方图"/"−3.9"/"-3.9" in docs and repo.
3. nuScenes data path — check Config.DATAROOT/NUSCENES_ROOT in datageneration config, and verify directories exist on disk (ls). Also .pcd.bin files and sample_annotation.
4. Also check whether eval predictions questions are actual raw questions — note in test_b4dl.py line 621-623: questions stored are the query with video/4dlidar tags removed and <meta> stripped. So questions in predictions.json = question text without meta. Good detail.

Let me run parallel:
- grep mIoU/IoU in evaluate_model.py with line numbers, and read relevant portion
- find the B3 doc file name and grep for TG 形态/直方图/-3.9
- search whole repo for analysis scripts containing "miou" or "hist" outside evaluation
- check datageneration config for NUSCENES_ROOT default; ls the nuScenes dirs

</details>

TG 预测/GT 是原始文本格式（"from frame 018 to frame 026."）。接下来查 evaluate_model.py 的 mIoU 计算与 TG 解析、B3 分析文档里的 TG 形态分析线索、nuScenes 数据路径实际存在性。

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -n "def \|miou\|IoU\|iou\|frame\s\+(\d\|parse" /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/evaluate_model.py | head -50; echo "===total lines==="; wc -l /root/autodl-tmp/wql/mmb4dl/mllm/evaluatio
…（已截断，原文 219 字符）
description: Find function definitions and IoU logic in evaluate_model.py
```

</details>

<details><summary>输出摘要</summary>

```
14:    - Time Grounding        : mIoU (overlap of predicted vs GT temporal segments)
20:  final score. Accuracy = mean(Existence, Binary QA). mIoU = Time Grounding.
31:import argparse
51:        def meteor_score(refs, hyp):
55:    def _ensure_nltk_data():
122:        def __init__(self):
128:        def close(self):
210:    def __init__(self, use_gpt: bool = False, gpt_api_key: Optional[str] = None,
251:    def _resolve_bertscore_model(model_path: str) -> Tuple[str, int]:
278:    def canonical_task(cls, task: str) -> str:
286:    def normalize_answer(answer: str) -> str:
294:    def extract_time_grounding_frames(text: str) -> Optional[Dict[str, int]]:
298:        parseable interval must score IoU 0, not be credited with frame 0.
310:    def compute_accuracy(self, predictions: List[str], gro
…（已截断，原文 3560 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && git show ff7293b --name-only | tail -5; echo "===docs grep==="; grep -rln "直方图\|−3.9\|-3.9\|偏早\|预测形态" /root/autodl-tmp/wql/mmb4dl/docs/ /root/autodl-tmp/wql/mmb4dl/wi
…（已截断，原文 226 字符）
description: Find B3 analysis doc file and grep TG shape analysis mentions
```

</details>

<details><summary>输出摘要</summary>

```
Date:   Sun Sep 6 10:08:19 2026 +0800

    docs: B3 整场景+meta 修复评测分析归档——mIoU 0.3467 超论文 0.311 全链复盘（B1 错位分析→B2 整场景反证→官方模型评测 0.174 证实论文完整模型未发布→ego_text relative-to-previous 修复→B3 闭环）；附 TG 预测形态验证与残余问题清单

"docs/learn docs/B4DL_B3\346\225\264\345\234\272\346\231\257meta\344\277\256\345\244\215mIoU\350\266\205\350\256\272\346\226\207_20260906.md"
===docs grep===
/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_B3整场景meta修复mIoU超论文_20260906.md
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -n "DATAROOT\|NUSCENES_ROOT\|NUSCENES_VERSION" /root/autodl-tmp/wql/mmb4dl/datageneration/config.py 2>/dev/null || find /root/autodl-tmp/wql/mmb4dl/datageneration -maxdepth 1 -name "*.py" | head;
…（已截断，原文 273 字符）
description: Find datageneration config with data roots
```

</details>

<details><summary>输出摘要</summary>

```
34:    NUSCENES_ROOT = "/root/autodl-tmp/Datasets/nuScenes/"
35:    NUSCENES_VERSION = "v1.0-trainval" #v1.0-mini
37:    DATAROOT = "./data"
39:    METADATA_DIR = DATAROOT + "/metadata"
43:    GENERATED_DESCRIPTION_DIR = DATAROOT + "/generated_description"
44:    GENERATED_DATASET_DIR = DATAROOT + "/generated_dataset"
===Config file===
README.md
__pycache__
config.py
data
datanew
generate_dataset.py
generate_description.py
prompts.py
scripts
tools
utils.py
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls /root/autodl-tmp/Datasets/nuScenes/ 2>/dev/null; echo "===v1.0-trainval==="; ls /root/autodl-tmp/Datasets/nuScenes/v1.0-trainval/ 2>/dev/null | head -20; echo "===samples lidar==="; ls /root/autodl
…（已截断，原文 341 字符）
description: Verify nuScenes dataset exists on disk
```

</details>

<details><summary>输出摘要</summary>

```
LICENSE
dataset_info.list
maps
nuScenes-lidarseg-all-v1.0
samples
sweeps
test
train
trainval
v1.0-trainval
===v1.0-trainval===
attribute.json
calibrated_sensor.json
category.json
ego_pose.json
instance.json
log.json
map.json
sample.json
sample_annotation.json
sample_data.json
scene.json
sensor.json
v1.0-trainval
visibility.json
===samples lidar===
n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin
n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243547836.pcd.bin
n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915244048280.pcd.bin
n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915244548143.pcd.bin
n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915245048008.pcd.bin
34149
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/evaluate_model.py
```

</details>

<details><summary>输出摘要</summary>

```
278	    def canonical_task(cls, task: str) -> str:
279	        """Map any common spelling to one of the 6 canonical task keys."""
280	        if task is None:
281	            return task
282	        key = task.strip().lower().replace('-', '_').replace(' ', '_')
283	        return cls.TASK_ALIASES.get(key, key)
284	
285	    @staticmethod
286	    def normalize_answer(answer: str) -> str:
287	        """Normalize for exact-match accuracy (Existence / Binary QA)."""
288	        answer = answer.strip().lower()
289	        answer = re.sub(r'^(the answer is|answer:|a:)\s*', '', answer)
290	        answer = re.sub(r'[^\w\s]', '', answer)
291	        return answer.strip()
292	
293	    @staticmethod
294	    def extract_time_grounding_frames(text: str) -> Optional[Dict[str, int]]:
295	        """Extr
…（已截断，原文 3549 字符）
```

</details>

<details><summary>思考过程</summary>

Excellent. Almost everything found. Remaining:

1. Read the TG extraction + mIoU code in evaluate_model.py (lines ~294-345).
2. Read the B3 analysis doc to see if the "TG 预测形态分析" was a one-off (no script) — grep the doc for script names/commands used.
3. Check analyze_behavior.py briefly (what it does — maybe it's the analysis script).
4. Check that sample_annotation.json exists (confirmed above), and check scene folder count for scenes used in B4DL (e.g. dataset dir). Also check nuscenes-devkit installed in wqlc env (for rendering). Check pcd.bin confirmed (34149 files). Also check if there are B4DL scene folders with annotations? The B4DL benchmark uses nuScenes scenes. Good enough.

Also confirm: does the doc mention the exact one-off analysis (e.g., inline python)? Let me read the doc's TG 形态 section.

Also check for scripts in Claude_record or backups that might contain the one-off analysis. And check "official_ckpt_nometa" etc. — not needed.

Also want the TG sample counts (2783) and parsed distribution — not strictly needed but the doc says −3.9 frames early bias. Let me verify quickly with python: parse predictions vs GT start frames, compute mean diff. That directly validates what the visualization task needs and the doc claim. Also verify matplotlib present (3.5.2 done). Check nuscenes devkit in wqlc.

Parallel batch:
1. Read evaluate_model.py lines 278-345 (or 280-350).
2. Read B3 doc (grep relevant section first).
3. python: parse TG preds/gts, compute start diff mean, count unparseable.
4. python: check nuscenes devkit import in wqlc.
5. head analyze_behavior.py docstring.

</details>

核心数据都拿到了。补最后几块：evaluate_model.py 的 TG 解析与 mIoU 代码段、B3 分析文档里的 TG 形态分析章节、实测解析 TG 区间验证 −3.9 偏早、nuscenes-devkit 可用性。

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -n "形态\|直方图\|3.9\|偏早\|脚本\|python\|分析" "/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_B3整场景meta修复mIoU超论文_20260906.md" | head -30
description: Grep TG shape analysis section in B3 doc
```

</details>

<details><summary>输出摘要</summary>

```
9:### 2.1 起点：B1 的 mIoU 错位分析（08-31）
11:- B1（seqv3 序列切片输入）TG 预测形态正常（99.9% 可解析、区间长度 11.0 vs GT 10.4），但 49.3% 与 GT 零重叠；
22:用官方仓库自带 checkpoint（stage1 projector + stage2 LoRA，68695 simple 数据 × 2 epochs = 1074 步，与官方脚本完全吻合）+ 官方特征直接评测：
63:### 3.3 TG 预测形态（真实定位确认）
67:- 残余偏置：整体偏早 mean −3.9 帧（std 10.2）；高帧段覆盖不足（GT 25-38 占 14%，预测仅 4%，30+ 为 0）——先验残留而非塌缩，仍有提升空间。
77:   - TG 预测仍有 −3.9 帧偏早与高帧段欠覆盖（可作后续优化方向：如 meta 措辞、数据重平衡）。
83:- `adb0d8f` B3 训练脚本（meta2 数据）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -c "
import json, re
p='/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b3/predictions.json'
tg=json.load(open(p))['time_ground
…（已截断，原文 819 字符）
description: Parse TG intervals and verify start-frame bias
```

</details>

<details><summary>输出摘要</summary>

```
TG n = 2783 | unparseable pred: 3 | unparseable gt: 0
mean(pred_start - gt_start) = -3.90 frames, n=2780
pred_start dist (first 12 bins): [(0, 451), (4, 336), (5, 327), (6, 322), (8, 5), (9, 1), (10, 14), (11, 24), (12, 659), (13, 22), (16, 41), (17, 17)]
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -c "
try:
    import nuscenes; print('nuscenes-devkit OK', nuscenes.__file__)
except Exception as e: print('nuscenes-devkit FAIL:', e)
try:
    impor
…（已截断，原文 411 字符）
description: Check nuscenes-devkit in wqlc and inspect analyze_behavior.py
```

</details>

<details><summary>输出摘要</summary>

```
nuscenes-devkit OK /root/autodl-tmp/.conda-stuff/envs/wqlc/lib/python3.10/site-packages/nuscenes/__init__.py
pyquaternion OK
===analyze_behavior head===
import json, re
from collections import Counter

with open('/root/autodl-tmp/wql/mmb4dl/mllm/evaluation/predictions_full.json') as f:
    data = json.load(f)

print("=" * 60)
print("1. 答案模式分布")
print("=" * 60)

for task in ['existence', 'binary_qa', 'time_grounding', 'description', 'temporal_understanding', 'comprehensive_reasoning']:
    info = data[task]
    preds = info['predictions']
    gts = info['ground_truths']

    yes_count = sum(1 for p in preds if str(p).strip() == 'Yes.')
    no_count = sum(1 for p in preds if str(p).strip() == 'No.')
    time_count = sum(1 for p in preds if 'from frame' in str(p))
    short_obj = sum(1 for p 
…（已截断，原文 1677 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Everything is answered. Let me do one final quick check: was there a TG-specific analysis script (the "TG 预测形态验证")? Check the doc section 3.3 fully, and search git history or untracked files for one-off scripts. Also check `git status` for untracked scripts. Quick check.

</details>

最后确认一下 TG 形态分析是否有对应脚本残留（git status 未跟踪文件 + 文档 3.3 节全文）。

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && git status --short 2>/dev/null | head -20; echo "===doc section 3.3==="; sed -n '55,80p' "/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_B3整场景meta修复mIoU超论文_20260906
…（已截断，原文 204 字符）
description: Check untracked files and read doc section 3.3
```

</details>

<details><summary>输出摘要</summary>

```
 M encoders/lidarclip/logs/val_mse_probe.csv
 M encoders/lidarclip/logs/val_mse_probe.log
?? .zcode/
?? encoders/lidarclip/logs/anneal_chain.log
?? encoders/lidarclip/logs/extract_stage1_sample_token_b1.log
?? encoders/lidarclip/logs/extract_stage2_b1.log
?? encoders/lidarclip/logs/train_anneal_0829_2036.log
?? encoders/lidarclip/logs/train_loss.csv
?? mllm/training_logs/b1_eval_pipeline.log
?? mllm/training_logs/b1_pipeline.log
?? mllm/training_logs/b2_pipeline.log
?? mllm/training_logs/b3_pipeline.log
?? mllm/training_logs/stage1_162k_b1_20260830_165411.log
?? mllm/training_logs/stage1_162k_b1_20260830_185010.log
?? mllm/training_logs/stage2_full_seqv3_mixed_b1_20260830_213612.log
?? mllm/training_logs/stage2_full_seqv3_mixed_b1_20260831_135828.log
?? mllm/training_logs/stage2_full_seqv3
…（已截断，原文 2311 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

All questions answered. Now compile the final report.

Summary of findings:

**1. predictions.json structure**
- Path: /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b3/predictions.json (8,385,367 bytes)
- Top-level: dict of 6 task keys: existence, binary_qa, time_grounding, description, temporal_understanding, comprehensive_reasoning
- Each task: {predictions: [...], ground_truths: [...], questions: [...]} — parallel string lists
- Counts: existence 3770, binary_qa 7525, time_grounding 2783, description 3770, temporal_understanding 4757, comprehensive_reasoning 7540. Total 30,145.
- TG records: predictions/GTs are raw strings "from frame 018 to frame 026." — NOT parsed start/end. Parsing must be done at analysis time (evaluate_model.py:294 extract_time_grounding_frames). Verified: 3/2783 unparseable predictions, 0 unparseable GTs. mean(pred_start - gt_start) = −3.90 frames reproduces the doc's −3.9 claim.
- Note: "questions" stored in predictions.json are the query with <video>/<4DLiDAR> removed and <meta> block stripped (test_b4dl.py:621-623), so they're the bare question text. No scene_id per record — alignment with scenes requires matching question text to test JSON.
- metrics.json full text provided.

**2. Evaluation code test_b4dl.py**
- parse_frame_numbers: line 115-126 — extracts min/max "frame N" from text
- lookup_ego_for_qa: 129-153
- build_sequence_ranges: 156-165, find_containing_sequence: 168-178
- resolve_feat_slice: 181-224 — feature frame selection priority; --answer_frames semantics: for time_grounding questions with no frame refs in question, recover sequence + meta range from GT answer's frame range (lines 211-217, 328-332); GT used only for input frame/meta selection, never shown to model (argparse help 440-448)
- build_query: 273-364
- run_inference: 382-384
- main: 388; predictions saved at 644-651 (clean_results: predictions/ground_truths/questions per task); metrics computed via B4DLEvaluator at 660-663; paper reference table printed at 674-683
- Metric computation lives in mllm/evaluation/evaluate_model.py:
  - extract_time_grounding_frames: lines 293-307 — 3 regex patterns for "from frame X to frame Y"
  - compute_miou: lines 317-340 — closed-interval IoU, inclusive; unparseable → 0
  - compute_accuracy: 310-315
  - evaluate_task: 474, evaluate_all: 501, compute_final_score: 522, save_results: 548

**3. Existing analysis scripts**
- No dedicated TG analysis script found. mllm/evaluation/analyze_behavior.py is a hardcoded one-off answer-pattern analysis (points at predictions_full.json, prints answer mode distributions; imports json/re/Counter, no argparse, no plot).
- The "TG 预测形态" stats in B3 doc (docs/learn docs/B4DL_B3整场景meta修复mIoU超论文_20260906.md section 3.3, lines 63-67) were a one-off inline analysis; doc records: 99.9% parse rate, start distribution alignment, mean −3.9 frames (std 10.2), GT 25-38 coverage 14% vs pred 4%, 30+ = 0. No script committed (git log shows only the doc commit ff7293b). So implementing TG prediction vs GT visualization = writing a fresh small script (low cost, ~1h), all data is in predictions.json.

**4. Visualization deps and assets**
- matplotlib 3.5.2 present in wqlc env. nuscenes-devkit + pyquaternion also installed in wqlc.
- nuScenes root: /root/autodl-tmp/Datasets/nuScenes/ (datageneration/config.py:34 NUSCENES_ROOT, :35 version v1.0-trainval)
  - v1.0-trainval/ contains sample_annotation.json, sample.json, sample_data.json, ego_pose.json, scene.json etc.
  - samples/LIDAR_TOP/ has 34,149 .pcd.bin files — accessible.
- Features: mllm/lidarclip stage2 features <scene_id>.npy loaded from --feat_folder (test_b4dl.py:367-371)

**5. eval_results other dirs**
- stage2_full_seqv3_mixed (B0): predictions.json 8,345,202 B + metrics.json + metrics_recomputed_20260829.json
- b1: predictions.json 8,332,277 B + metrics.json + eval_log.txt
- b2: predictions.json 8,370,863 B + metrics.json + eval_log.txt
- b3: predictions.json 8,385,367 B + metrics.json + eval_log.txt
All four present → cross-version comparison visualization ready. Also stage2_full_seq (8.3MB), stage2_full_seqv2 (8.3MB), stage23_seqv3 (8.9MB), official_ckpt_nometa (1.7MB) available.

Also mention: only 3 unparseable predictions out of 2783 → visualization parsing is essentially lossless.

One caveat for visualization: predictions.json has no scene_id / sample index → per-scene overlay onto point clouds requires re-running eval with logging or joining questions to test JSON (questions stored are stripped query text). Actually wait — questions are the bare question text; the test_data file (data/test_qa.json?) could be joined by question+gt text matching. That's worth mentioning as implementation cost.

Let me write the final report concisely with file:line references.

</details>

探索完成。以下为完整报告。

# B4DL 评测侧探索报告（TG 预测 vs GT 可视化实施成本评估）

## 1. predictions.json 结构

文件：`/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed_b3/predictions.json`（8,385,367 B）

- **顶层结构**：dict，6 个 task 键，每个 task 下是三个**平行字符串列表**（无逐条 id、无 scene_id 字段）：
  ```
  {existence: {predictions: [], ground_truths: [], questions: []},
   binary_qa: {...}, time_grounding: {...}, description: {...},
   temporal_understanding: {...}, comprehensive_reasoning: {...}}
  ```
- **条数分布**：existence 3770 / binary_qa 7525 / **time_grounding 2783** / description 3770 / temporal_understanding 4757 / comprehensive_reasoning 7540（合计 30,145）
- **TG 样本实际内容**（区间是**原始文本，未解析**，需分析时自行 regex）：
  ```
  sample 0: pred="from frame 018 to frame 026."  gt="from frame 006 to frame 014."
            q="When do the pedestrians appear in the scene?"
  sample 1: pred="from frame 012 to frame 020."  gt="from frame 006 to frame 014."
  ```
- **实测解析验证**（wqlc python 正则 `frame\s+(\d+)`）：2783 条 TG 中仅 **3 条预测不可解析**（GT 0 条），`mean(pred_start − gt_start) = −3.90 帧`，与 B3 文档声称的 −3.9 完全吻合——可视化脚本解析成本几乎为零。
- **metrics.json 全文**（b3）：per_task 里 `time_grounding: {miou: 0.3467}`，existence acc 0.6761，binary_qa 0.8291，final_scores 含 accuracy/miou/bleu4/meteor/rouge_l/bertscore，附 metric_backend（BERTScore 用本地 `models/roberta-large` 17 层、BLEU=pycocoevalcap-corpus、METEOR=Meteor-1.5）。
- **注意**：`questions` 字段存的是去掉 `<video>/<4DLiDAR>/<meta>` 后的裸问题文本（`test_b4dl.py:621-623`），predictions.json 里**没有 scene_id**——做“逐场景叠加点云”需要拿 questions+gt 文本与 test JSON 反向 join。

## 2. 评测代码

**`/root/autodl-tmp/wql/mmb4dl/mllm/evaluation/test_b4dl.py`**（686 行，推理侧）：
- `parse_frame_numbers()` L115-126：从文本提 min/max frame 号
- `resolve_feat_slice()` L181-224：特征帧选择优先级（feat_indices > feat_range > containing sequence > answer_frames GT 回退 > 全场景）
- **`--answer_frames` oracle 语义**（L440-448 argparse 帮助 + L211-217/L328-332 实现）：对问题中无帧号引用的 TG 样本，用 **GT 答案里的帧区间**恢复该 QA 的输入序列切片和 metatoken 锚定范围；GT **只用于选输入帧/meta，不展示给模型**。这是评测与论文口径对齐的关键，也是可视化时需标注的“oracle 泄漏”点
- `build_query()` L273-364：拼 `<4DLiDAR>\n<video>\n<q>\n<meta> ...`
- 落盘字段：L644-651 `clean_results`（predictions/ground_truths/questions 三列表）；metrics 由 L660-663 `B4DLEvaluator.evaluate_all` 计算

**`/root/autodl-tmp/wql/mmb4dl/mllm/evaluation/evaluate_model.py`**（761 行，指标侧，TG 核心都在这）：
- `extract_time_grounding_frames()` **L293-307**：3 种 regex 模式解析 `{start_frame, end_frame}`，解析失败返回 None（计 IoU 0）
- `compute_miou()` **L317-340**：闭区间 IoU（inclusive，frame 6..14 = 9 帧），不可解析记 0
- `compute_accuracy()` L310-315、`evaluate_task()` L474、`evaluate_all()` L501、`compute_final_score()` L522、`save_results()` L548

## 3. 现有分析脚本：不存在，是 one-off

- 全仓库（scripts/、mllm/、datageneration/）**没有**专门的 TG 分布/直方图/偏置分析脚本；`git log` 里也只有文档 commit ff7293b（仅新增 1 个 md）
- 唯一近似物：`/root/autodl-tmp/wql/mmb4dl/mllm/evaluation/analyze_behavior.py` —— 硬编码路径指向旧 `predictions_full.json` 的答案模式统计（Yes/No/from-frame 占比），无 argparse、无画图，是更早的一次性脚本
- B3 文档 `/root/autodl-tmp/wql/mmb4dl/docs/learn docs/B4DL_B3整场景meta修复mIoU超论文_20260906.md` §3.3（L63-67）记录了 TG 形态结论：解析率 99.9%、start 分布形状对齐、mean −3.9 帧（std 10.2）、GT 25-38 段占 14% vs 预测 4%、30+ 为 0——统计是临时 inline 跑的，**TG 预测 vs GT 可视化需要新写脚本**（成本很低：数据全在 predictions.json，我已验证 5 分钟内可复算核心统计）

## 4. 可视化依赖与素材：全部就绪

- **matplotlib 3.5.2 存在**于 wqlc 环境；**nuscenes-devkit + pyquaternion 也已装好**（`/root/autodl-tmp/.conda-stuff/envs/wqlc/lib/python3.10/site-packages/nuscenes/`）
- **nuScenes 数据根**：`/root/autodl-tmp/Datasets/nuScenes/`（配置在 `/root/autodl-tmp/wql/mmb4dl/datageneration/config.py:34-35`，`NUSCENES_ROOT` + `v1.0-trainval`）
  - `v1.0-trainval/` 下 `sample_annotation.json`、`sample.json`、`sample_data.json`、`ego_pose.json`、`scene.json` 等齐全
  - `samples/LIDAR_TOP/` 有 **34,149 个 `.pcd.bin`** 点云文件，可直接访问
- 特征文件路径约定：`<feat_folder>/<scene_id>.npy`（`test_b4dl.py:367-371`）

## 5. eval_results 跨版本对比素材：b0/b1/b2/b3 全在

| 目录 | predictions.json | metrics.json |
|---|---|---|
| `stage2_full_seqv3_mixed/`（B0） | 8,345,202 B | 有 + `metrics_recomputed_20260829.json` |
| `..._b1/` | 8,332,277 B | 有 |
| `..._b2/` | 8,370,863 B | 有 |
| `..._b3/` | 8,385,367 B | 有 |

另有 stage2_full_seq / seqv2 / stage23_seqv3（8.3-8.9 MB）、`official_ckpt_nometa/`（1.7 MB，官方 checkpoint 评测）可用。

## 实施成本结论

TG 预测 vs GT 可视化**成本很低**：(a) 数据解析只需 regex（99.9% 可解析，已实测 −3.9 偏差可复现）；(b) matplotlib/devkit 已装；(c) b0-b3 四版 predictions.json 同构可直接叠对比直方图/散点。唯一成本点是 predictions.json 无 scene_id——若要做“具体场景的点云+标注帧区间叠加”，需用 questions 文本 join 回 test JSON 取 scene_id，再从 nuScenes samples/LIDAR_TOP 取对应帧渲染（devkit 可用，素材在盘）。

