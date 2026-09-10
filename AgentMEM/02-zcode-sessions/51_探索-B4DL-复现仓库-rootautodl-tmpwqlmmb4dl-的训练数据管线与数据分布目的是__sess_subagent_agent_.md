# 探索 B4DL 复现仓库 /root/autodl-tmp/wql/mmb4dl 的训练数据管线与数据分布，目的是...

| 项 | 值 |
|---|---|
| 会话 ID | `sess_subagent_agent_dcd244ce-0a56-4c56-bff7-39b24b842d14` |
| 工作目录 | `/root/autodl-tmp/wql/mmb4dl` |
| 父会话 | `sess_cbeefbda-5c90-4e6b-b689-01f850ea4973` |
| 时间 | 2026-09-06 15:22 → 2026-09-06 15:28 |
| 模型 | GLM-5.3-Flash |
| 消息数 | 11（文本块 5，工具调用 23） |

---

### ASSISTANT  ·  `GLM-5.3-Flash`

> 🔀 模型切换：None → builtin:bigmodel-start-plan/GLM-5.3-Flash


### USER

探索 B4DL 复现仓库 /root/autodl-tmp/wql/mmb4dl 的训练数据管线与数据分布，目的是评估两个 B4 实验的实施成本：(a) metatoken 措辞调优；(b) TG 高帧段数据重平衡。搜索广度：medium。只读操作（可以用 python 只读统计 JSON，但不要写任何文件/不要修改数据）。

背景：148k 训练数据 /root/autodl-tmp/wql/mmb4dl/mllm/b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json（B3 用），metatoken 由 ego_text.py 渲染（relative-to-previous 语义），TG 答案形如 "from frame XXX to frame YYY"。B3 残余问题：TG 预测整体偏早 mean −3.9 帧、GT 25-38 帧段占 14% 但预测只覆盖 4%（30+ 帧为 0）；existence 任务 acc 0.676 比 B0 低 0.05。

需要回答的问题：

1. **数据构建链**：metatoken 注入脚本（inject_metatoken*，可能在 datageneration/ 或 scripts/）和 scripts/re_render_meta.py 的工作方式：输入什么 JSON、按什么规则选 meta 帧号（问题帧号/TG 用 GT 答案帧）、输出格式。"措辞调优"实验需要改哪个文件的哪个函数（ego_text.py 的 render_meta_texts?）——给出文件:行号和当前措辞模板原文（截取关键短语即可）。

2. **数据分布统计**（用 /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python 只读统计 stage2_full_train_seqv3_meta2_148k.json，文件约几十 MB，直接 json.load 即可）：
   - 总条数、按 task 字段的分布（time_grounding/existence 等各多少条；无 task 字段的条目按问题文本关键词归类）；
   - TG 样本 GT 区间分布：start 帧直方图（0-4/5-9/10-14/15-19/20-24/25-29/30-38 桶），区间长度分布（mean/median）；
   - GT start≥25 的"高帧段" TG 样本绝对数量（判断过采样空间）；
   - TG 样本的 scene_id 去重数；高帧段样本集中在哪些场景（top 5 场景及条数）；
   - existence 样本量与问题模板（截取 1 条原文）。

3. **重平衡可行性**：datageneration/tools/ 下的构建脚本（build_stage2*、build_stage1* 等）有哪些现成的过滤/采样结构可复用；如果要做"对 TG 高帧段过采样 2-3 倍"，最短路径是改哪个脚本还是写个独立采样脚本（从 148k json 里复制条目即可）？注意判断：直接在 148k json 上做条目级复制/过滤是否可行（条目之间是否独立、有没有跨条目依赖如顺序）。

4. **评测集**：test 评测用的 JSON（b4dl_dataset/ 下 test 相关文件）里各任务样本数；TG 测试样本 GT start 分布（与训练集对比是否同分布）；评测 JSON 里 TG 条目有没有 scene_id/feat_indices 字段（可视化 join 用）。

输出：构建链文件:行号、ego_text 措辞模板位置、上述统计数字（精确到条数）、对 (a) 措辞调优 (b) 高帧段过采样 两个实验各自的最短实施路径判断。


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Let me break down the task:

1. Explore the data construction chain: metatoken injection scripts (inject_metatoken*), scripts/re_render_meta.py, ego_text.py — find how they work, which function to modify for wording tuning, with file:line and current wording template.

2. Data distribution statistics using /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python on stage2_full_train_seqv3_meta2_148k.json (read-only).

3. Rebalancing feasibility: datageneration/tools/ scripts, whether entries are independent.

4. Test eval JSON: task sample counts, TG GT start distribution, scene_id/feat_indices fields.

Let me start with parallel exploration: find relevant files, and list directories.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: find /root/autodl-tmp/wql/mmb4dl -maxdepth 2 -type d | head -50
description: List directory structure two levels deep
```

</details>

<details><summary>输出摘要</summary>

```
/root/autodl-tmp/wql/mmb4dl
/root/autodl-tmp/wql/mmb4dl/.claude
/root/autodl-tmp/wql/mmb4dl/.git
/root/autodl-tmp/wql/mmb4dl/.ssh
/root/autodl-tmp/wql/mmb4dl/.vscode
/root/autodl-tmp/wql/mmb4dl/Claude_record
/root/autodl-tmp/wql/mmb4dl/assets
/root/autodl-tmp/wql/mmb4dl/datageneration
/root/autodl-tmp/wql/mmb4dl/dataset
/root/autodl-tmp/wql/mmb4dl/docs
/root/autodl-tmp/wql/mmb4dl/encoders
/root/autodl-tmp/wql/mmb4dl/mllm
/root/autodl-tmp/wql/mmb4dl/models
/root/autodl-tmp/wql/mmb4dl/training_logs
/root/autodl-tmp/wql/mmb4dl/wandb
/root/autodl-tmp/wql/mmb4dl/backups
/root/autodl-tmp/wql/mmb4dl/wiki
/root/autodl-tmp/wql/mmb4dl/.zcode
/root/autodl-tmp/wql/mmb4dl/.zcode/plans
/root/autodl-tmp/wql/mmb4dl/.claude/2479a242-c128-407e-a416-636c22376b32
/root/autodl-tmp/wql/mmb4dl/.claude/302e0a03-0
…（已截断，原文 2561 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: find /root/autodl-tmp/wql/mmb4dl -iname "*metatoken*" -o -iname "*inject*" -o -iname "*ego_text*" -o -iname "*re_render*" 2>/dev/null | grep -v ".git"
description: Find metatoken/injection/ego_text related files
```

</details>

<details><summary>输出摘要</summary>

```
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_metatoken.sh
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/ego_text.py
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/inject_metatoken.py
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/re_render_meta.py
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/__pycache__/ego_text.cpython-310.pyc
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/__pycache__/ego_text.cpython-314.pyc
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/__pycache__/inject_metatoken.cpython-314.pyc
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/__pycache__/inject_metatoken.cpython-310.pyc
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls -la /root/autodl-tmp/wql/mmb4dl/mllm/b4dl_dataset/ 2>/dev/null | head -40
description: List b4dl_dataset directory contents
```

</details>

<details><summary>输出摘要</summary>

```
total 1463504
drwxr-xr-x  4 root root      4096 Sep  2 16:37 .
drwxr-xr-x 13 root root      4096 Aug 29 21:20 ..
drwxr-xr-x  3 root root        33 May 12 13:42 .cache
drwxr-xr-x  2 root root        66 Aug  8 11:15 __pycache__
-rw-r--r--  1 root root      8499 Aug  8 11:14 convert_raw_to_conversations.py
-rw-r--r--  1 root root   9639747 Aug 24 01:02 ego_frame_motion.json
-rw-r--r--  1 root root   2479630 Aug 12 12:27 ego_metadata.json
-rw-r--r--  1 root root    350799 Aug 12 12:24 ego_metadata.json.bak.per_scene
-rw-r--r--  1 root root       130 Aug  7 22:25 split_scenes.json
-rw-r--r--  1 root root 127499469 Aug 28 19:27 stage1_train.json
-rw-r--r--  1 root root 127688944 Aug 28 19:27 stage1_train_700scenes_161845.json.bak
-rw-r--r--  1 root root  72805495 Aug 28 13:45 stage1_train_framei
…（已截断，原文 2492 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls -la /root/autodl-tmp/wql/mmb4dl/datageneration/tools/ 2>/dev/null | head -60; echo ---; ls /root/autodl-tmp/wql/mmb4dl/scripts/ 2>/dev/null | head -40
description: List datageneration/tools and scripts directories
```

</details>

<details><summary>输出摘要</summary>

```
total 28
drwxr-xr-x 3 root root  187 Aug 30 12:49 .
drwxr-xr-x 7 root root 4096 Aug 29 21:17 ..
drwxr-xr-x 2 root root   83 Aug 28 19:26 assets
-rw-r--r-- 1 root root 4242 Aug 28 13:45 build_stage1_from_lidarllm.py
-rw-r--r-- 1 root root 3584 Aug 30 12:49 build_stage1_from_lidarllm_official.py
-rw-r--r-- 1 root root 5233 Aug 28 19:26 create_metadata.py
-rw-r--r-- 1 root root 3754 Aug 28 19:26 generate_stage1_caption.py
---
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/scripts/ego_text.py
```

</details>

<details><summary>输出摘要</summary>

```
1	"""
2	ego_text.py — B4DL Metatoken 文本渲染共享模块（论文 §4.1 / Appendix C / Figure 6）
3	==================================================================================
4	ego 运动状态 → 自然语言模板的单一实现来源，供三方共用，保证训练注入与推理注入
5	产生逐字符一致的 metatoken：
6	
7	  - generate_ego_metadata.py  （per-scene / per-sequence 预生成条目 + 逐帧运动表）
8	  - inject_metatoken.py       （训练数据 metatoken 注入）
9	  - evaluation/test_b4dl.py   （推理时 metatoken 注入）
10	
11	论文要求（Appendix C）："the metatoken descriptions of the first and last frames
12	referenced in the QA pair are concatenated" —— metatoken 描述的是 **QA 文本所引用
13	的首帧与末帧**（不一定是序列边界帧），因此需要一个能对任意 (first, last) 帧对
14	渲染文本的模块，而不只是预生成的序列边界条目。
15	
16	逐帧运动表（ego_frame_motion.json，由 generate_ego_metadata.py --frame_motion 产出）：
17	    {
18	      "scene_id": [
19	        {"x": .., "y": .., "z": .., "y
…（已截断，原文 10385 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/scripts/inject_metatoken.py
```

</details>

<details><summary>输出摘要</summary>

```
1	#!/usr/bin/env python3
2	"""
3	inject_metatoken.py — 将 Metatoken 前缀注入训练/测试数据
4	============================================================
5	根据论文 §4.1 / Appendix C / Figure 6，在每条 QA 的 human question 中
6	拼接 Metatoken 前缀。论文格式（Figure 6）：
7	
8	    <4DLiDAR> <video> <question>
9	    <meta> The metadata of the first frame is '<ego state at first frame>'
10	    and the metadata of the last frame is '<ego state at last frame>'
11	
12	其中 "The metadata of the first frame is" 和 "and the metadata of the last
13	frame is" 为 Figure 6 中红色高亮的连接词。
14	
15	需要先运行 generate_ego_metadata.py 生成 ego_metadata.json（+ 可选
16	ego_frame_motion.json）。
17	
18	论文 Appendix C："the metatoken descriptions of the first and last frames
19	referenced in the QA pair are concatenated" —— metatoken 描述的是 QA 文本
20	所引用的首/末帧。两种实现模式：

…（已截断，原文 20321 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/scripts/re_render_meta.py
```

</details>

<details><summary>输出摘要</summary>

```
1	#!/usr/bin/env python
2	"""B3: re-render the <meta> body of an injected training JSON in place.
3	
4	ego_text.render_meta_texts was fixed to the paper's relative-to-previous
5	semantics (per-frame position/speed/turn now vary with the frame). The injected
6	JSONs carry the old render inline, so this re-renders every sample whose frame
7	reference can be resolved — same resolution rules as inject_metatoken.py:
8	question frame numbers, else (task==time_grounding) the GT answer range.
9	Samples without a resolvable range are left untouched (per-scene fallback).
10	
11	Usage: python scripts/re_render_meta.py \
12	    --data b4dl_dataset/stage2_full_train_seqv3_148k.json \
13	    --frame_motion b4dl_dataset/ego_frame_motion.json
14	"""
15	import argparse
16	import json
17	import re
18	import
…（已截断，原文 2783 字符）
```

</details>

<details><summary>思考过程</summary>

The datageneration/tools listing seems off — it showed a directory listing that looks like b4dl_dataset contents. Wait, actually the third command output shows tools has build_stage1_from_lidarllm.py etc. And scripts/ dir at repo root seems empty (the "---" separator shows nothing after). Let me check where scripts are: mllm/scripts likely. Let me look at datageneration structure and mllm/scripts.

Let me now read the key files:
- /root/autodl-tmp/wql/mmb4dl/mllm/scripts/ego_text.py
- /root/autodl-tmp/wql/mmb4dl/mllm/scripts/inject_metatoken.py
- /root/autodl-tmp/wql/mmb4dl/mllm/scripts/re_render_meta.py

And list datageneration/ fully.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls -la /root/autodl-tmp/wql/mmb4dl/datageneration/ /root/autodl-tmp/wql/mmb4dl/mllm/scripts/ 2>/dev/null
description: List datageneration and mllm/scripts directories
```

</details>

<details><summary>输出摘要</summary>

```
/root/autodl-tmp/wql/mmb4dl/datageneration/:
total 72
drwxr-xr-x  7 root root  4096 Aug 29 21:17 .
drwxr-xr-x 19 root root  4096 Sep  1 11:12 ..
-rw-r--r--  1 root root   949 May 12 13:02 README.md
drwxr-xr-x  2 root root  4096 Aug 29 21:23 __pycache__
-rw-r--r--  1 root root  1712 Jul 12 14:50 config.py
drwxr-xr-x  5 root root    92 May 12 13:02 data
drwxr-xr-x  2 root root    10 Jun  5 16:21 datanew
-rwxr-xr-x  1 root root 15200 Aug 29 21:17 generate_dataset.py
-rwxr-xr-x  1 root root  8338 Aug 29 21:15 generate_description.py
-rw-r--r--  1 root root 14933 Aug 29 21:17 prompts.py
drwxr-xr-x  2 root root    76 Jun  5 16:19 scripts
drwxr-xr-x  3 root root   187 Aug 30 12:49 tools
-rwxr-xr-x  1 root root  7723 May 12 13:02 utils.py

/root/autodl-tmp/wql/mmb4dl/mllm/scripts/:
total 252
drwxr
…（已截断，原文 3285 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/scripts/build_stage2_full_train.py
```

</details>

<details><summary>输出摘要</summary>

```
1	#!/usr/bin/env python3
2	"""
3	build_stage2_full_train.py — 按论文官方划分构建 Stage2 全量训练数据
4	====================================================================
5	论文 §3.3：B4DL 数据集由 850 个 nuScenes scene 组成，官方划分为
6	700 train / 150 test，train 共 148,271 条 QA（stage2 68,695 + stage3 79,576）。
7	
8	HF 发布的 `ccho4702/nuScenes-B4DL` train 目录（stage2.json + stage3.json）
9	本身就是论文的官方训练集：699 个 train scene 与官方 test_qa.json 的
10	150 个 scene 零重叠，无需（也不应）再做任何额外划分。
11	
12	本脚本替代旧的 create_splits.py（80/10/10 自创划分会把 850 个 scene
13	混在一起切，切出的内部 test 与官方测试集冲突，导致训练数据被砍到
14	118,722 条/559 scenes）：
15	
16	1. 读取 HF 原始平铺格式 {question, answer, human_annotation, ...}
17	2. 转为 mllm 训练所需的 conversations 格式（保留 scene_id 等字段）
18	3. 输出 stage2_full_train_148k.json（148,271 条，未注入 metatoken）
19	
20	后续注入 metatoken（per-seq 或 --frame_ctx）由 inje
…（已截断，原文 3847 字符）
```

</details>

<details><summary>思考过程</summary>

I have the three key files. Now I need:
- build_stage2_full_train.py in mllm/scripts (filtering/sampling structure)
- datageneration/tools/ scripts (build_stage1_from_lidarllm.py etc.)
- The 148k JSON statistics via python
- Test eval JSON stats

Note: the file I read for ego_text.py may differ from what I read — wait, I read /root/autodl-tmp/wql/mmb4dl/mllm/scripts/ego_text.py and got 233 lines. But the ls says ego_text.py is 10115 bytes, and the file read ended at line 233. OK that's consistent.

Wait — the re_render_meta.py I read is 2572 bytes / 74 lines. Good.

Interesting: re_render_meta.py docstring says "re-render... in place" but it actually takes --out. Fine.

Now let me read build_stage2_full_train.py and datageneration/tools scripts, and simultaneously run the python statistics. Let me first check the structure of the 148k JSON with a quick python one-liner, then do full stats.

Also need to check test files: stage2_test.json, stage3_test.json, test_qa.json, test_qa_simple_only.json — which is used for eval? Check evaluation/test_b4dl.py to see which test files it loads.

Let me do parallel calls:
1. Read /root/autodl-tmp/wql/mmb4dl/mllm/scripts/build_stage2_full_train.py
2. ls datageneration/tools (already done — shows build_stage1_from_lidarllm.py, build_stage1_from_lidarllm_official.py, create_metadata.py, generate_stage1_caption.py, assets). Wait, earlier output for tools listing was mixed with the b4dl_dataset listing. Actually the 4th command output showed tools listing: build_stage1_from_lidarllm.py, build_stage1_from_lidarllm_official.py, create_metadata.py, generate_stage1_caption.py. Good.
3. Check evaluation dir.
4. Run python structure probe on the 148k json.

</details>

我已获取这三个 metatoken 文件。现在我来查看一下 build/eval 脚本，并探查 JSON 结构。

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/ 2>/dev/null; echo ---; grep -rn "test_qa\|stage2_test\|stage3_test\|simple_only" /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/*.py 2>/dev/null | head -30
description: List evaluation dir and find test file references
```

</details>

<details><summary>输出摘要</summary>

```
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
---
/root/autodl-tmp/wql/mmb4dl/mllm/evaluation/build_test_split.py:27:        --output ../data/test_qa.json
/root/autodl-tmp/wql/mmb4dl/mllm/evaluation/build_test_split.py:30:    from evaluation.build_test_split import assemble_test_qa
/root/autodl-tmp/wql/mmb4dl/mllm/evaluation/build_test_split.py:31:    items = assemble_test_qa(predictions_dir, scene_metadata_pat
…（已截断，原文 2097 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python -c "
import json
p='/root/autodl-tmp/wql/mmb4dl/mllm/b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json'
data=json.load(open(p))
print('type:',type
…（已截断，原文 486 字符）
description: Probe structure of 148k training JSON
```

</details>

<details><summary>输出摘要</summary>

```
type: <class 'list'> len: 148271
first item keys: ['scene_id', 'scene_token', 'split', 'human_annotation', 'conversations', 'feat_range', 'feat_indices']
{"scene_id": "003833660", "scene_token": "2ffd7e2a1daf4b928464ddb2ed3dca59", "split": "train", "human_annotation": "Densely parked trucks", "conversations": [{"from": "human", "value": "<4DLiDAR>\n<video>\nWas a pedestrian present in front of the ego vehicle between frame 30 and frame 38?\n<meta> The metadata of the first frame is 'The ego vehicle is slightly ahead on flat ground, moving quickly at about 9 meters per second, with a slight turn to the left by about 3 degrees, with an acceleration of about 0.6 meters per second squared.' and the metadata of the last frame is 'The ego vehicle is slightly ahead on flat ground, moving quickly 
…（已截断，原文 1010 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b3_pipeline.sh
```

</details>

<details><summary>输出摘要</summary>

```
1	#!/bin/bash
2	# B3 两阶段链（2026-09-02）：整场景输入 + 修复版 metatoken（relative-to-previous，meta2 数据）。
3	#   阶段1: stage2 混合重训（run_stage2_full_seqv3_mixed_b3.sh，148k meta2 × 3 epochs）
4	#          —— 论文完整配置（整场景视觉 + 论文语义 meta）的最忠实复现
5	#   阶段2: 同口径评测（--whole_scene --per_sequence --answer_frames，meta 走修复版渲染）
6	# 显存门控窗口 72h（CoRViD/xmuda 共用 GPU）；断点续训（sort -V）兜底。
7	# tmux 会话 b3pipeline 运行。
8	set -u
9	START_STAGE=${1:-1}
10	case "$START_STAGE" in 1|2) ;; *) echo "用法: bash $0 [1|2]"; exit 1 ;; esac
11	cd /root/autodl-tmp/wql/mmb4dl/mllm
12	eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)"
13	conda activate wqlc
14	export HF_HUB_OFFLINE=1 HF_OFFLINE=1 TRANSFORMERS_OFFLINE=1
15	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
16	export WANDB_MODE=offline PYTHONUNBUFFERED=1
17	
18	if [ "$STA
…（已截断，原文 3944 字符）
```

</details>

<details><summary>思考过程</summary>

Now I understand the item structure. Let me run the full statistics on the training JSON. I need:

1. Total count: 148,271 (confirmed)
2. Task distribution: task field, else classify by question keywords
3. TG GT interval distribution: start histogram buckets, interval length mean/median
4. GT start≥25 TG count
5. TG unique scene_id count; top-5 scenes for high-frame samples
6. Existence sample count and a sample question template

Also test eval JSON stats. Which test file is used? test_qa.json and test_qa_simple_only.json. Let me check test_b4dl.py args default and the B3 run script to see which test file is used. Also check run_stage2_full_seqv3_mixed_b3.sh for the data pipeline used (which confirms the chain for the 148k meta2 file).

Let me check how stage2_full_train_seqv3_meta2_148k.json was built: seqv3 file → re_render_meta.py → meta2? Let me check run_b3_pipeline.sh.

Also, for the test JSON: need TG start distribution and fields (scene_id, feat_indices). Let me run python stats in parallel with reading the pipeline scripts.

</details>

JSON 结构已确认。现在我来并行运行完整的统计和流水线脚本分析。

**🔧 Bash**

<details><summary>输入</summary>

```
command: /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python << 'EOF'
import json, re, statistics
from collections import Counter
p='/root/autodl-tmp/wql/mmb4dl/mllm/b4dl_dataset/stage2_full_train_seqv3_meta2_1
…（已截断，原文 2622 字符）
description: Compute task distribution and TG GT interval stats on 148k JSON
```

</details>

<details><summary>输出摘要</summary>

```
total: 148271
task distribution: {'(no task field) yes/no-like': 43606, '(no task field) frame-ref question': 66496, '(no task field) other': 25045, 'time_grounding': 13124}
TG parsed: 13124 skipped: 0
GT start histogram:
  0-4: 2535
  5-9: 2561
  10-14: 2090
  15-19: 2063
  20-24: 1924
  25-29: 789
  30-38: 1162
  min start: 0 max start: 38
interval len mean: 9.82 median: 9.0
len buckets: {'6-10': 8617, '1-5': 1271, '11-15': 1929, '16-20': 1307}
TG with start>=25: 1951  start>=30: 1162
start 25-29: 789 start 30-38: 1162
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -n "task\|answer_frames\|seqv3\|meta" /root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_stage2_full_seqv3_mixed_b3.sh | head -30
description: Check B3 training script data references
```

</details>

<details><summary>输出摘要</summary>

```
2:# B2 混合训练（整场景输入，对齐官方 B4DL）：与 B1（run_stage2_full_seqv3_mixed_b1.sh）唯一差异
5:# 动机：seqv3 的序列切片使 time_grounding 学会"序列局部帧编号"，评测按场景全局编号解析，
15:echo "Data: b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json"
18:export WANDB_PROJECT=b4dl-stage2-full-seqv3-mixed-b3
23:OUT=vtimellm-$MODEL_VERSION-stage2-full-seqv3-mixed-b3
25:LOG=./training_logs/stage2_full_seqv3_mixed_b3_$(date +%Y%m%d_%H%M%S).log
36:    --data_path ./b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json \
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Now I have the training stats. The "no task field" classification I did was crude — the 148k set includes stage2 (simple tasks) + stage3 (complex tasks). Let me improve classification. Actually the key task types per the paper: stage2 = existence (yes/no), time_grounding, maybe counting; stage3 = complex tasks (description, planning, etc.). Let me look at what question types exist. Also, `task` field only tags TG (13124). Other items have no task field.

Let me get a better classification by question prefixes and also count distinct question templates. Also need:
- TG unique scene count; top-5 scenes for start>=25
- existence sample count + a sample question
- TG answers format check ("from frame XXX to frame YYY")

Also, test files: test_qa.json (11MB) and test_qa_simple_only.json (4MB, Sep 2 15:57 — recent). The B3 eval uses test_qa.json. Let me check both: task counts, TG start distribution, fields (scene_id, feat_indices, feat_range, task).

Also check test_qa_simple_only.json structure and how it's generated (maybe via evaluate_model.py or a script). Let me grep for "simple_only".

Also check datageneration/tools scripts briefly for reusable filter/sampling structure.

Also verify whether entries in 148k JSON are independent (each has its own conversations, scene_id, feat_indices — yes, list of dicts, no ordering dependencies). Also check dataset.py how it loads (any shuffling/grouping? does it group by scene?). Let me check the training dataset loading code to see if there is cross-item dependency (e.g., ordering matters for seq). Quick grep in mllm for data_path loading.

Let me do these in parallel.

</details>

训练统计数据已完成。接下来我将进行更精细的任务分类、场景统计、测试 JSON 统计，以及检查 dataset loader 的独立性。

**🔧 Bash**

<details><summary>输入</summary>

```
command: /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python << 'EOF'
import json, re
from collections import Counter
p='/root/autodl-tmp/wql/mmb4dl/mllm/b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json'
da
…（已截断，原文 2136 字符）
description: Classify question templates, TG scene concentration, sample examples
```

</details>

<details><summary>输出摘要</summary>

```
=== non-TG question templates (normalized prefix, top 25) ===
  8031  'Describe the LiDAR sequence at frame N.'
  6696  'Describe the LiDAR sequence between frame N and frame N.'
  3722  'Describe the LiDAR sequence.'
   169  'Was a pedestrian present in the front view between frame N a'
   169  'Describe the scene in frame N.'
   160  'Was a pedestrian present in front of the ego vehicle between'
   159  'Did a pedestrian appear in the back view between frame N and'
   138  'Which object was in front of the ego vehicle in frame N?'
   120  'Did a truck appear in the back view between frame N and fram'
   108  'Was a pedestrian present in frame N?'
   106  'Was a truck present in front of the ego vehicle between fram'
   102  'Did a bus appear in the back view between frame N and frame '
 
…（已截断，原文 2322 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python << 'EOF'
import json, re
from collections import Counter
tg_re = re.compile(r'from\s+frames?\s+(\d+)\s+to\s+frames?\s+(\d+)', re.I)
for name in ['tes
…（已截断，原文 1910 字符）
description: Analyze test JSON files: task counts, TG start distribution, fields
```

</details>

<details><summary>输出摘要</summary>

```
=== test_qa.json === type=list len=30145
keys: ['scene_id', 'scene_token', 'split', 'task', 'conversations']
task field counts: {'existence': 3770, 'binary_qa': 7525, 'time_grounding': 2783, 'description': 3770, 'temporal_understanding': 4757, 'comprehensive_reasoning': 7540} items with task: 30145
unique scenes: 150 feat_indices set: 0 feat_range set: 0
TG-shaped answers (short answer matching from..to): 2915
test GT start histogram: {'5-9': 543, '10-14': 469, '15-19': 424, '20-24': 424, '30-38': 237, '0-4': 627, '25-29': 191}
other question templates top 10:
   1632  'Describe the LiDAR sequence at frame N.'
   1363  'Describe the LiDAR sequence between frame N and frame N'
    756  'Describe the LiDAR sequence.'
     37  'Was a pedestrian present in front of the ego vehicle be'
     35 
…（已截断，原文 2313 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -rn "simple_only" /root/autodl-tmp/wql/mmb4dl/mllm/ --include="*.py" --include="*.sh" -l 2>/dev/null | grep -v __pycache__ | grep -v b4dl_dataset; echo ---; grep -n "data_path\|json.load\|shuffle
…（已截断，原文 356 字符）
description: Find simple_only generator and dataset loading code
```

</details>

<details><summary>输出摘要</summary>

```
---
---
__pycache__
dataset.py
llama_flash_attn_monkey_patch.py
train.py
train_mem.py
vtimellm_trainer.py
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Good progress. Now a few remaining checks:

1. The dataset.py loading — check for cross-item dependency (e.g., grouping, ordering). Check vtimellm/train/dataset.py briefly for how data_path JSON is loaded and whether shuffle happens (lazy random sampling).

2. Check test_b4dl.py: how it identifies TG items (task field?) and how it resolves feat_indices/feat_range for test items (test items have no feat_indices — they're computed at eval time via answer_frames + sequence metadata?). Let me check test_b4dl.py relevant lines.

3. datageneration/tools scripts — quick scan of build_stage1_from_lidarllm.py and generate_stage1_caption.py for reusable filtering structure. Also check stage1_train.json building — is there any dedup/oversample logic? Quick look.

4. Also confirm the "meta2" file was produced by re_render_meta.py from seqv3_148k (check file dates: seqv3_148k Aug 25 16:07, meta2 Sep 2 16:37 — matches re_render run). Check any log or git log for the command. Let me grep git log for re_render_meta.

5. Also check existence task count in training: the paper says stage2 has existence tasks. From the training data: no task field except TG. Existence = yes/no questions ("Was a pedestrian present...", "Did a truck appear..."). My earlier classification lumped things. Let me count yes/no-style questions (answer in Yes./No.) to give the existence count. Also "binary_qa" in test. Let me count training items whose answer is "Yes."/"No.".

Also count how many TG samples have answer format "from frame 000 to frame 008" (zero-padded). And check whether question for TG is "When are..." style.

Also for the oversampling feasibility: entries are independent list items. dataset.py — check if it uses LazySupervisedDataset with random access and __getitem__ shuffling via dataloader shuffle; no grouping. Let me confirm quickly.

Let me also check test_b4dl.py for TG evaluation: mean error computation and existence acc, and how feat selection works (--per_sequence --answer_frames). Just enough to confirm test JSON fields needed for join: scene_id present; feat_indices absent (computed at eval time). For visualization join, scene_id + task + conversations (question/answer with frame numbers) are available.

Let me run these in parallel.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -n "json.load\|data_path\|shuffle\|random\|group" /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/train/dataset.py | head -25; echo ===; grep -n "task\|time_grounding\|feat_indices\|feat_range\|scene_i
…（已截断，原文 270 字符）
description: Check dataset loader and eval script field usage
```

</details>

<details><summary>输出摘要</summary>

```
2:import random
67:    data_path: str = field(default=None,
421:    # random.choice 随机替换坏样本，会污染训练分布且不留任何记录。
424:    def __init__(self, data_path: str,
430:        self.list_data_dict = json.load(open(data_path, "r"))
486:            return random.choice(self)
541:                                data_path=data_args.data_path,
===
6:six tasks, writes per-task predictions, and finally computes the paper metrics
11:used by mllm/vtimellm/inference.py, but targets the B4DL 6-task benchmark
20:      "scene_id": "scene-0001",            # or "id": "scene-0001"
21:      "task": "temporal_understanding",     # one of the 6 tasks
29:If `task` is missing it is read from a per-file directory name expected by the
38:are loaded from `--feat_folder/<scene_id>.npy` as a [N_frames, 768] float16
65:`--checkp
…（已截断，原文 3628 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python << 'EOF'
import json, re
from collections import Counter
p='/root/autodl-tmp/wql/mmb4dl/mllm/b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json'
da
…（已截断，原文 1847 字符）
description: Count existence/yes-no items, TG question templates, meta coverage
```

</details>

<details><summary>输出摘要</summary>

```
yes/no (existence+binary) items: 50634
TG question templates (top 6):
      13  'When did a human pedestrian adult approach the ego vehicle?'
      13  'When did the human pedestrian adult move away from the ego vehicle?'
      12  'When did the human pedestrian adult approach the ego vehicle?'
      11  'When did the ego vehicle move forward?'
       9  'When does the ego vehicle move forward?'
       8  'When did the vehicle car approach the ego vehicle?'
items without <meta>: 0 | fallback "No ego motion metadata": 0
TG answer length distribution (chars): {24: 353, 25: 406, 26: 4375, 28: 7990}
TG Q sample: When are the vehicles in front of the ego vehicle getting closer?
TG Q sample: What object is initially stationed at the intersection behind the ego vehicle?
TG Q sample: When does the
…（已截断，原文 1015 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && git log --oneline --all -8 -- mllm/scripts/re_render_meta.py mllm/b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json 2>/dev/null; echo ===; head -50 datageneration/
…（已截断，原文 347 字符）
description: Check git history and datageneration tools structure
```

</details>

<details><summary>输出摘要</summary>

```
4413b6b fix: ego_text.py metatoken 渲染对齐论文 relative-to-previous 语义——原 first frame 位置恒 'at the starting position'、运动用前向(next)差分；论文 §4.1/Figure 6 定义为每帧相对前一帧的方向/位置/速度/转向/加速度（样例 'slightly ahead and to the right'）。first>0 时改用 spd_prev/yaw_prev/acc_prev + 相对前一帧位移方向，帧 0 保留 fallback；新增 re_render_meta.py 对已注入 JSON 重渲染 meta 段（113053/148271 条，35000+ 条无帧号 fallback 保留）
===
#!/usr/bin/env python3
"""
按官方仓库逻辑生成 Stage 1 训练数据（对齐论文 162K）。

官方逻辑（build_stage1_from_lidarllm.py）:
    - 直接读 HF Senqiao/LiDAR-LLM-Nu-Caption 全量数据（train.json, 161,845 条）
    - 按 sample_token → scene 过滤（官方用 assets/sample_token_to_scene.json +
      assets/train_scene_tokens.json 的 699 个训练 scene；本地用
      scene_metadata.json 的 700 个 train scenes 近似，实测 0 泄漏）
    - 不做任何帧映射，conversation id (scene_id) 就是 sample_token 本身

输出格式 (与 mllm/vtimel
…（已截断，原文 5234 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Almost complete. A few final details:

1. dataset.py line 486: `return random.choice(self)` — bad sample replacement. Let me look at the dataset.py around lines 420-500 to understand cross-item dependency: it's LazySupervisedDataset with __getitem__ random access; the random.choice is for replacing bad samples. Entries are independent — item-level duplication is safe. Let me quickly view that region to be precise.

2. The TG question in training has no frame numbers ("When are the vehicles...") — so metatoken for TG comes from GT answer frames (answer_frames). Good.

3. Confirm exact line numbers for the wording templates in ego_text.py: describe_position (line 39), frame_text (line 116-121), build_metatoken_line (line 215-218), render_meta_texts (line 142). Got those.

4. Check test_qa_simple_only.json provenance — generated how? grep "simple_only" found nothing in mllm py/sh. Maybe generated ad hoc. Let me search wider (docs, Claude_record). Not critical. Actually it may be created by a filter script in evaluation or by a one-off. Quick grep in the whole repo excluding dataset files.

5. Also confirm how eval computes TG mean error / whether test TG items get metatoken via answer_frames (yes, --answer_frames in eval). And whether test items lack feat_indices (confirmed: 0 set), meaning eval resolves frames from GT answer at runtime — fine for visualization join: scene_id + conversations present.

Let me read dataset.py lines 415-500 and do the simple_only grep.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: sed -n 415,500p /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/train/dataset.py
description: Read dataset.py lazy loading section
```

</details>

<details><summary>输出摘要</summary>

```
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    # 运行期特征加载累计失败超过该阈值即 fail-fast。原行为是静默
    # random.choice 随机替换坏样本，会污染训练分布且不留任何记录。
    MAX_FEATURE_FAILURES = 8

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = json.load(open(data_path, "r"))
        self.data_args = data_args
        self._feature_failures = 0
        self._preflight_check_features()

    def _preflight_check_features(self):
        """启动时校验数据里每个 scene_id 的特征文件都存在（纯 os.path，
        秒级）。缺文件若拖到训练中途才暴露，代价是静默替换或无限重试。"""
        feat_folder = getattr(self.data_args, "feat_fold
…（已截断，原文 3633 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -rn "simple_only" /root/autodl-tmp/wql/mmb4dl --include="*.py" --include="*.sh" --include="*.md" 2>/dev/null | grep -v b4dl_dataset | grep -v ".git" | head; echo ===; sed -n 380,470p /root/autodl
…（已截断，原文 244 字符）
description: Find simple_only origin and read eval script task section
```

</details>

<details><summary>输出摘要</summary>

```
===


def run_inference(model, tokenizer, features: torch.Tensor, query: str) -> str:
    """Run the B4DL autoregressive generation for one QA. Mirrors inference()."""
    return inference(model, features, query, tokenizer)


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="B4DL test/evaluation harness")
    parser.add_argument("--model_base", type=str, required=True,
                        help="Path to base LLM (e.g. ./base_model/vicuna-v1-5-7b)")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default=None,
                        help="Stage-1 mm_projector.bin path")
    parser.add_argument("--stage2", type=str, default=None,
                        help="Stage-2 checkpoint directo
…（已截断，原文 6194 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

The grep for "simple_only" in the repo found nothing outside the dataset dir (the output shows "===" immediately, meaning no matches). So test_qa_simple_only.json was likely created by an ad-hoc command not committed. Fine.

I have everything I need. Let me quickly double check a couple of remaining specifics:

1. Exact wording template locations in ego_text.py with line numbers — done:
   - describe_position: line 39-65
   - describe_terrain: 68-78
   - describe_turn: 81-91
   - describe_speed: 94-103
   - describe_acceleration: 106-113
   - frame_text: 116-121 ("The ego vehicle is {position} {terrain}, {speed}, {turn}, {accel}.")
   - render_meta_texts: 142-212 (relative-to-previous semantics; prev_motion at 177-190; "at the starting position" fallback at 187)
   - build_metatoken_line: 215-218 ("The metadata of the first frame is '{first_text}' and the metadata of the last frame is '{last_text}'")

2. Training chain for meta2 file: seqv3_148k (built by run_stage2_full_seqv3.sh presumably calling inject_metatoken.py with --answer_frames --sequence_metadata) → re_render_meta.py → meta2. Let me verify run_stage2_full_seqv3.sh briefly to confirm the inject command used (with --answer_frames). Actually the header of run_stage2_full_seqv3.sh. Quick check. Also to double check the git log commit 4413b6b mention: re_render re-rendered 113053/148271 entries, 35000+ fallback kept.

3. One more stat maybe worth having: distribution of TG by whether question contains frames — TG questions have no frame refs (all "When ..."), fine.

Also confirm the test TG entries: 2783 with task field 'time_grounding'; but "TG-shaped answers" 2915 in test_qa.json vs 2783 in simple_only — in test_qa.json there may be some complex-task items with short from..to answers. Also note test_qa.json includes 6 tasks; simple_only has 3.

Let me quickly verify run_stage2_full_seqv3.sh inject command line, then compose the final report. Also check how many TG training samples are in seqv3 (should be same 13124). Not needed.

One more useful number for oversampling: 2-3x oversample of start>=25 → 1951*2 = 3902 or *3 = 5853 added entries. Also mention that len 26+ = 0 bucket (max interval length 20, since len buckets show max 16-20; no 26+). Interesting: TG interval length max 20, so "30+ frames" in the user's B3 residual refers to start frame position (start 30-38), not length. The mean length 9.82.

Wait, the user said "GT 25-38 帧段占 14% 但预测只覆盖 4%（30+ 帧为 0）" — 1951/13124 = 14.9%. Consistent.

Let me verify the seqv3 pipeline script quickly.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -n "inject_metatoken\|re_render\|answer_frames\|sequence_metadata\|output" /root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_stage2_full_seqv3.sh | head -20
description: Check seqv3 pipeline inject command
```

</details>

<details><summary>输出摘要</summary>

```
30:    deepspeed --include localhost:0 --master_port 29578 vtimellm/train/train_mem.py     --deepspeed ./scripts/zero3.json     --lora_enable True     --model_name_or_path ./base_model/vicuna-v1-5-7b     --version v1     --data_path ./b4dl_dataset/stage2_train_seqv3.json     --feat_folder ../encoders/lidarclip/b4dl/stage2_features     --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin     --output_dir ./checkpoints/vtimellm-$MODEL_VERSION-stage2-seqv3     --bf16 True     --num_train_epochs 2     --per_device_train_batch_size 8     --gradient_accumulation_steps 16     --evaluation_strategy no     --save_strategy steps     --save_steps 200     --save_total_limit 3     --learning_rate 1e-4     --freeze_mm_mlp_adapter True     --lora_r 64     --lora_alpha 1
…（已截断，原文 2520 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

The seqv3 file itself was built elsewhere (the inject command must have been run ad hoc — seqv3 files are stage2_train_seqv3.json etc.). The meta2 file was produced by re_render_meta.py from stage2_full_train_seqv3_148k.json (git commit 4413b6b confirms: 113053/148271 re-rendered). The seqv3 injection itself: inject_metatoken.py with --frame_motion --sequence_metadata --answer_frames, per docstring examples. Good enough.

Now I have all the info. Let me write the final report.

Key numbers to report:

**1. Construction chain**
- /root/autodl-tmp/wql/mmb4dl/mllm/scripts/build_stage2_full_train.py — builds stage2_full_train_148k.json (148,271) from HF stage2.json+stage3.json; tags task='time_grounding' via regex `from frames? X to frames? Y` only for stage2 source (line 32 `_TG_PATTERN`, line 57-58).
- /root/autodl-tmp/wql/mmb4dl/mllm/scripts/inject_metatoken.py — injects `<4DLiDAR>\n<video>\n<question>\n<meta> ...` into first human turn; frame resolution: parse question "frame N" (parse_frame_numbers, lines 107-118); fallback for task=='time_grounding': parse GT answer frames (lines 263-269, gated by --answer_frames); feat_range/feat_indices from sequence_metadata containing-sequence (lines 272-284, find_containing_sequence 165-182); meta body via ego_text.render_metatoken_for_range (v2, lines 286-291) with per-seq/per-scene ego_metadata fallback.
- /root/autodl-tmp/wql/mmb4dl/mllm/scripts/re_render_meta.py — B3 fix script; re-renders <meta> body of an injected JSON: same resolution (question frames, else GT answer for time_grounding, lines 55-57); samples without resolvable range left untouched (per-scene fallback); produced stage2_full_train_seqv3_meta2_148k.json (113,053/148,271 re-rendered, per git commit 4413b6b).
- /root/autodl-tmp/wql/mmb4dl/mllm/scripts/ego_text.py — shared rendering module. Wording templates:
  - `frame_text` lines 116-121: "The ego vehicle is {position} {terrain}, {speed}, {turn}, {accel}."
  - `describe_position` lines 39-65: "slightly ahead"/"ahead"/"behind", "and to the left/right", "nearly at the same position", and for first frame fallback "at the starting position" (line 65 & 187)
  - `describe_terrain` 68-78: "on flat ground"/"on an uphill slope"/"on a downhill slope"
  - `describe_turn` 81-91: "moving straight ahead" / "with a slight turn to the left by about N degrees" / "performing a left turn by about N degrees"
  - `describe_speed` 94-103: "nearly stationary" / "moving slowly at about X.X meters per second" / "moving at about X meters per second" / "moving quickly at about X meters per second"
  - `describe_acceleration` 106-113: "with nearly constant speed" / "with an acceleration of about ..." / "with a deceleration of about ..."
  - `build_metatoken_line` 215-218: "The metadata of the first frame is '...' and the metadata of the last frame is '...'"
  - render_meta_texts lines 142-212 = relative-to-previous semantics; prev_motion 177-190.

For wording tuning: modify ego_text.py, then re-run re_render_meta.py on stage2_full_train_seqv3_148k.json → new meta3 json. But note: eval-time injection (test_b4dl.py lines 324-346) imports ego_text too, so a single edit keeps train/eval consistent. Caveat: re_render only touches items with resolvable frame range (113,053); the 35k+ per-scene fallback items retain old wording — to change those too, rerun inject_metatoken.py from the pre-injection stage2_full_train_148k.json. Also caveat: train/eval must use same ego_text version at both train and eval time (test_b4dl renders at inference from --frame_motion).

**2. Stats (stage2_full_train_seqv3_meta2_148k.json, 148,271 items)**
- task field: only 'time_grounding' = 13,124; the rest 135,147 have no task field. Classified by question text:
  - yes/no answers ("Yes."/"No."): 50,634 items (existence + binary style)
  - "Describe the LiDAR sequence..." captions: 8,031 + 6,696 + 3,722 ≈ 18.4k for the 3 main templates (total distinct templates 87k because complex stage3 prose)
  - Note: my keyword bucketing: yes/no-like 43,606 + frame-ref 66,496 + other 25,045 — better to give the cleaner numbers: task field 13,124 TG; yes/no answer 50,634; the remainder are stage3 complex tasks (caption/description/temporal_understanding/comprehensive).
  
  Actually let me be careful about presenting: the test set's 6 tasks are existence(3770), binary_qa(7525), time_grounding(2783), description(3770), temporal_understanding(4757), comprehensive_reasoning(7540) out of 30,145. Training has no task labels except TG.

- TG GT start histogram (13,124): 0-4: 2535, 5-9: 2561, 10-14: 2090, 15-19: 2063, 20-24: 1924, 25-29: 789, 30-38: 1162. start≥25 = 1,951 (14.9%); start≥30 = 1,162.
- Interval length: mean 9.82, median 9; max 20; buckets 1-5: 1271, 6-10: 8617, 11-15: 1929, 16-20: 1307.
- TG unique scenes: 698; high-frame (start≥25) spread across 478 scenes, top5: 008119752 (17), 005894281 (15), 006777915 (15), 008754860 (14), 001466411/004923482/005471112/004486356 (12). So no single-scene concentration — duplication at item level is the right granularity, not scene-level.
- TG questions are frame-free ("When ..."), answers zero-padded "from frame 000 to frame 008." (24-28 chars).
- existence-style example: Q "Was a pedestrian present in front of the ego vehicle between frame 30 and frame 38?" A "Yes."
- meta coverage: 0 items without <meta>, 0 fallback "No ego motion metadata".

**3. Rebalance feasibility**
- Entries are fully independent flat list items (list of dicts). Loader LazySupervisedDataset (vtimellm/train/dataset.py line ~430 json.load; __getitem__ deepcopy per index; shuffle by dataloader). No cross-item ordering dependencies; feat_range/feat_indices are per-item. Duplicate entries are safe (the only caveat: no dedup logic in loader; random.shuffle handled by trainer sampler).
- datageneration/tools has only stage1 build scripts (build_stage1_from_lidarllm.py / _official.py: scene filtering; create_metadata.py; generate_stage1_caption.py) — no stage2 sampling structure to reuse. mllm/scripts/build_stage2_full_train.py is a format converter, no sampling.
- Shortest path: standalone script that json.loads the meta2 file, selects items with task=='time_grounding' & start>=25, appends deepcopies (2x → +1,951; 3x → +3,902 extra entries), json.dump. ~20 lines, no changes to training code. Alternative: also consider downweighting low-start TG (0-4: 2535).
- Caveat: re-render is not needed since we copy entries verbatim (meta text stays consistent with ego_text.py version). If wording tuning is done in the same run, apply oversampling AFTER re-render/inject so the duplicated entries carry the same wording.

**4. Test sets**
- test_qa.json: 30,145 items, 150 scenes, all with task field: existence 3770, binary_qa 7525, time_grounding 2783, description 3770, temporal_understanding 4757, comprehensive_reasoning 7540.
- test_qa_simple_only.json: 14,078 items (existence 3770 + binary_qa 7525 + TG 2783); no committed generator script (created ad hoc).
- B3 eval (run_b3_pipeline.sh line 61) uses test_qa.json with --whole_scene --per_sequence --answer_frames.
- Test TG GT start histogram (2783, simple_only): 0-4: 578, 5-9: 511, 10-14: 460, 15-19: 411, 20-24: 410, 25-29: 180, 30-38: 233. Compare train: train 25-29 = 6.0%, test 25-29 = 6.5%; train 30-38 = 8.9%, test 30-38 = 8.4% — roughly same distribution.
- Test items have NO feat_indices/feat_range (0 set) — frames resolved at eval time from GT answer (answer_frames) in test_b4dl.py (lines 203-213 _select_features; 324-346 meta render). Fields available for visualization join: scene_id, scene_token, task, conversations (Q/A text with frame numbers). scene_id is the key to feat .npy and ego_frame_motion.json.

Experiment cost judgments:

(a) metatoken wording tuning: edit ego_text.py template functions (frame_text line 116, describe_* 39-113, build_metatoken_line 215). Then regenerate training JSON via re_render_meta.py (--data stage2_full_train_seqv3_148k.json --frame_motion ego_frame_motion.json --out ...meta3...). Must use the same ego_text.py at eval time (test_b4dl.py imports it live, so it's automatic but also means you can't A/B eval wording without switching files). Note re_render leaves 35,218 per-scene-fallback items (148,271-113,053) untouched — if wording change must be global, rerun inject_metatoken.py from stage2_full_train_148k.json with --frame_motion --sequence_metadata --answer_frames. Single-file edit + one command; lowest-cost experiment. Important: keep semantics relative-to-previous (render_meta_texts) as that's the paper's definition; wording changes should be inside describe_* / frame_text / build_metatoken_line strings.

(b) high-frame oversampling: standalone ~20-line script copying task=='time_grounding' items with GT start>=25 (1,951 items) 2-3x → +1,951/+3,902 entries on top of 148,271. Entries independent; no infra change needed. Note the data is spread across 478 scenes so scene-level upsampling is not a thing. Also consider that the test distribution matches train, so oversampling shifts the prior — this is the intended intervention given B3's under-prediction of high starts.

Also mention: TG mean bias −3.9 frames may also stem from feat_indices resolution; but that's beyond scope.

Now write final answer, structured, with file:line refs. Keep concise but complete. No emojis.

</details>

所有数据已收集完毕。这是完整的分析报告。

# B4DL 数据管线探索报告

## 1. 数据构建链（metatoken 注入）

链路（B3 用 `stage2_full_train_seqv3_meta2_148k.json` 的实际来源）：

```
dataset/nuScenes-B4DL/{stage2,stage3}.json
  → mllm/scripts/build_stage2_full_train.py          # 平铺 QA → conversations 格式, 148,271 条
  → mllm/b4dl_dataset/stage2_full_train_148k.json    # 未注入 meta
  → mllm/scripts/inject_metatoken.py --frame_motion --sequence_metadata --answer_frames
  → stage2_full_train_seqv3_148k.json
  → mllm/scripts/re_render_meta.py                   # git 4413b6b, 重渲染 113,053/148,271 条
  → stage2_full_train_seqv3_meta2_148k.json          # B3 训练用
```

各环节工作方式：

- **`/root/autodl-tmp/wql/mmb4dl/mllm/scripts/build_stage2_full_train.py`**：`:32` `_TG_PATTERN = r'from\s+frames?\s+\d+\s+to\s+frames?\s+\d+'`；`:57-58` 仅对 stage2 来源、答案匹配该 pattern 的条目打 `task='time_grounding'` 标签（13,124 条）。其余条目无 task 字段。
- **`/root/autodl-tmp/wql/mmb4dl/mllm/scripts/inject_metatoken.py`**：把首条 human 消息改写为 `<4DLiDAR>\n<video>\n<question>\n<meta> ...`（`:246-252, :335`）。meta 帧号解析规则：先从**问题文本**抓 `frame N`（`parse_frame_numbers` `:107-118`，取 min/max）；问题无帧号且 `task=='time_grounding'` 时回退解析 **GT 答案帧号**（`:263-269`，需 `--answer_frames`）。同时按 sequence_metadata 的包含序列写 per-item `feat_range`/`feat_indices`（`:275-284`，`find_containing_sequence` `:165-182`：优先最窄完全包含，否则最大重叠）。
- **`/root/autodl-tmp/wql/mmb4dl/mllm/scripts/re_render_meta.py`**：对已注入 JSON 重渲染 `<meta>` body。`:55-57` 与 inject 同规则（问题帧号，否则 TG 用 GT 答案帧）；`:58-64` 无法解析帧号的样本原样保留（per-scene fallback，约 35,218 条）；输出到 `--out`（meta2 文件即由此产出）。

### 措辞模板位置（`/root/autodl-tmp/wql/mmb4dl/mllm/scripts/ego_text.py`）

| 函数 | 行号 | 当前措辞（关键短语） |
|---|---|---|
| `frame_text` | 116-121 | `"The ego vehicle is {position} {terrain}, {speed}, {turn}, {accel}."` |
| `describe_position` | 39-65 | `"slightly ahead"/"ahead"/"behind"`, `"and to the left/right"`, `"nearly at the same position"`；首帧 fallback `"at the starting position"`（`:65` 和 `:187`） |
| `describe_terrain` | 68-78 | `"on flat ground"/"on an uphill slope"/"on a downhill slope"` |
| `describe_turn` | 81-91 | `"moving straight ahead"` / `"with a slight turn to the {dir} by about N degrees"` / `"performing a {dir} turn by about N degrees"` |
| `describe_speed` | 94-103 | `"nearly stationary"` / `"moving slowly at about X.X meters per second"` / `"moving at about X meters per second"` / `"moving quickly..."` |
| `describe_acceleration` | 106-113 | `"with nearly constant speed"` / `"with an acceleration/deceleration of about X.X meters per second squared"` |
| `build_metatoken_line` | 215-218 | `"The metadata of the first frame is '...' and the metadata of the last frame is '...'"` |
| `render_meta_texts` | 142-212 | relative-to-previous 语义核心（`prev_motion` `:177-190`，用 `spd_prev/yaw_prev/acc_prev` + 相对前一帧位移） |

## 2. 训练数据分布（`stage2_full_train_seqv3_meta2_148k.json`，148,271 条）

**任务分布**：仅 TG 有 task 字段（13,124 条）；其余 135,147 条无 task 字段，按答案/问题归类：
- Yes/No 短答案（existence + binary 类）：**50,634 条**
- "Describe the LiDAR sequence..." 描述类三大模板：8,031（单帧）+ 6,696（区间）+ 3,722（无帧）≈ 18.4k
- 其余为 stage3 复杂任务 prose（temporal_understanding / comprehensive 类，问题模板去重后 87,445 种，长尾分散）

**TG GT 区间分布**（13,124 条全部解析成功）：

| start 桶 | 0-4 | 5-9 | 10-14 | 15-19 | 20-24 | 25-29 | 30-38 |
|---|---|---|---|---|---|---|---|
| 条数 | 2,535 | 2,561 | 2,090 | 2,063 | 1,924 | **789** | **1,162** |

- 区间长度：mean **9.82**，median 9，max 20（桶：1-5: 1,271 / 6-10: 8,617 / 11-15: 1,929 / 16-20: 1,307）
- **GT start≥25 高帧段：1,951 条（14.9%）**；start≥30：1,162 条
- TG 去重场景：**698 个**；高帧段分散在 **478 个场景**，top5：008119752(17)、005894281(15)、006777915(15)、008754860(14)、001466411(12)——**无场景级集中，过采样只能在条目级做**
- TG 问题均为无帧号的 "When..." 型；答案零填充格式如 `from frame 000 to frame 008.`
- existence 样例：Q `Was a pedestrian present in front of the ego vehicle between frame 30 and frame 38?` A `Yes.`
- meta 覆盖：无 `<meta>` 的条目 0 条，"No ego motion metadata" fallback 0 条

## 3. 重平衡可行性

- **条目完全独立**：`vtimellm/train/dataset.py:430` `json.load` 整表 → `__getitem__` 按 index `deepcopy` 单条（`:470`），shuffle 由 trainer sampler 做，无跨条目顺序依赖；`feat_range`/`feat_indices` 是 per-item 字段。**条目级复制/过滤完全可行**。
- `datageneration/tools/` 只有 stage1 构建脚本（`build_stage1_from_lidarllm.py`/`_official.py` 是 scene 过滤，`create_metadata.py`、`generate_stage1_caption.py`），**没有可复用的 stage2 采样结构**；`mllm/scripts/build_stage2_full_train.py` 只是格式转换器。
- **最短路径：写独立采样脚本**（约 20 行）：`json.load` meta2 文件 → 选 `task=='time_grounding'` 且 GT start≥25 的 1,951 条 → deepcopy 追加（2x +1,951 条 / 3x +3,902 条）→ `json.dump`。不需要动任何训练代码。注意：复制的是已注入 meta 的完整条目，措辞与原文件一致；若与实验 (a) 同批做，先重渲染再复制。

## 4. 评测集（`mllm/b4dl_dataset/`）

- **`test_qa.json`**：30,145 条 / 150 scenes，全部带 task 字段：existence 3,770、binary_qa 7,525、**time_grounding 2,783**、description 3,770、temporal_understanding 4,757、comprehensive_reasoning 7,540。B3 评测即用它（`run_b3_pipeline.sh:61`，`--whole_scene --per_sequence --answer_frames`）。
- **`test_qa_simple_only.json`**：14,078 条（existence 3,770 + binary_qa 7,525 + TG 2,783）；仓库内无生成脚本（临时过滤产物）。
- **测试 TG GT start 分布**（2,783 条）：0-4: 578 / 5-9: 511 / 10-14: 460 / 15-19: 411 / 20-24: 410 / 25-29: 180 / 30-38: 233。与训练集同分布（train 25-29 占 6.0% vs test 6.5%；30-38 占 8.9% vs 8.4%）——**测试集不会惩罚过采样带来的先验偏移，方向上正是 B3 欠预测高帧段想要的干预**。
- **评测 JSON 无 `feat_indices`/`feat_range`**（两者均为 0 条）；帧号由 `test_b4dl.py:203-213`（`_select_features` GT-answer fallback）和 `:324-346`（meta 渲染）在评测时从 GT 答案现场解析。可视化 join 可用字段：`scene_id`（关联 `feat_folder/<scene_id>.npy` 与 `ego_frame_motion.json`）、`scene_token`、`task`、`conversations`（问题+GT 答案含帧号）。

## 两个实验的最短实施路径

**(a) metatoken 措辞调优 — 成本最低**：只改 `mllm/scripts/ego_text.py` 的字符串模板（建议只动 `describe_*`/`frame_text`/`build_metatoken_line` 的措辞，保留 `render_meta_texts:142-212` 的 relative-to-previous 语义不动，那是论文定义）。然后一条命令重生成训练数据：`python scripts/re_render_meta.py --data b4dl_dataset/stage2_full_train_seqv3_148k.json --frame_motion b4dl_dataset/ego_frame_motion.json --out <meta3>.json`。注意两点：(1) 35,218 条无帧号 fallback 条目不会被 re_render 触及，若要求全局措辞一致需从 `stage2_full_train_148k.json` 重跑 inject_metatoken.py；(2) `test_b4dl.py` 在推理时 live import 同一 ego_text.py（`:324-346`），训练/评测措辞自动保持一致，但也意味着 A/B 评测不同措辞需要切换文件版本。

**(b) TG 高帧段过采样 — 独立小脚本即可**：如上第 3 节，1,951 条 start≥25 样本复制 2-3x（+1,951~3,902 条），条目独立、无基础设施改动。可与 (a) 正交叠加（先措辞重渲染、后过采样复制）。额外可考虑的对照杠杆：0-4 桶 2,535 条（19%）是最大的低帧桶，若过采样不够可配对降采样。

