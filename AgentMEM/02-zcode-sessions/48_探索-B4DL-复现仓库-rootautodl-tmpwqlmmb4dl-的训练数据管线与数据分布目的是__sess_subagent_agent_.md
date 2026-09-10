# 探索 B4DL 复现仓库 /root/autodl-tmp/wql/mmb4dl 的训练数据管线与数据分布，目的是...

| 项 | 值 |
|---|---|
| 会话 ID | `sess_subagent_agent_ffdcd493-855e-405b-9c1a-8edfd051d34e` |
| 工作目录 | `/root/autodl-tmp/wql/mmb4dl` |
| 父会话 | `sess_cbeefbda-5c90-4e6b-b689-01f850ea4973` |
| 时间 | 2026-09-06 15:16 → 2026-09-06 15:16 |
| 模型 | GLM-5.3-Flash |
| 消息数 | 6（文本块 1，工具调用 6） |

---

### ASSISTANT  ·  `GLM-5.3-Flash`

> 🔀 模型切换：None → builtin:bigmodel-start-plan/GLM-5.3-Flash


### USER

探索 B4DL 复现仓库 /root/autodl-tmp/wql/mmb4dl 的训练数据管线与数据分布，目的是评估两个 B4 实验的实施成本：(a) metatoken 措辞调优；(b) TG 高帧段数据重平衡。搜索广度：medium。只读操作（可以用 python 只读统计 JSON，但不要写任何文件/不要修改数据）。

背景：148k 训练数据 /root/autodl-tmp/wql/mmb4dl/mllm/b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json（B3 用），metatoken 由 ego_text.py 渲染（relative-to-previous 语义），TG 答案形如 "from frame XXX to frame YYY"。B3 残余问题：TG 预测整体偏早 mean −3.9 帧、GT 25-38 帧段占 14% 但预测只覆盖 4%（30+ 帧为 0）；existence 任务 acc 0.676 比 B0 低 0.05。

需要回答的问题：

1. **数据构建链**：metatoken 注入脚本（inject_metatoken*，可能在 datageneration/ 或 scripts/）和 re_render_meta.py 的工作方式：输入什么 JSON、按什么规则选 meta 帧号（问题帧号/TG 用 GT 答案帧）、输出格式。"措辞调优"实验需要改哪个文件的哪个函数（ego_text.py 的 render_meta_texts?）——给出文件:行号和当前措辞模板原文。

2. **数据分布统计**（用 /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python 只读统计 stage2_full_train_seqv3_meta2_148k.json）：
   - 总条数、按 task 字段的分布（existence/binary_qa/time_grounding/description/temporal/comprehensive 各多少条）；
   - TG 样本 GT 区间分布：start 帧直方图（0-4/5-9/10-14/.../35-38 桶），start+interval 的联合特点（区间长度分布）；
   - TG 样本里"高帧段"（GT start≥25）占比；这些样本的 metatoken 是否有帧号（还是 fallback 无帧号）；
   - TG 的场景来源：是全部场景还是部分场景有 TG 样本（scene_id 去重数）；
   - 文件太大就流式处理，报告统计数字即可。

3. **重平衡可行性**：数据构建脚本（build_stage2* / datageneration/tools/）是否支持按条件过滤/采样 TG 样本（例如控制每场景 TG 数量、或对高帧段过采样）——现有脚本里有哪些现成参数/结构可复用；TG 样本总量里高帧段绝对数量是多少（判断过采样空间）。

4. **existence 任务**：existence 样本的问题模板和答案格式；B3 整场景输入下 existence 退化（0.676）的可能数据侧原因——existence 样本的视觉输入是整场景还是切片（结合 dataset.py 的 whole_scene 逻辑和 existence 样本有无帧号字段判断）；existence 样本量多少。

5. **评测集**：test 评测用的 JSON（b4dl_dataset/test_qa.json 或类似）里各任务样本数，TG 测试样本 GT 分布（和训练集对比是否同分布）。

输出：构建链文件:行号、ego_text 措辞模板原文位置、上述统计数字（精确到条数）、对 (a)(b) 两个实验各自的最短实施路径判断。


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Let me break down the task:

1. Explore the repo `/root/autodl-tmp/wql/mmb4dl` for training data pipeline and data distribution
2. Find metatoken injection scripts (inject_metatoken*, re_render_meta.py) and how they work
3. Statistics on stage2_full_train_seqv3_meta2_148k.json using python (read-only)
4. Rebalancing feasibility in build_stage2* / datageneration/tools/
5. Existence task analysis
6. Test eval set JSON

Let me start with parallel exploration: list the directory structure, find relevant files.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls /root/autodl-tmp/wql/mmb4dl && echo "---" && ls /root/autodl-tmp/wql/mmb4dl/mllm 2>/dev/null | head -50
description: List repo root and mllm directory
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
---
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
command: find /root/autodl-tmp/wql/mmb4dl -maxdepth 3 -iname "*inject*" -o -maxdepth 3 -iname "*re_render*" -o -maxdepth 3 -iname "*ego_text*" -o -maxdepth 3 -iname "build_stage2*" 2>/dev/null
description: Find metatoken injection and build scripts
```

</details>

<details><summary>输出摘要</summary>

```
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/ego_text.py
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/build_stage2_full_train.py
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/inject_metatoken.py
/root/autodl-tmp/wql/mmb4dl/mllm/scripts/re_render_meta.py
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

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

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls /root/autodl-tmp/wql/mmb4dl/mllm/scripts/ && echo "---" && ls /root/autodl-tmp/wql/mmb4dl/mllm/b4dl_dataset/ | head -40
description: List scripts and dataset directories
```

</details>

<details><summary>输出摘要</summary>

```
__pycache__
build_stage2_full_train.py
convert_lidarllm_to_stage1.py
convert_raw_to_conversations.py
create_splits.py
ego_text.py
eval_stage1_ppl.py
generate_ego_metadata.py
inject_metatoken.py
merge_stage2.py
re_render_meta.py
resume_stage2_full.sh
run_b1_pipeline.sh
run_b2_pipeline.sh
run_b3_pipeline.sh
run_b4dl_eval.sh
run_metatoken.sh
run_mixed_eval_after_train.sh
run_stage2_full.sh
run_stage2_full_seq.sh
run_stage2_full_seqv2.sh
run_stage2_full_seqv3.sh
run_stage2_full_seqv3_mixed.sh
run_stage2_full_seqv3_mixed_b1.sh
run_stage2_full_seqv3_mixed_b2.sh
run_stage2_full_seqv3_mixed_b3.sh
stage1.sh
stage1_glm.sh
stage2.sh
stage2_glm.sh
stage3.sh
test_sample.py
verify.sh
verify_stage1_sample_data.py
verify_stage2.py
zero2.json
zero3.json
zero3_offload.json
---
__pycache__
convert_raw_to_con
…（已截断，原文 1544 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

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

<details><summary>思考过程</summary>

Now read re_render_meta.py and ego_text.py.

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

