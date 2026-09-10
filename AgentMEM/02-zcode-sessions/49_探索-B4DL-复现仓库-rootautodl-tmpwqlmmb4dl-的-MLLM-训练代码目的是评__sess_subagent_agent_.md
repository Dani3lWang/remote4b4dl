# 探索 B4DL 复现仓库 /root/autodl-tmp/wql/mmb4dl 的 MLLM 训练代码，目的是评...

| 项 | 值 |
|---|---|
| 会话 ID | `sess_subagent_agent_c14623fd-89e9-448f-8ac5-332c968f176b` |
| 工作目录 | `/root/autodl-tmp/wql/mmb4dl` |
| 父会话 | `sess_cbeefbda-5c90-4e6b-b689-01f850ea4973` |
| 时间 | 2026-09-06 15:16 → 2026-09-06 15:21 |
| 模型 | GLM-5.3-Flash |
| 消息数 | 14（文本块 6，工具调用 24） |

---

### ASSISTANT  ·  `GLM-5.3-Flash`

> 🔀 模型切换：None → builtin:bigmodel-start-plan/GLM-5.3-Flash


### USER

探索 B4DL 复现仓库 /root/autodl-tmp/wql/mmb4dl 的 MLLM 训练代码，目的是评估"加入强化学习损失函数"（针对 time grounding 任务的 IoU 回报）的实现路径与工作量。搜索广度：medium。只读操作，不要修改任何文件。

背景：项目基于 VTimeLLM/LiDAR-LLM 架构（Vicuna-7B + LoRA + 视觉 projector），两阶段训练；stage2 用 HF Trainer 训 LoRA；TG（time grounding）答案是文本 "from frame XXX to frame YYY"。目前 B3 模型 mIoU 0.3467，想尝试 RL 损失改进定位。

需要回答的问题：

1. **训练框架基座**：找到 stage2 训练入口（mllm/vtimellm/train/train.py 或类似路径）和自定义 trainer（vtimellm_trainer.py 之类）。确认：继承的是不是 transformers.Trainer？有没有重写 compute_loss / prediction_step？loss 在哪里算（labels shift + CrossEntropy？）。LLaVA 式代码通常在 train.py 里直接调 Trainer——报告实际类结构和关键文件:行号。

2. **数据流**：训练 dataset 类（vtimellm/train/dataset.py）返回什么（input_ids, labels, attention_mask, images/features?），TG 样本的答案文本如何进 labels，能否在 batch 里识别出"这是 TG 样本"（有没有 task 字段透传）。

3. **LoRA 配置**：stage2 训练脚本（mllm/scripts 里 run_stage2_full_seqv3_mixed*.sh 或 b2/b3 变体）里 LoRA target_modules、r/alpha、lr、batch、epochs、梯度累积；用没用 deepspeed ZeRO-3（RL 训练时生成 rollout 的可行性）。

4. **生成能力**：仓库里评测端（test_b4dl.py）怎么调 model.generate——训练代码里有没有现成的 generate/rollout 基建可复用（比如 val 生成、beam search）；batch generate 是否可用（attention mask、feature 注入路径 mm_projector 在 generate 时怎么走）。

5. **deepspeed 配置文件**：找到 stage2 用的 ds_config（zero3?），RL 训练若需要"生成+逐步 logprob"两阶段，ZeRO-3 会有什么已知坑（只需指出配置文件路径和 stage 设置）。

6. **现有 RL/强化痕迹**：grep 全仓库 reward|reinforce|grpo|dpo|rlhf，确认有没有已开始的 RL 相关代码。

输出：训练代码结构图（文件:行号）、compute_loss 位置、LoRA/deepspeed 配置要点、对"在现有代码上加 RL 损失"的最短路径判断（例如：自定义 Trainer.compute_loss 内做 rollout + IoU reward vs 离线生成再 DPO/加权 SFT），以及主要风险点。


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Let me start exploring the repository structure to understand the codebase. I'll do multiple parallel searches.

The user wants me to explore the MLLM training code in /root/autodl-tmp/wql/mmb4dl to evaluate the implementation path for adding an RL loss function (IoU reward for time grounding task).

Let me plan:
1. Find stage2 training entry (mllm/vtimellm/train/train.py) and custom trainer
2. Dataset class
3. LoRA config in scripts
4. Generation capability (test_b4dl.py)
5. DeepSpeed configs
6. Grep for RL traces

Let me start with parallel bash calls.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: find /root/autodl-tmp/wql/mmb4dl -maxdepth 2 -type d | head -50
description: List top-level directory structure
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
…（已截断，原文 2588 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls /root/autodl-tmp/wql/mmb4dl/mllm/ 2>/dev/null; echo "---"; ls /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/ 2>/dev/null; echo "---"; ls /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/train/ 2>/dev/null
description: List mllm and vtimellm train directories
```

</details>

<details><summary>输出摘要</summary>

```
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
---
__init__.py
__pycache__
constants.py
conversation.py
demo_gradio.py
inference.py
mm_utils.py
model
train
utils.py
---
__pycache__
dataset.py
llama_flash_attn_monkey_patch.py
train.py
train_mem.py
vtimellm_trainer.py
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -rniE "reward|reinforce|grpo|dpo|rlhf|policy_loss|rollout" /root/autodl-tmp/wql/mmb4dl --include="*.py" --include="*.sh" --include="*.json" -l 2>/dev/null | grep -v ".git/" | head -30
description: Grep for RL-related keywords across repo
```

</details>

<details><summary>输出摘要</summary>

```
/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/mmdetection3d/configs/_base_/datasets/kitti-3d-3class.py
/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/mmdetection3d/configs/_base_/datasets/s3dis-seg.py
/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/mmdetection3d/configs/_base_/datasets/nus-3d.py
/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/mmdetection3d/configs/_base_/datasets/semantickitti.py
/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/mmdetection3d/configs/benchmark/hv_second_secfpn_4x8_80e_pcdet_kitti-3d-3class.py
/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/mmdetection3d/configs/benchmark/hv_pointpillars_secfpn_3x8_100e_det3d_kitti-3d-car.py
/root/autodl-tmp/wql/mmb4dl/encoders/lidarclip/mmdetection3d/configs/benchmark/hv_pointpillars_secfpn_4x8_80e_pcdet_kitti-3d-3class.py
/roo
…（已截断，原文 3549 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/train/train.py
```

</details>

<details><summary>输出摘要</summary>

```
1	# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
2	# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
3	#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
4	#
5	#    Licensed under the Apache License, Version 2.0 (the "License");
6	#    you may not use this file except in compliance with the License.
7	#    You may obtain a copy of the License at
8	#
9	#        http://www.apache.org/licenses/LICENSE-2.0
10	#
11	#    Unless required by applicable law or agreed to in writing, software
12	#    distributed under the License is distributed on an "AS IS" BASIS,
13	#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
14	#    See the License for the specific language governin
…（已截断，原文 22597 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/train/vtimellm_trainer.py
```

</details>

<details><summary>输出摘要</summary>

```
1	import os
2	import torch
3	
4	from transformers import Trainer
5	from typing import Optional
6	
7	
8	def maybe_zero_3(param, ignore_status=False, name=None):
9	    from deepspeed import zero
10	    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
11	    if hasattr(param, "ds_id"):
12	        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
13	            if not ignore_status:
14	                print(name, 'no ignore status')
15	        with zero.GatheredParameters([param]):
16	            param = param.data.detach().cpu().clone()
17	    else:
18	        param = param.detach().cpu().clone()
19	    return param
20	
21	
22	def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
23	    to_return = {k: t for k, t in named_params if any(key_match in k fo
…（已截断，原文 2669 字符）
```

</details>

<details><summary>思考过程</summary>

The grep matched mostly unrelated mmdetection3d files. Let me restrict the RL grep to mllm directory. Also read the key files: train.py, vtimellm_trainer.py, dataset.py. And check scripts dir.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -rniE "reward|reinforce|grpo|\bdpo\b|rlhf|policy_loss|rollout| advantage" /root/autodl-tmp/wql/mmb4dl/mllm --include="*.py" --include="*.sh" --include="*.json" 2>/dev/null | head -20; echo "=== s
…（已截断，原文 261 字符）
description: Grep RL keywords in mllm dir and list scripts
```

</details>

<details><summary>输出摘要</summary>

```
<persisted-output>
Output too large (60.1MB). Full output saved to: /root/.zcode/cli/exec/sess_subagent_agent_c14623fd-89e9-448f-8ac5-332c968f176b/call_ebcaaf0e0f314ac4b121dadf-stdout.log

Preview (first 2KB):
/root/autodl-tmp/wql/mmb4dl/mllm/b4dl_dataset/stage3_val.json:1:[{"scene_id": "004432952", "scene_token": "03ee880dd4e348f4b3407f0d073c7c70", "split": "train", "conversations": [{"from": "human", "value": "<video>\nDescribe the LiDAR sequence."}, {"from": "gpt", "value": "The ego vehicle is situated on a moderately busy road. It navigates past parked vehicles, including cars and a bus, primarily on its right, while a moving car ahead gradually decreases in distance over the sequence."}]}, {"scene_id": "004432952", "scene_token": "03ee880dd4e348f4b3407f0d073c7c70", "split": "train", "
…（已截断，原文 2234 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/train/dataset.py
```

</details>

<details><summary>输出摘要</summary>

```
1	import os
2	import random
3	import copy
4	import json
5	import re
6	import torch
7	import transformers
8	from PIL import Image
9	import numpy as np
10	from torch.utils.data import Dataset
11	from dataclasses import dataclass, field
12	from typing import Dict, Optional, Sequence, List, Tuple
13	
14	from vtimellm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
15	from vtimellm import conversation as conversation_lib
16	from vtimellm.mm_utils import tokenizer_image_token
17	
18	
19	def _parse_frame_numbers(text: str) -> Tuple[Optional[int], Optional[int]]:
20	    """Extract first and last frame numbers from a question text.
21	
22	    Looks for patterns like "frame 030", "frame 6", "frame 008".
23	    Returns (first_frame, last_frame) as integers, or (None, None).
24	 
…（已截断，原文 22355 字符）
```

</details>

<details><summary>思考过程</summary>

The grep hit the JSON dataset files. I need to restrict the grep to exclude b4dl_dataset JSON files. Let me redo the RL grep restricted to .py and .sh files only.

Now read dataset.py, and list the scripts. Also look for evaluation/test_b4dl.py.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls /root/autodl-tmp/wql/mmb4dl/mllm/scripts/ 2>/dev/null; echo "==="; ls /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/ 2>/dev/null; echo "==="; grep -rniE "reward|reinforc|grpo|dpo|rlhf|policy_loss|rol
…（已截断，原文 352 字符）
description: List scripts/evaluation dirs and grep RL keywords in code files
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
===
README.md
__pycache__
analyze_
…（已截断，原文 1205 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_stage2_full_seqv3_mixed_b3.sh
```

</details>

<details><summary>输出摘要</summary>

```
1	#!/bin/bash
2	# B2 混合训练（整场景输入，对齐官方 B4DL）：与 B1（run_stage2_full_seqv3_mixed_b1.sh）唯一差异
3	# 是 --whole_scene True——视觉输入不再按 QA 序列切片，而是喂整场景 (39/40/41, 768) 特征，
4	# 与官方实现（ccho4702/B4DL，dataset.py 对 np.load 结果直接用）一致。
5	# 动机：seqv3 的序列切片使 time_grounding 学会"序列局部帧编号"，评测按场景全局编号解析，
6	# mIoU 被系统性压低（B1 0.2653 vs 论文 0.311）；整场景输入下模型学到全局编号。
7	# 其余（数据 148k、162K projector、3ep、lr 1e-4、LoRA r64/α128、ZeRO-3）与 B1 完全一致。
8	# 断点续训：mllm train.py 已按步数数值排序取最新（e8639e2），此处 sort -V 与其对齐。
9	cd /root/autodl-tmp/wql/mmb4dl/mllm
10	eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)"
11	conda activate wqlc
12	
13	echo "===== B2 MIXED TRAINING (whole-scene input, paper-aligned) ====="
14	echo "Start: $(date)"
15	echo "Data: b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json"
16	
17	export WANDB_MODE=offline
18	ex
…（已截断，原文 2611 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/scripts/zero3.json
```

</details>

<details><summary>输出摘要</summary>

```
1	{
2	    "fp16": {
3	        "enabled": "auto",
4	        "loss_scale": 0,
5	        "loss_scale_window": 1000,
6	        "initial_scale_power": 16,
7	        "hysteresis": 2,
8	        "min_loss_scale": 1
9	    },
10	    "bf16": {
11	        "enabled": "auto"
12	    },
13	    "train_micro_batch_size_per_gpu": "auto",
14	    "train_batch_size": "auto",
15	    "gradient_accumulation_steps": "auto",
16	    "zero_optimization": {
17	        "stage": 3,
18	        "overlap_comm": true,
19	        "contiguous_gradients": true,
20	        "sub_group_size": 1e9,
21	        "reduce_bucket_size": "auto",
22	        "stage3_prefetch_bucket_size": 5e8,
23	        "stage3_param_persistence_threshold": "auto",
24	        "stage3_max_live_parameters": 1e9,
25	        "stage3_max_reuse_distance": 1e9,
2
…（已截断，原文 1101 字符）
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

No RL traces in code files (grep found nothing in .py/.sh — output shows nothing after the second "===", meaning no matches).

Now let me look at:
1. The stage2 scripts: run_stage2_full_seqv3_mixed_b3.sh (B3 variant) and run_stage2_full_seqv3_mixed.sh
2. test_b4dl.py to see generate usage
3. deepspeed configs zero3.json
4. The model forward (prepare_inputs_labels_for_multimodal) to understand feature injection

Let me read the scripts in parallel.

</details>

在代码文件中未发现 RL 痕迹（仅有的匹配项是无关的 mmdetection3d 配置和数据 JSON 文件）。现在我来检查一下 stage2 脚本、评测代码以及 deepspeed 配置。


### ASSISTANT  ·  `GLM-5.3-Flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/inference.py
```

</details>

<details><summary>输出摘要</summary>

```
1	import os
2	import sys
3	import argparse
4	import torch
5	from vtimellm.constants import IMAGE_TOKEN_INDEX
6	from vtimellm.conversation import conv_templates, SeparatorStyle
7	from vtimellm.model.builder import load_pretrained_model, load_lora
8	from vtimellm.utils import disable_torch_init
9	from vtimellm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
10	from PIL import Image
11	import requests
12	from io import BytesIO
13	from transformers import TextStreamer
14	from easydict import EasyDict as edict
15	try:
16	    from torchvision.transforms import InterpolationMode
17	    BICUBIC = InterpolationMode.BICUBIC
18	except ImportError:
19	    from PIL import Image
20	    BICUBIC = Image.BICUBIC
21	from torchvision.transforms import Compose, Resize, CenterCr
…（已截断，原文 4463 字符）
```

</details>

<details><summary>思考过程</summary>

Now let me look at:
1. vtimellm/inference.py — the inference() function (generate usage)
2. vtimellm/model/ — the forward pass, prepare_inputs_labels_for_multimodal
3. train_mem.py
4. The mixed dataset script (which dataset file used for TG) — check what tasks are in stage2_full_train_seqv3_meta2_148k.json and whether there's a "task" field
5. evaluate_model.py — IoU computation for TG

Let me read these in parallel.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/model/; echo "==="; cat /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/train/train_mem.py
description: List model dir and show train_mem.py
```

</details>

<details><summary>输出摘要</summary>

```
__init__.py
__pycache__
builder.py
chatglm
vtimellm_arch.py
vtimellm_chatglm.py
vtimellm_llama.py
===
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.

import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
print(root_dir)
import sys
sys.path.append(root_dir)

# flash_attn 2.x 不支持 RTX 5090 (sm_120)，改用 transformers 原生 SDPA
# 需要 transformers >= 4.45
try:
    from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
except ImportError:
    pass  # flash_attn 
…（已截断，原文 910 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -n "task" /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/train/dataset.py | head; echo "=== check train json task fields ==="; python3 -c "
import json
p='/root/autodl-tmp/wql/mmb4dl/mllm/b4dl_dataset
…（已截断，原文 811 字符）
description: Inspect training JSON structure and task field distribution
```

</details>

<details><summary>输出摘要</summary>

```
=== check train json task fields ===
total items: 148271
item keys: {'scene_id', 'split', 'feat_indices', 'feat_range', 'human_annotation', 'conversations', 'scene_token'}
task counts: Counter({None: 135147, 'time_grounding': 13124})
TG sample: {"scene_id": "008395103", "scene_token": "73030fb67d3c46cfb5e590168088ae39", "split": "train", "human_annotation": "Construction, maneuver between several trucks", "conversations": [{"from": "human", "value": "<4DLiDAR>\n<video>\nWhen are the vehicles in front of the ego vehicle getting closer?\n<meta> The metadata of the first frame is 'The ego vehicle is at the starting position on flat ground, moving at about 4 meters per second, performing a left turn by about 9 degrees, with a deceleration of about 0.5 meters per second squared.' and the metada
…（已截断，原文 845 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/model/vtimellm_llama.py
```

</details>

<details><summary>输出摘要</summary>

```
1	import torch
2	import torch.nn as nn
3	from typing import List, Optional, Tuple, Union
4	from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
5	from transformers.modeling_outputs import CausalLMOutputWithPast
6	from .vtimellm_arch import VTimeLLMMetaModel, VTimeLLMMetaForCausalLM
7	
8	class VTimeLLMConfig(LlamaConfig):
9	    model_type = "VTimeLLM"
10	
11	class VTimeLLMLlamaModel(LlamaModel, VTimeLLMMetaModel):
12	    config_class = VTimeLLMConfig
13	
14	    def __init__(self, config: LlamaConfig):
15	        super(VTimeLLMLlamaModel, self).__init__(config)
16	
17	class VTimeLLMLlamaForCausalLM(LlamaForCausalLM, VTimeLLMMetaForCausalLM):
18	    config_class = VTimeLLMConfig
19	
20	    def __init__(self, config):
21	        super(LlamaForCau
…（已截断，原文 3597 字符）
```

</details>

<details><summary>思考过程</summary>

Key findings so far:
- Training JSON has NO "task" field for most items, but TG items DO have `task: 'time_grounding'` (13,124 of 148,271). But dataset.py's `LazySupervisedDataset.__getitem__` doesn't pass the `task` field through to the batch — it only returns input_ids, labels, image. So to identify TG samples in a batch, you'd need to either add a `task` passthrough or regex-match the answer text "from frame XXX to frame YYY".

Let me now check:
1. vtimellm_llama.py forward + prepare_inputs_labels_for_multimodal in vtimellm_arch.py (feature injection path)
2. evaluate_model.py IoU computation (for reward function reuse)
3. Check builder.py load_pretrained_model
4. Check mixed script run_stage2_full_seqv3_mixed.sh (the base variant)

Let me grep the answer format in TG samples: "from frame XXX to frame YYY". Check evaluate_model.py for IoU parsing.

</details>

TG 样本确实带有 `task: 'time_grounding'` 字段（13,124 / 148,271 项），但 dataset.py 中并没有将其传递到 batch 中。现在我来检查一下模型架构（特征注入 + generate 路径）和 IoU 评估器。

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -n "def \|IoU\|iou\|from frame\|parse" /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/evaluate_model.py | head -60
description: List functions and IoU logic in evaluate_model.py
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
166:Answer: The car in front moves slightly forward from frame 000 to frame 008.
167:GT: The car in front moves slightly forward until it stops again from frame 0 to frame 8.
173:Question: How do the cars to the right change over time from frame 00 to frame 16?
210:    def __init__(self, use_gpt: bool = False, gpt_api_key: Optional[str] = None,
251:    def _resolve_bertscore_model(model_path: str) -> Tuple[str, int]:
278:    def canonical_task(cls, task: str) -> str:
286:    def
…（已截断，原文 4218 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/model/vtimellm_arch.py
```

</details>

<details><summary>输出摘要</summary>

```
1	import torch
2	import torch.nn as nn
3	from vtimellm.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
4	from abc import ABC, abstractmethod
5	
6	class VTimeLLMMetaModel:
7	
8	    def initialize_vision_modules(self, model_args):
9	        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
10	
11	        if not hasattr(self, 'mm_projector'):
12	            self.mm_projector = nn.Linear(768, self.config.hidden_size)
13	
14	        if pretrain_mm_mlp_adapter is not None:
15	            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
16	            def get_w(weights, keyword):
17	                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
18	
19	            self.mm_projector.load_state_dict(get_w(mm_projector_weigh
…（已截断，原文 9794 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/evaluation/evaluate_model.py
```

</details>

<details><summary>输出摘要</summary>

```
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
295	        """Extract {start_frame, end_frame} from text like 'from frame 006 to frame 014'.
296	
297	        Returns None when no frame interval is
…（已截断，原文 3260 字符）
```

</details>

<details><summary>思考过程</summary>

Now let me look at:
1. vtimellm_arch.py — prepare_inputs_labels_for_multimodal (feature injection, mm_projector)
2. extract_time_grounding_frames in evaluate_model.py (lines 294-340) for the reward function
3. Check if there's a val/eval generate in training (evaluation_strategy no → no prediction_step used)
4. builder.py load_pretrained_model (for how LoRA + mm_projector load at inference)

Let me read those parts.

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/model/builder.py
```

</details>

<details><summary>输出摘要</summary>

```
1	import os
2	import shutil
3	
4	from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
5	import torch
6	from vtimellm.model import *
7	from peft import PeftModel
8	
9	def load_lora(model, lora_path):
10	    non_lora_trainables_path = os.path.join(lora_path, 'non_lora_trainables.bin')
11	    if os.path.exists(non_lora_trainables_path):
12	        non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
13	        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
14	        if any(k.startswith('model.model.') for k in non_lora_trainables):
15	            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
16	     
…（已截断，原文 3981 字符）
```

</details>

<details><summary>思考过程</summary>

I now have almost everything. A few remaining checks:

1. builder.py — how load_pretrained_model loads stage2 LoRA (to confirm generate-time path & that model.generate works with mm_projector). Also check whether batch generate works — prepare_inputs_for_generation passes images through, and prepare_inputs_labels_for_multimodal handles input_ids.shape[1]==1 for decode steps. Batch generate with different-length features: images is a list (in inference it's `image[None,]` — a tensor?). Actually in inference.py: `images=image[None,].cuda()` — a 3D tensor [1, N, 768]. In prepare_inputs_labels_for_multimodal, if not list, `image_features = self.get_model().mm_projector(images)` — for batch B with different N frames, a stacked tensor would need equal N. The collator in dataset.py stacks images if same shape else keeps list. So batch generate is possible if features are passed as a list; the arch handles list of variable-length tensors. For generation steps (input_ids.shape[1]==1), prepare_inputs_for_generation keeps images in _inputs, and forward handles the cache-extension path. OK.

2. Check transformers version installed (affects Trainer API, e.g. compute_loss signature). Let me check pip show transformers in the conda env wqlc. Actually may take time; check requirements.txt and possibly `conda run` or pip list. Let me check quickly.

3. Check whether there's an eval during training (evaluation_strategy no → no prediction_step). Confirmed in script: `--evaluation_strategy no`.

4. Check run_stage2_full_seqv3_mixed.sh briefly for differences (the b3 is basically B2's script with different wandb project — interesting: comment says B2). Actually run_stage2_full_seqv3_mixed_b3.sh is nearly identical to b2 variant. Note batch size 8, grad accum 16, effective batch 128. lr 1e-4, 3 epochs, LoRA r=64 alpha=128, target_modules=find_all_linear_names → all Linear except lm_head (i.e., q,k,v,o,gate,up,down proj).

5. Check checkpoints dir for the B3 model and whether stage3 exists. Also check run_b3_pipeline.sh and run_mixed_eval_after_train.sh to see the full B3 pipeline and eval args (metatoken injection at eval). This matters for RL: reward needs GT frame range; also rollout must replicate eval-time prompt (metatoken).

6. Check zero3_offload.json and zero2.json exist (already listed). The b3 script uses zero3.json with CPU offload of optimizer + params. Single GPU (--include localhost:0). ZeRO-3 with single device: params partitioned, so rollout generation would need GatheredParameters or use trainer.model with deepspeed engine — generation under ZeRO-3 works if using trainer.model (deepspeed engine handles gather on forward). Actually for generation under ZeRO-3, model.forward works (params gathered on the fly), so model.generate should also work since it calls forward repeatedly. Known pitfalls: use_cache with ZeRO-3 is fine on single GPU; but gather overhead per step. Also KV-cache. The bigger issue: with gradient checkpointing + use_cache=False in training config, need to re-enable use_cache for generate.

Let me quickly check: requirements.txt, pip list for transformers version, the run_b3_pipeline.sh, and builder.py load_pretrained_model.

</details>

整体架构很清晰。还有几项最后的检查：builder（推理时的 LoRA 加载）、B3 pipeline/eval 脚本，以及已安装的 transformers 版本。

**🔧 Bash**

<details><summary>输入</summary>

```
command: cat /root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_b3_pipeline.sh 2>/dev/null | head -80; echo "====== mixed eval script ======"; head -60 /root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_mixed_eval_after_t
…（已截断，原文 219 字符）
description: Show B3 pipeline and eval scripts
```

</details>

<details><summary>输出摘要</summary>

```
#!/bin/bash
# B3 两阶段链（2026-09-02）：整场景输入 + 修复版 metatoken（relative-to-previous，meta2 数据）。
#   阶段1: stage2 混合重训（run_stage2_full_seqv3_mixed_b3.sh，148k meta2 × 3 epochs）
#          —— 论文完整配置（整场景视觉 + 论文语义 meta）的最忠实复现
#   阶段2: 同口径评测（--whole_scene --per_sequence --answer_frames，meta 走修复版渲染）
# 显存门控窗口 72h（CoRViD/xmuda 共用 GPU）；断点续训（sort -V）兜底。
# tmux 会话 b3pipeline 运行。
set -u
START_STAGE=${1:-1}
case "$START_STAGE" in 1|2) ;; *) echo "用法: bash $0 [1|2]"; exit 1 ;; esac
cd /root/autodl-tmp/wql/mmb4dl/mllm
eval "$(/root/autodl-tmp/miniconda3/bin/conda shell.bash hook)"
conda activate wqlc
export HF_HUB_OFFLINE=1 HF_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline PYTHONUNBUFFERED=1

if [ "$START_STAGE" -le 1 ]; then
echo "===== 阶段1: mixe
…（已截断，原文 5581 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -iE "transformers|peft|deepspeed|torch" /root/autodl-tmp/wql/mmb4dl/mllm/requirements.txt; echo "=== installed ==="; /root/autodl-tmp/miniconda3/envs/wqlc/bin/pip list 2>/dev/null | grep -iE "^(t
…（已截断，原文 264 字符）
description: Check transformers/peft/deepspeed versions
```

</details>

<details><summary>输出摘要</summary>

```
deepspeed==0.16.4
peft==0.4.0
torch==2.5.1
torchvision==0.20.1
transformers==4.31.0
=== installed ===
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

requirements.txt says transformers==4.31.0, peft==0.4.0, deepspeed==0.16.4, torch==2.5.1. The installed pip list didn't return (maybe the pip path is wrong). Let me check the actual installed version quickly. Also confirm GPU count (single GPU training: `--include localhost:0`). And check that `trl` is not installed.

Also want to double check the mixed script base variant (run_stage2_full_seqv3_mixed.sh) is same config. And check zero2.json stage setting (stage 2) for completeness. Also check how many GPUs (nvidia-smi) — read-only OK. And check GPU VRAM since RL rollout feasibility depends on it.

Also want to check whether dataset.py filters 'split' field — the JSON has 'split' key with 'train'. Dataset doesn't filter. Fine.

One more thing: does train.py have any eval dataset? make_supervised_data_module returns eval_dataset=None. So no prediction_step usage.

Let me also verify tokenizer pad token = unk, and model_max_length 2048, TG answer "from frame XXX to frame YYY" — check GT answer of TG sample to confirm format. Also verify that the collator passes images as list when shapes differ (whole_scene -> all scenes 39/40/41 frames → shapes differ → list of tensors). Note: under whole_scene, features per scene are 39/40/41 frames so images can't be stacked; the collator keeps them as a list. That matters for batch rollout generation: model.forward with images list works in prepare_inputs_labels_for_multimodal. But generate with different-length features per sample... prepare_inputs_for_generation passes `images` through kwargs; during decode steps input_ids.shape[1]==1 → arch returns early path. OK, batch generate should work in principle, though VTimeLLM upstream never does batch generate in eval.

Let me run the remaining checks.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: /root/autodl-tmp/miniconda3/envs/wqlc/bin/python -c "import transformers, peft, deepspeed, torch; print('transformers', transformers.__version__); print('peft', peft.__version__); print('deepspeed', d
…（已截断，原文 321 字符）
description: Print installed library versions
```

</details>

<details><summary>输出摘要</summary>

```
(Bash completed with no output)
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null | head -5; echo "=== zero2 stage ==="; grep -A2 '"stage"' /root/autodl-tmp/wql/mmb4dl/mllm/scripts/zero2.json; echo "=== t
…（已截断，原文 721 字符）
description: Check GPU, zero2 stage, trl presence, TG answer format
```

</details>

<details><summary>输出摘要</summary>

```
Exit code 127
name, memory.total [MiB], memory.free [MiB]
NVIDIA GeForce RTX 5090, 32607 MiB, 9714 MiB
=== zero2 stage ===
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true,
=== trl? ===
=== TG answer format ===
/bin/bash: line 3: /root/autodl-tmp/miniconda3/envs/wqlc/bin/python: No such file or directory
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

The conda env is at /root/autodl-tmp/.conda-stuff/envs/wqlc (as seen in run_b3_pipeline.sh). Let me use that python. Also GPU: single RTX 5090 32GB, currently 9.7GB free.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: PY=/root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python; $PY -c "import transformers, peft, deepspeed, torch; print('transformers', transformers.__version__); print('peft', peft.__version__); print('dee
…（已截断，原文 666 字符）
description: Check versions and TG ground truth format with correct python
```

</details>

<details><summary>输出摘要</summary>

```
[2026-09-06 15:18:41,023] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
transformers 4.47.0
peft 0.13.2
deepspeed 0.16.4
torch 2.8.0+cu128
TG count: 13124
GT: from frame 000 to frame 008.
GT: from frame 000 to frame 008.
GT: from frame 000 to frame 008.
split values: {'train'}
accelerate                1.3.0
```

</details>


### USER

The TodoWrite tool hasn't been used recently. If you're working on tasks that would benefit from tracking progress, consider using the TodoWrite tool to track progress. Also consider cleaning up the todo list if has become stale and no longer matches what you are working on. Only use it if it's relevant to the current work. This is just a gentle reminder - ignore if not applicable.


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

I have everything I need. No `trl` installed. transformers 4.47.0 installed (requirements.txt says 4.31.0 but env has 4.47.0), peft 0.13.2, deepspeed 0.16.4, torch 2.8.0. accelerate 1.3.0. Single RTX 5090 32GB.

Let me compile the final report. Key points:

1. Training framework:
- Entry: /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/train/train.py (train() at L245), launched via train_mem.py (flash-attn patch attempt, falls back to SDPA; imports train from train.py).
- Trainer: /root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/train/vtimellm_trainer.py — `class VTimeLLMTrainer(Trainer)` (L28). Inherits transformers.Trainer directly. Only overrides `_save_checkpoint` (L30) and `_save` (L53) for stage1 mm_projector saving. NO compute_loss override, NO prediction_step override. Loss computed by vanilla HF Trainer → model forward with labels → LlamaForCausalLM.forward does shift + CrossEntropy internally. instantiated at train.py L439-442. eval_dataset=None (dataset.py L545) → no eval loop, no prediction_step.
- Model forward: vtimellm_llama.py L33-78 — VTimeLLMLlamaForCausalLM.forward calls prepare_inputs_labels_for_multimodal then super().forward (LlamaForCausalLM) with labels → HF loss (shift + CE) inside LlamaForCausalLM.

2. Data flow:
- dataset.py LazySupervisedDataset (L417), __getitem__ (L456-500) returns dict(input_ids, labels, image). DataCollatorForSupervisedDataset (L503-534) → batch: input_ids, labels, attention_mask, images (stacked tensor if all same shape, else list).
- Labels masking: preprocess_v1 (L265-344) — human turns + system masked IGNORE_INDEX (L328), only gpt answer tokens supervised; answer text enters labels verbatim via tokenizer_image_token on full conversation.
- TG identification: training JSON has `task` field on TG items only (13,124/148,271 = ~8.9%), plus scene_id/scene_token/feat_indices/feat_range/human_annotation. But dataset.py does NOT pass `task` through — __getitem__ only keeps conversations-derived fields + image. To identify TG in batch: (a) add task passthrough in __getitem__+collator, or (b) regex the GT answer text "from frame XXX to frame YYY" (GT format confirmed: 'from frame 000 to frame 008.'). Also GT needs to be accessible for reward → must add gt/gt_frames to the item dict.

3. LoRA config (run_stage2_full_seqv3_mixed_b3.sh):
- LoRA enabled, r=64, alpha=128, dropout 0.05 default, bias none; target_modules=find_all_linear_names(model) (train.py L167-178) → ALL nn.Linear except lm_head (q/k/v/o/gate/up/down proj).
- lr 1e-4, cosine, warmup 3%, epochs 3, per_device_bs 8, grad_accum 16 (eff. 128), bf16, model_max_length 2048, gradient_checkpointing True, weight_decay 0, dataloader workers 4, save every 200 steps keep 3.
- DeepSpeed ZeRO-3 (scripts/zero3.json stage 3) + CPU offload of optimizer AND params, single GPU: `deepspeed --include localhost:0 --master_port 29583`. Also metatoken embedding rows trainable via grad hook (train.py L377-397) — hook not ZeRO-partition aware (comment says assumes single-device ZeRO-3).
- Freeze mm_projector (freeze_mm_mlp_adapter True), projector loaded from stage1 mm_projector.bin.

4. Generation:
- inference() in vtimellm/inference.py L26-61: single-sample generate, do_sample=False default, temperature 0.05 if sampling, num_beams=1, max_new_tokens=1024, use_cache=True, KeywordsStoppingCriteria on stop_str. test_b4dl.py calls run_inference → inference() per sample (L382-384, L614).
- prepare_inputs_for_generation (vtimellm_llama.py L80-87) passes images through; prepare_inputs_labels_for_multimodal handles decode steps (input_ids.shape[1]==1, arch L36-50). Batch generate is plausible (arch handles images as list with variable frame counts, L52-56) but untested in repo; collator already produces list-of-tensors images for variable-length scenes.
- No val generation in training (eval_dataset None, evaluation_strategy no).
- Feature injection at generate: images (N,768) → mm_projector (768→hidden 4096) → inserted at <video> token position (IMAGE_TOKEN_INDEX -200), labels for feature positions = IGNORE_INDEX.
- Reward infrastructure: evaluate_model.py extract_time_grounding_frames (L294-307) + compute_miou (L317-340) — directly reusable as reward function (pure python, no deps).

5. DeepSpeed configs: scripts/zero3.json (stage 3 + CPU offload optimizer+param), zero2.json (stage 2), zero3_offload.json. Stage2 uses zero3.json.
ZeRO-3 pitfalls for generate+logprob: params partitioned; model.generate under ZeRO-3 engine works on forward (params gathered on the fly) but each forward triggers gather — slow with CPU offload (params offloaded to CPU! zero3.json offloads params to CPU → generation would be extremely slow, every layer gather from CPU each step). Options: use zero2 for RL stage (LoRA-only training, ZeRO-2 sufficient — optimizer states partitioned; base model frozen except LoRA + 2 embedding rows), or disable param offload. Also the embedding grad hook (train.py L386-395) not ZeRO-partition aware. Also use_cache=False set in train.py L282 for training; generate needs use_cache=True (inference() passes it explicitly in gen_kwargs, so OK). Gradient checkpointing conflicts with generate — should call model.eval() / disable. Known ZeRO-3 + KV cache issue is fine on single GPU. Also with ZeRO-3 the ref model for DPO would double memory unless using the frozen base (LoRA adapter can be disabled via peft `disable_adapter` context — that's the natural ref policy, no extra memory).
- VRAM: RTX 5090 32GB single card; currently 9.7GB free (other processes). Training at bs8×2048 with ZeRO-3+offload fits.

6. RL traces: none. grep across mllm *.py/*.sh/*.yaml found zero matches for reward/reinforc/grpo/dpo/rlhf/policy_loss/rollout/advantage/ref_model (only dataset JSONs and unrelated mmdetection3d configs). trl not installed.

Shortest path assessment:
Option A (online RL in Trainer.compute_loss): implement custom Trainer.compute_loss that, for TG-only batches, runs model.generate on prompt part, parses frames, IoU reward vs GT, computes policy gradient / GRPO-style token-level loss on rollout tokens (need per-token logprobs = second forward with labels=None or gather logits), plus keep SFT CE on the rest/mixed. Complexity: high-medium. Key hooks: VTimeLLMTrainer.compute_loss, dataset passthrough of task+GT, batch generate with images list. VRAM: rollout + logprob forward with 32GB single card, ZeRO-3 + CPU param offload makes generation very slow (per-layer CPU gather); recommend switching to zero2.json for RL stage (LoRA-only trainable → ZeRO-2 fine; note zero2.json exists) and/or turning off param offload. Because base is frozen + LoRA, "ref model" via adapter disable is free.
Option B (offline: generate N rollouts per TG prompt once, compute IoU, then reweight-SFT / DPO / best-of-n SFT): much lower risk; reuses test_b4dl.py generate loop and evaluate_model.py parsing/IoU verbatim; then trains with existing trainer by replacing labels/weights (need small trainer change for per-sample weights or just filtering/preference pairs). DPO needs TRL (not installed) or ~150 lines of custom loss in VTimeLLMTrainer.compute_loss using two forwards (policy + frozen ref via adapter-disable). 

Estimate workload:
- Option A GRPO-ish on TG only: dataset passthrough (~20 lines), collator (~10), VTimeLLMTrainer.compute_loss with rollout+reward+logprob (~200-300 lines), config switch to zero2/no-offload, eval harness for mid-train mIoU (~reuse test_b4dl.py). Roughly 2-4 days incl. debugging ZeRO-3/generation interactions, VRAM tuning.
- Option B offline DPO: generation script (~reuse, 0.5 day), DPO trainer (~150-250 lines in compute_loss, 1 day), total ~1.5-2.5 days.
- Weighted SFT (REINFORCE-style with offline samples): simplest, ~0.5-1 day.

Risks:
- transformers 4.47.0 Trainer API: compute_loss signature (model, inputs, return_outputs=False, num_items_in_batch) — newer versions compute token-averaged loss; if overriding compute_loss must handle num_items_in_batch to avoid breaking grad accum normalization. Note requirements.txt says 4.31.0 but env has 4.47.0 — the code was adapted (train.py comments mention transformers >= 4.45 SDPA).
- ZeRO-3 + generate: param CPU offload in zero3.json makes rollout generation prohibitively slow; ZeRO-3 partition + gradient hook for meta embed rows not partition-aware (already assumed single GPU).
- Single 32GB GPU currently shared (9.7GB free) — RL rollout needs exclusive GPU.
- Images list batching: whole_scene features are 39/40/41 frames → collator passes list; batch generate requires per-sample variable-length handling — works in arch but padding for batch generate: prepare_inputs_labels_for_multimodal pads to max_len in batch — OK for prefill; decode steps use attention_mask cat logic (arch L37-49) which assumes images present in past_key_values path. Batch generate is the riskiest untested piece; fallback = per-sample rollout (slow, batch 1).
- TG answers are short (~10 tokens): sequence-level RL signal per sample is limited to ~6-8 answer tokens; parsing failures (no "from frame X to frame Y") must be rewarded 0 (evaluator already does that).
- Mixed-task data: 91% non-TG; RL only on TG subset (8.9%) — need to guard against catastrophic drift on other 5 tasks (keep SFT loss on other tasks or KL anchor).
- Also note: reward needs GT; training JSON has GT answers in conversations[1] — for TG, GT text contains frames; parse with extract_time_grounding_frames.
- Checkpoint/resume logic in train.py L448-466 auto-resumes latest checkpoint — RL run should use a fresh output_dir to avoid silently resuming SFT.

Also mention: the eval-time prompt building (metatoken <4DLiDAR>/<meta> injection) is already IN the training data (TG sample human value includes <4DLiDAR>\n<video>\n...\n<meta> ...) — so rollouts at training time can reuse the dataset's question text directly; no need to replicate build_query, provided the RL dataset is the same meta2 json. Good for rollout fidelity.

Now, also verify quickly: what is the actual prompt template at training (preprocess_v1 uses vicuna_v1 "v1" conv: "A chat between... USER: ... ASSISTANT:"). At inference conv_templates["v1"] too. So rollout prompt = conv template + question from dataset. Consistent.

One more detail: model_max_length 2048; whole_scene features ~40 frames × 4096 dims after projector = 40 tokens? Wait — each frame is a 768-dim vector → mm_projector → single 4096 embedding per frame. So each frame = 1 token. 40 frames = 40 tokens. Sequence length is dominated by text (metatoken descriptions). So generation is cheap; rollout cost is dominated by LLM forwards with ~300-600 token prompts.

Wait, actually each frame is one 768-d vector → one token. Yes.

I'm confident in the report now. Format: training code structure map with file:line, compute_loss location, LoRA/deepspeed config highlights, shortest path judgment, risks. In Chinese since the question is in Chinese.

</details>

所有问题均已解答。以下是完整报告。

---

# B4DL 复现仓库 MLLM 训练代码勘察报告（RL 损失接入评估）

## 0. 结论速览

- 训练基座是**原封不动的 `transformers.Trainer`**（子类只重写了 checkpoint 保存），loss 完全走 HF 默认路径（LlamaForCausalLM 内部 shift+CE），**没有任何自定义 compute_loss / prediction_step**。
- **全仓库无任何 RL 痕迹**（reward/grpo/dpo/rlhf/rollout/policy_loss 均零命中，仅数据 JSON 与无关的 mmdetection3d 配置误匹配）；`trl` 未安装。
- 加 RL 的最短路径判断见 §7：**离线 rollout + IoU 加权 SFT/DPO 风险最低**；在线 GRPO 式 compute_loss 可行但要换 deepspeed 配置（zero3.json 把参数 offload 到 CPU，rollout 生成会慢到不可用）。
- **评测端的 IoU reward 函数可直接复用**（纯 Python，零依赖）：`evaluate_model.py:294-340`。

---

## 1. 训练框架基座

**入口链**：`mllm/scripts/run_stage2_full_seqv3_mixed_b3.sh:31` → `deepspeed --include localhost:0 vtimellm/train/train_mem.py` → `train_mem.py`（仅尝试 flash-attn monkey patch，失败则 SDPA）→ `from train import train`。

**关键文件：行号**

| 内容 | 位置 |
|---|---|
| `train()` 主函数 | `/root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/train/train.py:245-488` |
| `VTimeLLMTrainer(Trainer)` 类定义 | `/root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/train/vtimellm_trainer.py:28` |
| Trainer 实例化 | `train.py:439-442` |
| 模型 forward（多模态注入 + labels 透传） | `/root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/model/vtimellm_llama.py:33-78` |
| 特征注入 `prepare_inputs_labels_for_multimodal` | `/root/autodl-tmp/wql/mmb4dl/mllm/vtimellm/model/vtimellm_arch.py:29-179` |

- **继承的就是 `transformers.Trainer`**。`VTimeLLMTrainer` 只重写了 `_save_checkpoint`（trainer.py:30，stage1 只存 mm_projector.bin）和 `_save`（:53）。**没有重写 compute_loss，也没有 prediction_step**。
- **loss 位置**：`eval_dataset=None`（dataset.py:545）+ 脚本 `--evaluation_strategy no`，因此无 eval 循环。训练 loss 走 HF 默认：`VTimeLLMLlamaForCausalLM.forward`（llama.py:66）调 `LlamaForCausalLM.forward`，labels 由 arch 层在特征插入位填 `IGNORE_INDEX`（arch.py:112）后透传，**shift + CrossEntropy 全在 HF LlamaForCausalLM 内部完成**——即加自定义 RL 项必须自己重写 `compute_loss`（当前为零覆盖，改起来干净）。
- 环境实装版本：**transformers 4.47.0 / peft 0.13.2 / deepspeed 0.16.4 / torch 2.8.0+cu128 / accelerate 1.3.0**（requirements.txt 写 4.31.0 已过期，代码注释也表明按 ≥4.45 适配过）。注意 4.47 的 `Trainer.compute_loss(model, inputs, return_outputs=False, num_items_in_batch=...)` 签名与梯度累积 loss 归一化问题，重写时要兼容。

## 2. 数据流

- **Dataset**：`LazySupervisedDataset`（dataset.py:417），`__getitem__`（:456-500）加载 `<feat_folder>/<scene_id>.npy`（[N,768] fp16，每帧=1个token），`whole_scene=True` 时不切片喂整场景（39/40/41 帧）；返回 `{input_ids, labels, image}`。
- **Collator**（dataset.py:503-534）：padding 后返回 `input_ids / labels / attention_mask / images`；**images 若各样本形状不同（whole_scene 下必然不同）则为 list[Tensor] 而非 stack**——这对 batch generate 是好消息，arch.py:52-56 原生支持 list 变长输入。
- **labels 构造**：`preprocess_v1`（dataset.py:265-344），vicuna_v1 模板，human/system 全部 `IGNORE_INDEX`（:328），**只监督 gpt 回答 token**；TG 答案文本 `"from frame 000 to frame 008."` 按 GT 逐 token 进 labels。
- **TG 样本识别**：训练 JSON `b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json`（148,271 条）中 **TG 条目带 `task: 'time_grounding'` 字段（13,124 条，8.9%）**；其余条目无 task 字段。item 另有 `scene_id/scene_token/feat_indices/feat_range/human_annotation`。**但 dataset.py 没有把 `task` 透传进 batch**——需在 `__getitem__`/collator 加 2 个字段（task + gt 文本），或按答案正则 `from\s+frame\s+(\d+)\s+to\s+frame\s+(\d+)` 识别。奖励计算所需的 GT 就在 `conversations[1]['value']` 里。
- 附带利好：**metatoken（`<4DLiDAR>`/`<meta>` 块）已在训练数据的问题文本里**（meta2 版本），训练时 rollout 直接复用 dataset 的 question 文本即可与评测 prompt（test_b4dl.py `build_query`）同分布，无需重建 prompt。

## 3. LoRA / 训练超参（B3 脚本）

`/root/autodl-tmp/wql/mmb4dl/mllm/scripts/run_stage2_full_seqv3_mixed_b3.sh`：

- LoRA：`--lora_enable True --lora_r 64 --lora_alpha 128`，dropout 0.05、bias none（默认）；**target_modules = `find_all_linear_names`（train.py:167-178）= 除 lm_head 外所有 nn.Linear**（q/k/v/o/gate/up/down）。
- lr 1e-4 cosine，warmup 3%，**3 epochs，bs 8 × grad_accum 16（有效 128）**，bf16，`model_max_length 2048`，`gradient_checkpointing True`，weight_decay 0（metatoken embedding 行训练的前提），save 每 200 步留 3 份。
- **DeepSpeed ZeRO-3**（`--deepspeed ./scripts/zero3.json`），单卡 `--include localhost:0`。
- mm_projector 冻结（`--freeze_mm_mlp_adapter True`，加载 stage1 的 `mm_projector.bin`）。
- 特殊点：**metatoken embedding 两行可训**，用梯度 hook 掩码实现（train.py:377-397），注释明言"hook 非 ZeRO-partition 感知、假设单卡 ZeRO-3"。
- 自动断点续训：train.py:448-466 会**自动找 output_dir 下最新 checkpoint 续训**——RL 实验务必用全新 output_dir，否则会静默接上 SFT。

## 4. 生成能力（rollout 基建）

- **评测端生成**：`evaluation/test_b4dl.py:382-384,614` → `vtimellm/inference.py:26-61` `inference()`：单样本，`do_sample=False`（可开采样 temperature=0.05）、num_beams=1、`max_new_tokens=1024`、`use_cache=True`、`KeywordsStoppingCriteria`（###/sep2）。**这就是现成的 rollout 参考实现**（离线批量生成脚本可基本照抄）。
- **generate 时的特征注入**：`prepare_inputs_for_generation`（llama.py:80-87）把 `images` 塞回每步 forward；arch.py:36-50 处理解码步（`input_ids.shape[1]==1` + KV cache 扩展 attention_mask）；prefill 走 arch.py:29-179，features 经 `mm_projector`（768→4096 Linear，arch.py:12）插到 `<video>`（IMAGE_TOKEN_INDEX=-200）位置。
- **batch generate**：架构上可行（arch 支持变长 list images + batch padding，collator 也产出 list），但**仓库内从未跑过 batch generate，属未验证路径**；且 batch 内变长场景 prefill padding 已处理、解码步 mask 扩展逻辑对 batch 的一致性需实测。稳妥起见 rollout 可先 batch=1（TG 答案仅 ~10 token，瓶颈在 prefill，prompt ~300-600 token）。
- 训练代码里**没有**任何 val 生成/beam search 基建；`model.config.use_cache=False`（train.py:282）训练时关 cache，generate 时 gen_kwargs 显式传 `use_cache=True` 可覆盖，但 gradient checkpointing 下 generate 前需 `model.eval()` 并注意冲突。

## 5. DeepSpeed 配置与 RL 坑

- stage2 用 `scripts/zero3.json`：**stage 3 + `offload_optimizer` 和 `offload_param` 都到 CPU**（zero3.json:26-34）。另有 `zero2.json`（stage 2）与 `zero3_offload.json` 备选。
- RL 若需"生成 + 逐步 logprob"：
  - **ZeRO-3 + param CPU offload 下 generate 极慢**（每层每步从 CPU gather 参数）——在线 RL 几乎必须换 `zero2.json`（LoRA-only 训练本就够用）或去掉 offload_param。
  - ZeRO-3 下 forward 本身可行（DS engine 按需 gather），单卡无多卡 gather 复杂度；embedding 梯度 hook 已声明只兼容单卡。
  - 硬件：单张 **RTX 5090 32GB**（当前被占用仅 9.7GB 空闲，run_b3_pipeline.sh 有 28GB 显存门控），rollout 期间要独占；若 DPO 需要 ref model，**可直接用 peft `disable_adapter()` 上下文复用同一模型当参考策略**（base 冻结 + LoRA），零额外显存。

## 6. 现有 RL 痕迹

无。`grep -riE "reward|reinforc|grpo|dpo|rlhf|policy_loss|rollout|advantage|ref_model"` 在 `mllm/` 全部 .py/.sh/.yaml 中**零命中**（.git 与 datageneration JSON 里的命中均为无关误匹配）。trl 未安装。

## 7. 加 RL 的最短路径判断与工作量

**可复用资产**：reward 解析+IoU（`evaluation/evaluate_model.py:294-307` 正则解析、:317-340 闭区间 IoU，含"解析失败=0分"语义）→ 直接 import；rollout 参考（inference.py + test_b4dl.py）；task 字段已在数据里（只差透传）。

**推荐路径（按工作量升序）**：

1. **离线 REINFORCE 式加权 SFT（~0.5-1.5 天）**：用 test_b4dl.py 的生成路径对 TG 训练集采样 K 个 rollout（可开 temperature），算 IoU 作权重，筛/加权后用现有 Trainer 直接 SFT。改动：一个离线生成脚本 + dataset 里给样本乘权（或简单过滤低分样本）。风险最低，最接近"改进定位"的增量验证。
2. **离线 DPO / best-of-n（~1.5-2.5 天）**：同上生成 chosen/rejected 对，在 `VTimeLLMTrainer` 重写 `compute_loss` 加 DPO 项（~150-250 行：policy+ref 两次 forward，ref 用 `disable_adapter()`），换 zero2.json。无 trl 依赖也能写。
3. **在线 GRPO 式（~3-5 天，含调试）**：`VTimeLLMTrainer.compute_loss` 内对 TG-only batch 做：组内采样 rollout → IoU reward → advantage → token logprob loss，其余 91% 非 TG batch 保留原 CE（防其他 5 任务漂移）。改动点：dataset 透传 task+gt（~20 行）、collator（~10 行）、compute_loss（~250-350 行）、deepspeed 换 zero2、中途评测脚本复用 test_b4dl.py。

**主要风险点**：
- zero3.json 的 **param CPU offload 使在线 rollout 不可用**（必须换 zero2/去 offload）；
- **batch generate 未验证**（变长 list images + 解码步 mask 扩展），可能只能 batch=1；
- transformers 4.47 的 `compute_loss` 需处理 `num_items_in_batch` 否则梯度累积 loss 归一化出错；
- 单卡 32GB 需独占（现仅 9.7GB 空闲），rollout+logprob 双 forward 的显存峰值需实测；
- train.py:448-466 的**自动断点续训**会把 RL run 静默接到 SFT checkpoint 上——用新 output_dir；
- TG 答案仅 ~10 个 token，每样本 RL 信号极短，且 8.9% 的 TG 占比下需要锚定损失防其他任务退化；
- metatoken embedding 梯度 hook（train.py:386-395）与任何新梯度逻辑共存时注意 mask 时序。

