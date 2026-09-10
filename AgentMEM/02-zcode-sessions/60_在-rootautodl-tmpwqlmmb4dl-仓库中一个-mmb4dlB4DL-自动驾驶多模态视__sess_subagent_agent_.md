# 在 /root/autodl-tmp/wql/mmb4dl 仓库中（一个 mmb4dl/B4DL 自动驾驶多模态视...

| 项 | 值 |
|---|---|
| 会话 ID | `sess_subagent_agent_86dfd096-bb03-4cfc-89c4-e19fd3bb4cc1` |
| 工作目录 | `/root/autodl-tmp/wql/mmb4dl` |
| 父会话 | `sess_5267559b-7b26-42ef-840c-42f9b265d1dd` |
| 时间 | 2026-09-08 23:22 → 2026-09-08 23:24 |
| 模型 | deepseek-v4-flash |
| 消息数 | 10（文本块 7，工具调用 21） |

---

### ASSISTANT  ·  `deepseek-v4-flash`

> 🔀 模型切换：None → 2801678c-7b5c-456f-ad57-1e9cb4a76bba/deepseek-v4-flash


### USER

在 /root/autodl-tmp/wql/mmb4dl 仓库中（一个 mmb4dl/B4DL 自动驾驶多模态视频问答复现项目），wiki/ 目录包含 13 个 GitHub 风格 wiki 页面 markdown 文件，全部最后修改于 2026-08-31。

请逐个阅读 wiki/*.md 全部文件（都很短，3-7KB），然后给出：
1. 每个文件的主要内容概述（2-3 句/文件），特别标注其中记载的"当前状态"类信息：基线版本号（B0/B1/B2/B3）、评测指标数值（accuracy、mIoU 等）、声称的"最新进展"、里程碑/路线图状态、以及 FAQ 或已知问题列表里的事项。
2. _Sidebar.md 与 Home.md 的结构（它们如何串联其他页面）。
3. 判断哪些页面已明显过时：例如仍在说"B3 训练中"、"B4 未启动"或引用旧的评测数字（对照：截至 2026-09-08，仓库已推进到 B4a 实验训练完成并正在评测，B3 mIoU 0.3467 已超论文 0.311，RL 引进已做可行性深度分析）。对每个过时页面列出具体过时点（哪一节/哪句话提到什么旧状态）。
4. 另外：检查 docs/ 目录（含 docs/learn docs/ 等子目录）里有什么文档文件（ls -la 递归列出文件名、大小、日期），特别找最近几次 commit 加入的分析文档（RL 引进可行性深度分析、训练优化角度分析、B4DL 训练全流程分步详解、全量工作进度总结等，时间 2026-09-06~09-08），看这些文档存放在哪里（哪个路径），文件名是什么。

只读调查，不要修改任何文件。输出用中文。


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Let me start by exploring the repository structure, the wiki directory, and docs directory.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls -la /root/autodl-tmp/wql/mmb4dl/wiki/
description: List files in wiki directory
```

</details>

<details><summary>输出摘要</summary>

```
total 100
drwxr-xr-x  2 root root 4096 Aug 31 00:33 .
drwxr-xr-x 19 root root 4096 Sep  8 12:49 ..
-rw-r--r--  1 root root 4861 Aug 31 00:33 Architecture.md
-rw-r--r--  1 root root 5735 Aug 31 00:33 Data-Generation.md
-rw-r--r--  1 root root 4092 Aug 31 00:33 FAQ-and-Known-Issues.md
-rw-r--r--  1 root root 4629 Aug 31 00:33 Home.md
-rw-r--r--  1 root root 6090 Aug 31 00:33 Inference-and-Evaluation.md
-rw-r--r--  1 root root 2952 Aug 31 00:33 Installation.md
-rw-r--r--  1 root root 6143 Aug 31 00:33 LiDAR-CLIP-Encoder.md
-rw-r--r--  1 root root 4504 Aug 31 00:33 Paper-vs-Reproduction.md
-rw-r--r--  1 root root 4671 Aug 31 00:33 Repository-Structure.md
-rw-r--r--  1 root root 5791 Aug 31 00:33 Reproduction-Guide.md
-rw-r--r--  1 root root 5857 Aug 31 00:33 Reproduction-Log.md
-rw-r--r--  1 r
…（已截断，原文 892 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls -la /root/autodl-tmp/wql/mmb4dl/
description: List files in repository root
```

</details>

<details><summary>输出摘要</summary>

```
total 104
drwxr-xr-x 19 root root  4096 Sep  8 12:49 .
drwxr-xr-x  5 root root    66 May 18 10:59 ..
drwxr-xr-x 25 root root  4096 Aug 12 17:37 .claude
drwxr-xr-x 10 root root  4096 Sep  8 23:19 .git
-rw-r--r--  1 root root   114 Aug  7 22:25 .gitattributes
-rw-r--r--  1 root root   848 Sep  6 16:07 .gitignore
drwxr-xr-x  2 root root   120 Aug  7 22:13 .ssh
drwxr-xr-x  2 root root    35 May 27 20:58 .vscode
drwxr-xr-x  3 root root    27 Sep  1 11:12 .zcode
-rw-r--r--  1 root root 18847 Sep  8 12:49 CLAUDE.md
drwxr-xr-x  2 root root  4096 Jun  5 16:28 Claude_record
-rw-r--r--  1 root root  4301 May 12 13:02 README.md
drwxr-xr-x  2 root root  4096 May 12 13:02 assets
drwxr-xr-x  2 root root   153 Sep  7 00:25 backups
drwxr-xr-x  7 root root  4096 Aug 29 21:17 datageneration
drwxr-xr-x  4 roo
…（已截断，原文 1370 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/wiki/Home.md
```

</details>

<details><summary>输出摘要</summary>

```
1	# B4DL 复现仓库 Wiki
2	
3	本仓库是对论文 **B4DL: A Benchmark for 4D LiDAR LLM in Spatio-Temporal Understanding**（ACM Multimedia 2025）的**独立复现**：基于官方发布的数据集与论文描述，完整重建了数据生成、LiDAR-CLIP 特征编码、VTimeLLM 多阶段训练与六任务评测的端到端流水线，并以论文指标为基准验证复现质量。
4	
5	> ⚠️ **本仓库非官方实现**（官方代码与权重从未完整公开）。复现的端到端步骤见 [[Reproduction-Guide]]，与论文的逐项差异见 [[Paper-vs-Reproduction]]，复现历程与基线锁定见 [[Reproduction-Log]]。
6	
7	- 📄 论文：[arXiv:2508.05269](https://arxiv.org/abs/2508.05269) ｜ ACM MM 2025（pp. 3399–3407）
8	- 🤗 官方数据集：[HuggingFace ccho4702/nuScenes-B4DL](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL)（metadata + train/test QA 数据，本仓库复现的直接输入）
9	- 📜 原论文许可：CC BY-NC-ND 4.0
10	
11	## 三大核心模块
12	
13	| 模块 | 目录 | 职责 |
14	|------|------|------|
15	| 数据生成管线 | `datageneration/` | 调用 GPT-4o API，从 nuScenes 相机图像生成 LiDAR 场景描述，再转化为 6 类任务的问答对 |
16	| LiDAR-
…（已截断，原文 3571 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/wiki/_Sidebar.md
```

</details>

<details><summary>输出摘要</summary>

```
1	**B4DL 复现 Wiki**
2	
3	- [[Home]] — 项目概览与复现状态
4	- [[Installation]] — 环境搭建
5	- [[Architecture]] — 架构总览
6	
7	**流水线**
8	
9	- [[Data-Generation]] — 数据生成（GPT-4o）
10	- [[LiDAR-CLIP-Encoder]] — 编码器与特征提取
11	- [[Training]] — 模型训练
12	- [[Inference-and-Evaluation]] — 推理与评测
13	
14	**复现专项**
15	
16	- [[Reproduction-Guide]] — 端到端复现指南
17	- [[Paper-vs-Reproduction]] — 与论文差异对照
18	- [[Reproduction-Log]] — 复现记录与基线 B0
19	
20	**参考**
21	
22	- [[Repository-Structure]] — 目录结构
23	- [[FAQ-and-Known-Issues]] — 已知问题
24	
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/wiki/Architecture.md
```

</details>

<details><summary>输出摘要</summary>

```
1	# 架构总览
2	
3	## 数据流全景
4	
5	```
6	nuScenes 数据集（相机图像 + LiDAR 点云）
7	  │
8	  ├─ [datageneration/] 相机图像 → GPT-4o → 场景描述 JSON → QA 数据集 JSON
9	  │     数据格式：{"from": "human/gpt", "value": "..."}
10	  │
11	  ├─ [encoders/lidarclip/] LiDAR 点云 → SST编码器 → CLIP 特征 (.npy)
12	  │     Stage1 特征：每帧独立 .npy，shape (1, 768)
13	  │       - stage1_features/（旧，frame_id 键控，95K 数据用）
14	  │       - stage1_features_sample/（新，sample_token 键控，对齐官方 162K 方案）
15	  │     Stage2 特征：每场景拼接 .npy，shape (N_frames, 768)
16	  │
17	  └─ [mllm/] 预提取特征 + QA 数据 → VTimeLLM → 训练 / 推理 / 评估
18	```
19	
20	三个模块之间只通过**文件**耦合：特征文件名 = 数据 JSON 中的 `scene_id`（详见 [[Training]] 的特征查找约定）。
21	
22	## 三阶段训练流程
23	
24	| 阶段 | 目标 | 关键配置 |
25	|------|------|----------|
26	| **Stage 1** | 对齐 mm_projector | 只训练 `nn.Linear(768, 4096)`，LLM 冻结。`--tune_mm_mlp_ada
…（已截断，原文 3640 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/wiki/Data-Generation.md
```

</details>

<details><summary>输出摘要</summary>

```
1	# 数据生成管线（datageneration/）
2	
3	调用 GPT-4o API，从 nuScenes 相机图像生成 LiDAR 视角的场景描述（Step 1），再转化为 6 类任务的问答对（Step 2）。对应论文 §3 的数据集构建流程。
4	
5	## 运行前准备
6	
7	- 设置环境变量：`OPENAI_API_KEY`（必填）、`OPENAI_BASE_URL`（可选，默认官方 API）、`B4DL_GPT_MODEL`（可选，默认 `gpt-4o`）
8	- 或通过 `--api_key` 参数直接传入（两个主脚本均必填）
9	- 下载 nuScenes v1.0-trainval，通过 `--nuscenes_root` 指定路径
10	
11	## Step 1 — 4D LiDAR 上下文提取
12	
13	```bash
14	cd datageneration
15	bash scripts/generate_description.sh
16	# 或
17	python generate_description.py \
18	    --start_index 10 --end_index 20 \
19	    --api_key YOUR_API_KEY \
20	    --nuscenes_root /path/to/nuScenes \
21	    --dataroot ./data
22	```
23	
24	**处理流程**：
25	- 每段序列取 `LOAD_N_FRAMES=5` 帧（`FRAME_INTERVAL=2` 间隔，如 index [0,2,4,6,8]）
26	- 6 个相机视图分两组：**前组**（FRONT/FRONT_LEFT/FRONT_RIGHT，15 张图）、**后组**（BACK/BAC
…（已截断，原文 4388 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/wiki/FAQ-and-Known-Issues.md
```

</details>

<details><summary>输出摘要</summary>

```
1	# 已知问题与注意事项
2	
3	按模块整理的踩坑清单。多数问题已在代码中修复，此处保留背景与检查方法。
4	
5	## 环境
6	
7	| 问题 | 说明 |
8	|------|------|
9	| 依赖文件选择 | 只用 `mllm/requirements.txt`；根目录 `requirements.txt` 版本过新且冲突 |
10	| peft 0.4.0 | Stage3 的 merge + 重加 LoRA 模式依赖旧式 API，不可升级 |
11	| transformers 4.31.0 | monkey-patch 与旧版 generation API 绑定；`merge_stage2.py` 里已专门修了新版 generation_config 校验报错的兼容分支 |
12	| flash-attn 与 RTX 5090 | sm_120 不支持 flash-attn 2.x，`train_mem.py` ImportError 时自动回退 SDPA |
13	| METEOR 需要 java | `apt-get install default-jre-headless`；缺 jar 会静默回退 NLTK（与论文不可比），看结果 JSON 的 `metric_backend` 字段确认 |
14	| clip.load 不转 fp16 | train.py 已加 `clip.model.convert_weights()` |
15	
16	## 数据生成（datageneration/）
17	
18	- `--start_index/--end_index` 必须是 `SAVE_TERM`(10) 的倍数，否则直接退出
19	- `--api_key` 必填（或设 `OPENAI_API_KEY`）；历史提交中有旧 key，公开发布前需清
…（已截断，原文 2681 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/wiki/Inference-and-Evaluation.md
```

</details>

<details><summary>输出摘要</summary>

```
1	# 推理与评测（mllm/）
2	
3	## 单条推理
4	
5	```bash
6	cd mllm
7	python vtimellm/inference.py \
8	    --model_base ./base_model/vicuna-v1-5-7b \
9	    --clip_path checkpoints/clip/ViT-L-14.pt \
10	    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
11	    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
12	    [--stage3 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3] \
13	    --video_path /path/to/video.mp4
14	```
15	
16	流程：`load_pretrained_model` 完整加载链 → `VideoExtractor(N=100)` 均匀抽帧 → CLIP ViT-L/14 编码 → v1 模板拼 `<video>\n`+query → greedy 生成（温度 0.05，max_new_tokens 1024）。
17	
18	## Gradio Web Demo
19	
20	```bash
21	python vtimellm/demo_gradio.py --model_base ... --clip_path ... \
22	    --pretrain_mm_mlp_adapter ... --stage2 ... [--share]
23	```
24	
25	支持
…（已截断，原文 4747 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/wiki/Installation.md
```

</details>

<details><summary>输出摘要</summary>

```
1	# 环境搭建
2	
3	## Conda 环境
4	
5	项目统一使用名为 **`wqlc`** 的 Conda 环境（Python 3.10），所有 Python 命令都必须在该环境中执行。
6	
7	```bash
8	conda create -n wqlc python=3.10 -y && conda activate wqlc
9	pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
10	pip install -r mllm/requirements.txt
11	```
12	
13	## ⚠️ 依赖文件选择
14	
15	- **权威依赖文件是 `mllm/requirements.txt`**。
16	- 根目录的 `requirements.txt` **版本过新且与 mllm 模块冲突，不要使用**。
17	- 编码器模块另有 `encoders/lidarclip/requirements.txt`（mmcv_full 1.3.9、mmdet 2.14、pytorch_lightning 1.6.2、openai_clip 等）。
18	
19	## 关键版本锁定及原因
20	
21	| 包 | 锁定版本 | 原因 |
22	|----|---------|------|
23	| torch / torchvision | 2.5.1 / 0.20.1 | CUDA 12.4 构建基线 |
24	| transformers | 4.31.0 | VTimeLLM 依赖旧版 Trainer/generation 内部 API 与 monkey-patch；新版会破坏兼容 |
25	| peft | 0.4
…（已截断，原文 2372 字符）
```

</details>

<details><summary>思考过程</summary>

Let me read all the wiki files in parallel. They're short (3-7KB). Let me read them all.

</details>

Now let me read all wiki files in parallel.


### ASSISTANT  ·  `deepseek-v4-flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/wiki/LiDAR-CLIP-Encoder.md
```

</details>

<details><summary>输出摘要</summary>

```
1	# LiDAR-CLIP 编码器（encoders/lidarclip/）
2	
3	基于 SST（Sparse Swin Transformer，CVPR 2022）backbone 的 LiDAR 编码器，训练目标是将点云序列特征对齐到冻结 CLIP 图像编码器（ViT-L/14）的输出空间，为下游 VTimeLLM 提供视觉特征。
4	
5	## 架构
6	
7	```
8	点云帧列表 (每帧 N×4: x,y,z,intensity，已投影到相机坐标系裁剪)
9	  → voxelization (0.5m voxel, 80×80 网格)
10	  → DynamicVFE (4→64→128 通道)
11	  → SSTInputLayerV2 (12×12 窗口) + SSTv2 4 层 Transformer (d_model=128)
12	  → 特征图 (bs, 128, 80, 80)
13	  → in_proj (128→768) + AttentionPool2d (CLIP 风格空间注意力池化)
14	  → (bs, 768)  ← 与 CLIP ViT-L/14 图像特征逐样本计算 MSE
15	```
16	
17	- **对齐方式是非对比损失**：CLIP 图像编码器完全冻结，训练目标为 LiDAR 特征与图像特征的 MSE（默认）或负余弦相似度（`--loss-function cosine`）
18	- 核心类：`LidarEncoderSST` / `SSTEncoder`（`lidarclip/model/sst.py`）、`AttentionPool2d`（`lidarclip/model/attention_pool.py`）；训练 LightningModule 为 train.py 内的 `LidarClip`
1
…（已截断，原文 4536 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/wiki/Paper-vs-Reproduction.md
```

</details>

<details><summary>输出摘要</summary>

```
1	# 与论文的对照（差异清单）
2	
3	复现 ≠ 复刻。本页列出复现实现与论文（arXiv:2508.05269）的全部已知一致项与偏差项，依据 2026-08-24 与 2026-08-29 两轮全面审计（详见 `docs/learn docs/B4DL_论文对齐审计_20260824.md`、`B4DL_全仓库论文对齐审查_20260829.md`）。
4	
5	## 结果对比（B0 vs 论文）
6	
7	| 指标 | 复现 B0 | 论文 | 状态 |
8	|------|---------|------|------|
9	| accuracy（existence + binary 平均） | 0.7629 | 0.762 | ✅ 持平 |
10	| mIoU（time_grounding） | 0.2696 | 0.311 | ❌ 差 0.041（≈3.1× 显著性半宽） |
11	| BLEU-4（语料级） | 0.0973 | 0.095 | ✅ |
12	| METEOR（Meteor-1.5 jar） | 0.1729 | 0.275 | ❌ 口径一致仍偏低，唯一遗留考证项 |
13	| ROUGE-L | 0.3244 | — | 口径一致 |
14	| BERTScore（roberta-large L17） | 0.8973 | 0.897 | ✅ 精确命中 |
15	
16	分任务 accuracy：existence 0.7072、binary 0.8187。
17	
18	## 一致项
19	
20	- **数据集产物 100% 对齐**：官方发布 train 148,271 条 / test 30,145 条，六任务条数与论文 Table 2 的 14 个数字精确吻合；官方 train 与 test 零重叠（700/150 scene 
…（已截断，原文 2926 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/wiki/Repository-Structure.md
```

</details>

<details><summary>输出摘要</summary>

```
1	# 目录结构
2	
3	```
4	mmb4dl/
5	├── README.md                    # 官方 README（管线概览、Demo、引用）
6	├── CLAUDE.md                    # Claude Code 工作指引（项目速查权威文件）
7	├── requirements.txt             # ⚠️ 已废弃，勿用（版本过新，与 mllm 冲突）
8	├── smoke_test.log               # 冒烟测试日志
9	│
10	├── datageneration/              # 数据生成管线（GPT-4o）
11	│   ├── config.py                # 全局配置：模型名、环境变量、任务类型、帧采样
12	│   ├── generate_description.py  # Step 1：相机图 → 场景描述 JSON
13	│   ├── generate_dataset.py      # Step 2：描述 → 6 类任务 QA JSON
14	│   ├── prompts.py               # 全部 prompt 模板（1 描述 + 6 QA）
15	│   ├── utils.py                 # ReadJson / base64 / QA 解析 / conversation 组装
16	│   ├── scripts/                 # generate_description.sh / generate_dataset.sh
17	│   └── tools/                   # metadata 构建、stage1 数据转换（LiDAR
…（已截断，原文 3555 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/wiki/Reproduction-Guide.md
```

</details>

<details><summary>输出摘要</summary>

```
1	# 复现实施指南
2	
3	从零复现到基线 B0 的端到端步骤。模块级细节见各分页，本页只讲**按什么顺序做什么、每步产出什么、如何验收**。
4	
5	> 论文原文的逐行解析与资源清单见 `docs/B4DL_复现方案.md`；本页是该方案的执行摘要。
6	
7	## 前置资源
8	
9	| 资源 | 说明 |
10	|------|------|
11	| 硬件 | 参考配置 2×RTX 5090（32GB）；CLIP ViT-L/14 + batch 32 即显存上限，超出用梯度累积 |
12	| nuScenes v1.0-trainval | 相机图像 + LiDAR 点云 + metadata（编码器与数据生成的输入） |
13	| HF `ccho4702/nuScenes-B4DL` | 官方发布：`metadata/`（scene/sequence_metadata.json）、`dataset/train/`（148,271 条 QA）、test_qa.json |
14	| OpenAI API Key | 仅数据生成路线需要；直接用官方数据可跳过 |
15	| 环境 | `wqlc` conda 环境，见 [[Installation]] |
16	
17	## Step 1 — 数据生成（可跳过）
18	
19	两条路线二选一：
20	
21	- **路线 A（推荐，B0 采用）**：直接使用官方发布的 train/test 数据——数据集产物经审计已与论文 100% 对齐，重跑 GPT-4o 管线只会引入随机性
22	- **路线 B（验证管线）**：完整重跑两步生成（`generate_description.sh` → `generate_dataset.sh`，六任务逐个跑），见 [[Data-Generation]]
23	
24	#
…（已截断，原文 4582 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/wiki/Reproduction-Log.md
```

</details>

<details><summary>输出摘要</summary>

```
1	# 复现记录与基线
2	
3	本页汇总 `docs/` 与 `docs/learn docs/` 中的复现历程、当前基线锁定与文档索引。**这些文档是排查"为什么这样做"的第一手资料**。
4	
5	## 复现时间线（2026-07 ~ 2026-08）
6	
7	| 阶段 | 内容 |
8	|------|------|
9	| 07 初 | plan1：打通训练管线（修复 mm_projector 768 维、验证特征链路） |
10	| 07-12 | VTimeLLM 源码分析；时序能力弱的根因分析（metatoken 缺失致命等 7 项） |
11	| 08-07~10 | 首次全量训练评测：定位 `<4DLiDAR>` token 未训练 bug；发现数据失衡（Yes 45.5%）导致模板化输出 |
12	| 08-24 | 论文对齐审计（A1-A9/B1-B6/R1-R4）：mIoU 半开区间 bug、per-scene 回退错配等修复；seqv3 数据格式引入 |
13	| 08-25 | seqv2 评测 acc 0.7647 首超论文但 TG 坍缩（(0,8) 占 87%）；两阶段法实测失败（acc 0.0001），决策回混合法 |
14	| 08-27 | seqv3-mixed 全量评测：acc 0.7629 达标、mIoU 0.2696 未达 |
15	| 08-28 | 权重核对：本地 ckpt 实为 ONCE 权重（domain gap）；评测数值口径考证（BERTScore 层号 bug 等）；启动 nuScenes 编码器自训 |
16	| 08-29 | 全仓库论文对齐审查；**基线 B0 正式锁定**；编码器退火链与早停监控上线；B1 全流水线启动 |
17	
18	## 基线 B0（seqv3-mixed）锁定
19	
20	20
…（已截断，原文 3915 字符）
```

</details>

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/autodl-tmp/wql/mmb4dl/wiki/Training.md
```

</details>

<details><summary>输出摘要</summary>

```
1	# 模型训练（mllm/）
2	
3	基于 VTimeLLM 架构（Vicuna-7B backbone）的多模态 LLM 训练，输入为预提取的 LiDAR-CLIP 特征 + QA 对话数据。
4	
5	## vtimellm/ 包核心文件
6	
7	| 文件 | 职责 |
8	|------|------|
9	| `train/train.py` | 训练主程序：参数、LoRA 装配、stage3 merge 逻辑、metatoken 梯度掩码、断点续训（校验 trainer_state.json，缺失 fail-fast） |
10	| `train/train_mem.py` | 脚本实际入口（先打 flash-attn monkey-patch 再调 train.py） |
11	| `train/dataset.py` | `LazySupervisedDataset`：特征加载、`_select_features` 帧切片（feat_indices > feat_range > 问题文本帧号）、对话预处理；启动时预校验特征文件存在（fail-fast） |
12	| `train/vtimellm_trainer.py` | 自定义 Trainer，Stage1 只按步落盘 `mm_projector.bin`（规避 ZeRO-3 全量 checkpoint 冲突） |
13	| `model/vtimellm_llama.py` / `vtimellm_arch.py` | 模型与多模态融合（见 [[Architecture]]） |
14	| `model/builder.py` | `load_pretrained_model` 加载链：base → 注册 `<4DLiDAR>`/`<meta>` 特殊 token → stage1 project
…（已截断，原文 5629 字符）
```

</details>

<details><summary>思考过程</summary>

Let me now read the remaining wiki files: LiDAR-CLIP-Encoder.md, Paper-vs-Reproduction.md, Repository-Structure.md, Reproduction-Guide.md, Reproduction-Log.md, Training.md.

</details>

Now let me read the remaining wiki files.


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

All wiki files read. Now let me explore docs/ directory recursively with ls -la, and look at git log for recent commits to identify newly-added analysis documents (2026-09-06 to 09-08).

</details>

All wiki files read. Now let me investigate the docs/ directory and recent git commits.

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls -laR /root/autodl-tmp/wql/mmb4dl/docs/ 2>/dev/null
description: Recursively list docs directory with sizes and dates
```

</details>

<details><summary>输出摘要</summary>

```
/root/autodl-tmp/wql/mmb4dl/docs/:
total 6876
drwxr-xr-x  5 root root     125 Aug 31 00:33 .
drwxr-xr-x 19 root root    4096 Sep  8 12:49 ..
-rw-r--r--  1 root root   70320 Aug 28 13:50 B4DL_复现方案.md
drwxr-xr-x  2 root root    4096 Sep  8 13:13 learn docs
drwxr-xr-x  3 root root      49 Aug  8 11:05 mmb4dl-md
-rw-r--r--  1 root root 6957414 May 28 13:18 mmb4dl.pdf
drwxr-xr-x  2 root root      55 Jul 12 16:36 参考

/root/autodl-tmp/wql/mmb4dl/docs/learn docs:
total 332
drwxr-xr-x 2 root root  4096 Sep  8 13:13 .
drwxr-xr-x 5 root root   125 Aug 31 00:33 ..
-rw-r--r-- 1 root root  8058 Sep  6 10:08 B4DL_B3整场景meta修复mIoU超论文_20260906.md
-rw-r--r-- 1 root root  8827 Aug 29 15:27 B4DL_LiDARCLIP权重核对_20260828.md
-rw-r--r-- 1 root root  5743 Sep  7 21:33 B4DL_METEOR双口径溯源与评测规则_20260907.md
-rw-r--r-- 1 r
…（已截断，原文 3368 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && git log --oneline -40 --date=format:'%Y-%m-%d %H:%M' --pretty=format:'%h %ad %s' 2>/dev/null
description: Show last 40 commits with dates
```

</details>

<details><summary>输出摘要</summary>

```
842f6a2 2026-09-08 13:14 docs: B4DL 训练优化角度分析（除 RL 之外的策略盘点）——以已观测病灶为锚点：TG 输入无显式帧号信号（B2 坍缩/B3 偏早根因）→ A 帧身份锚点三方案（文本锚点/可学习帧嵌入/帧间注意力）；B TG 任务重加权（B4a 过采样的全局推广，TG 占比 8.9%→20%）；C RFT/DPO 奖励函数平价用法（训练集采样+污染红线：测试集 predictions 禁入训练数据）；D 动态 padding/序列 packing 压缩单卡 24-35h 迭代成本；E DoRA/r·α 扫描/解冻 mm_projector（peft 0.13 use_dora 已实测可用）；F NEFTune（transformers 4.47 原生）；G 格式变体/帧号抖动/全帧 meta 增广；行动序列=等 B4a→M1 审计校准→2 个单变量消融→RFT→组合接 GRPO
083f5e7 2026-09-08 12:47 docs: B4DL RL 引进可行性深度分析——实测环境（transformers 4.47.0/torch 2.8.0/peft 0.13.2，CLAUDE.md 旧依赖清单已同步修正；TRL 0.15-0.17 与 4.47 兼容且 PyPI 可达但集成改造量≈自写最小 GRPO）；奖励语义逐任务目录（TG IoU 与论文指标逐位同构、existence/binary 归一化精确匹配即奖励源）；CPU 实测奖励空间（TG 13,124 条 100% 可解析、start≥25 占 14.9%、简单/复杂官方文件级分割现成、特征帧数 39/40/41 三桶）；rollout 批量改造点与实测速率（TG 2.3 条/s vs description 2.45s/条→第一批只做短答案任务）；logprob 三途
…（已截断，原文 9511 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Now I have a comprehensive picture. Let me get the exact list of recently-added docs (from commits 2026-09-06 to 09-08) and figure out which files were added in those commits. Let me also check git log for files changed in specific commits to know which doc files map to which commits, and verify mtime dates.

Actually the ls -laR already shows modification dates:
- B4DL_B3整场景meta修复mIoU超论文_20260906.md (Sep 6 10:08, 8058)
- B4DL_METEOR双口径溯源与评测规则_20260907.md (Sep 7 21:33, 5743)
- B4DL_RL引进可行性深度分析_20260908.md (Sep 8 12:49, 18578)
- B4DL_工作进度总结_20260906.md (Sep 7 21:33, 11536)
- B4DL_训练代码全景解析_20260907.md (Sep 7 21:50, 65912)
- B4DL_训练优化角度分析与策略_20260908.md (Sep 8 13:13, 12207)
- B4DL_训练全流程分步详解_RL引进挂载点_20260907.md (Sep 7 23:23, 24247)
- B4DL_学习计划与知识前提_20260831.md (Sep 8 12:49, 21070)

Also the user mentioned "B4DL 训练全流程分步详解" (2026-09-07), "全量工作进度总结" (2026-09-06), "RL 引进可行性深度分析" (2026-09-08), "训练优化角度分析" (2026-09-08). All located in docs/learn docs/.

Let me check which files were added in each of the recent commits to be thorough, using git show --stat for commits from 09-06 to 09-08. Let me also quickly verify the file paths tracked in git.

Let me check the git log with file names for those specific commits.

</details>

Let me verify which files the recent commits (09-06 to 09-08) added or touched, to precisely map new docs to paths.

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && for c in 842f6a2 083f5e7 99928e7 d49c78e a4db31e f2619a9 ff7293b; do echo "=== $c ==="; git show --stat --format='%h %s' $c | head -30; done
description: Show files changed in recent commits from Sep 6-8
```

</details>

<details><summary>输出摘要</summary>

```
=== 842f6a2 ===
842f6a2 docs: B4DL 训练优化角度分析（除 RL 之外的策略盘点）——以已观测病灶为锚点：TG 输入无显式帧号信号（B2 坍缩/B3 偏早根因）→ A 帧身份锚点三方案（文本锚点/可学习帧嵌入/帧间注意力）；B TG 任务重加权（B4a 过采样的全局推广，TG 占比 8.9%→20%）；C RFT/DPO 奖励函数平价用法（训练集采样+污染红线：测试集 predictions 禁入训练数据）；D 动态 padding/序列 packing 压缩单卡 24-35h 迭代成本；E DoRA/r·α 扫描/解冻 mm_projector（peft 0.13 use_dora 已实测可用）；F NEFTune（transformers 4.47 原生）；G 格式变体/帧号抖动/全帧 meta 增广；行动序列=等 B4a→M1 审计校准→2 个单变量消融→RFT→组合接 GRPO

 ...44\270\216\347\255\226\347\225\245_20260908.md" | 191 +++++++++++++++++++++
 1 file changed, 191 insertions(+)
=== 083f5e7 ===
083f5e7 docs: B4DL RL 引进可行性深度分析——实测环境（transformers 4.47.0/torch 2.8.0/peft 0.13.2，CLAUDE.md 旧依赖清单已同步修正；TRL 0.15-0.17 与 4.47 兼容且 PyPI 可达但集成改造量≈自写最小 GRPO）；奖励语义逐任务目录（TG IoU 与论文指标逐位同构、existence/binary 归一化精确匹配即奖励源）；CPU 实测奖励空间（TG 13,124 条 100% 可解析、start≥25 占 
…（已截断，原文 3248 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Now I have everything I need. Let me also check the untracked file (B4DL_学习计划与知识前提_20260831.md seems dated 20260831 but mtime Sep 8 12:49 — probably modified in commit 083f5e7 along with CLAUDE.md). Actually git show for 083f5e7 says 2 files changed: CLAUDE.md and 分析_20260908.md. Hmm. But B4DL_学习计划与知识前提_20260831.md has mtime Sep 8 12:49. Let me check git log for that file. Actually it's not important for the parent question — but let me quickly check which commit last touched the two files with mtime Sep 8 12:49 (RL 分析 18578 bytes and 学习计划 21070 bytes). Wait ls shows B4DL_学习计划与知识前提_20260831.md mtime Sep 8 12:49 too. Let me check whether it was modified in commit 083f5e7 or another commit. Actually git show --stat 083f5e7 shows only CLAUDE.md + one new doc (322 insertions = RL 分析, 18578 bytes ≈ 322 lines). Hmm 18578 bytes... the file with 191 insertions was the 训练优化分析 (842f6a2), 421 insertions = 训练全流程分步详解 (24247 bytes? hmm 421 lines * ~57 bytes). 1332 insertions = 训练代码全景解析 65912 bytes — yes.

Wait: 学习计划 mtime Sep 8 12:49 — maybe it was touched when merging in commit 083f5e7's parent via 99928e7 merge? Merges don't change mtimes unless checkout updates them. mtime Sep 8 12:49 for B4DL_学习计划... and B4DL_RL引进... — the RL file commit was 083f5e7 at 12:47, and the merge or 842f6a2 at 13:14... hmm 12:49 is odd. Whatever — the git log last-touched info is more reliable. Not essential.

Let me double-check the untracked docs/eval_results type paths mentioned. Also the user asked to check "docs/learn docs/" subdirectories — I did ls -laR.

The user asked to identify, for each over-dated wiki page, specific outdated points (section/quote with stale status). Reference point: repo as of 2026-09-08 has advanced to B4a experiment training completed and being evaluated, B3 mIoU 0.3467 has exceeded paper 0.311, RL introduction feasibility deep analysis done.

Stale elements in wiki pages:

1. Home.md:
   - Section "复现状态速览（基线 B0，2026-08-29 锁定）" — entire page reports B0 as the current status; no mention of B1/B2/B3/B4a; "编码器自训中" — line: "主要差距在编码器（官方未发布权重，自训中）与 time_grounding 定位精度" — encoder self-training has been completed (B2 encoder final + features re-extracted since 09-01..09-06).
   - mIoU 0.2696 vs paper 0.311 ❌ — outdated; B3 = 0.3467 > 0.311.
   - METEOR 0.1729 遗留口径考证项 — was resolved on 09-07 (METEOR dual-backend, NLTK-2005 is paper-cited [2] 口径; 0.3344~0.3371 超论文).
   - accuracy table 0.7629.
   - "基线 B0" claims current.

2. Architecture.md — mostly structural, but mentions "实际复现采用 Stage2+Stage3 混合单 LoRA 或两阶段法" — fine-ish; no version numbers. Metatoken section describes old semantics ("<meta> The metadata of the first frame is '...' and the metadata of the last frame is '...'" — that's the first/last frame absolute-at-start semantics; B3 fixed relative-to-previous semantics 09-02). So Architecture's metatoken description is pre-meta2. Also stage1_features_sample keyed by sample_token. Not much status claim. Slight staleness in metatoken semantics description (论文 Appendix C format shown with first/last frame meta) — but that matches meta format pre-fix. Actually the format string with first/last frame metadata matches paper's textual description. The semantic bug fix doesn't change the template; rendering semantics changed. So not badly outdated. No baseline numbers claimed.

3. Data-Generation.md — describes pipeline, mostly code-level stable. Notes: tools/generate_stage1_caption.py create_metadata.py referencing config issues — status quo. No baseline status claims. Not outdated regarding B versions. It does not mention the datageneration fixes as completed? It mentions in 注意事项 with "Git 历史中曾有硬编码 API key…公开发布前需清理" fine. No B-version claims.

4. FAQ-and-Known-Issues.md — clearly outdated:
   - "论文 vs 复现的已知差异" section: "编码器权重：官方从未发布…自训 nuScenes 编码器（B2）进行中" — B2 encoder done (encoding finished 09-06 logs archived; anneal 3 epoch complete).
   - "METEOR 0.1729 vs 论文 0.275：修正口径后仍偏低，是唯一遗留的口径考证项" — resolved 09-07 via dual-backend: NLTK-2005 = 论文 [2] 口径, B0-B3 0.3344-0.3371 > 0.275.
   - "mIoU 0.2696 vs 论文 0.311：TG 时间定位是当前最大差距项（曾出现…坍缩，seqv3 + answer_frames 后缓解）" — superseded: B3 0.3467 > 0.311.
   - 训练 (mllm/) items: "seqv3 训练的模型评测必须带全参数" & 代际表引用 — seqv3 is no longer "current"; B1/B2/B3 use --whole_scene; B4a ongoing. Also item "stage1...新旧数据配 features" — fine. Actually Inference-and-Evaluation 代际表 says "seqv3（当前）" — outdated since B1/B2/B3.
   - Also earlier 编码器 section: "**旧特征全部作废**：现存 stage1/stage2 特征均为 ONCE 旧编码器产物，编码器定稿后必须全量重提，不可增量混提" — that historical state is resolved; encoder 定稿 and features re-extracted. The statement as written "现存...均为 ONCE 产物" no longer true but presented as historical warning... it says "编码器定稿后必须全量重提" — done. Text is stale in describing current state.

5. Inference-and-Evaluation.md:
   - "seqv3（当前）" in the model-generation table — outdated (current is b4a/whole_scene); missing --whole_scene row; B1-B4a eval commands now use --whole_scene --per_sequence --answer_frames.
   - No baseline numbers other than reference. 评测代际表 missing B1/B2/B3/B4a.

6. Installation.md — version lock table transformers 4.31.0 etc. CLAUDE.md says old dependency list corrected 09-08 (transformers 4.47.0/torch 2.8.0/peft 0.13.2 in current env per RL analysis commit message: "CLAUDE.md 旧依赖清单已同步修正"). Wiki Installation.md lists torch 2.5.1 cu124/transformers 4.31.0/peft 0.4.0/ds 0.16.4/flash-attn 2.7 — these are the "old dependency list" that was 已修正 in CLAUDE.md; the wiki likely still cites old versions. Also flash-attn RTX 5090 remark. So stale relative to 09-08 environment. I should note that Installation locks env versions that the 09-08 commits describe as outdated (transformers 4.47.0 / torch 2.8.0 / peft 0.13.2 实测环境), though the wiki's versions are what B0-B3 might have used. I'll flag it as "likely stale — conflicts with 09-08 update", with caution since wiki doesn't claim a state update date. Hmm, need care: the env on this machine is transformers 4.47/torch 2.8? Actually the commit message says "实测环境（transformers 4.47.0/torch 2.8.0/peft 0.13.2，CLAUDE.md 旧依赖清单已同步修正）". So yes current environment differs. But Installation.md may be describing a conda env "wqlc" that got upgraded. As of 09-08 the CLAUDE.md was updated; wiki not. This is staleness but versions-lock is not "status info" per se. I'll mention briefly.

7. LiDAR-CLIP-Encoder.md:
   - "⚠️ 权重现状：仓库原有的 vit_l_14.ckpt 是 ONCE 权重…此前所有特征均为旧 ONCE 编码器产物，编码器定稿后需全量重提" — outdated: encoder trained (anneal chain complete 09-06, val MSE epoch2 0.0992 < baseline 0.1061), features re-extracted 28130+850 (09-06 archived).
   - It describes run_anneal_chain as the process; no claim about result. No status "训练中" claim? "自训 nuScenes 编码器（B2）进行中" appears in FAQ & Paper-vs-Repro. In this page the phrase "此前所有特征均为旧 ONCE 编码器产物，编码器定稿后需**全量重提**" is a to-do that is now done.

8. Paper-vs-Reproduction.md:
   - Section "结果对比（B0 vs 论文）" mIoU 0.2696 ❌ 差0.041 — outdated (B3 0.3467 exceeds).
   - METEOR 0.1729 偏低 唯一遗留考证项 (#4) — resolved 09-07.
   - 偏差 1: "自训进行中（B2），将触发基线重置" — done; features re-extracted.
   - 偏差 6 stage1 数据量: "B1 修复中（162K 数据已构建）" — completed.
   - 偏差 5 mIoU 差距: "待编码器（B2）与 B1 改进" — resolved via B2/B3.
   - "对比规则（冻结）基线 B0 四位一体" — superseded by later baselines (B1/B2/B3 locked with whole_scene). Well, as a rule page it says "基线 B0 四位一体锁定" — this is历史 but page frames B0 as the baseline.

9. Reproduction-Guide.md:
   - Step 8 验收: "与论文 Table 3 及基线 B0 对比（见 Reproduction-Log 的 B0 数值与 MD5 清单）" — treats B0 as latest baseline.
   - Step 5c inject metatoken with --answer_frames; no --whole_scene mention (B2+).
   - Reference to B0 数值与 MD5 — fine historically but as "how to reproduce", the guide ends at B0 while repo is at B4a.
   - "参考判据：自训新权重 MSE 0.083 vs 旧 ONCE 权重 0.169" — that's the earlier checkpoint validation (08-29); final annealed ckpt val MSE 0.0992 epoch2. Minor.

10. Reproduction-Log.md:
    - Timeline ends 08-29 ("B1 全流水线启动"); B0/B1/B2 naming section; no B3/B4a; states "B2：nuScenes 自训编码器 → 新特征（触发基线重置）" as future plan. B3 mIoU 0.3467 not present; page title "复现记录与基线" current baseline = B0. Also states "evaluation 代码版本 8b85cd2 起零变化" — since then evaluate_model.py got METEOR dual-backend change 09-07 (5db0b05). So MD5/代码版本 four-part lock is stale.
    - docs index lists 16 documents in learn docs; actual directory now contains 21 files (16 original minus plan1? Actually original list had 16 entries incl plan1; now file listing shows 21 files; Reproduction-Log references e.g. "B4DL_待修改清单_20260829.md", no B3/RL docs; plan1* files were deleted 08-31 per commit 19bb843. Also mentions docs top-level includes "B4DL_训练评测报告_20260810.md" which was deleted 08-31.) So Reproduction-Log's document index is outdated in several spots (references deleted files 训练评测报告_20260810, plan1, and lacks newer docs).

11. Repository-Structure.md:
    - Lists docs/ content: "B4DL_复现方案.md, B4DL_训练评测报告_20260810.md, learn docs 16 篇" — 训练评测报告_20260810 deleted (08-31); learn docs count now 21; missing eval_results? Also mllm/scripts listing lacks b1-b4a scripts (run_stage2_full_seqv3_mixed_b1/b2/b3/b4a, run_b1_pipeline/b2/b3/b4a), oversample_tg_highframe.py; missing backups/. Also says requirements.txt 已废弃 fine. And FAQ. Minor staleness but structural listing incomplete (new scripts exist since 09-01..09-06).
    - Also docs/ now includes mmb4dl-md/, learn docs has new files. Also missing "B4DL_全流程..." new docs list.
    - scripts quick-ref table lists b1 variants only up to run_b1_pipeline.sh; b2/b3/b4a missing; stage3 etc. Also mentions "B1 变体（独立 output_dir 保 B0 可比）" fine.

12. Training.md:
    - "混合法（当前基线 B0 采用）" — "当前基线" is B0; outdated: current B3/B4a. stage1 projector 用 95K — B0; B1+ uses 162K.
    - 代际说法 stage2_train_seqv3.json 等 seqv3 references, no whole_scene/整场景 b2/b3 scripts (run_stage2_full_seqv3_mixed_b2.sh etc.), no meta2 (relative-to-previous) description — ego_text semantics update 09-02 not reflected.
    - 其他脚本 list mentions run_b1_pipeline.sh only.
    - claims Stage1 数据（官方 162K 方案）as one plan; B1 completed it.
    - Training.md ends "训练日志统一 tee 到 mllm/training_logs/".

13. Home.md additionally says wiki navigation fine. And "GitHub 风格 wiki 页面 13 files, all modified 2026-08-31 00:33" confirmed. They were added 08-30/08-31 (commit 19bb843 at 00:32 deleted old docs, and wiki added 08-30 14:30 commit c3226bc created 11 pages; maybe additional pages later on 08-31 00:33). Wait: commit c3226bc (08-30 14:30) added wiki with 11 pages: "新增仓库 Wiki（wiki/ 目录，11 页...）" then later commit 9f31aa4 (08-30 19:36) added Reproduction-Guide + Paper-vs-Reproduction + Home rewrite. And commit 19bb843 at 08-31 00:32. Total 13 pages. Fine.

Now, which pages are outdated "明显": Home, FAQ-and-Known-Issues, Paper-vs-Reproduction, Reproduction-Log, Inference-and-Evaluation (generation table "seqv3（当前）"), LiDAR-CLIP-Encoder (weight status paragraph), Training (当前基线 B0 采用 + 缺 B2/B3 代际), Reproduction-Guide (B0 验收步骤 + seqv3 注入命令无 whole_scene), Repository-Structure (docs/learn docs 16 篇 → now 21, deleted files still listed), Architecture (metatoken old semantics; 可选 stage3 and 复现采用 stage2+3混合 — moderately stale), Installation (version lock outdated per 09-08 CLAUDE.md sync; but wiki may describe training machine environment... Actually the note "CLAUDE.md 旧依赖清单已同步修正" implies env changed. Installation page lists old list as authoritative. I'd flag lightly with caution), Data-Generation (no version status — current).

Let me quickly check CLAUDE.md head to confirm the "current status" per CLAUDE.md (updated 09-08 12:49) so I can corroborate B4a status: B4a training completed (3519/3519) evaluation 29805/30145 in progress, B3 mIoU 0.3467. CLAUDE.md was 18847 bytes modified Sep 8 12:49. Let me grep a few key lines to ground claims about "current state as of 09-08" so the report precisely contrasts. Let me also verify wiki file git history: which files modified 08-31 00:33 commit.

Let me check git log for wiki path, and grep CLAUDE.md key lines.

</details>

Let me corroborate the repo's current state from CLAUDE.md (updated 09-08) and check the wiki files' git history.

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && git log --format='%h %ad %s' --date=format:'%m-%d %H:%M' -- wiki/ | head -20
description: Show git history for wiki directory
```

</details>

<details><summary>输出摘要</summary>

```
9f31aa4 08-30 19:36 docs: Wiki 复现定位补充——Home 重写为独立复现仓库框架（非官方实现声明 + B0 复现状态速览表）；新增 Reproduction-Guide（Step0-8 端到端复现指南：每步命令/产出/验收判据）与 Paper-vs-Reproduction（B0 vs 论文结果对比、7 项偏差清单、关键 bug 修复史、冻结对比规则）；_Sidebar 增复现专项分组
643f079 08-30 19:30 docs: Wiki 首页删除致谢部分（用户要求）
c3226bc 08-30 14:30 docs: 新增仓库 Wiki（wiki/ 目录，11 页，GitHub Wiki 就绪）——Home/_Sidebar 导航 + Installation/Architecture/Data-Generation/LiDAR-CLIP-Encoder/Training/Inference-and-Evaluation 分模块页 + Repository-Structure 速查 + Reproduction-Log（B0 基线锁定与文档索引）+ FAQ-and-Known-Issues；内容依据 CLAUDE.md 与 2026-08 复现记录整理
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && grep -n -E 'B4a|B3|B4 |mIoU|0\.3467|0\.311|评测|当前|RL' CLAUDE.md | head -60
description: Grep current status markers in CLAUDE.md
```

</details>

<details><summary>输出摘要</summary>

```
19:- **权威依赖文件**：`mllm/requirements.txt`（不要用根目录的 `requirements.txt`，其版本过新且与 mllm 模块冲突）。评测 METEOR 为**双后端**（`B4DLEvaluator(meteor_backend='dual')` 默认）：`meteor` 主口径 = NLTK-2005（论文引用 [2] Banerjee & Lavie 2005 参数 α0.9/β3.0/γ0.5，B3=0.3366 超论文 0.275），`meteor_pycocoevalcap` 参考口径 = Meteor-1.5 jar（需系统 java：`apt-get install -y --no-install-recommends default-jre-headless`，与 B0-B3 旧表衔接）。详见 `docs/learn docs/B4DL_METEOR双口径溯源与评测规则_20260907.md`
84:评估指标：captioning 任务用 SODA_c / METEOR / CIDEr；grounding 任务用 mIoU / R@n（n=0.3,0.5,0.7）。
184:# B4DL 六任务评测（论文 Table 3 对齐，含 metatoken 注入）
198:#   ⚠️ 仅在模型用 per-sequence 数据训练后才启用，否则会有训练-评测不匹配
203:# 一键脚本（数据划分 + 评测 + 指标）
232:- **API key 配置**（已修复）：`config.py` 现在从环境变量 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 读取，不再硬编码。默认模型改为 `gpt-4o`，可通过 `B4DL_GPT_MODEL` 环境变量覆盖。但 git 历史中仍有旧 key，公
…（已截断，原文 2271 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Let me get more precise current status markers from CLAUDE.md around lines 230-280 and check where B4a status is noted, to correctly contrast.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && grep -n -E 'B0|B1|B2|B3|B4|基线|编码器.*(定稿|重提)|whole_scene|0\.3467|0\.311' CLAUDE.md | head -50
description: Grep baseline version markers in CLAUDE.md
```

</details>

<details><summary>输出摘要</summary>

```
7:B4DL（Benchmark for 4D LiDAR LLM）是 ACM Multimedia 2025 论文的官方 PyTorch 实现，目标是为 4D LiDAR 点云数据的时空理解构建基准数据集，并训练多模态大语言模型回答关于 LiDAR 序列的问题。
19:- **权威依赖文件**：`mllm/requirements.txt`（不要用根目录的 `requirements.txt`，其版本过新且与 mllm 模块冲突）。评测 METEOR 为**双后端**（`B4DLEvaluator(meteor_backend='dual')` 默认）：`meteor` 主口径 = NLTK-2005（论文引用 [2] Banerjee & Lavie 2005 参数 α0.9/β3.0/γ0.5，B3=0.3366 超论文 0.275），`meteor_pycocoevalcap` 参考口径 = Meteor-1.5 jar（需系统 java：`apt-get install -y --no-install-recommends default-jre-headless`，与 B0-B3 旧表衔接）。详见 `docs/learn docs/B4DL_METEOR双口径溯源与评测规则_20260907.md`
21:- **数据集**：B4DL 数据集托管在 [HuggingFace](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL)；nuScenes 需自行下载
129:- **退火链** `run_anneal_chain.sh`（tmux `b4dlanneal`）：等高 LR 续训（`--name lidarclip_nuscenes`，v2 监控 step≥31,550 硬停）退出 → 基线 prob
…（已截断，原文 1981 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

CLAUDE.md head (lines 1-30 or so) may have a status section. Let me look at its top and check whether there's a "当前状态/进度" section near the end referencing B4a. Grep only found B3=0.3366 meteor. Let me grep for "进行中" or "B4a" more specifically and read line 130 vicinity — line 130 in CLAUDE.md still says features are ONCE-old encoder products... wait, but commit ef85115 (09-06 16:07) archived the re-extraction b1 logs ("stage1 sample_token/stage2 特征提取 b1 日志"). Hmm — CLAUDE.md line 130 says "现存全部特征均为 ONCE 旧编码器产物——编码器定稿后必须全量重提". That contradicts the re-extraction on 09-06? Actually ef85115 archived logs of the feature re-extraction (b1). Wait, it says "LiDAR-CLIP anneal 训练（3 epoch 完成）... stage1 sample_token/stage2 特征提取 b1 日志". And a4db31e (09-06 15:53 全量工作进度总结) says "编码器定稿与特征重提" as part of 复现闭环. CLAUDE.md mtime Sep 8 12:49 was updated in 083f5e7 (CLAUDE.md + RL doc). Line 130 still says old features... maybe CLAUDE.md wasn't fully updated (only 依赖清单 synced). Not crucial — CLAUDE.md's line 130 might be stale itself. But wait, maybe the re-extraction happened but features live in non-repo dirs and CLAUDE.md's statement is stale. Anyway, the parent's prompt states as-of 09-08 facts: "B4a 实验训练完成并正在评测，B3 mIoU 0.3467 已超论文 0.311，RL 引进已做可行性深度分析" — I'll use those as ground truth for the report and not rely on CLAUDE.md line 130.

Let me quickly grep CLAUDE.md lines 100-135 for the encoder status and check for B4a mentions elsewhere. Also let me confirm where "B3 mIoU 0.3467" numbers and B4a status appear in docs — but the parent prompt already establishes them. Quick check of eval_results dir? Not needed.

Now for repo docs directory completeness: mention the top-level file evaluation_results.json exists too (not in docs/). The user's question #4 is: check docs/ directory, find recent analysis docs from commits 2026-09-06~09-08 (RL feasibility deep analysis, training optimization analysis, B4DL training full-process step-by-step detail, 全量工作进度总结 etc.), where do they live, file names.

Findings (docs/learn docs/, all committed 09-06 to 09-08):
1. B4DL_RL引进可行性深度分析_20260908.md — 18,578 B, Sep 8 12:49 — commit 083f5e7 (09-08 12:47), docs/learn docs/
2. B4DL_训练优化角度分析与策略_20260908.md — 12,207 B, Sep 8 13:13 — commit 842f6a2 (09-08 13:14), docs/learn docs/
3. B4DL_训练全流程分步详解_RL引进挂载点_20260907.md — 24,247 B, Sep 7 23:23 — commit 99928e7 (09-07 23:23), docs/learn docs/
4. B4DL_工作进度总结_20260906.md — 11,536 B, Sep 7 21:33 (mtime) but commit a4db31e 09-06 15:53 — hmm mtime 21:33 Sep 7. Actually mtime Sep 7 21:33 matches commit 640beba (09-07 00:26) which "进度总结补备份记录" — modified later. The commit a4db31e added it 09-06; 640beba updated. mtime shows Sep 7 21:33? git log shows 640beba at 09-07 00:26. mtime 21:33 vs commit time 00:26 — whatever, the local repo was cloned/checked out at different times. mtime = when last checked out (commit 5db0b05 at 21:34 09-07). Right — mtimes reflect the last checkout commit that touched the file. E.g., files at Sep 8 12:49/13:13 reflect 842f6a2 checkout. Fine.

5. B4DL_训练代码全景解析_20260907.md — 65,912 B — commit d49c78e 09-07 21:50.
6. B4DL_B3整场景meta修复mIoU超论文_20260906.md — 8,058 B — commit ff7293b 09-06 10:08.
7. B4DL_METEOR双口径溯源与评测规则_20260907.md — 5,743 B — commit 5db0b05 (09-07 21:34, part of METEOR dual-backend change).
8. B4DL_学习计划与知识前提_20260831.md — 21,070 B — dated 0831 originally commit ea712dd 08-31 00:21; mtime Sep 8 12:49 (also updated in 083f5e7? git show said only CLAUDE.md + new file. Hmm mtime Sep 8 12:49 though. Maybe merge commit 99928e7 or the checkout of 842f6a2 touched it? mtime is per-file checkout time... Actually mtime only changes when file content changes. Sep 8 12:49 for 学习计划 — let me not worry; maybe it was edited and committed in 99928e7's merge? Merge 99928e7 shows no stat (merge commit). Possibly the merge brought changes from remote that modified it. Not needed for answer.)

Note: 用户说的"全量工作进度总结"→ B4DL_工作进度总结_20260906.md ✓; "RL 引进可行性深度分析" → B4DL_RL引进可行性深度分析_20260908.md ✓; "训练优化角度分析" → B4DL_训练优化角度分析与策略_20260908.md ✓; "B4DL 训练全流程分步详解" → B4DL_训练全流程分步详解_RL引进挂载点_20260907.md ✓.

All in docs/learn docs/ (directory name contains a space). Also backups/ dir at repo root got B3 archive (commit 640beba) — separate.

Also note git commits 09-06~09-08 that changed docs:
- ff7293b 09-06 10:08 (docs: B3 分析归档)
- a4db31e 09-06 15:53 (docs: 全量工作进度总结)
- f2619a9 09-06 15:53 (feat B4a scripts, not docs)
- ef85115 09-06 16:07 (归档 logs — writes into repository? "归档 anneal 链与 B1-B4a 管线日志" — where did it archive? Possibly backups/ or training_logs). Let me check where those logs were archived... could matter for #4 ("最近几次 commit 加入的分析文档"). Commit ef85115's stat not shown. Not necessary.
- 5db0b05 09-07 21:34 (METEOR dual-backend; docs B4DL_METEOR双口径溯源与评测规则_20260907.md)
- d49c78e 09-07 21:50 (训练代码全景解析)
- 99928e7 09-07 23:23 (训练全流程分步详解_RL引进挂载点)
- 083f5e7 09-08 12:47 (RL 引进可行性深度分析 + CLAUDE.md 修正)
- 842f6a2 09-08 13:14 (训练优化角度分析与策略)

Enough. Now also confirm current repo status claims for B4a: parent says B4a training done, evaluation in progress (29805/30145). Commit message 083f5e7 confirms "B4a 训练完成（3519/3519）评测 29805/30145 进行中". Good.

One more check: does docs/learn docs have exactly 21 files? Yes per ls. And Reproduction-Log claims 16 篇. And docs top-level has B4DL_复现方案.md 70,320 B (28 Aug), mmb4dl.pdf 6,957,414 B, mmb4dl-md/, 参考/. There's no B4DL_训练评测报告_20260810.md anymore (deleted 08-31 commit 19bb843: "删除 plan1 系列学习计划/复现指南、0802 时序弱点分析、0810 旧管线训练评测报告"). Also note doc at docs/learn docs/B4DL_训练评测报告...? No. There is an eval_results? no.

Wait — Repository-Structure.md docs listing shows B4DL_训练评测报告_20260810.md under docs/ (still listed) and CLAUDE.md says docs/learn docs has "16 篇" (line?). Wiki states 16. Actual count 21.

Also top-level dirs: backups/ (Sep 7 00:25) contains B3 archive. 

I now have enough for a complete answer. Also wiki pages were created 08-30/08-31, and repo timeline advanced hugely since 08-31. I'll write the report in Chinese, structured by the user's 4 questions.

Structure:
1. Per-file summary (13 files) with status-like info flagged.
2. _Sidebar + Home structure.
3. Outdated pages list with concrete stale points.
4. docs/ listing details + recently added docs mapping to commits/paths.

Let me now compose a careful final response in Chinese. Keep it organized, comprehensive but readable.

</details>

调查完毕。以下为完整报告（全部为只读调查结果）。

# 一、wiki/*.md 逐文件内容概述（13 个文件，均 2026-08-31 00:33 修改，由 2026-08-30~31 的三个 commit 创建：c3226bc、643f079、9f31aa4）

**1. Home.md（4.6KB）— 项目首页**
- 声明本仓库是对论文 B4DL（ACM MM 2025）的"独立复现"（非官方实现），三大核心模块（datageneration/LiDAR-CLIP/mllm）+ 端到端流水线图 + 快速开始命令 + Wiki 导航 + bibtex。
- 状态信息：标题即"复现状态速览（**基线 B0，2026-08-29 锁定**）"；表格：accuracy 0.7629✅、mIoU **0.2696 ❌ Δ-0.041**、BLEU-4 0.0973✅、BERTScore 0.8973✅、ROUGE-L 0.3244、METEOR 0.1729❌"唯一遗留口径考证项"；正文称"编码器（官方未发布权重，自训中）"、数据集产物 100% 对齐。

**2. _Sidebar.md（0.6KB）— 侧边导航**（详见下文第 2 节）

**3. Architecture.md（4.9KB）— 架构总览**
- 纯技术描述：数据流、三阶段训练表、多模态融合机制（`<video>` token 替换）、关键模型组件（VTimeLLMLlamaForCausalLM、mm_projector 单层 Linear(768,4096)、ChatGLM 备选 backbone）、对话模板系统、六种 QA 任务、Metatoken 机制。
- 状态信息：无基线数字；"实际复现采用 Stage2+Stage3 混合单 LoRA 或两阶段法"；Metatoken 例子写的是 `The metadata of the first frame is ... and the metadata of the last frame is ...`（未包含 09-02 修复的 relative-to-previous 语义描述）。

**4. Data-Generation.md（5.7KB）— 数据生成管线**
- 两步管线（GPT-4o：Step1 场景描述 → Step2 六任务 QA）、CLI 参数、tools/ 辅助脚本（create_metadata、build_stage1_from_lidarllm 等）、注意事项。
- 状态信息：无基线/进度声明，内容基本是代码事实；"时间线"类信息只有"论文所用 metadata 已在 HF 发布"。本页是全 wiki 中过时度最低的一页。

**5. FAQ-and-Known-Issues.md（4.1KB）— 已知问题清单**
- 按环境/数据生成/编码器/训练/推理评测/论文差异分类的踩坑清单。
- 状态信息（含大量"进行中/遗留"类现状声明）：环境问题表（METEOR 需 java、peft 0.4.0、transformers 4.31.0）；编码器节"**旧特征全部作废**…现存 stage1/stage2 特征均为 ONCE 旧编码器产物，编码器定稿后必须全量重提"；训练节 seqv3 相关约定；"论文 vs 复现的已知差异"节三条：①"自训 nuScenes 编码器（**B2）进行中**"；②"METEOR 0.1729 vs 论文 0.275：…是**唯一遗留的口径考证项**"；③"mIoU 0.2696 vs 论文 0.311：TG 时间定位是**当前最大差距项**"。

**6. Inference-and-Evaluation.md（6.1KB）— 推理与评测**
- 单条推理、Gradio demo、test_b4dl.py 参数与特征切片优先级、metatoken 注入、B4DLEvaluator 指标表、口径冻结说明（2026-08-29 修正）、输出格式、辅助脚本、评测数据文件、模型代际表。
- 状态信息：指标口径标注"2026-08-29 修正后冻结"；代际表最后一行 "**seqv3（当前）**：`--per_sequence --frame_motion --sequence_metadata --answer_frames`"——即把 seqv3 当作"当前"代际，表内没有 B1/B2/B3/B4a 及 `--whole_scene`。

**7. Installation.md（3.0KB）— 环境搭建**
- conda 环境 wqlc、依赖文件选择、版本锁定表（torch 2.5.1 cu124 / transformers 4.31.0 / peft 0.4.0 / deepspeed 0.16.4 / flash-attn 2.7.0.post2 / decord / clip 0.2.0）、METEOR java、数据与模型准备、硬件参考。
- 状态信息：无基线声明，纯环境事实；但版本锁定与 09-08 实测环境不符（见第 3 节）。

**8. LiDAR-CLIP-Encoder.md（6.1KB）— 编码器**
- SST 架构、训练命令与参数、early_stop/val_mse_probe 探针、退火链脚本、特征提取两脚本、硬性约定、annotations/ 元数据。
- 状态信息：⚠️"**权重现状**：仓库原有的 `vit_l_14.ckpt` 是 ONCE 数据集权重（非 nuScenes）…此前所有特征均为旧 ONCE 编码器产物，**编码器定稿后需全量重提，不可增量混提**"——把"自训未完成"当作现状描述。

**9. Paper-vs-Reproduction.md（4.5KB）— 与论文差异对照**
- B0 vs 论文结果表、一致项 4 条、偏差项 7 条表格（带"状态"列）、修复的关键 bug 史、对比规则。
- 状态信息：结果表 mIoU **0.2696 ❌ 差 0.041（≈3.1× 显著性半宽）**、METEOR 0.1729 口径一致仍偏低；偏差 #1 状态"自训进行中（B2），将触发基线重置"；#5 mIoU 差距"待编码器（B2）与 B1 改进"；#6 stage1 数据量"B1 修复中（162K 数据已构建）"；#7 GPT-4o Score 可选；文首引用审计依据 08-24/08-29 两文档。

**10. Repository-Structure.md（4.7KB)— 目录速查**
- 完整目录树 + mllm/scripts/ 速查表 + 大文件存放约定。
- 状态信息：docs/ 段列出的内容停留在 08 月底（含已删除的 `B4DL_训练评测报告_20260810.md`、"learn docs 16 篇"）；scripts 表只到 b1 变体（`run_stage2_full_seqv3_mixed_b1.sh`、`run_b1_pipeline.sh`），无 b2/b3/b4a 脚本与 `oversample_tg_highframe.py`、无 `backups/` 目录。

**11. Reproduction-Guide.md（5.8KB）— 从零到 B0 的实施指南**
- Step1 数据生成（可跳过）→ Step2 编码器训练（参考判据 val MSE 0.083 vs ONCE 0.169）→ Step3 特征提取 → Step4 stage1（"B0 用的 projector 由 95K 数据训练（B1 升级为 162K）"）→ Step5 seqv3 数据构建 → Step6 混合法单 LoRA（B0 方案）→ Step7 评测（四个对齐参数缺一不可）→ Step8 对比验收（"与论文 Table 3 及基线 B0 对比"、显著性阈值）。
- 状态信息：整页把 B0 当作终点基线；命令均为 seqv3/无 `--whole_scene`；多处"（B1 升级为 162K，见 Reproduction-Log）"式的"待办"表述。

**12. Reproduction-Log.md（5.9KB）— 复现记录与基线**
- 时间线表（2026-07 ~ 08-29 止）、基线 B0 四位一体锁定（6 组件 MD5 + 冻结命令 + 评测代码版本 8b85cd2 + 特征目录）、B0 数值表、显著性阈值、"B0/B1/B2 命名"段（B1=162K projector 重训；B2=自训编码器→新特征，均标注为计划）、文档索引（docs/ 顶层 + "docs/learn docs/ 16 篇"清单）。
- 状态信息：这是状态信息最密集且最过时的一页（见第 3 节）。

**13. Training.md（7.0KB）— 模型训练**
- vtimellm/ 包核心文件表、三阶段标准训练、训练数据格式（seqv3 字段）、metatoken 注入、复现采用的方案：两阶段法（"⚠️ 实测失败…已回退混合法"）vs "**混合法（当前基线 B0 采用）**"、Stage1 官方 162K 方案、其他脚本表。
- 状态信息："当前基线 B0 采用"；projector"用 95K nu-caption 数据重训"；脚本表只有 `run_b1_pipeline.sh`；无整场景（whole_scene）输入与 meta2/relative-to-previous 修复的记载。

# 二、_Sidebar.md 与 Home.md 的串联结构

- **_Sidebar.md**：GitHub Wiki 侧栏，全站导航模板。四组共 11 个链接：
  - 顶部：Home（项目概览与复现状态）、Installation、Architecture；
  - "流水线"组：Data-Generation、LiDAR-CLIP-Encoder、Training、Inference-and-Evaluation；
  - "复现专项"组：Reproduction-Guide、Paper-vs-Reproduction、Reproduction-Log（"复现记录与基线 B0"）；
  - "参考"组：Repository-Structure、FAQ-and-Known-Issues。
- **Home.md**：作为入口页，正文首段用醒目提示串联三大专项页——"端到端步骤见 Reproduction-Guide、与论文逐项差异见 Paper-vs-Reproduction、复现历程与基线锁定见 Reproduction-Log"；随后给出论文/HF 数据集外部链接与三大模块表；"Wiki 导航"一节按同样分组（流水线 6 页 / 复现专项 3 页 / 参考 2 页）逐条列出，等于把 _Sidebar 结构又用列表复述一遍。侧栏与首页结构一致，互相参照；Reproduction-Log 承担"文档索引中枢"角色（列出 docs/ 与 docs/learn docs/ 全部文档名）。

# 三、明显过时的页面及具体过时点

对照基准（截至 2026-09-08）：仓库已推进到 **B4a 实验（TG 高帧段过采样 150,222 条）训练完成（3519/3519），评测 29805/30145 进行中**；**B3 整场景+meta2 方案 mIoU 0.3467 已超论文 0.311**；编码器 anneal 定稿+特征全量重提已完成；METEOR 口径已于 09-07 双后端化闭环（B0-B3 离线补算 0.3344~0.3371 超论文 0.275）；RL 引进可行性深度分析已出（09-08）。13 页全部停留在 08-30/31 的 B0 时代，按过时严重度排序：

**1. Reproduction-Log.md（过时最重）**
- "复现时间线（2026-07 ~ 2026-08）"表止于 08-29"B1 全流水线启动"，无任何 09 月条目。
- "基线 B0（seqv3-mixed）锁定"整节把 B0 当作当前基线，其后已发生 B1/B2/B3 多次基线更迭（B3 为 `--whole_scene` 整场景口径）。
- "B0/B1/B2 命名"段："B2：nuScenes 自训编码器 → 新特征（触发**基线重置**）"——B2 早已完成并重置，且没有 B3/B4a 命名条目。
- B0 数值表：mIoU 0.2696 ❌ Δ-0.041、METEOR 0.1729"唯一遗留口径考证项"——均已被 B3（0.3467）与 09-07 METEOR 双口径结论取代。
- "四位一体"第 3 条"评测代码版本：8b85cd2 起 evaluation/ 零变化"——09-07 commit 5db0b05 已改 evaluate_model.py（METEOR 双后端）。
- "docs/learn docs/（16 篇开发记录）"文档索引：目录实际已有 **21 个文件**，缺 09-06~09-08 新增的 5 篇（B3 复盘、工作进度总结、METEOR 溯源、训练代码全景解析、训练全流程分步详解、RL 分析、训练优化分析——共 7 篇新文档）；且列出的部分旧文档已被删。

**2. Home.md（状态速览即 B0，且自我定位为"当前"）**
- "复现状态速览（基线 B0，2026-08-29 锁定）"表：mIoU 0.2696 ❌ Δ-0.041（应为 B3 0.3467 ✅ 超论文 0.311）、METEOR 0.1729"唯一遗留口径考证项"（09-07 已闭环：NLTK-2005 主口径 0.3344~0.3371 超论文 0.275）、accuracy 0.7629（后续基线未更新）。
- 正文"主要差距在编码器（官方未发布权重，**自训中**）与 time_grounding 定位精度"——编码器已定稿、特征已重提、mIoU 已反超。

**3. FAQ-and-Known-Issues.md**
- "论文 vs 复现的已知差异"节三条全部过时：①"自训 nuScenes 编码器（**B2）进行中**"（已完成）；②"METEOR 0.1729 vs 论文 0.275：…**唯一遗留的口径考证项**"（09-07 双后端闭环，paper 口径实为 NLTK-2005，B0-B3 全超论文）；③"mIoU 0.2696 vs 论文 0.311：TG 时间定位是**当前最大差距项**"（B3 已超论文，当前差距项是高帧段欠覆盖/B4a 方向）。
- "训练（mllm/）"节第 5 条"seqv3 训练的模型评测必须带全参数，见 Inference-and-Evaluation 的代际表"——seqv3 已不是当前代际。
- 编码器节"旧特征全部作废：现存 stage1/stage2 特征均为 ONCE 旧编码器产物，编码器定稿后必须全量重提"——该 to-do 已于 09-06 完成，表述仍像现状。
- 环境表"peft 0.4.0 / transformers 4.31.0"——09-08 实测环境已升 peft 0.13.2（CLAUDE.md 已同步修正，旧锁定说明过期）。

**4. Paper-vs-Reproduction.md**
- "结果对比（B0 vs 论文）"表：mIoU 0.2696 ❌"差 0.041（≈3.1× 显著性半宽）"、METEOR 0.1729"唯一遗留考证项"（同前，均过时）。
- 偏差项 #1 状态列"**自训进行中（B2）**，将触发基线重置"——已完成且已重置两次（B2/B3）。
- #5 "mIoU 差距…**待编码器（B2）与 B1 改进**"——该改进路径已走完并超论文。
- #6 "stage1 数据量…**B1 修复中**（162K 数据已构建）"——B1 早已完成。
- "对比规则（冻结）"节"基线 B0 四位一体锁定"——B0 已被 B1/B2/B3 取代，且 B3 起冻结命令新增 `--whole_scene`。

**5. Inference-and-Evaluation.md**
- 模型代际表末行"**seqv3（当前）**：`--per_sequence --frame_motion --sequence_metadata --answer_frames`"——实际当前代际是 whole_scene 系（B2/B3/B4a 用 `--whole_scene --per_sequence --answer_frames`），表缺 B1-B4a 各行与 `--whole_scene` 参数。
- "指标口径（2026-08-29 修正后冻结）"节 METEOR 行——09-07 起默认改 dual 双后端（主报 NLTK-2005），口径已非"修正后冻结"的单一 jar 口径。
- 评测数据文件/断点等其余内容仍有效。

**6. LiDAR-CLIP-Encoder.md**
- "⚠️ 权重现状"段整体过时："仓库原有 vit_l_14.ckpt 是 ONCE 权重…此前所有特征均为旧 ONCE 编码器产物，**编码器定稿后需全量重提**"——anneal 链已于 09-06 完成 3 epoch（val MSE epoch2 0.0992 < baseline 0.1061），特征已按 b1 全量重提归档。该页没有任何"已定稿"状态的表述。

**7. Training.md**
- "混合法（**当前基线 B0 采用**）…stage1 projector 用 95K nu-caption 数据重训"——当前基线已是 B3（整场景+meta2+162K projector），B0 仅为历史。
- 整个训练方案叙述停在 seqv3（feat_indices 切片、`--answer_frames`），无 09-01 起 B2 的 `--whole_scene` 整场景输入、无 09-02 ego_text relative-to-previous（meta2）修复、无 b2/b3/b4a 训练脚本（`run_stage2_full_seqv3_mixed_b2/b3/b4a.sh`、`run_b2/b3/b4a_pipeline.sh`）与 TG 过采样（`oversample_tg_highframe.py`）记载。

**8. Reproduction-Guide.md**
- 全页止于 B0：Step 4 注释"B0 用的 projector 由 95K 数据训练（**B1 升级为 162K**）"（已完成）；Step 5/7 的注入与评测命令全是 seqv3 四参数、无 `--whole_scene`（B2+ 必需）；Step 8"与…**基线 B0** 对比（见 Reproduction-Log 的 B0 数值与 MD5 清单）"——对比基准应已是 B3。

**9. Repository-Structure.md**
- docs/ 树过时：仍列 `B4DL_训练评测报告_20260810.md`（08-31 commit 19bb843 已删除）、"learn docs/ 16 篇"（现 21 篇，缺全部 09-06~09-08 新文档）；未列 `docs/mmb4dl-md/`。
- scripts 速查表缺 b2/b3/b4a 系列脚本与 `oversample_tg_highframe.py`、缺仓库根 `backups/`（09-07 B3 备份归档）。

**10. Architecture.md / Installation.md（轻度过时）**
- Architecture：Metatoken 节按修复前语义描述（首帧"恒 at the starting position"+前向差分），未反映 09-02 对齐论文的 relative-to-previous（帧 0 fallback）语义修复；无基线/进度声明，其余结构描述仍有效。
- Installation：版本锁定表（transformers 4.31.0 / peft 0.4.0 / torch 2.5.1 cu124）与 09-08 起实测环境不符（peft 0.13.2、transformers 4.47.0、torch 2.8.0，见 commit 083f5e7 "CLAUDE.md 旧依赖清单已同步修正"——同步的是 CLAUDE.md，wiki 此页未同步）。

**11. Data-Generation.md（基本不过时）**：纯代码事实描述（命令、任务格式、已知坑），不含版本状态声明；唯一可留意的是 git 历史 API key 清理提示与"论文所用 metadata 已在 HF 发布"等静态事实，仍有效。

# 四、docs/ 目录内容与近期新增文档

**docs/ 顶层**（4 项 + 子目录）：
- `docs/B4DL_复现方案.md`（70,320 B，08-28 13:50）
- `docs/mmb4dl.pdf`（6,957,414 B，05-28）
- `docs/mmb4dl-md/`（论文 markdown 版：`mmb4dl.md` 74,415 B + `images/` 10 张 jpg，08-02）
- `docs/参考/`（`LiDAR LLM.py` 15,586 B、`记录.md` 1,472 B，07-12）
- 注意：08-31 commit 19bb843 已删除 `docs/learn docs` 中的旧文档及 `docs/evl`、`docs/eval_results` 等目录，故现在顶层只余上述内容。

**docs/learn docs/（21 个 md 文件，其中 09-06~09-08 新增/更新的 7 篇分析文档都在此目录）**：

| 文件（绝对路径前缀 /root/autodl-tmp/wql/mmb4dl/docs/learn docs/） | 大小 | mtime | 对应 commit |
|---|---|---|---|
| `B4DL_RL引进可行性深度分析_20260908.md`（**RL 引进可行性深度分析**） | 18,578 B | 09-08 12:49 | 083f5e7（09-08 12:47） |
| `B4DL_训练优化角度分析与策略_20260908.md`（**训练优化角度分析**） | 12,207 B | 09-08 13:13 | 842f6a2（09-08 13:14） |
| `B4DL_训练全流程分步详解_RL引进挂载点_20260907.md`（**B4DL 训练全流程分步详解**） | 24,247 B | 09-07 23:23 | 99928e7（09-07 23:23） |
| `B4DL_训练代码全景解析_20260907.md` | 65,912 B | 09-07 21:50 | d49c78e（09-07 21:50） |
| `B4DL_METEOR双口径溯源与评测规则_20260907.md` | 5,743 B | 09-07 21:33 | 5db0b05（09-07 21:34） |
| `B4DL_工作进度总结_20260906.md`（**全量工作进度总结**，09-07 补备份记录） | 11,536 B | 09-07 21:33 | a4db31e（09-06 15:53）+ 640beba |
| `B4DL_B3整场景meta修复mIoU超论文_20260906.md` | 8,058 B | 09-06 10:08 | ff7293b（09-06 10:08） |
| 其余 14 个 08 月旧文档（B4DL_基线锁定…、全仓库论文对齐审查…、各训练评测分析等，均已列出在上方 ls 输出中，含 `B4DL_学习计划与知识前提_20260831.md` 21,070 B） | — | 08-24 ~ 08-31 | — |

即用户问到的四份关键分析文档全部位于 **`docs/learn docs/`**（注意目录名含空格）：
1. RL 引进可行性深度分析 → `docs/learn docs/B4DL_RL引进可行性深度分析_20260908.md`
2. 训练优化角度分析 → `docs/learn docs/B4DL_训练优化角度分析与策略_20260908.md`
3. B4DL 训练全流程分步详解 → `docs/learn docs/B4DL_训练全流程分步详解_RL引进挂载点_20260907.md`
4. 全量工作进度总结 → `docs/learn docs/B4DL_工作进度总结_20260906.md`

另外可佐证 09-06~09-08 状态的相关产物：仓库根 `backups/`（09-07 00:25，B3 产物 tar.gz 归档，commit 640beba）；mllm/scripts 下 09-06 新增 `oversample_tg_highframe.py`、`run_b4a_pipeline.sh`、`run_stage2_full_seqv3_mixed_b4a.sh`（commit f2619a9）；CLAUDE.md（09-08 12:49 同步修正依赖清单，commit 083f5e7）。wiki/ 自 08-31 起**零 commit 更新**（git log 显示 wiki 最后变更即 9f31aa4，08-30 19:36），与仓库实际进度已脱节 9 天。

