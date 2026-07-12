# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

B4DL（Benchmark for 4D LiDAR LLM）是 ACM Multimedia 2025 论文的官方 PyTorch 实现，目标是为 4D LiDAR 点云数据的时空理解构建基准数据集，并训练多模态大语言模型回答关于 LiDAR 序列的问题。

三大核心模块：

1. **数据生成管线**（`datageneration/`）— 调用 GPT-4o API，从 nuScenes 相机图像生成 LiDAR 场景描述，再转化为问答对
2. **LiDAR-CLIP 编码器**（`encoders/lidarclip/`）— 基于 SST backbone，将点云序列编码为 CLIP 对齐的 768 维特征向量
3. **VTimeLLM**（`mllm/`）— 基于 Vicuna-7B + LoRA + DeepSpeed ZeRO-3，训练多模态 LLM 进行时空问答

## 环境

- **Conda 环境**：`wqlc`，Python 3.10
- **核心依赖**：PyTorch 2.5.1 CUDA 12.4、transformers 4.31.0、deepspeed 0.16.4、peft 0.4.0、flash-attn 2.7.0
- **权威依赖文件**：`requirements_sum/requirements_b4dl.txt`（不要用根目录的 `requirements.txt`，其版本过新且与 mllm 模块冲突）
- 所有 Python 命令必须在 `wqlc` 环境中执行
- **RTX 5090 / CUDA 13 环境**：详见 `requirements_sum/RTX5090_CUDA13_ENV_SETUP.md`，需特殊处理 flash-attn 和 CUDA 版本兼容性
- **数据集**：B4DL 数据集托管在 [HuggingFace](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL)；nuScenes 需自行下载

```bash
# 完整环境搭建
conda create -n wqlc python=3.10 -y && conda activate wqlc
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_sum/requirements_b4dl.txt
```

## 整体架构与数据流

```
nuScenes 数据集（相机图像 + LiDAR 点云）
  │
  ├─ [datageneration/] 相机图像 → GPT-4o → 场景描述 JSON → QA 数据集 JSON
  │     数据格式：{"from": "human/gpt", "value": "..."}
  │
  ├─ [encoders/lidarclip/] LiDAR 点云 → SST编码器 → CLIP 特征 (.npy)
  │     Stage1 特征：每帧独立 .npy，shape (1, 768)
  │     Stage2 特征：每场景拼接 .npy，shape (N_frames, 768)
  │
  └─ [mllm/] 预提取特征 + QA 数据 → VTimeLLM → 训练 / 推理 / 评估
```

### 三阶段训练流程

| 阶段 | 目标 | 关键配置 |
|------|------|----------|
| **Stage 1** | 对齐 mm_projector | 只训练 `nn.Linear(768, 4096)`，LLM 冻结。`--tune_mm_mlp_adapter True`，`--version plain` |
| **Stage 2** | LoRA 微调 LLM | 加载 Stage1 的 mm_projector 并冻结，对全部 Linear 层加 LoRA。`--lora_enable True`，`--freeze_mm_mlp_adapter True` |
| **Stage 3**（可选）| 二次 LoRA | 先 merge Stage2 的 LoRA 权重作为基座，再加新 LoRA，解冻 mm_projector |

### 多模态融合机制

训练数据中的 `<video>` token 在前向传播时被替换为投影后的视觉特征。核心逻辑在 `vtimellm_arch.py` 的 `prepare_inputs_labels_for_multimodal()`：找到 `<video>` token 位置 → 将 CLIP 特征经 `mm_projector` 投影后插入对应 embedding 位置 → labels 对应位置设为 `IGNORE_INDEX` → 填充/截断到统一长度。

### 关键模型组件

- **`VTimeLLMLlamaForCausalLM`**（`vtimellm_llama.py`）：继承 `LlamaForCausalLM` + `VTimeLLMMetaForCausalLM`，重写 `forward()` 在调用父类前先做多模态融合
- **`VTimeLLMMetaForCausalLM`**（`vtimellm_arch.py`）：多模态融合的核心，`prepare_inputs_labels_for_multimodal()` 实现图像 token 替换、序列填充、attention mask 生成
- **`mm_projector`**：单层 `nn.Linear(768, config.hidden_size)`，将 CLIP ViT-L/14 输出映射到 LLM 的 hidden space（Vicuna-7B 为 4096 维）
- **`VTimeLLMTrainer`**（`vtimellm_trainer.py`）：继承 HF Trainer，Stage1 时只保存 mm_projector 权重（非全量 checkpoint），Stage2 走父类逻辑
- 同时支持 **LLaMA**（Vicuna）和 **ChatGLM** 两种 backbone，通过模型名是否含 "chatglm" 自动选择

### 对话系统

`conversation.py` 定义了 Vicuna 和 ChatGLM 的对话模板，通过 `conv_templates` dict 管理：
- **`plain`**：仅图像描述（Stage1 用），无角色标记，`<video>\n` + 描述文本
- **`v1`** / **`vicuna_v1`**：标准 Vicuna 对话格式（Stage2 用），`USER: ... ASSISTANT: ...</s>`
- **`llama_2`**：LLaMA2 格式，`[INST] ... [/INST]`

训练数据中 `version` 参数选择模板，不同模板的 separator style 决定 `preprocess()` 函数的分支（`preprocess_v1` / `preprocess_llama_2` / `preprocess_plain`）。

### 五种 QA 任务类型

`existence`、`binary`、`time_grounding`（简单任务）和 `description`、`temporal_understanding`、`comprehensive`（复杂任务），在 `datageneration/config.py` 中定义。

每个任务类型对应 `prompts.py` 中的专用 prompt 方法和 `generate_dataset.py` 中的独立生成方法。数据生成流程：
1. **Step 1**（`generate_description.py`）：将 nuScenes 6 个相机视图（FRONT/FRONT_LEFT/FRONT_RIGHT/BACK/BACK_LEFT/BACK_RIGHT）分为前/后两组，每组 3 视图的图像编码为 base64 → 发送给 GPT → 生成前/后场景描述 → 存为 JSON
2. **Step 2**（`generate_dataset.py`）：读取 Step 1 的场景描述 JSON → 根据 `--task` 选择对应 prompt → 发送给 GPT → 解析 `Q:...A:...` 格式的回复 → 转为 Vicuna 对话格式 → 存为 JSON

评估指标：captioning 任务用 SODA_c / METEOR / CIDEr；grounding 任务用 mIoU / R@n（n=0.3,0.5,0.7）。

## 常用命令

### 数据生成（调用 GPT-4o API）

```bash
cd datageneration

# Step 1: 从 nuScenes 相机图生成场景描述
conda run -n wqlc python generate_description.py \
    --start_index 0 --end_index 100 \
    --api_key YOUR_API_KEY \
    --nuscenes_root /path/to/nuScenes \
    --dataroot ./data

# Step 2: 描述转 QA 数据集（每次指定一种任务类型）
conda run -n wqlc python generate_dataset.py \
    --start_index 0 --end_index 100 \
    --api_key YOUR_API_KEY \
    --task existence \
    --dataroot ./data
```

start_index / end_index 必须是 `SAVE_TERM`（10）的倍数。

### LiDAR-CLIP 特征提取

```bash
cd encoders/lidarclip
conda run -n wqlc python extract_pc_features.py \
    --checkpoint /path/to/lidarclip.ckpt \
    --scene-json-path ./annotations/scene_metadata.json \
    --frame-json-path ./annotations/sequence_metadata.json \
    --data-path /path/to/nuScenes \
    --stage1-save-dir ./b4dl/stage1_features/ \
    --stage2-save-dir ./b4dl/stage2_features/
```

checkpoint 是 PyTorch Lightning 格式，加载时需 `weights_only=False` 且 `strict=False`（旧版含 bbox_head 的 key 会忽略）。

### 训练

```bash
cd mllm

# 一键跑 Stage1 + Stage2：
bash run_stages.sh \
    --s1_data ./b4dl_dataset/stage1_lidarllm_mm.json \
    --s1_feat ../encoders/lidarclip/b4dl/stage1_features \
    --s2_data ./b4dl_dataset/stage2.json \
    --s2_feat ../encoders/lidarclip/b4dl/stage2_features \
    --model_name_or_path ./base_model/vicuna-v1-5-7b

# 单独跑某一阶段（支持额外的命令行覆盖）：
bash scripts/stage1.sh [额外参数]
bash scripts/stage2.sh [额外参数]
bash scripts/stage3.sh [额外参数]

# 例：自定义输出目录和 epoch
bash scripts/stage2.sh --output_dir ./checkpoints/custom --num_train_epochs 3
```

训练使用 DeepSpeed ZeRO-3（`scripts/zero3.json`）。`train.py` 中有 monkey-patch 修复 ZeRO-3 梯度分区与 accelerate `no_sync` 的兼容性问题。

### 推理与评估

```bash
cd mllm

# 单条推理
conda run -n wqlc python vtimellm/inference.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --clip_path checkpoints/clip/ViT-L-14.pt \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    --video_path /path/to/video.mp4

# 批量评估（captioning + grounding）
conda run -n wqlc python vtimellm/eval/eval.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --data_path vtimellm/eval/data_example.json \
    --feat_folder /path/to/features \
    --task all \
    --log_path vtimellm/eval/log/eval_log.txt

# 计算指标（METEOR、CIDEr、SODA_c、mIoU、R@n）
conda run -n wqlc python vtimellm/eval/metric.py \
    --log_path vtimellm/eval/log/eval_log.txt \
    --task all --data_path vtimellm/eval/data_example.json
```

### Gradio Web Demo

```bash
cd mllm
conda run -n wqlc python vtimellm/demo_gradio.py \
    --model_base /绝对路径/vicuna-7b-v1.5 \
    --clip_path ViT-L/14 \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2
```

⚠️ `demo_gradio.py` 中 `root_dir = os.path.join(os.getcwd(), "..")` 存在路径解析 bug，**必须显式传入所有绝对路径**，不要依赖默认值。

## 代码与配置约定

- 配置管理：`dataclass` 定义默认值 + `argparse` 命令行覆盖
- Python 文件：snake_case；类名：PascalCase
- 数据格式：对话统一为 `{"from": "human/gpt", "value": "..."}`，JSON 存储
- OpenAI API 调用：`client.chat.completions.create()`
- 特征文件：`.npy`，dtype float16
- 模型权重：预训练模型放 `./base_model/`，checkpoint 放 `./checkpoints/`
- **文档存放**：后续生成的所有说明文档（.md）统一保存到 `learn docs/` 目录下

## 已知问题与注意事项

- **API key 配置**（已修复）：`config.py` 现在从环境变量 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 读取，不再硬编码。默认模型改为 `gpt-4o`，可通过 `B4DL_GPT_MODEL` 环境变量覆盖。但 git 历史中仍有旧 key，公开发布前需清理历史
- **`generate_description.py` break bug**（已修复）：移除了第 142 行的 `break` 语句，现在可以正常处理多批数据
- **Gradio 路径 bug**（已修复）：`demo_gradio.py` 改为使用脚本自身目录计算路径（`_script_dir`），不再依赖 CWD
- **index 约束**：`start_index` / `end_index` 必须是 `SAVE_TERM`(10) 的倍数，否则脚本直接退出
- **数据生成两步使用不同的数据加载方式**：`generate_description.py` 用 `ReadJson.readFiles()` 解析 metadata 并加载 nuScenes 图像；`generate_dataset.py` 直接读取已生成的描述 JSON，不再访问 nuScenes
- **peft 版本锁定**：必须用 peft 0.4.0（旧式 API），Stage3 的 `merge_and_unload()` + 重新加 LoRA 的模式依赖此版本
- **Flash Attention**：通过 monkey-patch（`llama_flash_attn_monkey_patch.py`）注入，锁定 transformers 4.31.0
- **DeepSpeed no_sync**：`train.py` 中对 `DeepSpeedEngine.no_sync` 做了 monkey-patch，原版在 ZeRO-3 梯度分区下会崩溃
- **LiDAR-CLIP checkpoint**：PyTorch Lightning 格式，含非 tensor 对象（scheduler 等），加载需 `weights_only=False`，且用 `strict=False`（忽略旧版 bbox_head 的 key）
- **训练数据集容错**：`LazySupervisedDataset` 在特征文件缺失时返回随机其他样本（`random.choice(self)`），会导致静默的数据丢失，检查日志中的异常打印
- **特征文件命名约定**：Stage2/3 训练时以 `scene_id` 查找 `{feat_folder}/{scene_id}.npy`，需确保 LiDAR-CLIP 提取时 stage2-save-dir 下的文件名与数据 JSON 中的 `scene_id` 一致

## Git 提交规范

- 每次代码修改后自动 commit
- 使用中文提交信息
- 只 `git add` 具体修改的文件，不用 `git add -A`
