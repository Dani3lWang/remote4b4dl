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
- **核心依赖（2026-09-08 实测，此前记载的 4.31/0.4.0 已随 sm_120 SDPA 升级过期）**：PyTorch 2.8.0+cu128、transformers 4.47.0、deepspeed 0.16.4、peft 0.13.2、accelerate 1.3.0；flash-attn 未安装（2.x 不支持 RTX 5090 sm_120），训练/推理走 transformers 原生 SDPA
- **权威依赖文件**：`mllm/requirements.txt`（不要用根目录的 `requirements.txt`，其版本过新且与 mllm 模块冲突）。评测 METEOR 为**双后端**（`B4DLEvaluator(meteor_backend='dual')` 默认）：`meteor` 主口径 = NLTK-2005（论文引用 [2] Banerjee & Lavie 2005 参数 α0.9/β3.0/γ0.5，B3=0.3366 超论文 0.275），`meteor_pycocoevalcap` 参考口径 = Meteor-1.5 jar（需系统 java：`apt-get install -y --no-install-recommends default-jre-headless`，与 B0-B3 旧表衔接）。详见 `docs/learn docs/B4DL_METEOR双口径溯源与评测规则_20260907.md`
- 所有 Python 命令必须在 `wqlc` 环境中执行
- **数据集**：B4DL 数据集托管在 [HuggingFace](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL)；nuScenes 需自行下载

```bash
# 完整环境搭建
conda create -n wqlc python=3.10 -y && conda activate wqlc
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r mllm/requirements.txt
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
  │       - stage1_features/（旧，frame_id 键控，95k 数据用，保留给旧 checkpoint）
  │       - stage1_features_sample/（新，sample_token 键控，对齐官方 162K 方案）
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

### LiDAR-CLIP 编码器训练与退火（2026-08-29）

- `train.py` 新增 `--max-epochs`（默认 20）/`--scheduler-max-lr`（默认 1e-3）/`--seed`（默认不设）——默认值完全保持旧行为。
- `val_mse_probe.py`：val/test 场景（编码器未训练的 150 个）确定性子集上的 LiDAR→CLIP MSE 探针，收敛/选型判据；lidar encoder 保持 train 模式（与提取约定一致），固定 seed + shuffle=False 保证跨 ckpt 可比。`loader.build_loader` 新增 `val_mode/val_max_scenes` 透传（默认关闭）。
- **退火链** `run_anneal_chain.sh`（tmux `b4dlanneal`）：等高 LR 续训（`--name lidarclip_nuscenes`，v2 监控 step≥31,550 硬停）退出 → 基线 probe → 从 last.ckpt `--load-only-model` 短程退火（3 epoch、OneCycle 1e-4→0、seed 0、ckpt 存 `ckpt_anneal/`）→ 逐 ckpt probe。结果 `logs/val_mse_probe.csv`；每小时 cron 只读报告进度。
- 现存全部特征（stage1_features_sample 28,130 / stage2_features 850）均为 ONCE 旧编码器产物——编码器定稿后必须**全量重提**，不可增量混提。


**Stage1 官方对齐（162K 方案）特征**：`extract_pc_features_sample_token.py` 输出 `{sample_token}.npy`（键控与官方 stage1 数据一致，点云处理同 `with_path` loader）：

```bash
cd encoders/lidarclip
conda run -n wqlc python extract_pc_features_sample_token.py \
    --checkpoint ./lidarclip/checkpoint/vit_l_14.ckpt \
    --scene-metadata /root/autodl-tmp/wql/mmb4dl/dataset/nuScenes-B4DL/metadata/scene_metadata.json \
    --sample-json /root/autodl-tmp/Datasets/nuScenes/v1.0-trainval/sample.json \
    --data-path /root/autodl-tmp/Datasets/nuScenes \
    --save-dir ./b4dl/stage1_features_sample
```

只提取 `scene_metadata` 中 `split == 'train'`（700 scenes）的全部关键帧（28,130 帧）。配对的 stage1 数据由 `datageneration/tools/build_stage1_from_lidarllm.py` 生成（LiDAR-LLM 全量 161,845 条，scene_id=sample_token，与官方逻辑一致）。

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

# 加 --per_sequence 启用 per-sequence 模式（论文 Appendix C 对齐）：
#   - 特征切片到 QA 的包含序列（item['feat_range'] > --sequence_metadata + 问题帧号）
#   - metatoken：--frame_motion 时按 QA 引用帧渲染真实描述（与训练注入同源）
#   ⚠️ 仅在模型用 per-sequence 数据训练后才启用，否则会有训练-评测不匹配
conda run -n wqlc python evaluation/test_b4dl.py ... --per_sequence \
    --frame_motion ./b4dl_dataset/ego_frame_motion.json \
    --sequence_metadata ../encoders/lidarclip/annotations/sequence_metadata.json

# 官方训练数据已按 scene 划分；直接使用上面的 evaluation/test_b4dl.py 命令评测指定 checkpoint。
```

### TG 失败模式诊断（零训练成本，只看 predictions.json）

```bash
cd mllm
# 两次评测的分桶失败模式对比（GT start 分桶 / 中心偏移 / 众数坍缩）
python evaluation/analyze_tg_regression.py eval_results/migration_b3 eval_results/framepos3ep --labels B3 FP
# 解码端校正探测：把预测中心做移位/仿射/分位数映射后还能涨多少
python evaluation/tg_decode_probe.py eval_results/migration_b3 eval_results/framepos3ep \
    --labels B3 FP --prior eval_results/tg_prior_train.json
# 生成训练集 TG 先验（centers/lengths/starts/ends），供上面 --prior 做无泄漏参照
python evaluation/tg_prior_from_train.py \
    --data b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json \
    --out  eval_results/tg_prior_train.json
```

三个脚本都复用 `evaluate_model` 的区间正则与闭区间 IoU，所以输出的 mIoU 与 `metrics.json` 同口径、可逐位对上。

2026-09-13 结论：任何**不用测试集标签**的解码端校正（分位数映射 / 均值方差匹配 / 中位长度 / 全局移位）对 B3 与 framepos3ep 都是**负的**，中心展开系数扫描的峰值恰好在 k=1.0；用训练集先验挑出的单一常数区间 `[06-26]` 在测试集上就有 mIoU **0.2998**。因此 TG 的任何改进都必须对着这条先验地板报告，提升只能来自目标函数侧（见 `docs/` 与 RL 计划），而不是采样/后处理。

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
- **peft 版本**（2026-09 更新）：环境已升至 peft 0.13.2（旧文档称锁 0.4.0 已过期），Stage3 的 `merge_and_unload()` + 重新加 LoRA 模式在新版下验证可用（B4a 全链在跑为证）
- **Flash Attention**：flash-attn 未安装（2.x 不支持 sm_120）；`train_mem.py` 尝试 monkey-patch 注入、ImportError 时静默回退 transformers 原生 SDPA（需 transformers ≥4.45，现环境 4.47.0 满足）
- **DeepSpeed no_sync**：`train.py` 中对 `DeepSpeedEngine.no_sync` 做了 monkey-patch，原版在 ZeRO-3 梯度分区下会崩溃
- **LiDAR-CLIP checkpoint**：PyTorch Lightning 格式，含非 tensor 对象（scheduler 等），加载需 `weights_only=False`，且用 `strict=False`（忽略旧版 bbox_head 的 key）
- **训练数据集容错**：`LazySupervisedDataset` 在特征文件缺失时返回随机其他样本（`random.choice(self)`），会导致静默的数据丢失，检查日志中的异常打印
- **特征文件命名约定**：Stage2/3 训练时以 `scene_id` 查找 `{feat_folder}/{scene_id}.npy`，需确保 LiDAR-CLIP 提取时 stage2-save-dir 下的文件名与数据 JSON 中的 `scene_id` 一致
- **Per-sequence 特征切片**（2026-08-12 引入；2026-08-24 v2 对齐审计后升级）：论文 Appendix C 指出 metatoken 应描述"QA pair 中引用的首末帧"的 ego 状态，且输入 $S_L$ 是序列而非整个 scene。三层实现：
  - `generate_ego_metadata.py`：per-scene（`{scene_id}`）+ per-sequence（`{scene_id}_{first}_{last}`）文本条目，以及 `--frame_motion` 输出的**逐帧运动表** `ego_frame_motion.json`（每 scene 全帧 `{x,y,z,yaw,spd/yaw/acc_next/prev}`）
  - `scripts/ego_text.py`（2026-08-24 新增）：metatoken 文本渲染单一来源，`render_meta_texts()` 可对**任意** (first,last) 帧对渲染真实描述（单帧引用输出真实邻帧运动）；generate_ego_metadata / inject_metatoken / test_b4dl 三方共用
  - `inject_metatoken.py --frame_motion + --sequence_metadata`（v2）：对 QA 引用帧渲染真实 metatoken（79,975 条有帧号 QA 零 per-scene 回退），并写入 `item["feat_indices"] = [i0,i1,...]`（序列**精确采样帧**，如 `[0,2,4,6,8]`）与 `item["feat_range"] = [s,e]`——训练 `dataset.py` 与评测 `test_b4dl.py` 优先按 feat_indices 选帧
  - 训练数据：**两阶段（论文/官方方法，2026-08-25 起）**：`stage2_train_seqv3.json`（简单任务 68,695 条，含 TG 13,124 全部 GT 归属）→ merge → `stage3_train_seqv3.json`（复杂任务 79,576 条）。合并版 `stage2_full_train_seqv3_148k.json` 仅作对照
  - **帧位置实验（roadmap A.2）**：零初始化可学习的**绝对场景帧位置 embedding**（`nn.Embedding(frame_position_max=64, hidden_size)`）在 mm_projector 之后按 `feat_indices` 加到视觉 token 上，靠 `non_lora_trainables.bin` 落盘；开关 `--use_frame_position_embedding True`（默认 False），B3 脚本省略的 `lora_dropout/train_meta_embed/optim` 等与 train.py 默认值一致，故该实验对 B3 是单变量改动。CPU 侧自检：`python mllm/scripts/verify_frame_position.py`
  - **已弃用的 2ep 方案**：`scripts/run_stage2_full_seqv3_mixed_paper2ep.sh`（无帧位置基线）+ `..._framepos2ep.sh` 需各训 2ep（~24h）。2026-09-12 改为直接复用已训完的 B3（整场景+meta2, 3ep, acc 0.7526 / mIoU 0.3467）作对照，只跑 `framepos3ep` 一组，省 ~12h；两个 2ep 脚本与 `mllm/docs/train_2ep_frame_position.md` 保留备查
  - **迁移验收 → 帧位置链** `scripts/run_framepos3ep_migration_chain.sh`（tmux 会话 `b4dl3ep` 内跑，驱动日志 `mllm/training_logs/chain_framepos3ep_driver.log`）：阶段0 若 `eval_results/migration_b3/metrics.json` 已存在则复用，否则检测到在飞的 B3 评测就轮询等待（最长 9h，每 10 分钟打一次心跳），都没有时自己跑一次同口径评测，须复现 mIoU 0.3467（|Δ|>0.013 判定搬运/口径漂移，链停止）→ 阶段1 `scripts/smoke_framepos_train.sh`（2 步训练 + 续跑 2 步，验证帧位置开关进 config.json、权重进 non_lora_trainables.bin，本机训练从未跑过故必须先冒烟）→ 阶段2 `run_stage2_full_seqv3_mixed_framepos3ep.sh`（`trainer_state.epoch>=2.99` 判据 + 3 次断点续训重试 + 磁盘余量门槛）后同口径评测并回报显著性，成功后删除该实验自身中间 `checkpoint-*`（新机数据盘仅 19G，`KEEP_CKPT=1` 关闭清理、`SMOKE_KEEP=1` 保留冒烟产物）
  - **seqv3 --answer_frames（2026-08-25）**：发布的 benchmark 丢失每条 QA 的序列归属字段，TG 问题文本无帧号导致此前只能用整 scene 输入（模型坍缩为输出 (0,8)，占 87%）。现双侧回退解析 GT 答案帧范围恢复包含序列：训练侧 `inject_metatoken.py --answer_frames`（仅对 `task=='time_grounding'` 生效，build_stage2_full_train.py 已打 TG 标签 13,124 条），评测侧 `test_b4dl.py --answer_frames`（同样仅 TG）。TG feat_indices 覆盖 100%、GT 范围 100% ⊂ feat_range、训练/评测两侧 2783/2783 一致。⚠️ 评测侧用 GT 恢复归属是"还原论文原始评测设置"（oracle 输入选择），报告中须声明
  - **数据划分（2026-08-24 对齐）**：HF 发布的 `dataset/nuScenes-B4DL/dataset/train/{stage2,stage3}.json`（148,271 条）**本身就是论文官方训练集**（700/150 划分的 train 部分，与官方 test_qa.json 的 150 scenes 零重叠），无需再划分。旧的 80/10/10 自创划分会把 850 scenes 混切、与官方测试集冲突，已移除。一键重建：`python scripts/build_stage2_full_train.py --input_dir ../dataset/nuScenes-B4DL/dataset/train --output_dir ./b4dl_dataset` → 再跑 inject_metatoken 注入（seqv3 需 `--frame_motion --sequence_metadata --answer_frames`）
  - ⚠️ 训练和评测必须用相同代际的格式：seqv2 模型评测时加 `--per_sequence --frame_motion --sequence_metadata`；seq 模型只加 `--per_sequence`；旧模型都不加。time_grounding 类无帧号问题两侧都用全 scene 特征（benchmark 未提供序列归属，属数据级限制）

## 规划与拆解文档

- `docs/learn docs/B4DL_训练全流程分步详解_RL引进挂载点_20260907.md`：SFT 管线 A–E 阶段拆解 + RL（M1–M4）挂载点底稿
- `docs/learn docs/B4DL_分割模块移植方案_AB路线_20260909.md`：动态目标分类/分割模块移植（Reason3D/MORE3D 方法），实验编号 **B5 系列**（B5-P0/P1 门控 + B5a 感知前端 + B5b token 解码头）；时序硬约束：B5b-4 输出接口冻结先于 RL M2

## Git 提交规范

- 每次代码修改后自动 commit
- 使用中文提交信息
- 只 `git add` 具体修改的文件，不用 `git add -A`
