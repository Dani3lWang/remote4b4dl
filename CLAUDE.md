# CLAUDE.md

本文件记录当前仓库的有效开发约定。当前主线分支是 `upcore`，用于更新 B4DL 主模型及其完整实验链路；`main` 的基础实现、`4D_seg` 的分割研究和 `rf-grpo` 的强化学习保留各自历史定位。

## upcore 当前方向

- 优先更新 B4DL 的输入与数据质量、模型接入、训练稳定性、推理恢复和六任务评测，验收以回答质量、LiDAR 证据利用及复现结果为依据。
- 当前核心是 `mllm/b4dl_core/` 与 `mllm/vtimellm/`。已有时序几何适配器作为待完整验收的方案，时空理解和时间定位仍属于 B4DL 的任务范围。
- MaskPLS、Grounded4D、AnswerSeg 与伪掩码作为历史研究及可选扩展维护，不将增强时空分割能力作为当前主线目标。
- 开发顺序、模块职责与验收条件见 [upcore 开发方向](docs/upcore/README.md)；本轮审查见 [2026-10-27 代码审查](docs/research/upcore代码审查_20261027.md)。历史实验配置及 schema 不随分支名称变化。

## 项目概述

B4DL（Benchmark for 4D LiDAR LLM）是 ACM Multimedia 2025 论文的官方 PyTorch 实现，包含：

1. `datageneration/`：由 nuScenes 相机图像生成场景描述和问答数据。
2. `encoders/lidarclip/`：用 SST backbone 将 LiDAR 序列编码为 768 维特征。
3. `mllm/`：VTimeLLM 训练、推理和评测代码。

## 环境

- Conda 环境：`wqlc`，Python 3.10；所有 Python 命令必须在该环境执行。
- `mllm/requirements.txt` 已对齐本体 CPU 检查栈，CPU/CUDA wheel 与演示、指标、旧训练器依赖分层；运行 `python -m b4dl_core.runtime` 做版本/API 预检。安装步骤和验证边界见 [环境与恢复说明](mllm/docs/upcore_runtime.md)。
- 当前本地检查栈：PyTorch 2.8.0+cpu、transformers 4.47.0、peft 0.13.2、accelerate 1.15.0、numpy 1.26.4、sentencepiece 0.2.2。历史 CUDA 环境为 PyTorch 2.8.0+cu128、deepspeed 0.16.4、accelerate 1.3.0 等；本次未重验或升级该环境。
- flash-attn 未安装，训练和推理使用 transformers 原生 SDPA。

```bash
conda activate wqlc
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
```

以上用于核对已有环境；新建环境需要按实际 CPU/CUDA 用途锁定并验证依赖。

## 保留的 B3 两轮复现配方

B3 是 `main` 上唯一维护的实验版本：

- 数据：`mllm/b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json`
- 特征：`encoders/lidarclip/b4dl/stage2_features/`
- 输入：`--whole_scene True`
- 元数据：relative-to-previous 语义的 meta2
- LoRA：r=64，alpha=128
- 训练：统一为 **2 epochs**，lr=1e-4，batch 8，gradient accumulation 16，ZeRO-3
- 训练入口：`mllm/scripts/run_b3.sh`
- 单独评测：`mllm/run_baseline_eval.sh`

历史 3-epoch B3 checkpoint 的 mIoU 0.3467 / accuracy 0.7526 只作为历史结果；不能将它当作新 2-epoch 配方的已验证结果。

```bash
cd mllm
bash scripts/run_b3.sh
```

流水线以最终 adapter 和 `trainer_state.json` 中 `epoch >= 1.99` 共同判定完成，并支持从最新 checkpoint 恢复。

## 官方通用入口

保留 `stage1.sh`、`stage2.sh`、`stage3.sh` 及 ChatGLM 对应入口，用于官方三阶段能力和自定义实验。原 B3 复现沿用既有 B3 流水线；upcore 的模型更新从 `b4dl_core` 入口和独立实验配置开展。不再新增 B0/B1/B2/B4a 或 frame-position 变体入口。

## 数据与特征

```bash
cd datageneration
conda run -n wqlc python generate_description.py \
  --start_index 0 --end_index 100 \
  --api_key YOUR_API_KEY \
  --nuscenes_root /path/to/nuScenes \
  --dataroot ./data

conda run -n wqlc python generate_dataset.py \
  --start_index 0 --end_index 100 \
  --api_key YOUR_API_KEY \
  --task existence \
  --dataroot ./data
```

`start_index` 和 `end_index` 必须是 `SAVE_TERM`（10）的倍数。禁止提交 API 密钥。

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

checkpoint 是 PyTorch Lightning 格式，加载时使用 `weights_only=False` 和 `strict=False`。不同编码器生成的特征不可混用。

## B3 历史对照评测口径

B3 使用 `--whole_scene --per_sequence --answer_frames`。其中 `--answer_frames` 依据答案恢复序列归属，属于 oracle 输入选择，报告结果时必须声明。

```bash
cd mllm
bash run_baseline_eval.sh
```

METEOR 默认使用双后端；与论文对比采用 NLTK-2005 主口径。

## 关键实现

- `mllm/b4dl_core/b3.py`、`integration.py`、`temporal.py`：主模型接入、时序适配、数值保护及扩展 checkpoint。
- `mllm/b4dl_core/inputs.py`、`evaluation.py`、`six_task.py`：输入干预、完整分母几何诊断和原六任务回归。
- `mllm/scripts/inspect_core_progress.py`：只读核验训练、评测进度及完成凭证。
- `mllm/vtimellm/model/vtimellm_arch.py`：将 `<video>` token 替换为投影后的 LiDAR 特征。
- `mllm/vtimellm/train/dataset.py`：加载 QA 和场景特征，处理 whole-scene/per-sequence 输入。
- `mllm/vtimellm/train/train.py`：训练参数、LoRA、DeepSpeed 和 checkpoint 恢复。
- `mllm/evaluation/test_b4dl.py`：六任务推理及统一评测入口。
- `mllm/evaluation/analyze_tg_regression.py`：B3 与后续模型的 TG 失败模式对比。

## 约束

- 训练和评测必须使用同代际的数据构造和输入参数。
- Stage 2/3 以 `scene_id` 查找 `{feat_folder}/{scene_id}.npy`。
- 训练前必须运行 `scripts/preflight_b3.py` 检查环境、数据与特征。
- 训练日志、预测结果、checkpoint 和临时分析图均为运行产物，不提交到 Git。
- 功能代码或开发方向变更必须同步维护 [项目方法与落实情况](docs/项目方法与落实情况.md)。

## Git 规范

- 每次代码修改后自动提交。
- 提交标题使用 `type(scope): 中文目的`，标准英文类别加中文说明；正文用中文列出具体改动和目的。
- `git add` 必须列出具体文件，不使用 `git add -A`。
- 不在 `main` 新增 RL 代码；RL 变更提交到 `rf-grpo`。
