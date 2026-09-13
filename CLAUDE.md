# CLAUDE.md

本文件记录当前仓库的有效开发约定。`main` 只维护官方基础模块与最新 B3 基线；强化学习实现位于 `rf-grpo` 分支。

## 项目概述

B4DL（Benchmark for 4D LiDAR LLM）是 ACM Multimedia 2025 论文的官方 PyTorch 实现，包含：

1. `datageneration/`：由 nuScenes 相机图像生成场景描述和问答数据。
2. `encoders/lidarclip/`：用 SST backbone 将 LiDAR 序列编码为 768 维特征。
3. `mllm/`：VTimeLLM 训练、推理和评测代码。

## 环境

- Conda 环境：`wqlc`，Python 3.10；所有 Python 命令必须在该环境执行。
- 权威依赖文件：`mllm/requirements.txt`。
- 当前实测栈：PyTorch 2.8.0+cu128、transformers 4.47.0、deepspeed 0.16.4、peft 0.13.2、accelerate 1.3.0。
- flash-attn 未安装，训练和推理使用 transformers 原生 SDPA。

```bash
conda create -n wqlc python=3.10 -y
conda activate wqlc
pip install -r mllm/requirements.txt
```

## 最新 B3 基线

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

保留 `stage1.sh`、`stage2.sh`、`stage3.sh` 及 ChatGLM 对应入口，用于官方三阶段能力和自定义实验。新的基线复现优先使用 B3 流水线，不再新增 B0/B1/B2/B4a 或 frame-position 变体入口。

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

## B3 评测口径

B3 使用 `--whole_scene --per_sequence --answer_frames`。其中 `--answer_frames` 依据答案恢复序列归属，属于 oracle 输入选择，报告结果时必须声明。

```bash
cd mllm
bash run_baseline_eval.sh
```

METEOR 默认使用双后端；与论文对比采用 NLTK-2005 主口径。

## GRPO 强化学习（仅 `rf-grpo`）

GRPO 代码位于 `mllm/vtimellm/rl/`，训练入口为 `mllm/scripts/run_grpo_tg.sh`。RL-LoRA 叠加在当前 B3 adapter 合并后的模型上，参考策略通过同一模型的 `disable_adapter()` 获得，不常驻第二份基础模型。

```bash
cd mllm
python -m vtimellm.rl.smoke_m2
bash scripts/run_grpo_tg.sh m31
bash scripts/run_grpo_tg.sh m32
bash scripts/run_grpo_tg.sh full
bash scripts/run_grpo_tg.sh eval
```

- 奖励必须复用 `evaluation.evaluate_model` 的解析与指标实现。
- 训练数据只允许 `b4dl_dataset/rl_tg_train.jsonl`，测试集不可进入训练或调参。
- old/ref/actor log-probability 统一走 `logprob.forward_sequence`。
- rollout 使用 `--gen-max-batch` 分块生成，RL-LoRA 使用 fp32 参数与优化器更新。
- `eval` 通过 `test_b4dl.py --stage2 <B3> --stage3 <RL-LoRA>` 做同口径双合并评测。
- 历史 M2、M3.1、M3.2 的验证结果基于旧 3-epoch B3 checkpoint；换成新 2-epoch B3 后必须重新运行门控和全量评测。

## 关键实现

- `mllm/vtimellm/model/vtimellm_arch.py`：将 `<video>` token 替换为投影后的 LiDAR 特征。
- 帧位置嵌入（A.2）是默认关闭的可选开关：`--use_frame_position_embedding --frame_position_max 64`，零初始化，作为非 LoRA 参数存入 `non_lora_trainables.bin`。`builder.py` 会从 stage2 的 `config.json` 自动读取这两个键，所以加载 `framepos3ep` checkpoint 不需要额外传参。关闭时 B0/B3/B4a 的结构与数值与未引入该开关前完全一致；CPU 自检见 `scripts/verify_frame_position.py`。
- `mllm/vtimellm/train/dataset.py`：加载 QA 和场景特征，处理 whole-scene/per-sequence 输入。
- `mllm/vtimellm/train/train.py`：训练参数、LoRA、DeepSpeed 和 checkpoint 恢复。
- `mllm/evaluation/test_b4dl.py`：六任务推理及统一评测入口。
- `mllm/evaluation/analyze_tg_regression.py`：B3 与后续模型的 TG 失败模式对比。

## 约束

- 训练和评测必须使用同代际的数据构造和输入参数。
- Stage 2/3 以 `scene_id` 查找 `{feat_folder}/{scene_id}.npy`。
- 训练前必须运行 `scripts/preflight_b3.py` 检查环境、数据与特征。
- 训练日志、预测结果、checkpoint 和临时分析图均为运行产物，不提交到 Git。

## Git 规范

- 每次代码修改后自动提交。
- 提交信息使用中文。
- `git add` 必须列出具体文件，不使用 `git add -A`。
- 不在 `main` 新增 RL 代码；RL 变更提交到 `rf-grpo`。
