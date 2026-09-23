# CLAUDE.md

本文件记录当前仓库的有效开发约定。`main` 只维护官方基础模块与最新 B3 基线；强化学习实现位于 `rf-grpo` 分支；ReasonSeg 单帧点级分割位于 `seg` 分支。

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

## 关键实现

- `mllm/vtimellm/model/vtimellm_arch.py`：将 `<video>` token 替换为投影后的 LiDAR 特征。
- `mllm/vtimellm/train/dataset.py`：加载 QA 和场景特征，处理 whole-scene/per-sequence 输入。
- `mllm/vtimellm/train/train.py`：训练参数、LoRA、DeepSpeed 和 checkpoint 恢复。
- `mllm/evaluation/test_b4dl.py`：六任务推理及统一评测入口。
- `mllm/evaluation/analyze_tg_regression.py`：B3 与后续模型的 TG 失败模式对比。

## ReasonSeg（seg 分支）

ReasonSeg（单帧点级推理分割）在 `seg` 分支开发，运行环境是独立的 `reasonseg`（`/root/autodl-tmp/.conda-stuff/envs/reasonseg`），不复用 `wqlc`。

- 复算 teacher-forcing 验证用 `--validate-only --validation-samples 0 --validate-dtype-variants as_is --eval-checkpoint <best>`；**不可**与 `--spatial-checkpoint` 并用——ReasonSeg checkpoint 已含完整空间编码器，loader 会直接 raise。
- teacher-forcing mean IoU 的绝对值只在同一 manifest 内可比：query 构成主导，同一份权重在 `val_thin`(2248 条)/`val_internal_fresh`(19450 条)/`val_oov500`(1991 条) 上分别为 0.113/0.082/0.138。报数必须写明 manifest 名与记录数，模型间对比只能在同一 manifest 上做。
- 模型选择只能用 `reasonseg_val_internal*.jsonl`（官方 train 场景内划分，由 `scripts/reasonseg_experiments/repartition_reasonseg_val.py` 产出）。`reasonseg_val_thin.jsonl` 的 20 个场景属 nuScenes 官方 val，即 B4DL 测试集，只能作最终测试报告，不得用于 early stopping。
- 训练一律显式 `--validation-samples 0`（默认 32 只测 manifest 前 32 条，且几乎落在同一场景）。
- 自由生成评测固定 `--dtype fp16 --encoder-dtype fp32`：编码器进 fp16 会让部分帧特征 NaN，贪心解码随即锁死在 `<unk>`。metrics 的 `cIoU` 是面积加权、`gIoU` 是逐对均值，键名与惯例相反。
- 掩码与粗定位损失可调 `--bce-mode {plain,balanced}` 与 `--region-loss {dice,tversky}`（后者配 `--tversky-alpha/--tversky-beta`）。默认 `plain`+`dice` 与 2026-09-23 之前的实现逐位一致，由 `tests/test_reasonseg.py::MaskLossTests` 锁定，改损失前先跑它。掩码正例只占帧内点的 3e-4，`plain` 会让"全不触发"与"在所有同类候选上对冲"的总损失几乎持平（下坡仅占沉默损失的 4.9%），这是掩码塌缩的成因；`balanced` 把正负两侧各自归一后等权平均，实测把该下坡放大到 74.9%。
- 上述损失项只影响损失计算、不决定任何参数形状，故已列入 `config.py:LOSS_ONLY_FIELDS`，loader 的加载期比对只比 `architecture_dict()`。新增任何纯损失/权重字段都必须同步加进该元组，否则用非默认损失训出的档将无法被默认配置的评测入口加载。
- 接地诊断（秩 AUC、oracle top-K、LOC 包含率、相对阈值解码、LOC 先验置零臂）用 `scripts/reasonseg_experiments/diagnose_reasonseg_grounding.py`，同样必须 `--dtype fp16 --encoder-dtype fp32`——整模型 fp32 的 7B 需 28G，在 4090 上直接 OOM。
- 实验脚本归档在 `mllm/scripts/reasonseg_experiments/`；`mllm/reasonseg_data_trainval/` 与 `mllm/training_logs/` 整体被 gitignore，脚本必须放进前者才入库。

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
- ReasonSeg 代码与训练脚本变更提交到 `seg`，不混入 `main`；`seg` 晋级 `main` 前必须完成非泄漏数据闭包与消融。
- `push` 前先过 L3 深度安全扫描。
- autodl3 服务器无 GitHub 直连且是 blob 过滤的 partial clone：同步用区间 bundle（`git bundle create x.bundle <base>..seg`，全历史 bundle 会因缺 blob 反复回源卡死），服务器→本地经 scp+fetch 后由本地 push，本地→服务器经 scp+`git fetch <bundle> seg:tmp && git merge --ff-only tmp`。
