# 复现实施指南

从零复现到基线 B0 的端到端步骤。模块级细节见各分页，本页只讲**按什么顺序做什么、每步产出什么、如何验收**。

> 论文原文的逐行解析与资源清单见 `docs/B4DL_复现方案.md`；本页是该方案的执行摘要。

## 前置资源

| 资源 | 说明 |
|------|------|
| 硬件 | 参考配置 2×RTX 5090（32GB）；CLIP ViT-L/14 + batch 32 即显存上限，超出用梯度累积 |
| nuScenes v1.0-trainval | 相机图像 + LiDAR 点云 + metadata（编码器与数据生成的输入） |
| HF `ccho4702/nuScenes-B4DL` | 官方发布：`metadata/`（scene/sequence_metadata.json）、`dataset/train/`（148,271 条 QA）、test_qa.json |
| OpenAI API Key | 仅数据生成路线需要；直接用官方数据可跳过 |
| 环境 | `wqlc` conda 环境，见 [[Installation]] |

## Step 1 — 数据生成（可跳过）

两条路线二选一：

- **路线 A（推荐，B0 采用）**：直接使用官方发布的 train/test 数据——数据集产物经审计已与论文 100% 对齐，重跑 GPT-4o 管线只会引入随机性
- **路线 B（验证管线）**：完整重跑两步生成（`generate_description.sh` → `generate_dataset.sh`，六任务逐个跑），见 [[Data-Generation]]

## Step 2 — LiDAR-CLIP 编码器训练

官方**从未发布编码器权重**，必须自训（这是与论文最大的不可控差异源）：

```bash
cd encoders/lidarclip
python train.py --data-dir <nuScenes> --name lidarclip_nuscenes \
    --nuscenes-datadir <nuScenes> --batch-size 32
```

- 训练时挂 `early_stop_monitor_v2.py`（loss 真值看 `logs/train_loss.csv`，勿信 wandb offline）
- 高 LR 续训被硬停后跑 `run_anneal_chain.sh` 退火链
- **验收**：`val_mse_probe.py` 在 150 个 val 场景确定性子集上选 **val MSE 最低**的 checkpoint；参考判据：自训新权重 MSE 0.083 vs 旧 ONCE 权重 0.169

## Step 3 — 特征提取

编码器定稿后**全量重提**（新旧编码器特征不可混用）：

| 特征 | 脚本 | 产出 |
|------|------|------|
| Stage1（每帧） | `extract_pc_features_sample_token.py` | 700 训练场景 28,130 个 `{sample_token}.npy` (1,768) |
| Stage2（每场景） | `extract_pc_features.py` | 850 个 `{scene_id}.npy` (N,768) |

**验收**：文件名 = 数据 JSON 的 `scene_id`（stage1 为 sample_token）；dtype float16。

## Step 4 — Stage1：mm_projector 对齐

```bash
# 4a. Stage1 数据：LiDAR-LLM-Nu-Caption → 162K 对话格式（scene_id=sample_token）
python datageneration/tools/build_stage1_from_lidarllm_official.py --input <LiDAR-LLM train.json>
# 或 build_stage1_from_lidarllm.py（需本地 nuScenes sample.json）

# 4b. 训练（只训 nn.Linear(768,4096)，LLM 冻结）
cd mllm && bash scripts/stage1.sh --output_dir ./checkpoints/...-stage1 [数据/特征路径]
```

**产出**：`checkpoints/...-stage1/mm_projector.bin`。B0 用的 projector 由 95K 数据训练（B1 升级为 162K，见 [[Reproduction-Log]]）。

## Step 5 — Stage2 训练数据构建

```bash
cd mllm
# 5a. 官方 train 数据 → 训练格式（148,271 条 = 68,695 简单 + 79,576 复杂；TG 打标 13,124 条）
python scripts/build_stage2_full_train.py --input_dir <HF>/dataset/train --output_dir ./b4dl_dataset

# 5b. ego 运动元数据（从 nuScenes ego_pose 计算）
python scripts/generate_ego_metadata.py --frame_motion ...

# 5c. metatoken + 序列归属注入（seqv3 关键步骤）
python scripts/inject_metatoken.py --input ./b4dl_dataset/stage2_train.json \
    --output ./b4dl_dataset/stage2_train_seqv3.json \
    --frame_motion ./b4dl_dataset/ego_frame_motion.json \
    --sequence_metadata ../encoders/lidarclip/annotations/sequence_metadata.json --answer_frames
# stage3_train_seqv3.json 同理
```

**验收**：TG 条目的 feat_indices 覆盖 100%、GT 帧范围 100% ⊂ feat_range；训练/评测两侧 2783/2783 一致。

## Step 6 — 混合训练（B0 方案）

```bash
bash scripts/run_stage2_full_seqv3_mixed.sh
```

148,271 条全部任务混合，**单 LoRA**（r64/α128）3 epochs、lr 1e-4、bs 8×16；`<4DLiDAR>`/`<meta>` 为可训练 embedding 行。

> ⚠️ 不要用两阶段法（`run_stage2_full_seqv3.sh`，Phase A→merge→Phase B）：实测简单任务格式漂移、exact match 归零（acc 0.0001），已回退混合法，详见 [[Paper-vs-Reproduction]]。

**产出**：`checkpoints/...-stage2`（LoRA adapter + non_lora_trainables.bin）。断点续训由脚本自动处理（trainer_state.json 校验）。

## Step 7 — 六任务评测

```bash
python evaluation/test_b4dl.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter <stage1>/mm_projector.bin \
    --stage2 <stage2-mixed> \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --test_data ./b4dl_dataset/test_qa.json \
    --ego_meta ./b4dl_dataset/ego_metadata.json \
    --output ./evaluation/predictions.json --metrics_output ./evaluation/evaluation_results.json \
    --per_sequence --frame_motion --sequence_metadata --answer_frames
```

四个对齐参数**缺一不可**（与训练数据代际绑定，见 [[Inference-and-Evaluation]] 的代际表）。支持断点续跑（每 50 条存 ckpt）。

**产出**：predictions.json + evaluation_results.json（含 `per_task_metrics` / `final_scores` / `metric_backend`）。

## Step 8 — 对比验收

与论文 Table 3 及基线 B0 对比（见 [[Reproduction-Log]] 的 B0 数值与 MD5 清单）。**显著性阈值**（判真改进的门槛，95% CI）：

- ΔmIoU > +0.013（CI 半宽 ±0.0132）
- Δaccuracy > ±0.009
- 文本指标 ≥ 0.01

换编码器/重提特征/换 projector 数据版本会**触发基线重置**，必须与新基线重新对比，不可跨基线比数值。
