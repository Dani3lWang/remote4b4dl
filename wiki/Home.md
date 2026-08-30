# B4DL 复现仓库 Wiki

本仓库是对论文 **B4DL: A Benchmark for 4D LiDAR LLM in Spatio-Temporal Understanding**（ACM Multimedia 2025）的**独立复现**：基于官方发布的数据集与论文描述，完整重建了数据生成、LiDAR-CLIP 特征编码、VTimeLLM 多阶段训练与六任务评测的端到端流水线，并以论文指标为基准验证复现质量。

> ⚠️ **本仓库非官方实现**（官方代码与权重从未完整公开）。复现的端到端步骤见 [[Reproduction-Guide]]，与论文的逐项差异见 [[Paper-vs-Reproduction]]，复现历程与基线锁定见 [[Reproduction-Log]]。

- 📄 论文：[arXiv:2508.05269](https://arxiv.org/abs/2508.05269) ｜ ACM MM 2025（pp. 3399–3407）
- 🤗 官方数据集：[HuggingFace ccho4702/nuScenes-B4DL](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL)（metadata + train/test QA 数据，本仓库复现的直接输入）
- 📜 原论文许可：CC BY-NC-ND 4.0

## 三大核心模块

| 模块 | 目录 | 职责 |
|------|------|------|
| 数据生成管线 | `datageneration/` | 调用 GPT-4o API，从 nuScenes 相机图像生成 LiDAR 场景描述，再转化为 6 类任务的问答对 |
| LiDAR-CLIP 编码器 | `encoders/lidarclip/` | 基于 SST backbone，将点云序列编码为 CLIP 对齐的 768 维特征向量（.npy） |
| VTimeLLM | `mllm/` | Vicuna-7B + LoRA + DeepSpeed ZeRO-3，多阶段训练时空问答模型，含完整评测套件 |

## 端到端流水线

```
nuScenes 数据集（相机图像 + LiDAR 点云）
  │
  ├─ [datageneration/] 相机图像 → GPT-4o → 场景描述 JSON → QA 数据集 JSON
  │
  ├─ [encoders/lidarclip/] LiDAR 点云 → SST 编码器 → CLIP 特征 (.npy)
  │       Stage1：每帧独立 (1, 768)   Stage2：每场景拼接 (N_frames, 768)
  │
  └─ [mllm/] 预提取特征 + QA 数据 → VTimeLLM 三阶段训练 → 六任务评测
```

## 复现状态速览（基线 B0，2026-08-29 锁定）

| 指标 | 复现 B0 | 论文 | 状态 |
|------|---------|------|------|
| accuracy（existence+binary） | 0.7629 | 0.762 | ✅ 持平 |
| mIoU（time_grounding） | 0.2696 | 0.311 | ❌ Δ-0.041 |
| BLEU-4（语料级） | 0.0973 | 0.095 | ✅ |
| BERTScore（roberta-large L17） | 0.8973 | 0.897 | ✅ 精确命中 |
| ROUGE-L | 0.3244 | — | 口径一致 |
| METEOR（Meteor-1.5 jar） | 0.1729 | 0.275 | ❌ 唯一遗留口径考证项 |

数据集产物已 100% 对齐（论文 Table 2 的 14 个数字精确吻合）；主要差距在编码器（官方未发布权重，自训中）与 time_grounding 定位精度。完整对照见 [[Paper-vs-Reproduction]]。

## 快速开始

```bash
# 1. 环境（Python 3.10, PyTorch 2.5.1 cu124）
conda create -n wqlc python=3.10 -y && conda activate wqlc
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r mllm/requirements.txt   # ⚠️ 不要用根目录 requirements.txt

# 2. 数据生成（需 OpenAI API key）
cd datageneration && bash scripts/generate_description.sh && bash scripts/generate_dataset.sh

# 3. 特征提取（需 nuScenes + 编码器 checkpoint）
cd encoders/lidarclip && python extract_pc_features_sample_token.py ...

# 4. 训练（Stage1 对齐 + Stage2 LoRA）
cd mllm && bash run_stages.sh --s1_data ... --s1_feat ... --s2_data ... --s2_feat ...

# 5. 评测（六任务，论文 Table 3 对齐）
python evaluation/test_b4dl.py ... 
```

详细步骤见各分页。

## Wiki 导航

- [[Installation]] — 环境搭建与依赖版本锁定说明
- [[Architecture]] — 整体架构、模型组件与对话模板
- [[Data-Generation]] — 两步数据生成管线与 6 类任务
- [[LiDAR-CLIP-Encoder]] — 编码器训练、特征提取与退火链
- [[Training]] — VTimeLLM 三阶段训练与 metatoken 注入
- [[Inference-and-Evaluation]] — 推理、六任务评测与指标口径

**复现专项**

- [[Reproduction-Guide]] — 端到端复现实施指南（从零到 B0 基线）
- [[Paper-vs-Reproduction]] — 与论文的逐项差异对照
- [[Reproduction-Log]] — 复现时间线、基线 B0 锁定与文档索引

**参考**

- [[Repository-Structure]] — 目录结构速查
- [[FAQ-and-Known-Issues]] — 已知问题与注意事项

## 引用

```bibtex
@inproceedings{choi2025b4dl,
  title={B4DL: A Benchmark for 4D LiDAR LLM in Spatio-Temporal Understanding},
  author={Choi, Changho and Shin, Youngwoo and Han, Gyojin and Lee, Dong-Jae and Kim, Junmo},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={3399--3407},
  year={2025}
}
```
