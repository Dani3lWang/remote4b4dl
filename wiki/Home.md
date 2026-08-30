# B4DL 仓库 Wiki

**B4DL: A Benchmark for 4D LiDAR LLM in Spatio-Temporal Understanding**（ACM Multimedia 2025）的官方 PyTorch 实现。

本仓库构建了一个面向 4D LiDAR 点云序列时空理解的基准数据集，并训练多模态大语言模型（VTimeLLM 架构，Vicuna-7B backbone）回答关于 LiDAR 序列的问题。

- 📄 论文：[arXiv:2508.05269](https://arxiv.org/abs/2508.05269) ｜ ACM MM 2025（pp. 3399–3407）
- 🤗 数据集：[HuggingFace ccho4702/nuScenes-B4DL](https://huggingface.co/datasets/ccho4702/nuScenes-B4DL)
- 📜 许可：CC BY-NC-ND 4.0

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
- [[Repository-Structure]] — 目录结构速查
- [[Reproduction-Log]] — 复现时间线、基线 B0 锁定与文档索引
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

## 致谢

本工作部分由韩国政府（MSIT）资助的 IITP 资助（No.RS-2024-00439020、No.RS-2025-02283048）。架构基于 [VTimeLLM](https://github.com/HuangJulian/VTimeLLM)，编码器基于 [LiDAR-CLIP](https://github.com/divadiow/LiDAR-CLIP) 与 [SST](https://github.com/fanqi-no1/SST)（CVPR 2022）。
