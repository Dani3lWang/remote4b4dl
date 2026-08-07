# Stage1 训练报告

**日期**: 2026-08-02
**数据集**: LiDAR-LLM-Nu-Caption → `stage1_train.json` 95,048 条
**模型**: VTimeLLM (Vicuna-7B-v1.5) mm_projector (Linear 768→4096)
**GPU**: NVIDIA RTX 5090 (32GB)

---

## 训练配置

| 参数 | 值 |
|------|-----|
| DeepSpeed | ZeRO-3 + CPU offload |
| 精度 | bf16 |
| Epochs | 1 |
| per_device_batch_size | 16 |
| gradient_accumulation | 8 |
| 有效 batch | 128 |
| 总步数 | 742 |
| Learning Rate | 1e-3 (cosine + warmup 3%) |

## 训练结果

| 指标 | 值 |
|------|-----|
| **Final Loss** | **0.847** |
| 训练时长 | 8,650 秒 (2h 24min) |
| 吞吐量 | 10.99 samples/sec |
| 步速 | 0.086 steps/sec (~11.6s/step) |

## Loss 曲线概述

| 阶段 | 步数 | Loss 范围 |
|------|------|-----------|
| 初始 (step 0-5) | ~5 | 1.73 → 1.56 |
| 快速下降 (step 5-100) | ~95 | 1.56 → 1.05 |
| 平稳收敛 (step 100-500) | ~400 | 1.05 → 0.88 |
| 最终微调 (step 500-742) | ~242 | 0.88 → 0.85 |

## 保存文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `mm_projector.bin` | 6.1 MB | 训练后权重 (768→4096 bf16) |
| `trainer_state.json` | 127 KB | 743 步完整日志 |
| `config.json` | 856 B | 训练配置 |
| `stage1_training.log` | 213 KB | DeepSpeed 原始输出（含 loss 曲线） |

## 与旧版对比

| 指标 | 旧 Stage1 (699 条) | 新 Stage1 (95k 条) |
|------|-------------------|---------------------|
| 训练数据 | 699 | **95,048** (136×) |
| 特征类型 | stage2_features (场景级) | stage1_features (帧级) ✓ |
| 数据来源 | 手工整理 | LiDAR-LLM-Nu-Caption 官方 |
| Loss | 未记录 | 1.73 → 0.85 |

## 备注

- ZeRO-3 CPU offloading 导致训练较慢（~12s/步），若改用 ZeRO-2 可提速 3-5 倍
- 训练过程出现 1 次 PyTorch allocator cache flush 警告（显存压力），未影响训练
