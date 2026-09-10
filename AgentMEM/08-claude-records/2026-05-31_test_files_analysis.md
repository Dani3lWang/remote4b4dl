# 项目测试文件分析

日期: 2026-05-31

---

项目中有完整的测试/评估体系，分为 **B4DL 主模型评估** 和 **编码器单元测试** 两个层面。

---

## 一、B4DL 模型训练结果评估（核心）

这是直接测试模型训练结果的评估管线：

| 文件 | 作用 |
|------|------|
| `mllm/scripts/run_b4dl_eval.sh` | **一站式评估脚本**，自动完成数据划分 + Stage2/Stage3 推理 + 指标计算 |
| `mllm/vtimellm/eval/b4dl_eval.py` | **推理脚本**，加载训练好的 checkpoint，在测试集上逐条推理，输出 `jsonl` 日志 |
| `mllm/vtimellm/eval/b4dl_metrics.py` | **指标计算**，Stage2 算准确率/F1/IoU，Stage3 算 BLEU-1~4 / ROUGE-L / METEOR |
| `mllm/scripts/verify_stage2.py` | **快速抽查**，随机取 20 条样本推理并打印 PASS/FAIL |
| `mllm/scripts/create_splits.py` | **数据集划分**，按 scene 级别 80/10/10 划分 train/val/test |

### 测试数据

- `mllm/b4dl_dataset/stage2_test.json` — Stage2（QA 问答）测试集
- `mllm/b4dl_dataset/stage3_test.json` — Stage3（Captioning）测试集

### 已运行的评估结果

#### Stage2 指标 (`mllm/eval_results/stage2_metrics.json`)

| 指标 | 数值 |
|------|------|
| 二分类准确率 | 82.89% |
| 二分类 F1 | 86.84% |
| Frame Range Exact Match | 8.54% |
| Frame Range mIoU | 17.11% |
| Frame Range R1@0.5 | 13.95% |
| 分类准确率 | 39.22% |
| **整体准确率** | **64.67%** |

#### Stage3 指标 (`mllm/eval_results/stage3_metrics.json`)

| 指标 | 数值 |
|------|------|
| BLEU-1 | 33.22 |
| BLEU-2 | 20.61 |
| BLEU-3 | 13.83 |
| BLEU-4 | 9.64 |
| ROUGE-L | 28.04 |
| METEOR | 26.29 |

---

## 二、通用视频评估脚本

| 文件 | 作用 |
|------|------|
| `mllm/vtimellm/eval/eval.py` | 通用视频理解评估（grounding IoU + captioning），支持 video 输入或预提取特征 |
| `mllm/vtimellm/eval/metric.py` | SODA / DVC 指标计算（视频 captioning 专用指标） |

---

## 三、编码器层测试（LidarCLIP / MMDetection3D）

这些是底层编码器模块的测试，不直接测试 B4DL 训练结果：

- `encoders/lidarclip/sst/tools/test.py` — 标准 MMDet3D 测试脚本，加载 checkpoint 在测试集上评估
- `encoders/lidarclip/sst/tools/test_waymo.py` — Waymo 数据集专用测试
- `encoders/lidarclip/sst/tools/dist_test.sh` / `slurm_test.sh` — 分布式测试启动脚本
- `encoders/lidarclip/sst/tests/` — 单元测试（模型前向、loss、评估指标、anchor 等）
- `encoders/lidarclip/mmdetection3d/tools/test.py` — MMDetection3D 框架标准测试入口
- `encoders/lidarclip/mmdetection3d/tests/` — MMDetection3D 单元测试

---

## 四、使用方式

### 完整评估

```bash
# Stage2 完整评估
bash mllm/scripts/run_b4dl_eval.sh

# 包含 Stage3 评估
bash mllm/scripts/run_b4dl_eval.sh --stage3
```

### 快速抽查

```bash
python mllm/scripts/verify_stage2.py
```

### 编码器测试

```bash
# LidarCLIP SST 测试
python encoders/lidarclip/sst/tools/test.py <config> <checkpoint> --eval mAP

# 分布式测试
bash encoders/lidarclip/sst/tools/dist_test.sh <config> <checkpoint> <gpus> --eval mAP
```

---

## 五、总结

项目具备完整的测试体系：

1. **B4DL 端到端评估**：已有 `run_b4dl_eval.sh` 一键脚本，从数据划分到推理再到指标计算全自动
2. **已有评估结果**：Stage2 和 Stage3 都已运行过测试，日志和指标文件在 `mllm/eval_results/` 下
3. **编码器层测试**：LidarCLIP 和 MMDetection3D 均有完整的单元测试和推理测试脚本
