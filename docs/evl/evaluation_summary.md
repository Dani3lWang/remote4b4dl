# B4DL Stage2 评估报告

**评估时间**: 2026-07-26
**模型**: VTimeLLM + Vicuna-7B-v1.5 + LoRA (Stage2)
**GPU**: NVIDIA RTX 5090 (32GB)
**评估工具**: `mllm/vtimellm/eval/b4dl_eval.py` + `b4dl_metrics.py`

---

## 评估流程

### 1. 环境检查
- 确认 GPU 可用（RTX 5090, 32GB VRAM）
- 确认模型文件完整：base_model (Vicuna-7B-v1.5)、mm_projector.bin (Stage1)、Stage2 LoRA checkpoint
- 确认特征文件就绪：850 个 `.npy` 文件（`encoders/lidarclip/b4dl/stage2_features/`）
- 安装评估依赖：`nltk`、`rouge-score`（`bert-score` 已安装）

### 2. 快速验证（10 条样本）

验证命令：
```bash
cd mllm && conda run -n wqlc python vtimellm/eval/b4dl_eval.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    --data_path ./b4dl_dataset/stage2_test.json \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --log_path ../docs/evl/stage2_metrics.json \
    --max_samples 10
```

快速验证结果（10 条）：

| 类型 | 数量 | 正确 | Accuracy |
|------|------|------|----------|
| Binary QA | 7 | 7 | 100.00% |
| Categorical | 3 | 0 | 0.00% |
| **Overall** | **10** | **7** | **70.00%** |

模型加载和推理管线正常工作，确认可以进行完整评估。

### 3. 全量评估

```bash
# Test 集评估（7071 条）
cd mllm && conda run -n wqlc python vtimellm/eval/b4dl_eval.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    --data_path ./b4dl_dataset/stage2_test.json \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --log_path ../docs/evl/stage2_test_log.jsonl

# Val 集评估（6723 条）
# 同上，替换 --data_path 和 --log_path

# 计算指标
conda run -n wqlc python vtimellm/eval/b4dl_metrics.py \
    --log_path ../docs/evl/stage2_test_log.jsonl \
    --task stage2 --output ../docs/evl/stage2_test_metrics.json
```

- Test 集耗时：约 12.5 分钟（~3.5 it/s）
- Val 集耗时：约 12 分钟（~3.5 it/s）

---

## 数据概览

| 集合 | 总数 | Binary QA | Frame Range | Categorical |
|------|------|-----------|-------------|-------------|
| Test | 7071 | 5179 (73.2%) | 1405 (19.9%) | 487 (6.9%) |
| Val | 6723 | 4950 (73.6%) | 1315 (19.6%) | 458 (6.8%) |

---

## 评估结果

### Binary QA

| 指标 | Test | Val |
|------|------|-----|
| Accuracy | **83.59%** | **82.08%** |
| Precision | 84.25% | 82.42% |
| Recall | 90.92% | 89.81% |
| F1 | 87.46% | 85.95% |

### Frame Range (时间定位)

| 指标 | Test | Val |
|------|------|-----|
| Exact Match | 9.11% | 10.49% |
| Mean IoU | 18.23 | 20.35 |
| R1@0.5 | 15.44% | 17.57% |
| R1@0.7 | 10.96% | 12.17% |

### Categorical (描述类)

| 指标 | Test | Val |
|------|------|-----|
| Accuracy | **44.56%** | **39.30%** |
| Macro F1 | 13.47% | 13.96% |

### 总体

| 指标 | Test | Val |
|------|------|-----|
| Overall Exact Match | **65.63%** | **64.85%** |

---

## 分析

### 优势
- **Binary QA**: Test/Val 均 >82%，模型对是/否类问题理解较好
- **一致性**: Test 和 Val 差异 <1%，无明显过拟合

### 短板
- **Frame Range 时间定位很弱**: mIoU 仅 ~20，Exact Match ~10%。模型难以精确定位帧范围
- **Categorical 类别不平衡**: Accuracy ~40% 但 Macro F1 ~13%，说明小类别预测极差

### 建议
1. Frame Range 任务需要更多训练样本或用特定数据增强
2. Categorical 任务的类别分布需要重新审视
3. 考虑在 Stage3 中重点改善时间定位能力

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `stage2_test_log.jsonl` | Test 集推理日志（7071 条，1.3MB） |
| `stage2_test_metrics.json` | Test 集指标 JSON |
| `stage2_val_log.jsonl` | Val 集推理日志（6723 条，1.3MB） |
| `stage2_val_metrics.json` | Val 集指标 JSON |
| `stage2_metrics.json` | 快速验证指标（10 条样本） |
| `stage2_test_summary.md` | Test 集详细报告 |
| `evaluation_summary.md` | 本文件：综合评估报告（含验证流程记录） |

### 日志格式

每行一个 JSON 对象：
```json
{
  "scene_id": "008158330",
  "query": "<video>\nDid a barrier exist in front of the ego vehicle in frame 005?",
  "gt": "Yes.",
  "pred": "Yes.",
  "answer_type": "binary"
}
```

- `answer_type` 由 GT 答案内容自动分类：`binary`（Yes/No）、`frame_range`（from frame X to frame Y）、`categorical`（其他）
