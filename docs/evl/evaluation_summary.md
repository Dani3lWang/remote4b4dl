# B4DL Stage2 评估报告

**评估时间**: 2026-07-26
**模型**: VTimeLLM + Vicuna-7B-v1.5 + LoRA (Stage2)
**GPU**: NVIDIA RTX 5090 (32GB)

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
| `stage2_test_log.jsonl` | Test 集推理日志（7071 条） |
| `stage2_test_metrics.json` | Test 集指标 JSON |
| `stage2_val_log.jsonl` | Val 集推理日志（6723 条） |
| `stage2_val_metrics.json` | Val 集指标 JSON |
| `stage2_test_summary.md` | Test 集详细报告 |
| `evaluation_summary.md` | 本文件：综合评估报告 |
