# B4DL Stage2 测试集评估结果

**评估时间**: 2026-07-26
**数据集**: stage2_test.json (7071 条)
**模型**: VTimeLLM + Vicuna-7B-v1.5 + LoRA

## 数据分布

| 类型 | 数量 | 占比 |
|------|------|------|
| Binary QA | 5179 | 73.2% |
| Frame Range | 1405 | 19.9% |
| Categorical | 487 | 6.9% |

## 指标

### Binary QA

| 指标 | 值 |
|------|-----|
| Accuracy | 83.59% |
| Precision | 84.25% |
| Recall | 90.92% |
| F1 | 87.46% |

### Frame Range (时间定位)

| 指标 | 值 |
|------|-----|
| Exact Match | 9.11% |
| Mean IoU | 18.23 |
| R1@0.5 | 15.44% |
| R1@0.7 | 10.96% |

### Categorical (描述类)

| 指标 | 值 |
|------|-----|
| Accuracy | 44.56% |
| Macro F1 | 13.47% |

### 总体

| 指标 | 值 |
|------|-----|
| Overall Exact Match | **65.63%** |

## 分析

- **Binary QA 表现较好** (83.59% accuracy, 87.46% F1)，说明模型能有效理解简单的是/否类问题
- **Frame Range 表现较弱** (mIoU 18.23, Exact Match 9.11%)，时间定位是明显的短板
- **Categorical 描述类** accuracy 44.56% 但 Macro F1 仅 13.47%，说明对少类别预测较差
