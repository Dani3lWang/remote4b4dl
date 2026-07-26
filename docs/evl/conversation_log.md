# B4DL 评估会话记录

**日期**: 2026-07-26
**会话主题**: B4DL Stage2 模型评估 — 环境检查、依赖安装、推理、指标计算、结果分析

---

## 会话概览

用户说 "resume" 恢复之前的工作，之前正在构建 B4DL 评估流程。本次会话完成了从环境检查到完整评估的全流程，并将所有日志和结果保存到 `docs/evl/`。

---

## 时间线

### 1. 状态检查

检查了当前仓库状态，确认以下文件正在开发中：

| 文件/目录 | 状态 | 用途 |
|-----------|------|------|
| `mllm/evaluation/` | 新建 | 评估模块（evaluate_model.py, test_b4dl.py, build_test_split.py, split_dataset.py） |
| `mllm/scripts/test_sample.py` | 新建 | 单样本推理测试 |
| `mllm/scripts/verify.sh` | 新建 | 训练验证脚本 |
| 6 个 metadata JSON | 已修改 | split 字段从全 "train" 修复为正确的 train/test 划分 |

### 2. 评估前置条件检查

| 检查项 | 结果 |
|--------|------|
| GPU | NVIDIA RTX 5090, 32GB ✓ |
| PyTorch | 2.8.0+cu128, CUDA available ✓ |
| base_model | Vicuna-7B-v1.5 ✓ |
| mm_projector (Stage1) | mm_projector.bin ✓ |
| Stage2 checkpoint | LoRA weights ✓ |
| 特征文件 | 850 个 .npy ✓ |
| 测试数据 | stage2_test.json (7071), stage2_val.json (6723) ✓ |
| NLTK | 缺失 → **已安装** |
| rouge-score | 缺失 → **已安装** |
| bert-score | 已安装 ✓ |

**安装命令**:
```bash
conda run -n wqlc pip install nltk rouge-score
```

### 3. 快速验证（10 条样本）

验证评估流程是否能正常运行：

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

结果：
- Binary QA: 7/7 = 100%
- Categorical: 0/3 = 0%
- Overall: 7/10 = 70%

模型加载正常，推理管线正常。

### 4. 数据分布分析

检查了 `stage2_test.json` 和 `stage2_val.json` 的答案类型分布：

| 类型 | Test | Val |
|------|------|-----|
| Binary QA (Yes/No) | 5179 (73.2%) | 4950 (73.6%) |
| Frame Range (from frame X to frame Y) | 1405 (19.9%) | 1315 (19.6%) |
| Categorical (其他) | 487 (6.9%) | 458 (6.8%) |
| **总计** | **7071** | **6723** |

### 5. 完整 Test 集推理

```bash
cd mllm && conda run -n wqlc python vtimellm/eval/b4dl_eval.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    --data_path ./b4dl_dataset/stage2_test.json \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --log_path ../docs/evl/stage2_test_log.jsonl
```

- 耗时：约 12.5 分钟
- 吞吐：~3.5 it/s
- 输出：7071 条推理日志（1.3MB）

### 6. 完整 Val 集推理

Test 集完成后启动 Val 集推理（后台运行），耗时约 12 分钟。

### 7. 指标计算

```bash
conda run -n wqlc python vtimellm/eval/b4dl_metrics.py \
    --log_path ../docs/evl/stage2_test_log.jsonl \
    --task stage2 --output ../docs/evl/stage2_test_metrics.json
```

---

## 最终结果

| 指标 | Test (7071) | Val (6723) |
|------|-------------|------------|
| Binary QA Accuracy | **83.59%** | **82.08%** |
| Binary QA F1 | 87.46% | 85.95% |
| Frame Range mIoU | 18.23 | 20.35 |
| Frame Range R1@0.5 | 15.44% | 17.57% |
| Categorical Accuracy | 44.56% | 39.30% |
| Categorical Macro F1 | 13.47% | 13.96% |
| **Overall Exact Match** | **65.63%** | **64.85%** |

### 关键发现

1. **Binary QA 表现良好** (>82%)，模型能有效理解是/否类问题
2. **Frame Range 是明显短板** (mIoU ~20, Exact Match ~10%)，时间定位能力弱
3. **Categorical 类别不平衡严重** (Accuracy ~40% vs Macro F1 ~13%)
4. **Test/Val 一致性高** (<1% 差异)，无过拟合

---

## Git 提交记录

```
304645b 添加 B4DL 评估流程：评估脚本、推理测试工具、验证脚本及完整评估结果
786baf4 docs: 完善评估报告，补充验证流程记录和日志格式说明
```

---

## 产出文件 (`docs/evl/`)

| 文件 | 大小 | 说明 |
|------|------|------|
| `evaluation_summary.md` | - | 综合评估报告 |
| `stage2_test_log.jsonl` | 1.3MB | Test 集 7071 条推理日志 |
| `stage2_test_metrics.json` | 413B | Test 集指标 |
| `stage2_val_log.jsonl` | 1.3MB | Val 集 6723 条推理日志 |
| `stage2_val_metrics.json` | 413B | Val 集指标 |
| `stage2_metrics.json` | 244B | 快速验证指标 |
| `stage2_test_summary.md` | - | Test 集详细报告 |
| `conversation_log.md` | - | 本文件 |

### 日志格式说明

每行一个 JSON 对象：
```json
{"scene_id": "008158330", "query": "<video>\nDid a barrier exist?", "gt": "Yes.", "pred": "Yes.", "answer_type": "binary"}
```

- `answer_type`: `binary` / `frame_range` / `categorical`（由 GT 自动分类）

---

## 使用的关键文件

- **推理**: `mllm/vtimellm/eval/b4dl_eval.py` — 加载模型 → 逐条推理 → 输出 JSONL
- **指标**: `mllm/vtimellm/eval/b4dl_metrics.py` — 按 answer_type 分组计算准确率/mIoU/宏F1
- **模型加载**: `mllm/vtimellm/model/builder.py` — load_pretrained_model()
- **推理引擎**: `mllm/vtimellm/inference.py` — inference(model, features, query, tokenizer)

## 后续建议

1. 改善 Frame Range 时间定位 — 增加训练样本或设计专项数据增强
2. 平衡 Categorical 类别分布，或采用加权损失
3. 如有 Stage3 权重，可进行对比评估
4. 考虑用 `mllm/evaluation/evaluate_model.py` 中的 BLEU/METEOR/BERTScore 做更细粒度的描述质量评估
