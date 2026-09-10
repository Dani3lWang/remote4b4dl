# B4DL 项目分析报告

> 生成日期: 2026-05-30
> 项目: B4DL — A Benchmark for 4D LiDAR LLM in Spatio-Temporal Understanding
> 论文: ACM Multimedia 2025

---

## 1. 项目概述

B4DL 是一个将 **4D 激光雷达点云数据** 与大语言模型（LLM）结合的研究项目。其核心目标是通过 LLM 实现对自动驾驶场景中 LiDAR 序列的时空理解，包括场景描述、问答和时序定位。

### 1.1 核心思想

传统的自动驾驶感知模型只能输出固定的标签（如检测框），而 B4DL 利用 LLM 的生成能力，让模型能够用自然语言描述场景中的事件、物体关系和时序变化。

### 1.2 数据来源

基于 nuScenes 数据集，使用 OpenAI GPT 生成场景描述和问答对，构建 B4DL benchmark 数据集。

---

## 2. 模型架构

```
┌─────────────────────────┐
│  LiDAR 点云序列 (nuScenes) │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│  LiDARCLIP 特征提取器      │  ← 预训练 CLIP ViT-L-14
│  输出: [N, 768] float16   │     N = 每场景帧数 (通常 ~100)
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│  mm_projector (线性投影)  │  ← nn.Linear(768, 4096)
│  768 → LLM hidden_size   │     将视觉特征映射到 LLM 嵌入空间
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│  Vicuna-7B-v1.5 (LLM)   │  ← 基于 LLaMA 的对话模型
│  语言生成 / 推理          │     LoRA 微调 (r=64, alpha=128)
└───────────┬─────────────┘
            ↓
       文本输出 (描述/答案)
```

### 2.1 关键组件

| 组件 | 规格 | 说明 |
|------|------|------|
| 特征提取器 | CLIP ViT-L-14 | 768 维特征向量，冻结不训练 |
| 投影层 | Linear(768, 4096) | 特征空间对齐，Stage1 训练后冻结 |
| 基座模型 | Vicuna-7B-v1.5 | ~7B 参数，LoRA 高效微调 |
| 微调方法 | LoRA (r=64, α=128) | 仅训练 ~0.5% 参数 |
| 训练框架 | DeepSpeed ZeRO-3 | 显存优化 + CPU offload |
| 精度 | BF16 混合精度 | 节省显存 |

---

## 3. 三阶段训练管线

### 3.1 训练管线总览

```
Stage 1           Stage 2           Merge            Stage 3
特征对齐  ────→   QA 微调   ────→  LoRA合并入基座  ────→  描述生成微调
(1 epoch)        (2 epochs)         (推理前)          (3 epochs)
lr=1e-3          lr=1e-4                             lr=2e-5
仅投影层           LoRA on LLM                         新 LoRA on LLM
```

### 3.2 各阶段训练配置对比

| 参数 | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|
| **目的** | 特征对齐 | 问答微调 | 描述生成 |
| **可训练参数** | 仅 mm_projector | LoRA (r=64) | LoRA (r=64) |
| **训练数据** | 699 条对齐数据 | 68,695 条 QA 对 | 63,821 条描述 |
| **基础模型** | Vicuna-7B-v1.5 | Vicuna + Stage1 投影层 | Stage2 Merge 模型 |
| **学习率** | 1e-3 | 1e-4 | 2e-5 |
| **学习率调度** | cosine + warmup 3% | cosine + warmup 3% | cosine + warmup 3% |
| **Epochs** | 1 (6 steps) | 2 (1072 steps) | 3 (747 steps) |
| **Batch Size** | 16 × 8 = 128 | 8 × 16 = 128 | 8 × 16 × 2GPU = 256 |
| **GPU** | 单卡 (GPU 1) | 单卡 (GPU 1) | 双卡 (GPU 0,1) |
| **对话模板** | plain (无模板) | vicuna_v1 | vicuna_v1 |
| **mm_projector** | **训练** | 冻结 | 冻结 |
| **Max Length** | 2048 | 2048 | 2048 |

### 3.3 Stage 1 — 特征对齐

训练投影层，让 LiDARCLIP 的 768 维特征通过 `nn.Linear(768, 4096)` 映射到 Vicuna 的语义空间。

**训练过程**:

| Step | Loss | Grad Norm | LR |
|------|------|-----------|-----|
| 1 | 7.18 | 53.73 | 1.00e-3 |
| 2 | 7.21 | 51.97 | 8.54e-4 |
| 3 | 4.94 | 11.41 | 5.00e-4 |
| 4 | 4.51 | 6.46 | 1.46e-4 |
| 5 | 4.22 | 4.59 | 0 |

- 平均训练 Loss: 5.61
- 训练耗时: 16秒（仅 6 步）
- 显存占用: 极小（仅投影层 3.1M 参数参与训练）

### 3.4 Stage 2 — QA 微调

在 Stage 1 投影层基础上，用 LoRA 微调 LLM 实现 LiDAR 场景问答。

**训练 Loss 变化**:

- 初始 Loss: **3.35** → 最终 Loss: **0.28**
- 总训练步数: 1072（2 epochs）
- Checkpoint 大小: **2.4GB**（仅 LoRA adapter + non_lora_trainables）

**数据类型分布**:
- Binary QA: 5,179 条测试样本（判断物体存在性）
- Frame Range QA: 1,405 条测试样本（定位时序范围）
- Categorical QA: 487 条测试样本（类别判断）

### 3.5 Stage 3 — 描述生成微调

**关键设计 — Stage2 Merge + 新 LoRA**:

```python
# train.py, training_stage=3 时的逻辑:
model = load_lora(model, stage2_path)        # 加载 Stage2 LoRA
model = model.merge_and_unload()              # 合并到基座 → 13GB 完整模型
model = get_peft_model(model, lora_config)   # 添加新的 Stage3 LoRA
```

| Merge 后基座 | 大小 | 说明 |
|-------------|------|------|
| Stage2-merged | 13GB | Stage2 LoRA 已融入 Vicuna-7B 权重 |

**训练 Loss 变化**:

- 初始 Loss: **2.82** → 最终 Loss: **1.35**（未完全收敛，仍有下降空间）
- 总训练步数: 747（3 epochs）
- Checkpoint 大小: **2.4GB**（仅 Stage3 LoRA adapter）

**数据类型** (63,821 train / 7,710 val / 8,045 test):
- 全场景描述: `"Describe the LiDAR sequence."`
- 单帧描述: `"Describe the LiDAR sequence at frame 005."`
- 时间段描述: `"Describe the LiDAR sequence between frame 004 and frame 006."`

### 3.6 DeepSpeed 配置

所有阶段使用统一的 ZeRO-3 + CPU offload 配置:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": { "device": "cpu" },
        "offload_param": { "device": "cpu" }
    }
}
```

Monkey-Patch: 修复了 ZeRO-3 梯度分区与 `accelerate.no_sync` 的兼容性问题（train.py:29-40）。

---

## 4. 评估结果

### 4.1 Stage 2 — QA 任务评估

| 任务类型 | 指标 | 得分 |
|----------|------|------|
| **Binary QA** | Accuracy | **82.89%** |
| | Precision | 84.16% |
| | Recall | 89.69% |
| | F1 | 86.84% |
| **Frame Range** | Mean IoU | 17.11% |
| | Exact Match | 8.54% |
| | R1@0.5 | 13.95% |
| | R1@0.7 | 10.11% |
| **Categorical** | Accuracy | 39.22% |
| | Macro F1 | 6.99% |
| **Overall** | Exact Match | **64.67%** |

### 4.2 Stage 3 — 描述生成评估

| 指标 | 得分 |
|------|------|
| BLEU-1 | **33.22** |
| BLEU-2 | 20.61 |
| BLEU-3 | 13.83 |
| BLEU-4 | 9.64 |
| ROUGE-L | 28.04 |
| METEOR | 26.29 |

测试样本数: 8,045

### 4.3 Stage 2 与 Stage 3 结果并排对比

| 维度 | Stage 2 (QA) | Stage 3 (Captioning) |
|------|-------------|---------------------|
| 任务类型 | 结构化问答 | 开放式场景描述 |
| 准确率基准 | Binary: 82.89% | BLEU-1: 33.22 |
| 时序理解 | Frame IoU: 17.11% | 评估方式不同，不可直接对比 |
| 语言质量 | N/A | ROUGE-L: 28.04, METEOR: 26.29 |
| 测试样本量 | 7,071 (三类合计) | 8,045 |

---

## 5. 综合分析

### 5.1 训练收敛分析

```
Stage1: Loss 7.18 → 4.22 (↓41%, 5步, 16秒) — 快速收敛，投影层有效学习
Stage2: Loss 3.35 → 0.28 (↓92%, 1072步)     — 大幅降低，LoRA 充分拟合 QA 模式
Stage3: Loss 2.82 → 1.35 (↓52%, 747步)      — 下降明显但未完全收敛
```

- **Stage 1** 收敛极快：699 条数据、仅 6 步训练，Loss 已在下降趋势中，说明 LiDARCLIP 特征与 Vicuna 语义空间天然接近
- **Stage 2** 最充分：1072 步训练后 Loss 降至 0.28，模型已熟练记忆 QA 输出的固定模式（如 "Yes." / "No." / "from frame X to frame Y"）
- **Stage 3** 仍有潜力：最终 Loss 1.35 相对初始 2.82 下降 52%，但远未达到 Stage2 的水平。考虑因素：
  - 描述文本远比短答案复杂（平均长度远大于 "Yes/No"）
  - 学习率 2e-5 仅为 Stage2 的 1/5，收敛更保守
  - 3 epochs 可能不足以充分学习开放式描述

### 5.2 能力评估

**Stage 2 (QA) — 二元判断强，精确定位弱**:

- Binary QA (82.89%)：模型对"场景中是否有某物体"的判断可靠，这是自动驾驶基础感知的核心需求
- Frame Range IoU (17.11%)：精确的时序边界定位效果较差，模型难以准确判断"从第几帧到第几帧"
- Categorical (39.22%)：多类别推理能力有限，可能是类别不均衡或类别间区分度低导致
- Overall 64.67%：综合看，模型在结构化问答上已具备一定可用性

**Stage 3 (Captioning) — 描述内容基本准确，细节不足**:

- BLEU-1 33.22：1-gram 重叠约 1/3，说明关键词（物体、动作）匹配较好
- BLEU-4 9.64：4-gram 精确匹配不足 10%，说明短语级别的表述与 ground truth 差异大
- ROUGE-L 28.04 / METEOR 26.29：长文本召回和语义匹配约 1/4，符合开放式描述任务的正常水平
- 对于 LiDAR 场景描述这类开放式生成任务，n-gram 指标本身有天花板效应——场景描述不存在唯一正确答案

### 5.3 关键发现

1. **渐进式训练有效**：Stage2 的 strong QA 能力（82.89% binary accuracy）证明 Stage1 的特征对齐 + Stage2 LoRA 策略是成功的
2. **时序定位是核心难点**：Frame Range IoU 仅 17.11%，精确的时序边界预测对整个架构仍是挑战。可能原因：
   - 预计算特征丢失了帧级细腻度（整个场景压缩为 N×768 矩阵）
   - 训练数据中 Frame Range 样本可能相对困难
3. **描述生成已具备基本能力**：BLEU-1 33.22 + METEOR 26.29 表明模型能生成语义相关的场景描述，但细节丰富度有限
4. **Stage3 训练未完全收敛**：最终 Loss 1.35 仍有下降空间，增加训练 epochs 或提升学习率可能进一步改善描述质量

### 5.4 改进建议

| 方向 | 具体措施 |
|------|---------|
| 时序定位 | 增加时序位置编码，或引入帧级 attention mask 增强时间感知 |
| 描述丰富度 | Stage3 增加 epochs（5-10），尝试 lr=5e-5，或使用更大 LLM |
| 类别理解 | 数据增强平衡类别分布，增加类别相关训练样本 |
| 评估体系 | 引入 LLM-based 评估（如 GPT-score）作为 n-gram 指标的补充 |
| 对比实验 | 消融实验验证 LoRA merge 策略的有效性（对比不 merge 直接训练 Stage3） |

---

## 6. 数据集结构

### 6.1 数据格式

采用 LLaVA 风格的 `conversations` 格式:

```json
{
    "scene_id": "001314821",
    "scene_token": "...",
    "split": "train",
    "conversations": [
        {"from": "human", "value": "<video>\nDescribe the LiDAR sequence."},
        {"from": "gpt", "value": "The sequence captures an urban street scene..."}
    ]
}
```

### 6.2 特征存储

- 每个场景的 LiDAR 特征预计算为 `.npy` 文件
- 形状: `[N, 768]` (N ≈ 100帧, 768 维 CLIP 特征)
- 存储位置: `encoders/lidarclip/b4dl/stage2_features/{scene_id}.npy`
- 训练时动态加载 (LazySupervisedDataset)

### 6.3 数据集规模

| 数据集 | 训练集 | 验证集 | 测试集 |
|--------|--------|--------|--------|
| Stage1 对齐数据 | 699 | — | — |
| Stage2 QA 数据 | 68,695 | — | 7,071 |
| Stage3 描述数据 | 63,821 | 7,710 | 8,045 |

---

## 7. 工程实践

### 7.1 训练策略
1. **渐进式训练**: Stage1 对齐 → Stage2 QA → Stage3 描述，逐步增加任务难度
2. **参数高效微调**: 全程使用 LoRA（除 Stage1 投影层），仅训练 ~0.5% 参数
3. **LoRA 合并策略**: Stage3 在 Stage2 合并权重上训练，继承问答能力
4. **ZeRO-3 + CPU offload**: 7B 模型可在 24GB 显存的消费级 GPU 上训练

### 7.2 数据策略
1. **预计算特征**: LiDARCLIP 特征提前提取为 .npy 文件，训练时直接加载
2. **场景级特征**: 整个 LiDAR 序列编码为多帧特征矩阵，保留时序信息
3. **GPT 生成标注**: 使用 GPT-4 通过 API 生成高质量的问答和描述文本
4. **Lazy Loading**: 训练时按需加载特征文件，避免一次性全载入内存

### 7.3 显存与存储
- Stage1 checkpoint: **13MB**（仅投影层权重）
- Stage2 checkpoint: **2.4GB**（LoRA adapter + non_lora_trainables）
- Stage2-merged: **13GB**（完整 Vicuna-7B + 合并后的 LoRA）
- Stage3 checkpoint: **2.4GB**（新 LoRA adapter）
- 推理时仅需 base model (13GB) + Stage1 投影层 + Stage2 LoRA + Stage3 LoRA（LoRA 动态合并）

---

## 8. METEOR 指标修复

METEOR 指标存在两个已知问题:

**问题 1: NLTK wordnet 数据未下载**

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

**问题 2: `b4dl_metrics.py` 代码 Bug（已修复）**

`meteor_score()` 要求传入 tokenized 列表（`Iterable[str]`），原代码传入原始字符串:

```python
# 修复前
meteor_scores = [meteor_score([ref], pred) for ref, pred in zip(references, predictions)]

# 修复后（reference 和 hypothesis 都需要 .split()）
meteor_scores = [meteor_score([ref.split()], pred.split()) for ref, pred in zip(references, predictions)]
```

修复后 METEOR 正常输出: **26.29**

---

## 9. 评估工具链

### 9.1 运行命令

```bash
# 一键评估 (含 Stage2 + Stage3)
cd /root/autodl-tmp/wql/mmb4dl/mllm
bash scripts/run_b4dl_eval.sh --stage3

# 分步运行 Stage3:
# Step 1: 推理
python vtimellm/eval/b4dl_eval.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    --stage3 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3 \
    --data_path ./b4dl_dataset/stage3_test.json \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --log_path ./eval_results/stage3_test_log.jsonl

# Step 2: 指标计算
python vtimellm/eval/b4dl_metrics.py \
    --log_path ./eval_results/stage3_test_log.jsonl \
    --task stage3 \
    --output ./eval_results/stage3_metrics.json
```

### 9.2 评估指标说明

| 指标 | 含义 | 适用任务 |
|------|------|---------|
| BLEU-1/2/3/4 | n-gram 精确匹配 | Stage3 描述 |
| ROUGE-L | 最长公共子序列 F1 | Stage3 描述 |
| METEOR | 同义词/词形变化匹配 | Stage3 描述 |
| Binary Accuracy/F1 | 二分类精确度 | Stage2 QA |
| Frame Range IoU | 时序范围交并比 | Stage2 QA |
| Categorical Accuracy | 多分类准确率 | Stage2 QA |

---

## 10. 总结

B4DL 成功构建了一个完整的 4D LiDAR 多模态 LLM 训练与评估框架。

**已完成的三阶段训练成果**:

1. **Stage 1（特征对齐）**: 投影层有效建立 LiDARCLIP-Vicuna 语义桥梁，Loss 从 7.18 降至 4.22
2. **Stage 2（QA 微调）**: Binary 判断精度 82.89%，Overall 64.67%，模型具备基本的场景问答能力
3. **Stage 3（描述生成）**: BLEU-1 33.22, METEOR 26.29，模型能够生成语义相关的场景描述

**核心贡献**:
- 将自动驾驶 LiDAR 数据引入 LLM 开放式问答和描述场景
- 三阶段渐进训练策略使模型逐步从特征对齐到复杂描述生成
- 完整的评估体系覆盖结构化问答和开放式描述两个维度
- Benchmark 数据集（nuScenes-B4DL）覆盖 140K+ 样本

**待探索方向**: 时序定位精度提升、描述生成丰富度增强、更大规模 LLM 的迁移实验、端到端训练（不预计算特征）的可行性。
