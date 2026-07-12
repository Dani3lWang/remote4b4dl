# Plan1 学习结果与分析报告

> 基于 plan1.md 执行计划，逐项分析当前代码状态、发现的问题和学习结论。
> 分析日期：2026-07-12

---

## 第一阶段：清理代码层面的"阻塞问题"

### 任务 1.1：mm_projector 维度问题 ✅ 已修复

**状态**：问题已修复，无需再次修改。

**验证过程**：
1. 检查 `mllm/vtimellm/model/vtimellm_arch.py` 第 12 行，代码为 `nn.Linear(768, self.config.hidden_size)`，使用的是 768 而非 128。
2. 实际加载 Stage1 checkpoint 的 `mm_projector.bin`，形状确认为 `[4096, 768]`，与 CLIP ViT-L/14 输出维度（768）和 Vicuna-7B hidden_size（4096）一致。
3. 特征文件由 LiDAR-CLIP 的 AttentionPool2d 输出，`embed_dim=768`（由 CLIP ViT-L/14 决定），与 mm_projector 输入匹配。

**关键学习点**：
- AttentionPool2d 的 `output_dim` 默认等于 `embed_dim`，由初始化时的 `clip_model.visual.output_dim`（=768）决定
- SST backbone 输出经过 `in_proj(conv_out→768)` → `AttentionPool` → `c_proj(768→768)`，最终得到 768 维特征
- 此问题在 git commit `79c578f` 中已经修复

**验收标准达成情况**：
- [x] Stage1 训练已正常完成（5 steps，loss 从 7.21 降至 4.22）
- [x] 无维度不匹配错误
- [x] mm_projector.bin 形状正确

---

### 任务 1.2：三阶段学习对应关系 ✅ 已梳理

**状态**：已全部理清，model.py 不存在于当前仓库。

**核心发现**：

#### 1. 论文 B4DL Model 的两阶段训练 vs 代码实现的三阶段

| 维度 | 论文 B4DL Model | 当前代码实现 (VTimeLLM-based) |
|------|----------------|---------------------------|
| Stage 1 | 3D LiDAR Understanding：训练 LiDAR Aligner fp | `stage1.sh`：训练 mm_projector (Linear 768→4096) |
| Stage 2 | 4D LiDAR Understanding：冻结 fp，加 LoRA | `stage2.sh`：冻结 mm_projector，加 LoRA (r=64) |
| Stage 3 | 不存在 | `stage3.sh`：merge Stage2 LoRA → 重新加 LoRA → 训练 |

#### 2. model.py 分析

plan1 中引用的大量 `model.py` 代码（含 Qformer、VAT、view_masks、angle_pos_embd 等）**不存在于当前仓库**。这些代码来自原 LiDAR-LLM 论文的官方实现，是独立项目。

**当前 B4DL 架构**与 LiDAR-LLM 的根本差异：
- **LiDAR-LLM**：使用 Qformer + VAT（视角感知Token化）+ BEV 特征图 + angle embedding
- **B4DL**：使用 SST encoder + AttentionPool → CLIP 特征 → mm_projector → LLM（基于 VTimeLLM 架构）

#### 3. Stage 3 的设计理由

Stage 3 先 merge Stage 2 LoRA 再添加新 LoRA 的原因：
- Stage 2 的 LoRA 权重已经学到了时序理解能力
- Merge 后作为新的基座模型，再添加全新的 LoRA 进行二次微调
- 这种方式可以让 Stage 3 在新的数据分布上学习，同时保留 Stage 2 的知识
- 依赖 peft 0.4.0 的 `merge_and_unload()` API（旧式 API）

#### 4. 参数训练对照表

| 阶段 | 可训练参数 | 冻结参数 | 数据 | 学习率 |
|------|-----------|---------|------|--------|
| Stage 1 | mm_projector (Linear 768→4096) | LLM、SST encoder | stage1_lidarllm_mm.json (plain 格式) | 1e-3 |
| Stage 2 | LoRA (全部 Linear 层, r=64) | mm_projector、LLM base | stage2_conversations.json (v1 对话格式) | 1e-4 |
| Stage 3 | 新 LoRA (r=64) | mm_projector、LLM base (已含 merged Stage2 LoRA) | stage3_train.json | 2e-5 |

**验收标准达成情况**：
- [x] 能清楚解释每个阶段训练哪些参数
- [x] 能解释 Stage 3 的 merge + re-add 机制
- [x] 明确了 B4DL 当前三阶段与论文两阶段的对应关系（代码实现了比论文更细粒度的训练策略）

---

## 第二阶段：诊断 B4DL 时序理解不足的根因

### 任务 2.1：控制变量法排查

**状态**：已分析代码，得出初步结论。

#### 时间戳转换机制分析

`dataset.py` 第 385-398 行的时间戳转换逻辑：
```python
def convert(duration, x):
    x = x / duration * 100    # 归一化到 [0, 100]
    x = str(min(round(x), 99))  # 截断到 99
    if len(x) == 1:
        x = "0" + x            # 补零为两位数字符串
    return x
```

**问题**：时间信息被压缩为 00-99 的两位数字符串。这会丢失：
- 帧间绝对时间间隔（如 frame 03 到 frame 07 是 2 秒 vs 4 秒）
- 帧率信息
- 无法区分"帧内时间"和"帧间时间"

**对比论文**：论文使用 Metatoken 机制提供 ego vehicle 的运动元数据（相对方向、位置、速度、加速度），这比二维字符串编码更丰富。

#### 数据质量初步判断

从代码分析：
1. `generate_dataset.py` 只使用 front/back 描述生成 QA，GPT 无法直接看到原始 LiDAR 点云
2. 生成的 temporal QA 可能过度依赖文本描述中的时序信息，而非真正的多帧推理需求
3. 训练数据为 178k QA pairs，但简单任务（Existence/Binary/Time Grounding）占比较大

#### 训练轮次分析

当前配置：
- Stage 2：2 epochs（loss 从初始降至 ~0.28）
- Stage 3：3 epochs（loss 从 ~1.4 降至 ~1.35，下降缓慢）

Stage 2 的 loss 曲线（最后 5 步）：
- Step 1068: 0.3077 → Step 1072: 0.2813
- Loss 仍在下降但趋于平缓

Stage 3 的 loss 曲线（最后 5 步）：
- Step 743: 1.3944 → Step 747: 1.3524
- Higher baseline loss（新 LoRA 初始化导致），下降缓慢

**初步结论**：
- 根因可能是多方面的：数据质量（temporal QA 对多帧推理的依赖不足）+ 训练轮次（2 epoch 可能不够）+ 时序编码方式（时间戳压缩）
- 需要进一步量化实验验证

---

### 任务 2.2：VTimeLLM 的细粒度时序建模

**状态**：分析了 VTimeLLM 在 B4DL 中的使用方式。

#### B4DL 如何使用 VTimeLLM

当前代码实际上直接复用了 VTimeLLM 的架构（`vtimellm_arch.py`、`vtimellm_llama.py`），但做了以下适配：
1. 将视频帧替换为多帧 LiDAR 特征：`<video>` token → N 帧 LiDAR embedding
2. 去除了 VTimeLLM 的时间边界检测头（boundary head），仅保留 QA 能力
3. 时间信息通过 `<meta>` token 中的文本描述注入

#### 三个低成本改进点

| 改进点 | 实现位置 | 复杂度 | 预期收益 |
|--------|---------|--------|---------|
| 1. 帧间位置编码：为每帧 LiDAR embedding 添加可学习的时间位置编码 | `vtimellm_arch.py` 的 `prepare_inputs_labels_for_multimodal()` | 低（~10行代码） | 让模型感知帧间顺序和间隔 |
| 2. 时间戳保留为实数而非两位字符串 | `dataset.py` 的 `convert()` 函数 | 低（改3行） | 保留精确的时序信息 |
| 3. 增加时间 grounding 的辅助 loss head | `vtimellm_arch.py` + 新增 head | 中（~50行代码） | 加强模型的时序定位能力 |

---

## 第三阶段：数据生成优化

### 任务 3.1：增强 temporal QA

**状态**：分析了现有 prompt 和数据生成流程。

#### 当前 prompt 结构
- 使用 front/back 两组描述（各含 3 张相机图）作为 GPT 的输入
- 系统 prompt 为 "make simple QnA pairs about the entire scene"
- 没有显式要求生成"需要多帧推理"的问题

#### 关键发现
数据生成的两步流程存在信息瓶颈：
1. Step 1：6 相机图 → GPT → 文本场景描述（丢失了大量 3D 几何信息）
2. Step 2：文本描述 → GPT → QA pairs（进一步压缩信息）

**改进方向**：
- 在 prompt 中显式要求生成需要跨帧比较的问题
- 增加后处理过滤：移除答案可从单帧推断的 QA
- 考虑在 prompt 中直接提供帧索引和对应的时间戳

---

## 第四阶段：VAT 空间感知机制

### 任务 4.1 & 4.2：VAT 移植

**状态**：model.py 在当前仓库中不存在，VAT 源码缺失。

**关键发现**：
1. plan1 中引用的 `model.py`（含 Qformer、VAT、view_masks、angle_pos_embd）来自 LiDAR-LLM 原始仓库，不在当前 B4DL 代码中
2. 当前 B4DL 使用 SST encoder + AttentionPool → 768 维 CLIP 特征，这是一种**全局池化**方式，丢失了 BEV 空间结构
3. 要引入 VAT，需要：
   - 从 LiDAR-LLM 仓库获取 VAT 源码
   - 修改 SST encoder 使 `no_pooling=True` 输出 BEV 特征图
   - 将 VAT 插入在 Pooling 之前

**建议**：
- 方案 C（快速验证）：在现有 `[N_frames, 768]` 序列上为每帧学习"视角嵌入"——改动最小
- 方案 A（完整实现）：修改特征提取流程，保留 BEV 图 → 加 VAT → 再池化

---

## 第五阶段：对比实验框架

### 当前实验组状态

基于现有的训练 checkpoint，可以进行以下评估：

| 实验组 | 状态 | Checkpoint 路径 |
|--------|------|----------------|
| A (Stage1+Stage2 baseline) | ✅ 已完成 | `checkpoints/vtimellm-vicuna-v1-5-7b-stage2` |
| A+Stage3 | ✅ 已完成 | `checkpoints/vtimellm-vicuna-v1-5-7b-stage3` |
| B (增加 epoch) | ❌ 未执行 | — |
| C (增强 temporal QA) | ❌ 未执行 | — |
| D (VAT 移植) | ❌ 未执行 | — |

---

## 第六阶段：核心发现与下一步方向

### 1. 已确认的事实

1. **mm_projector 维度问题已在 git commit `79c578f` 中修复**，当前使用 768→4096
2. **model.py（含 VAT）不存在于本仓库**，这是 LiDAR-LLM 原项目的代码
3. **当前代码实现了三阶段训练**，比论文的两阶段更细粒度
4. **Stage1 仅训练了 5 step**（数据量很小，仅 1 epoch），Stage2 训练了 2 epochs（1072 steps），Stage3 训练了 3 epochs（747 steps）
5. **RTX 5090 环境** (CUDA 13/PyTorch 2.8.0)，原始脚本针对 CUDA 12.4 + PyTorch 2.5.1 设计

### 2. 推断和假设

1. **时序理解不足的根本原因**很可能是数据质量问题——生成的 QA 中真正需要多帧推理的比例可能不高
2. **Stage3 的 loss 高于 Stage2**（1.35 vs 0.28），这表明 Stage3 使用了不同的数据分布或更困难的任务
3. **VAT 在当前架构中难以直接移植**，因为 SST+AttentionPool 设计为输出全局特征向量，而非空间特征图

### 3. 下一步建议

- **短期**：运行现有模型的评估脚本，获取各任务（existence/description/temporal/comprehensive/binary）的 baseline 分数
- **中期**：改进数据生成 prompt → 重新生成更高质量 temporal QA → 重新训练
- **长期**：评估是否需要引入 VAT 类的空间感知机制，或者通过改进时序编码方式来提升性能

---

## 附录：实际代码与 Plan1 的文档对照

| Plan1 引用的文件 | 当前是否存在 | 备注 |
|-----------------|------------|------|
| `model.py` (Qformer/VAT) | ❌ 不存在 | 来自 LiDAR-LLM 原仓库 |
| `mllm/vtimellm/model/vtimellm_arch.py` | ✅ 存在，已修复 | mm_projector=768 |
| `encoders/lidarclip/extract_pc_features.py` | ✅ 存在 | 特征提取正常 |
| `encoders/lidarclip/lidarclip/model/sst.py` | ✅ 存在 | SST encoder |
| `mllm/vtimellm/train/train.py` | ✅ 存在 | 三阶段训练 |
| `mllm/scripts/stage1.sh` | ✅ 存在 | Stage1 训练脚本 |
| `mllm/scripts/stage2.sh` | ✅ 存在 | Stage2 训练脚本 |
| `mllm/scripts/stage3.sh` | ✅ 存在 | Stage3 训练脚本 |
| `datageneration/generate_dataset.py` | ✅ 存在 | 数据生成 |
| `datageneration/prompts.py` | ✅ 存在 | prompt 模板 |
