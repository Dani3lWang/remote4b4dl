# B4DL 模型真正训练方案与时序检测能力分析

**日期**: 2026-08-02
**目标**: 分析时序检测能力弱的根本原因，设计包含 Metatoken 注入的完整训练方案，规划六类任务测试方案

---

## 一、时序检测能力弱的根本原因

### 1.1 原因总览

| # | 原因 | 严重程度 | 论文依据 | 对应 mIoU 损失 |
|---|------|----------|----------|---------------|
| 1 | **Metatoken 缺失** | 🔴 致命 | Table 4 消融 | -0.093 (0.311→0.218) |
| 2 | **`<4DLiDAR>` token 缺失** | 🔴 严重 | Section 4.2 | 引导注意力缺失 |
| 3 | **帧特征无位置编码** | 🟠 严重 | 架构分析 | 无序帧特征，模型无法区分 t=0 和 t=8 |
| 4 | **训练数据无 `meta` 字段** | 🟠 严重 | 代码分析 | `stage2_train.json` 54,901 条中 0 条含 `meta` |
| 5 | **训练数据任务覆盖不全** | 🟠 严重 | Table 2 | 缺 4/6 任务类型，Time Grounding 无独立 prompt |
| 6 | **时间表示被压缩为百分位数** | 🟡 中等 | dataset.py | 丢失绝对帧间隔信息 |
| 7 | **模型实际是 VTimeLLM baseline** | 🟡 中等 | Table 3 | 本地 mIoU 0.1823 ≈ 论文 VTimeLLM 0.160 |

### 1.2 详细分析

#### 原因 1: Metatoken 缺失（最关键）

**论文设计** (Section 4.1, Appendix C, Figure 6)：
- `<meta>` token 前置于 QA 输入，后接自车传感器元数据的文本描述
- 元数据来源：nuScenes 的 ego_pose（translation/rotation/velocity）
- 文本格式：相对方向（forward/backward/left/right）、相对速度、相对位移、加速度
- 只包含 QA 所涉及的**首帧和尾帧**的描述（避免冗余），中间用连接词拼接
- 示例：`"The ego vehicle moved forward 3 meters between frame 12 and frame 20 at a speed of 5 m/s"`

**为什么重要**：
- 论文消融实验 (Table 4)：移除 Metatoken 导致 mIoU 从 0.311 降至 0.218（**下降 30%**）
- 原因：无 Metatoken 时，模型无法区分"物体在运动"和"自车在运动"
- 例如：前方车辆在 LiDAR 中变近，可能是车辆倒车，也可能是自车加速——只有 Metatoken 提供自车运动信息才能判别

**本地状态**：
- 代码中完全不存在 Metatoken 实现
- 训练数据 54,901 条中 `meta` 字段 = 0 条
- `<4DLiDAR>` token 存在于数据生成的 `utils.py`（仅首轮 QA 前置），但未注册为 special token

#### 原因 2: 帧特征无序

**当前架构**：
```
LiDAR-CLIP → 每帧独立提取 cls embedding → concat 为 (N_frames, 768)
```
- 每帧的 cls embedding 是独立提取的，**无位置编码 (positional encoding)**
- concat 后直接通过 `mm_projector` 投影，替换 `<video>` token
- 模型看到的是一组 **无序的帧特征包**（bag of frame features）

**后果**：
- 模型学会了输出格式 `"from frame 000 to frame 008"`（整个序列的固定范围）
- 但**无法真正定位事件发生在哪几帧**
- 评估日志确认：模型倾向于输出覆盖全序列的范围

#### 原因 3: 时间信息压缩

`dataset.py` 第 385-398 行的 metatoken 转换（当数据有 `meta` 字段时）：
```python
def convert(duration, x):
    x = x / duration * 100      # 归一化到百分位
    x = str(min(round(x), 99))
    if len(x) == 1:
        x = "0" + x              # 补零为两位数字符串
    return x
```
- 将时间戳压缩为 0-99 的百分位数字
- 丢失了绝对帧间隔（2Hz = 0.5s/frame）、帧数信息
- 模型无法从 "15" 推断出这是第几帧

#### 原因 4: 模型实际身份

本地模型 ≈ 论文的 **VTimeLLM baseline**（非完整 B4DL）：

| 组件 | 论文 B4DL | 本地实现 |
|------|-----------|----------|
| LiDAR 编码 | `E_L` 端到端处理原始点云 | LiDAR-CLIP 预提取 .npy |
| Metatoken `<meta>` | ✅ | ❌ |
| `<4DLiDAR>` token | ✅ 所有 QA 前置 | 仅数据生成侧部分使用 |
| LoRA | ✅ | ✅ |
| 训练数据 | 178k (六任务) | 55k (主要为 binary + frame_range) |

论文 Table 3 中 VTimeLLM 的 mIoU = 0.160，本地 mIoU = 0.1823——**基本吻合**。

---

## 二、真正的 B4DL 模型训练方案

### 2.1 整体架构与数据流

```
nuScenes ego_pose 数据 ──→ Metatoken 文本生成 ──┐
                                                  ├──→ 训练数据注入
LiDAR-CLIP 提取特征 (N_frames, 768) ──────────────┤
                                                  │
GPT-4o 生成六任务 QA ────────────────────────────┘
                                                  │
                                                  ▼
                              tokenizer 编码 → VTimeLLM 前向传播
```

**关键变化**：
- 训练数据中每条样本增加 `meta` 字段（自车运动元数据）
- 训练数据中每条样本增加 `<4DLiDAR>` token 前置
- `dataset.py` 的 metatoken 转换逻辑需要改进（见下）

### 2.2 实现 Metatoken（优先级最高）

#### Step 1: 从 nuScenes 提取 ego_pose 数据

```python
# 伪代码：为每个 sequence 提取首帧和尾帧的 ego_pose
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_root)

def extract_ego_metadata(scene_token, start_frame, end_frame):
    """提取首帧和尾帧的自车运动元数据，转为文本描述"""
    sample_first = nusc.get('sample', first_sample_token)
    sample_last = nusc.get('sample', last_sample_token)
    
    # 读取 ego_pose (lidar_top 传感器的标定位姿)
    ego_first = nusc.get('ego_pose', sample_first['data']['LIDAR_TOP'])
    ego_last = nusc.get('ego_pose', sample_last['data']['LIDAR_TOP'])
    
    # 计算相对变化
    dx = ego_last['translation'][0] - ego_first['translation'][0]
    dy = ego_last['translation'][1] - ego_first['translation'][1]
    dt = (end_frame - start_frame) * 0.5  # 2Hz → 0.5s/frame
    
    # 方向判断
    if abs(dx) > abs(dy):
        direction = "forward" if dx > 0 else "backward"
    else:
        direction = "right" if dy > 0 else "left"
    
    distance = np.sqrt(dx**2 + dy**2)
    speed = distance / dt
    
    return f"The ego vehicle moved {direction} {distance:.1f} meters between frame {start_frame} and frame {end_frame} at a speed of {speed:.1f} m/s."
```

#### Step 2: 训练数据注入

修改 `dataset.py` 的数据预处理，在 human value 前置 `<4DLiDAR><meta>`：

```python
# 当前格式（无 metatoken）:
"<video>\nWas a bus present in front of the ego vehicle between frame 0 and frame 8?"

# 目标格式（有 metatoken）:
"<4DLiDAR><meta>The ego vehicle moved forward 12.3 meters between frame 0 and frame 8 at a speed of 3.1 m/s. The ego vehicle continued moving forward, ending at a speed of 4.2 m/s.\n<video>\nWas a bus present in front of the ego vehicle between frame 0 and frame 8?"
```

#### Step 3: 注册 `<4DLiDAR>` 和 `<meta>` 为 special tokens

在 `train.py` 的 tokenizer 初始化后：
```python
special_tokens_dict = {'additional_special_tokens': ['<4DLiDAR>', '<meta>']}
smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
```

**注意**：`<video>` 已经是特殊 token（`IMAGE_TOKEN_INDEX = -200`），但 `<4DLiDAR>` 和 `<meta>` 是**普通文本 token**（不需要在 embedding 层特殊处理），只需要 tokenizer 将其作为一个完整 token 而非拆分为子词。

### 2.3 改进时间编码

**方案 A（保守）**：保留当前百分位数字方案，但增加帧率信息

在 metatoken 文本中显式包含帧率：
```
"The sequence covers frame 000 to frame 008 at 2Hz (0.5s per frame)."
```

**方案 B（推荐）**：引入 VTimeLLM 的边界 token 机制

VTimeLLM 原始设计中有 `<s0>`/`<e0>` 边界 token 系统——模型学习预测时间边界的特殊标记，而非生成纯文本数字。这需要在：
1. tokenizer 中添加 `<s0>`...`<sN>` 和 `<e0>`...`<eN>` 标记
2. 训练数据中将 "from frame 000 to frame 008" 替换为 "from `<s0>` to `<e0>`"
3. 推理时从输出中解析边界 token 的位置

### 2.4 完整的训练数据准备流程

```
Step 0: 修复 metadata split（已完成，scene_metadata.json 已有 700 train / 150 test）

Step 1: 从 nuScenes 提取所有 5100 个 sequence 的 ego_pose 元数据
        → 生成 meta_descriptions.json (scene_id → 首尾帧元数据文本)

Step 2: 运行完整六任务数据生成 (GPT-4o API)
        cd datageneration
        for task in existence binary temporal comprehensive; do
            python generate_dataset.py --task $task ...
        done
        # 注意：需要新增 time_grounding 的独立 prompt（当前 prompts.py 缺失）
        # temporal → temporal_understanding (当前代码用 "temporal" 而非 "temporal_understanding")

Step 3: 注入 metatoken 到训练数据
        python scripts/inject_metatoken.py \
            --data_dir datageneration/data/generated_dataset/ \
            --meta_file meta_descriptions.json \
            --output_dir mllm/b4dl_dataset/

Step 4: 重建 train/test 划分（按 nuScenes 官方 700/150 划分）
        python mllm/evaluation/build_test_split.py \
            --metadata b4dl_dataset/metadata/scene_metadata.json \
            --qa_dir ... --output test_qa.json

Step 5: 验证数据质量
        - 每条 QA 包含 task 标签
        - 每条 QA 包含 meta 字段（首尾帧元数据）
        - 每条 QA 以 <4DLiDAR><meta> 开头
        - test 集来自 nuScenes val (150 scenes)，train 集来自 train (700 scenes)
```

### 2.5 重训 Stage2

```bash
cd mllm

# 1. 确保 tokenizer 包含 <4DLiDAR> 和 <meta> token
# 2. 使用完整六任务 + metatoken 数据
bash scripts/stage2.sh \
    --s2_data ./b4dl_dataset/stage2_full.json \
    --s2_feat ../encoders/lidarclip/b4dl/stage2_features \
    --num_train_epochs 3
```

训练配置：
- 预计训练量：148k 条（与论文持平）
- GPU：RTX 5090 32GB
- 预计时间：1-2 天
- 关键超参：learning_rate=2e-5, lora_r=64, lora_alpha=16, model_max_length=2048 (需增大以容纳 metatoken 文本)

### 2.6 推理改动

`inference.py` 中需要在 query 前注入 Metatoken：
```python
def inference_with_metatoken(model, features, query, meta_text, tokenizer):
    full_query = f"<4DLiDAR><meta>{meta_text}\n<video>\n{query}"
    # 其余逻辑不变
```

---

## 三、六类问题测试与检验方案

### 3.1 论文评估框架

| 任务 | 类型 | 指标 | 测试样本数 | 评估方法 |
|------|------|------|-----------|----------|
| Existence | Simple | Accuracy | 3,770 | 与 GT 精确匹配 |
| Binary QA | Simple | Accuracy | 7,525 | 与 GT 精确匹配 (Yes/No) |
| Time Grounding | Simple | mIoU | 2,783 | 预测区间与 GT 区间的 IoU |
| Description | Complex | BLEU-4/METEOR/ROUGE-L/BERTScore/GPT | 3,770 | NLG 指标 + GPT 评分 |
| Temporal Understanding | Complex | 同上 | 4,757 | 同上 |
| Comprehensive Reasoning | Complex | 同上 | 7,540 | 同上 |

**聚合规则** (Section 5.1)：
- Simple Tasks Accuracy = mean(Existence accuracy, Binary QA accuracy)
- Time Grounding mIoU = 独立计算
- Complex Tasks 四个文本指标 = 三个复杂任务分别计算后取 mean
- GPT Score = 三个复杂任务分别计算后取 mean

### 3.2 六类任务的测试数据准备

#### 当前缺失情况

| 任务 | prompts.py 是否有 prompt | generate_dataset.py 是否有生成方法 | 备注 |
|------|--------------------------|-----------------------------------|------|
| Existence | ✅ | ✅ | 正常 |
| Binary QA | ✅ | ✅ | 正常 |
| **Time Grounding** | ❌ 缺失 | ❌ 缺失 | **需要新增！** |
| Description | ✅ | ✅ | 正常 |
| Temporal Understanding | ✅ | ✅ (通过 `temporal`) | prompt 生成 10 条，论文要求 5 条 |
| Comprehensive | ✅ | ✅ | 正常 |

#### Time Grounding prompt 设计

```python
def generate_time_grounding_dataset_prompt(self, front_description, back_description, gt_caption, start_index, end_index):
    TIME_GROUNDING_PROMPT = f"""...
    Generate 5 time grounding question-answer pairs. 
    Each question should ask WHEN a specific event occurred.
    The answer MUST be strictly in the format 'from frame XXX to frame XXX' 
    (three-digit zero-padded frame numbers).
    
    Example:
    Q: When did the pedestrian cross the road?
    A: from frame 004 to frame 008
    
    Q: During which frames was the truck visible in the front view?
    A: from frame 000 to frame 010
    """
```

### 3.3 测试流程

#### Phase 1: 数据完整性检查

```bash
# 检查每个任务的数据量和格式
python mllm/evaluation/check_data.py --data_path b4dl_dataset/stage2_test.json
# 输出: 各任务样本数、答案格式是否规范、meta 字段覆盖率
```

检测项目：
1. 每个 sequence 是否生成了完整的 40 条（5+10+5+5+5+10）
2. 答案格式校验：Time Grounding → `from frame \d{3} to frame \d{3}$`
3. Existence/Binary → `Yes|No` 或类别词
4. meta 字段存在且格式正确（首尾帧元数据文本）

#### Phase 2: 分任务评估

```bash
# 使用 evaluate_model.py（已实现全部论文指标）
python mllm/evaluation/evaluate_model.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    --data_path ./b4dl_dataset/stage2_test.json \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --use_gpt --gpt_api_key $OPENAI_API_KEY \
    --output_dir ./eval_results/
```

#### Phase 3: 消融实验

验证各组件的贡献（对应论文 Table 4）：

| 实验 | Metatoken | `<4DLiDAR>` | HA (GT caption) | 预期 mIoU |
|------|-----------|-------------|-----------------|-----------|
| 基线 (当前) | ❌ | ❌ | 部分 | ~0.18 |
| +Metatoken | ✅ | ❌ | 部分 | ~0.22 |
| +`<4DLiDAR>` | ❌ | ✅ | 部分 | ~0.20 |
| 完整模型 | ✅ | ✅ | 部分 | ~0.25-0.31 |

### 3.4 各任务的诊断方法

#### Existence / Binary QA
- **预期表现**：较好（无需时序推理，纯空间识别）
- **诊断**：按类别分组计算准确率，检查是否有特定类别始终预测错误
- **基线**：论文 0.762，当前本地 Binary 约 0.83（但测试集不同，不可直接对比）

#### Time Grounding（最薄弱环节）
- **预期表现**：差→有明显改善（加 Metatoken 后）
- **诊断**：
  - mIoU 分布直方图（检查是否大量为 0 或大量接近 1）
  - 按帧范围长度分组（短范围 (≤3帧) vs 长范围 (≥6帧)）
  - 按是否有自车运动分组（检查 Metatoken 是否真正提供了运动信息）
  - 预测偏差分析：模型倾向于预测全范围 ("from 000 to 008") 还是随机猜测

#### Description / Temporal Understanding / Comprehensive
- **预期表现**：中等
- **诊断**：
  - 逐指标分析（BLEU-4 通常很低因为生成文本长）
  - BERTScore 和 GPT Score 更能反映语义质量
  - 检查是否包含时序信息（"between frame X and Y"）

### 3.5 测试环境与自动化

```bash
# 完整的评估脚本
cd mllm

# 1. 分任务评估
for task in existence binary_qa time_grounding description temporal_understanding comprehensive; do
    python evaluation/evaluate_model.py \
        --task $task \
        --data_path ./b4dl_dataset/stage2_test.json \
        --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
        --output_dir ./eval_results/$task/
done

# 2. 汇总指标
python evaluation/aggregate_results.py --results_dir ./eval_results/

# 3. 生成对比表（与论文 Table 3）
python evaluation/compare_with_paper.py --results_dir ./eval_results/
```

---

## 四、执行计划

### 优先级排序

```
P0 (阻塞项):
  ├── B1: 实现 Metatoken（数据 + 模型）
  ├── B2: 实现 <4DLiDAR> token
  └── A2: 补全 Time Grounding prompt + 生成

P1 (数据项):
  ├── A1: 完整六任务数据生成（需要 GPT-4o API，成本 ~$100-200）
  ├── A3: 按 nuScenes 官方 700/150 重建 train/test 划分
  └── A4: 数据加 task 标签

P2 (验证项):
  ├── C2: 重训 Stage2（依赖 P0+P1 完成）
  ├── D1: 分任务评估
  └── D2: 消融实验

P3 (优化项):
  ├── 改进时间编码（边界 token）
  ├── 帧位置编码
  └── GPT Score 评估
```

### 建议节奏

| 天数 | 任务 | 产出 |
|------|------|------|
| Day 1-2 | B1+B2 实现 + A2 prompt | 代码完成，可生成 Metatoken |
| Day 3-5 | A1 完整数据生成（GPT-4o API 并行跑） | 完整 178k 数据集 |
| Day 3-4 | A3+A4 数据划分配置 | train/test 正确划分 |
| Day 6-7 | C2 重训 Stage2 | 完整 B4DL 模型 |
| Day 8 | D1+D2 评估 + 消融 | 评估报告 |

---

## 五、关键代码改动清单

### 必须修改的文件

| 文件 | 改动内容 |
|------|----------|
| `datageneration/prompts.py` | 新增 `generate_time_grounding_dataset_prompt()` |
| `datageneration/generate_dataset.py` | 新增 `time_grounding` case；修正 `temporal` → `temporal_understanding`；写入 `task` 字段 |
| `mllm/vtimellm/train/dataset.py` | 改进 metatoken 转换逻辑（保留帧率信息）；默认注入 `<4DLiDAR><meta>` |
| `mllm/vtimellm/train/train.py` | 注册 `<4DLiDAR>` 和 `<meta>` 为 special tokens；增大 model_max_length |
| `mllm/vtimellm/inference.py` | 支持 Metatoken 注入 |
| `mllm/vtimellm/eval/b4dl_eval.py` | 按任务类型分路评估，注入 metatoken |

### 新增文件

| 文件 | 用途 |
|------|------|
| `scripts/extract_ego_metadata.py` | 从 nuScenes 提取自车运动元数据 |
| `scripts/inject_metatoken.py` | 将元数据注入训练数据 |
| `datageneration/data/ego_meta/` | 存储提取的元数据 |

---

## 六、参考：论文完整评估框架

```
训练数据 (178,416 QA pairs, 850 scenes, 700 train / 150 test)
  │
  ├── Existence (22,315) ──── Accuracy (exact match)
  ├── Binary QA (44,551) ──── Accuracy (exact match)
  │                           └── Simple Tasks Accuracy = mean(两者)
  │
  ├── Time Grounding (15,907) ── mIoU (0-1 scale)
  │
  ├── Description (22,310) ──┐
  ├── Temporal (28,713) ─────┤ BLEU-4, METEOR, ROUGE-L, BERTScore, GPT Score
  └── Comprehensive (44,620) ┘ └── Complex Tasks Metrics = mean(三任务)
```
