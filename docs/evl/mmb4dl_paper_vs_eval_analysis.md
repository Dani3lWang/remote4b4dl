# 论文 (B4DL) 与本地评估结果无法对齐的原因分析

**日期**: 2026-08-02
**分析对象**: `docs/mmb4dl-md/mmb4dl.md`（论文）vs `docs/evl/`（本地评估）

---

## 结论先行

`docs/evl/` 中的评估结果（Overall 65.63%、Binary 83.59%、mIoU 18.23）**不能**与论文 Table 3（Accuracy 0.762 / mIoU 0.311 / B@4 0.095）直接对比。两者在**测试集、模型架构、指标口径、数据规模**四个层面均不对齐，属于"同名但不同物"的比较。

---

## 一、测试集完全不同（根本原因）

| 项目 | 论文 | 本地评估 |
|------|------|----------|
| 划分方式 | nuScenes 官方 train/val 划分 | `mllm/scripts/create_splits.py` 随机 80/10/10（seed=42） |
| Test scenes | 150 个 nuScenes **val** scenes | 71 个 scenes，**全部来自论文 train 划分** |
| Test 规模 | 30,145 条 QA | 7,071 条 QA（仅 23%） |
| 场景重叠 | — | 与论文 test **重叠为 0** |

### 证据链

1. `b4dl_dataset/metadata/scene_metadata.json`：850 scenes，split 字段正确标注 700 train / 150 test ✓
2. `stage2_test.json`：71 个 unique scenes，**全部命中 metadata 的 train 集合**（in_meta_test=0, in_meta_train=71）
3. 数据文件内 `split` 字段全部为 `"train"`（数据生成于 split 修复之前，文件名与内容不符）
4. `create_splits.py` 用 `random.shuffle(scenes)` + 80/10/10 随机切分，**未使用 nuScenes 官方划分**

### 影响

- 评估的不是论文定义的 test（nuScenes val），而是从训练场景中随机抽的子集
- 即使得分高（如 Binary 83.59%），也无参考价值，因为场景分布、难度与论文 test 不同
- 无任何一篇论文结果可与之对照（论文只报了 nuScenes val 上的成绩）

---

## 二、模型架构不是论文的 B4DL 模型

| 组件 | 论文 B4DL | 本地实现（VTimeLLM 架构） |
|------|-----------|---------------------------|
| 视觉编码 | LiDARCLIP encoder 直接处理原始点云 `P_t ∈ R^{N_t×4}` | LiDAR-CLIP 预提取 stage2 特征 `(N_frames, 768)` |
| 对齐器 | LiDAR Aligner `f_p`（linear） | mm_projector（linear） |
| **Metatoken** `<meta>` | ✅ 传感器元数据文本（速度/方向/位移/加速度，首尾帧拼接） | ❌ **未实现** |
| **`<4DLiDAR>` token** | ✅ QA 输入前置，提示 4D LiDAR 推理 | ❌ **未实现** |
| LoRA | ✅（4D 阶段） | ✅（Stage2） |
| LLM | Vicuna-7b-v1.5（冻结） | Vicuna-7b-v1.5（冻结） |

### Metatoken 的重要性（论文 Table 4 消融）

| 配置 | Accuracy | **mIoU** | B@4 |
|------|----------|----------|-----|
| 有 HA，**无 Metatoken** | 0.756 | **0.218** | 0.067 |
| 有 HA，有 Metatoken（完整） | 0.762 | **0.311** | 0.095 |

去掉 Metatoken 导致 **mIoU 下降 30%**（0.311 → 0.218），因为模型无法区分"物体运动"和"自车运动"。本地实现缺此模块，时间定位能力天然受限——这与本地 mIoU 仅 18.23（即 0.1823）的弱点吻合。

### 本地模型实际对应论文中的谁？

本地模型 ≈ **VTimeLLM baseline**（论文 Table 3）：

| | Accuracy | mIoU |
|------|----------|------|
| VTimeLLM [12]（论文） | 0.694 | 0.160 |
| B4DL（论文） | 0.762 | 0.311 |

---

## 三、评估指标口径不一致

### 论文（Section 3.1 / 5.1）

- **Simple Tasks**：Accuracy = mean(Existence acc, Binary QA acc)，**两个任务分别计算**
- **Time Grounding**：mIoU（0–1 刻度）
- **Complex Tasks**：BLEU-4, METEOR, ROUGE-L, BERTScore, GPT-4o score（0–100）

### 本地评估（`b4dl_metrics.py`）

- 数据无 `task` 标签 → 只能按 **GT 答案格式** 分类：
  - `yes`/`no` → binary（5179 条）
  - `from frame X to frame Y` → frame_range（1405 条）
  - 其他 → categorical（487 条）
- **Existence 与 Binary QA 无法区分**，混在一个 binary 组里算准确率
- **categorical 组实际主要是 Existence 类问题**（"Which object was near the ego vehicle?" 答案为类别词），却用 exact match 而非文本生成指标
- **mIoU 刻度 ×100**（18.23 对应 0.1823），且公式带 +1 帧计数，与论文标准 mIoU 可能不同
- **未计算** BLEU/METEOR/ROUGE-L/BERTScore/GPT score——三个复杂任务（Description/Temporal/Comprehensive）的论文指标完全缺失

---

## 四、数据规模与任务覆盖不足

| 项目 | 论文 | 本地 |
|------|------|------|
| Train 总量 | 148,271 | 54,901（stage2_train.json） |
| Test 总量 | 30,145 | 7,071 |
| 每 sequence 样本 | 40（5/10/5/5/5/10 六任务） | 不固定，任务分布未知 |
| Test 中复杂任务占比 | **53.3%**（16,067/30,145） | **6.9%**（487/7,071） |

- 本地生成的测试数据中描述类任务样本严重缺失（487 条），无法支撑 BLEU/METEOR 等文本指标的计算
- 数据生成流程（`datageneration/`）虽含 `gt_description`（HA 参数），但生成的数据集任务覆盖不完整

---

## 五、原因总结表

| # | 原因 | 严重程度 | 修复方向 |
|---|------|----------|----------|
| 1 | test 集不是 nuScenes val（随机划分，来自 train scenes） | 🔴 致命 | 用 nuScenes 官方 val 150 scenes 重建 test，用 `split_dataset.py --fix_split` 保证元数据正确 |
| 2 | 模型缺 Metatoken + `<4DLiDAR>` token | 🔴 致命 | 实现论文 Section 4.1 的 Metatoken 模块 |
| 3 | 指标口径不同（任务混算、刻度、缺文本指标） | 🟠 严重 | 数据加 task 标签；区分 Existence/Binary；实现 BLEU/METEOR/ROUGE-L/BERTScore/GPT |
| 4 | 数据规模与复杂任务覆盖不足 | 🟠 严重 | 扩充数据生成（尤其 description/temporal/comprehensive） |
| 5 | mIoU 计算口径（+1 帧计数、×100 刻度） | 🟡 中等 | 对齐论文公式 |

---

## 六、验证建议（如需对齐论文结果）

1. **重建 test 集**：以 metadata 中 split="test"（150 scenes）为准过滤生成数据，产出约 30k 条
2. **补 task 标签**：从 `datageneration/` 的生成流程中保留任务类型到最终 JSON
3. **实现 Metatoken**：读取 nuScenes ego 位姿/速度，按论文格式转文本，前置 `<4DLiDAR><meta>` 到输入
4. **用论文指标脚本**：`mllm/evaluation/evaluate_model.py`（已实现全部论文指标）替代或补充 `b4dl_metrics.py`
5. **重训后评估**：Metatoken 需要重新训练 Stage2 才能生效
