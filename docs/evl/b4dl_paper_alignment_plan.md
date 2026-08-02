# B4DL 论文对齐差距清单与处理方法

**日期**: 2026-08-02
**目标**: 让本地实现（模型 + 数据 + 评估）与论文 `docs/mmb4dl-md/mmb4dl.md` 对齐，使评估结果可对比 Table 3

---

## 差距总览

| 模块 | 论文要求 | 本地现状 | 差距倍数 |
|------|----------|----------|----------|
| 场景描述数据 | 5,100 条（4200 train + 900 test） | **20 条** | 255× |
| QA 数据 | 178,416 条（train 148,271 + test 30,145） | 68,695 条（conversations） | 2.6× |
| QA 任务覆盖 | 6 任务全生成 | 只生成过 description + existence | 缺 4/6 |
| Stage1 数据 | LiDAR-LLM-Nu-Caption 162k | 699 条 | 232× |
| Test 划分 | nuScenes val 150 scenes / 30,145 条 | 71 scenes（来自 train）/ 7,071 条 | — |
| Metatoken | ✅ | ❌ 无 | 缺 |
| `<4DLiDAR>` token | ✅ | ❌ 无 | 缺 |
| 评估指标 | 6 任务分开 + 7 项指标 | 按答案格式分类 + 3 项 | 口径不同 |

---

## A. 数据侧（最基础，优先做）

### A1. 补全场景描述生成（差 255 倍）🔴

**现状**: `datageneration/data/generated_description/` 只有 2 个文件、20 条描述（10 批 × 每批 10 条？），论文需要 5100 条（850 scenes × 6 sequences）。

**处理方法**:
```bash
cd datageneration
# 需要运行 850 scenes × 6 sequences 的完整描述生成
# generate_description.py 的 start/end index 按 10 的倍数批量跑
conda run -n wqlc python generate_description.py \
    --start_index 0 --end_index 850 \
    --api_key $OPENAI_API_KEY \
    --nuscenes_root /path/to/nuScenes \
    --dataroot ./data
```
- 每次调用 API 的帧序列 = 每个 sequence 的 3-10 帧 × 6 相机视图
- **成本预估**: 5100 条描述 × 每条 ~2 次 GPT-4o 调用（front+back）≈ 1 万次调用，约 $100-200
- **注意**: 论文 prompt 要求输出 `[1] Description of the Scene / [2] Key Changes Over Time / [3] Important Objects and Events` 三段式，本地 `prompts.py` 已具备（FRONT_PROMPT/BACK_PROMPT），且含 `gt_caption`（HA）✓

### A2. 补全 6 任务 QA 生成（缺 4/6）🔴

**现状**: `datageneration/data/generated_dataset/` 只有 `description/`、`existence/` 各 1 个文件。

**处理方法**: 对每批描述运行其余 4 个任务：
```bash
for task in binary temporal comprehensive; do
    conda run -n wqlc python generate_dataset.py \
        --start_index 0 --end_index 850 \
        --api_key $OPENAI_API_KEY \
        --task $task \
        --dataroot ./data
done
```

**注意**: 论文各任务样本数（40/sequence）:
| 任务 | 论文 | 本地 prompt 生成数 |
|------|------|-------------------|
| Existence | 5 | 5 ✓ |
| Binary QA | 10 | 10 ✓ |
| **Time Grounding** | **5** | **❌ 无独立 prompt**（并入 temporal 10 条中） |
| Description | 5 | 5 ✓ |
| Temporal Understanding | 5 | 10 ✗（多 5 条） |
| Comprehensive | 10 | 10 ✓ |

→ **需要新增独立 Time Grounding prompt（5 条/seq），并把 Temporal 改为 5 条/seq**，才能对齐论文 Table 2 的数量分布（test: 3770/7525/2783/3770/4757/7540）。

### A3. 重建 train/test 划分（根本问题）🔴

**现状**: `create_splits.py` 随机 80/10/10（seed=42）；`stage2_test.json` 的 71 scenes 全部来自论文 train 集合。

**处理方法**:
1. 删除/废弃 `create_splits.py` 随机划分流程
2. 按已修复的 `b4dl_dataset/metadata/scene_metadata.json`（700 train / 150 test）过滤：
```python
# 以 metadata split="test" 的 150 scenes 为 test，split="train" 的 700 scenes 为 train
import json
scenes = json.load(open('b4dl_dataset/metadata/scene_metadata.json'))
test_ids = {s['scene_id'] for s in scenes if s['split'] == 'test'}
# 用 build_test_split.py 按 test_ids 组装 test QA
```
3. 用 `split_dataset.py --fix_split --nuscenes_root <path>` 保证元数据与官方划分一致
4. **验收**: test = 150 scenes / ~30k 条，train = 700 scenes / ~148k 条

### A4. 数据加 task 标签 🔴

**现状**: `stage2_*.json` 无 `task` 字段，评估时只能按答案格式猜任务。

**处理方法**: 在 `generate_dataset.py` 的 `preprocessing()` 中给每条样本写入 `"task": self.task` 字段；历史数据可用文件名/答案格式批量回填。

### A5. 修正数据内 split 字段 🟡

**现状**: `stage2_train/val/test.json` 内 `split` 全为 `"train"`（生成早于 metadata 修复）。

**处理方法**: 按 metadata 重新标注，或在新划分流程（A3）中一并写入正确 split。

---

## B. 模型侧

### B1. 实现 Metatoken（论文 4.1 节）🔴

**作用**: 把自车传感器元数据（速度/方向/位置/加速度，相对前帧）转为文本，首尾帧拼接，提供运动上下文。论文消融显示它贡献 mIoU 的 30%（0.218→0.311）。

**实现步骤**:
1. 从 nuScenes 读取 ego 数据（`lidar_top` pose 的 translation/rotation/velocity）
2. 按论文格式转文本：相对方向（forward/backward/left/right）、相对速度、相对位移、加速度，如 `"The ego vehicle moved forward 3 meters between frame 12 and frame 20 at a speed of 5 m/s"`
3. 格式: `<4DLiDAR><meta>{首帧元数据描述}...{尾帧元数据描述}\n{QA}`
4. 输入侧: 在 `prepare_inputs_labels_for_multimodal()` 之前的 prompt 组装处注入；训练与推理同步改

### B2. 实现 `<4DLiDAR>` token 🔴

**作用**: 提示模型输入为 4D LiDAR 数据，引导注意力到时空关系。

**实现步骤**:
1. `tokenizer.add_special_tokens({'additional_special_tokens': ['<4DLiDAR>']})`
2. `model.resize_token_embeddings(len(tokenizer))`（新 token embedding 随机初始化）
3. QA 输入统一前置 `<4DLiDAR>`（训练数据的 human value 和推理时都要加）
4. 注意与 `<video>` 特征插入逻辑共存（`<4DLiDAR>` 是纯文本 token，不影响特征插入位置）

### B3. 特征侧（基本对齐，可选优化）🟢

**现状**: LiDAR-CLIP 预提取 (N_frames, 768) 特征 ≈ 论文 E_L 输出的各帧 cls embedding 拼接。**基本等价**，无需改动。
可选: 若用 RTX 5090 环境重训 LiDARCLIP encoder 以完全复现论文的 similarity loss 预训练（当前 checkpoint 来自公开权重）。

---

## C. 训练侧

### C1. 补 Stage1 数据（差 232 倍）🔴

**现状**: `mllm/data/stage1_lidarllm_mm.json` 仅 699 条；论文用 LiDAR-LLM-Nu-Caption 162k 条训练 aligner。

**处理方法**:
- 从 HuggingFace 下载 LiDAR-LLM Nu-Caption 数据集（官方 repo: Senqiao Yang/LiDAR-LLM）
- 或联系论文作者获取（B4DL 官方 GitHub: ccho4702/B4DL）
- 若无法获取: 用 A1-A2 生成的单帧描述数据替代（质量会打折扣）

### C2. 重训 Stage2 🔴

**现状**: train 54,901 条（46%），且输入无 `<4DLiDAR>` + Metatoken。

**处理方法**: B1-B2 完成后，用完整 6 任务数据重训:
```bash
bash scripts/stage2.sh \
    --s2_data ./b4dl_dataset/stage2_full.json \
    --s2_feat ../encoders/lidarclip/b4dl/stage2_features
```
- 训练时 human value 前置 `<4DLiDAR><meta>...`（B1/B2 实现后自动生效）
- 预计训练量 148k 条，RTX 5090 上约 1-2 天（论文单 4090 24h 内）

---

## D. 评估侧

### D1. 按论文口径评估 🔴

**现状**: `b4dl_metrics.py` 按答案格式分类（binary/frame_range/categorical），无任务标签。

**处理方法**: A4 加 task 标签后，用 `mllm/evaluation/evaluate_model.py`（已实现全部论文指标）:
- Existence / Binary QA 分别算 accuracy → 论文 Accuracy = mean(两者)
- Time Grounding 算 mIoU（0-1 刻度）
- Description / Temporal / Comprehensive 算 BLEU-4 / METEOR / ROUGE-L / BERTScore / GPT-4o

### D2. 对齐 mIoU 计算 🟡

**现状**: `b4dl_metrics.py` 的 mIoU 带 +1 帧计数、刻度 ×100（18.23 = 0.1823）。

**处理方法**: 统一按论文标准（0-1 刻度、区间重叠/并集），用 evaluate_model.py 的 `compute_miou`（已按论文实现）。

### D3. GPT score 需要 API key 🟡

**现状**: evaluate_model.py 支持 `--use_gpt --gpt_api_key`，但需要 OPENAI_API_KEY 且按样本计费（30k 条 test ≈ $30-60）。

**处理方法**: 评估时加 `--use_gpt --gpt_api_key $OPENAI_API_KEY`；或先跑 1k 子集。

---

## E. 对齐验收标准

完成后应接近（无需完全一致，硬件/随机性有差异）：

| 指标 | 论文 B4DL | 对齐目标 |
|------|-----------|----------|
| Accuracy | 0.762 | ≥ 0.72 |
| mIoU | 0.311 | ≥ 0.25（有 Metatoken 才有戏） |
| B@4 | 0.095 | ≥ 0.08 |
| ROUGE-L | 0.322 | ≥ 0.30 |
| METEOR | 0.275 | ≥ 0.25 |
| BERTScore | 0.897 | ≥ 0.88 |
| GPT Score | 59.513 | ≥ 55 |

---

## 执行顺序（依赖关系）

```
A1 描述生成 ──→ A2 六任务 QA ──→ A3 官方划分 ──→ A4 task 标签
                                                  │
B1 Metatoken ←── nuScenes ego 数据 ───────────────┤
B2 <4DLiDAR> token ───────────────────────────────┤
                                                  ▼
C1 Stage1 数据（LiDAR-LLM）──→ C2 重训 Stage2 ←──┘
                                                  │
D1 论文口径评估 ←── D2 mIoU 对齐 ←── D3 GPT key ──┘
```

**建议节奏**:
1. 先做 A3 + A4（纯数据处理，1 天内，评估立刻可对齐口径）
2. 再做 B1 + B2（代码改动，2-3 天）
3. C2 重训（1-2 天 GPU）
4. 数据生成 A1/A2 成本高（GPT-4o API），可并行启动，按 100 批跑
