# B4DL 评测模块

对训练完成的 B4DL 模型进行论文 Table 3 的六任务评测。

## 文件说明

| 文件 | 功能 |
|------|------|
| `test_b4dl.py` | **端到端评测**：加载模型 → 逐条推理 → 计算全部指标（支持断点续跑） |
| `evaluate_model.py` | **指标库**：Accuracy、mIoU、BLEU-4、METEOR、ROUGE-L、BERTScore、GPT Score |
| `build_test_split.py` | **测试集构建**：从 `datageneration/` 生成的原始数据中提取测试集 |
| `split_dataset.py` | **划分校验**：验证 train/test 划分与论文 Table 2 一致 |

## 快速开始

### 1. 准备测试数据

测试数据需为 conversations 格式的 JSON，每条包含 `scene_id`、`task`、`conversations`：

```json
[
  {
    "scene_id": "005745653",
    "task": "existence",
    "conversations": [
      {"from": "human", "value": "<video>\nWas a pedestrian present in frame 006?"},
      {"from": "gpt", "value": "Yes."}
    ]
  }
]
```

仓库已提供现成的测试集：`mllm/b4dl_dataset/test_qa.json`（30,145 条，6 任务合并）。

如需从 HF 原始数据重新构建：

```bash
cd mllm/b4dl_dataset
python convert_raw_to_conversations.py
```

### 2. 确保特征文件就位

评测需要 `stage2_features/` 目录下有所有测试场景的 `.npy` 文件（由 `encoders/lidarclip/extract_pc_features.py` 生成）：

```bash
ls ../encoders/lidarclip/b4dl/stage2_features/ | head -5
# 预期: 003833660.npy, 005745653.npy, ...
```

### 3. 运行评测

```bash
cd mllm

# 完整评测（30k 条，数小时）
python evaluation/test_b4dl.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --test_data ./b4dl_dataset/test_qa.json \
    --output ./evaluation/predictions.json \
    --metrics_output ./evaluation/evaluation_results.json

# 快速冒烟（10 条）
python evaluation/test_b4dl.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --test_data ./b4dl_dataset/test_qa.json \
    --max_samples 10

# 含 GPT Score（需要 OPENAI_API_KEY）
python evaluation/test_b4dl.py \
    ... \
    --use_gpt --gpt_api_key $OPENAI_API_KEY
```

### 4. 断点续跑

推理过程中每 50 条自动保存 checkpoint 到 `<output>.ckpt`。若中途中断，使用**完全相同的命令**重新运行即可自动恢复。

```bash
# 中断后直接重新运行，自动从 checkpoint 恢复
python evaluation/test_b4dl.py --output ./evaluation/predictions.json ...

# 手动调整 checkpoint 保存频率
python evaluation/test_b4dl.py --checkpoint_interval 100 ...
```

### 5. 仅计算指标（已有预测结果）

```bash
# 方式 A：对 test_b4dl.py 输出的 predictions.json 直接计算指标
# （predictions.json 已包含 ground_truths）
python evaluation/evaluate_model.py \
    --predictions ./evaluation/predictions.json

# 方式 B：单独指定 predictions 和 ground_truth
python evaluation/evaluate_model.py \
    --predictions ./evaluation/predictions.json \
    --ground_truth ./b4dl_dataset/test_qa.json

# 方式 C：快速验证 — 只加载 test_qa.json 查看数据统计
python evaluation/evaluate_model.py \
    --ground_truth ./b4dl_dataset/test_qa.json

# 方式 D：内置 demo 数据测试
python evaluation/evaluate_model.py --demo
```

> **格式自动检测**：`evaluate_model.py` 会自动识别输入格式 —
> `test_b4dl.py` 输出的 dict 格式（`{task: {predictions:[], ground_truths:[]}}`）
> 和 `test_qa.json` 的 list 格式（`[{task, conversations}]`）均支持。

## 验收标准（论文 Table 3 参考值）

| Accuracy | mIoU | B@4 | ROUGE-L | METEOR | BERTScore | GPT Score |
|----------|------|-----|---------|--------|-----------|-----------|
| 0.762 | 0.311 | 0.095 | 0.322 | 0.275 | 0.897 | 59.513 |

建议容差：±5%（相对值）。

## 依赖

```bash
pip install pycocoevalcap nltk rouge-score bert-score openai
```

首次运行时会自动下载 NLTK 数据（punkt、wordnet、omw-1.4）。

| 依赖 | 说明 | 缺失时影响 |
|------|------|-----------|
| `pycocoevalcap` | BLEU-4、METEOR 计算后端 | BLEU-4 / METEOR 返回 0 |
| `nltk` | 分词 + METEOR | BLEU-4 / METEOR 返回 0 |
| `rouge-score` | ROUGE-L F1 | ROUGE-L 返回 0 |
| `bert-score` | BERTScore F1（deberta-xlarge-mnli ~1.5GB） | BERTScore 返回 0 |
| `openai` | GPT-4o 参考无关打分（需 API Key） | GPT Score 返回 0 |
