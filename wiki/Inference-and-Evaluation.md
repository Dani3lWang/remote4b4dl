# 推理与评测（mllm/）

## 单条推理

```bash
cd mllm
python vtimellm/inference.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --clip_path checkpoints/clip/ViT-L-14.pt \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    [--stage3 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3] \
    --video_path /path/to/video.mp4
```

流程：`load_pretrained_model` 完整加载链 → `VideoExtractor(N=100)` 均匀抽帧 → CLIP ViT-L/14 编码 → v1 模板拼 `<video>\n`+query → greedy 生成（温度 0.05，max_new_tokens 1024）。

## Gradio Web Demo

```bash
python vtimellm/demo_gradio.py --model_base ... --clip_path ... \
    --pretrain_mm_mlp_adapter ... --stage2 ... [--share]
```

支持视频上传 + 多轮对话（首轮自动插 `<video>\n` 前缀，流式输出）。⚠️ 已知问题：`gr.Examples` 引用了未定义的 `root_dir`，运行会 NameError，需手动修正该行；且不要依赖默认路径，显式传入所有绝对路径。

## B4DL 六任务评测（test_b4dl.py）

论文 Table 3 对齐的端到端评测：加载测试 QA → 加载特征 → 加载模型 → 逐任务逐条 greedy 推理 → 写 predictions → 计算指标 → 与论文参考值对比。

```bash
python evaluation/test_b4dl.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/.../mm_projector.bin \
    --stage2 ./checkpoints/...-stage2 \
    [--stage3 ./checkpoints/...-stage3] \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --test_data ./b4dl_dataset/test_qa.json \
    --ego_meta ./b4dl_dataset/ego_metadata.json \
    --output ./evaluation/predictions.json \
    --metrics_output ./evaluation/evaluation_results.json
```

**关键参数**：

- `--per_sequence`：论文 Appendix C 对齐的 per-sequence 模式（特征切片到 QA 的包含序列）。⚠️ 仅在模型用 per-sequence 数据训练后才启用，否则训练-评测不匹配
- `--frame_motion` + `--sequence_metadata`：配合 per_sequence，metatoken 按 QA 引用帧渲染真实描述（与训练注入同源）
- `--answer_frames`：对 time_grounding 类问题用 GT 答案帧范围恢复包含序列（benchmark 丢失序列归属字段的双侧回退方案）。⚠️ 属 oracle 输入选择，报告中须声明
- 消融开关：`--no_4dlidar` / `--no_meta`；其他：`--max_samples`、`--dtype`、`--use_gpt`（GPT-4o Score）
- **断点续跑**：每 50 条写 `<output>.ckpt`，中断后同命令自动恢复

**特征切片优先级**（per_sequence 模式）：item 的 `feat_indices`（精确帧）> `feat_range` > 问题帧号的包含序列 > `--answer_frames` 的 GT 范围（仅 TG）> 全场景特征。

**Metatoken 注入**：prompt 拼为 `<4DLiDAR>\n<video>\n{question}\n<meta> ...`，meta 文本三级来源：① per_sequence + frame_motion 时按 QA 引用帧现场渲染（与训练逐字符一致）→ ② per-sequence 键 → ③ per-scene 回退。

## 评测指标（evaluate_model.py，B4DLEvaluator）

| 任务 | 指标 | 说明 |
|------|------|------|
| existence / binary_qa | Accuracy | 小写、去标点、剥前缀后整串精确匹配 |
| time_grounding | mIoU | 从答案正则提取 `from frame X to frame Y` 闭区间算交并比 |
| description / temporal / comprehensive | BLEU-4、METEOR、ROUGE-L、BERTScore（可选 GPT-4o Score） | 三任务平均 |

**指标口径（2026-08-29 修正后冻结）**：

- BLEU-4：pycocoevalcap **语料级**（NLTK 句级偏高 ~15-17%，不可比）
- METEOR：pycocoevalcap **Meteor-1.5 jar**（需系统 java；NLTK 1.0 式系统性偏高）——仓库修复了上游 stderr 管道死锁 bug
- BERTScore：本地 roberta-large，**强制取第 17 层**（按 config 的 num_hidden_layers=24 取层会虚高 ~0.07）
- GPT 缺失记 null 而非 0；实际使用的后端记录在结果 JSON 的 `metric_backend` 字段

聚合口径（论文 §5.1）：accuracy = mean(existence, binary)；mIoU 为 TG 单独值；4 个文本指标在 3 个复杂任务间平均。

## 输出格式

- `evaluation_results.json` / `eval_*_metrics.json`：`{per_task_metrics, final_scores, metric_backend}` 三层结构
- `predictions_*.json`：`{canonical_task: {predictions, ground_truths, questions}}`

## 辅助脚本与一键评测

| 脚本 | 用途 |
|------|------|
| `scripts/run_b4dl_eval.sh` | 一键：建划分 → 评测 → 打印论文 Table 3 参考值；`--stage3`/`--no_meta` 等透传 |
| `run_baseline_eval.sh` | baseline 版（无 per_sequence，stage2-full checkpoint） |
| `evaluation/build_test_split.py` | 把生成数据聚合为单文件 test_qa.json（优先 nuScenes 官方 val split 的 150 场景） |
| `evaluation/split_dataset.py` | 划分校验：700/150 scene、各任务条数对齐论文 Table 2（test 六任务 3770/7525/2783/3770/4757/7540，合计 30,145） |
| `evaluation/analyze_behavior.py` | 答案模式分布 / 混淆矩阵 / 模板复用度分析（诊断模型坍缩） |

## 评测数据文件

| 文件 | 说明 |
|------|------|
| `test_qa.json` | 30,145 条六任务测试集（HF 官方 test_qa），list 格式，human 首条含 `<video>\n` |
| `ego_metadata.json` | per-scene / per-sequence 两级 ego 运动自然语言描述 |
| `ego_frame_motion.json` | 逐帧运动表 `{scene_id: [{x,y,z,yaw,spd,yaw_next/prev,acc_next/prev}, ...]}` |
| `sequence_metadata.json` | 850 scene / 5100 sequence 元数据（含采样 indices 与 split） |

这些大文件本体在远端训练机（`mllm/b4dl_dataset/`），仓库以 MD5 锁定版本（见 [[Reproduction-Log]]）。

## 模型代际与评测命令匹配

⚠️ 训练和评测必须用相同代际的格式：

| 模型代际 | 评测参数 |
|---------|---------|
| seqv2 | `--per_sequence --frame_motion --sequence_metadata` |
| seq（旧 per-sequence） | 只加 `--per_sequence` |
| 旧模型（全 scene 输入） | 都不加 |
| seqv3（当前） | `--per_sequence --frame_motion --sequence_metadata --answer_frames` |
