---
name: b4dl-codex-project-memory
description: Codex 对 AutoDL 远程 mmb4dl 项目的全量盘点与 AgentMEM 导入摘要（2026-09-10）
metadata:
  node_type: memory
  type: project
  source: /root/autodl-tmp/wql/mmb4dl
  remote_head: 11326063abdbb4cc8575da1f461ba24f6fa9efcd
---

# B4DL 项目记忆（Codex 导入版）

## 1. 当前项目与盘点范围

- 远程项目：`/root/autodl-tmp/wql/mmb4dl`；当前 SSH 会话已进入该目录。
- 远程 Git：`main` 与 `origin/main` 对齐，工作树干净；当前 HEAD 为 `1132606`。
- 远程目录约 45G、122,579 个文件、1,414 个目录。主要空间由预计算特征、模型权重、评测/训练产物和备份占用。
- 当前盘点时间：2026-09-10（Asia/Shanghai）。项目中存在共享 GPU 上其他用户/项目的运行进程，训练或评测前必须先检查显存与进程，不要误杀非本项目任务。

## 2. 项目目标与架构

B4DL 是 ACM MM 2025 的 4D LiDAR 多模态 LLM benchmark 复现。主链路为：

`nuScenes LiDAR 序列 → LiDAR-CLIP/SST（768 维/帧）→ mm_projector Linear(768,4096) → Vicuna-v1.5-7B → LoRA QA/描述生成`

主要任务：existence、binary QA、time grounding（mIoU）、description、temporal understanding、comprehensive reasoning。评测代码集中在 `mllm/evaluation/`，训练/推理模型在 `mllm/vtimellm/`，数据生成在 `datageneration/`，编码器和特征提取在 `encoders/lidarclip/`。

## 3. 目录职责与当前体量

| 目录 | 职责 | 远程体量/备注 |
|---|---|---|
| `datageneration/` | GPT 场景描述、QA 转换、nuScenes 元数据与 Stage1 构建工具 | 76M；含 `scene_metadata.json`、`sequence_metadata.json` 与 Stage1 官方筛选脚本 |
| `mllm/` | VTimeLLM 模型、训练、评测、数据和 checkpoint | 约 15G；项目主运行目录 |
| `encoders/lidarclip/` | LiDAR-CLIP/SST 源码、编码器 checkpoint、特征 | 约 11G；含上游 mmdetection3d/SST 代码，约 117,684 个文件在 `b4dl/` 及相关树中 |
| `dataset/` | nuScenes-B4DL 与 LiDAR-LLM-Nu-Caption 数据 | 384M；仓库内为已清理/抽取版本，不等同于完整 nuScenes 原始数据 |
| `models/` | `roberta-large` 评测模型，多种格式副本 | 15G；多个权重格式会重复占空间 |
| `AgentMEM/` | ZCode/Claude 记忆、计划、记录、快照和导出工具 | 约 59M；长期记忆见 `01-zcode-memory/` |
| `docs/`、`wiki/`、`Claude_record/` | 复现方案、架构/训练/评测说明、历史记录 | 文档入口优先看 `wiki/Home.md`、`wiki/Reproduction-Log.md` |
| `backups/` | checkpoint 与阶段性项目备份 | 约 2.3G；不要未经确认删除 |
| `.claude/`、`.zcode/` | agent 运行记录/计划/运行时状态 | `.claude` 约 601M，不应当作项目源码或长期记忆整体读取 |

核心源码规模：`datageneration` 约 1,414 行、`mllm/vtimellm` 约 4,461 行、`mllm/evaluation` 约 2,648 行、`mllm/scripts` 约 4,087 行；`encoders/lidarclip` 的上游实现和测试约 232k 行。

## 4. 数据、特征与模型资产

- `mllm/b4dl_dataset/`：29 个文件、约 501M。关键文件是 `stage1_train.json`（官方 162K 方案，161,629 条）、`stage1_val.json`、`stage2_full_train_148k.json`、`stage2_full_train_seqv3_meta2_148k.json`、`stage2_full_train_seqv3_meta2_oversampled_150k.json`、`test_qa.json`、`ego_metadata.json`、`ego_frame_motion.json`。
- `dataset/LiDAR-LLM-Nu-Caption/`：LiDAR-LLM-Nu-Caption 的 `train.json`/`val.json`，用于 Stage1；`dataset/nuScenes-B4DL/` 是 B4DL 数据集抽取/清理副本。
- `encoders/lidarclip/b4dl/` 有 `stage1_features`、`stage1_features_once`、`stage1_features_sample`、`stage1_features_sample_once`、`stage2_features`、`stage2_features_once` 六类特征目录。Stage1 新规范使用 sample token 命名；旧 frame-id/ONCE 版本必须按记忆中的来源说明区分，不能混用。
- `mllm/base_model/vicuna-v1-5-7b/` 是本地 Vicuna 7B 基座，两个分片约 13.5G。
- 当前保留 checkpoint：Stage1 投影层，以及 B0、B3、B4a 三个 Stage2 顶层 adapter；`mllm/checkpoints/` 清理后约 1.7G。B3/B4a adapter 与 `non_lora_trainables.bin` 均约 320M/262M 级别。
- `models/roberta-large/` 同时存在 safetensors、PyTorch、ONNX、TF、Flax 等重复格式；评测只需与当前 BERTScore 配置匹配的一个格式，空间回收前需确认引用。

## 5. 训练/评测结论（当前最重要的事实）

- B0：seqv3-mixed，acc 0.7629、mIoU 0.2696，作为 per-sequence 基线。
- B1：官方 Stage1 162K projector 重训，acc 0.7787、mIoU 0.2653；预测体现窗口内局部帧号，不具备场景全局位置感。
- B2：整场景输入但旧 meta，mIoU 0.1992，出现大量 `[000-008]` 先验坍缩。
- 官方无 meta checkpoint：acc 0.7542、mIoU 0.1737，说明论文 Table 3 的完整高分模型并未随官方仓库发布。
- B3：整场景输入 + relative-to-previous 修复后的 meta2，acc 0.7526、mIoU 0.3467，超过论文 0.311，是当前最优基线；BLEU-4 0.0965、ROUGE-L 0.3234、BERTScore 0.8967，NLTK-2005 METEOR 0.3366。
- B4a：在 TG 高帧段过采样后，acc 0.7775 但 mIoU 0.3271，显著低于 B3；高帧段专项更差。结论是缺显式帧身份/位置锚点，而不是简单的数据分布不足。
- 评测必须与训练数据格式配对：`--per_sequence`、`--answer_frames`、`--frame_motion`、`--whole_scene` 的组合不能随意套到旧 checkpoint；旧模型与 per-sequence 模型不可混评。

## 6. 关键实现约束

- `mllm/scripts/ego_text.py` 是训练注入与推理注入共用的 metatoken 渲染源；论文要求使用 QA 所引用的首尾帧描述，并按 relative-to-previous 运动语义渲染。
- `mllm/scripts/inject_metatoken.py` 负责训练数据注入；`mllm/evaluation/test_b4dl.py` 负责推理时相同格式的 query 构造和特征切片；`mllm/vtimellm/train/dataset.py` 负责训练时特征选择。三处修改必须保持一致。
- per-sequence 模式应依据 QA 的包含序列和 `feat_indices` 精确取样；不能把整个 scene 的特征或不属于序列的中间帧混入。
- `mllm/evaluation/evaluate_model.py` 同时支持 NLTK-2005 与 pycocoevalcap Meteor 后端；报告指标时必须注明后端，不能直接比较不同后端的 METEOR。
- BERTScore 可能 OOM；评测脚本支持断点续跑，长任务优先使用 checkpoint 间隔和独立后台会话。
- 当前实测环境：wqlc conda 环境、torch 2.8.0+cu128、transformers 4.47.0、peft 0.13.2、accelerate 1.3.0、SDPA（未安装 flash-attn）。官方 peft 0.19.1 的 adapter 配置可能需要最小化兼容处理。

## 7. 当前工作规则与安全边界

- 远端多项目共用，写操作只允许落在 `/root/autodl-tmp/wql/mmb4dl`；跨项目只读检查，不要修改或清理其他项目。
- 运行训练/评测前先检查 `nvidia-smi`、进程、磁盘和目标 checkpoint；共享 GPU 可能长时间被其他实验占用。
- 大文件（权重、特征、备份、评测预测）只做清单/统计，除非任务明确要求不要全文加载或搬运。
- AgentMEM 的 `01-zcode-memory/`、`03-zcode-plans/`、`04-claude-code-memory/`、`08-claude-records/`、`09-history-snapshots/` 已从远程更新到当前本地项目；`02-zcode-sessions/` 与 `05-claude-code-sessions/` 的原始会话没有导入 Codex 长期记忆层。
- 不在项目记忆或回复中记录 SSH 密码、API key 或原始 secret；安全扫描报告只用于提示路径和风险。

## 8. 后续优先方向

1. 先做帧身份锚点消融：文本锚点与帧嵌入，针对 B3 的 TG 残余偏早和高帧段失败。
2. 在 B3 稳定基线之上再做 RFT/DPO/GRPO；奖励函数可复用 `evaluate_model.py` 的答案规范化、帧范围和文本质量规则。
3. 补充简单 baseline、消融和跨阶段遗忘检查，再考虑论文实验整理。

关联记忆：`MEMORY.md`、`b4dl-project-overview.md`、`b4dl-training-eval-history.md`、`b4dl-per-sequence-refactor.md`、`b4dl-eval-methodology-caveats.md`、`b4dl-server-access-workflow.md`。
