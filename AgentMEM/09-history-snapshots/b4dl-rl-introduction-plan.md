---
name: b4dl-rl-introduction-plan
description: 用户计划在 B4DL SFT 管线（B3/B4a 之后）引入强化学习，拆解底稿文档已产出（2026-09-07）
metadata:
  node_type: memory
  type: project
  originSessionId: sess_115a31ac-a60b-49a3-92d4-6bdd0206206f
---

用户（Daniel）计划在 mmb4dl 的 SFT 训练管线之后**引入强化学习（RL/RLVR）**，方向是"先给详尽的训练流程步骤以便自己拆解"（2026-09-07 会话）。

**Why:** SFT 链已闭环（B3 mIoU 0.3467 超论文），下一步想用 RL 优化六任务（尤其可验证的 existence/binary 与 time_grounding）。

**How to apply:**
- 拆解底稿文档：`docs/learn docs/B4DL_训练全流程分步详解_RL引进挂载点_20260907.md`（A 数据→B stage1→C SFT-LoRA→D 评测逐段命令/产物/耗时 + RL-锚点标注）；代码级解读见同目录 `B4DL_训练代码全景解析_20260907.md`；**可行性深度分析**（2026-09-08，实测数字）见同目录 `B4DL_RL引进可行性深度分析_20260908.md`。
- 实测关键事实（2026-09-08）：wqlc 环境实际为 transformers 4.47.0 / torch 2.8.0+cu128 / peft 0.13.2（CLAUDE.md 旧清单已同步修正）；TRL 0.15-0.17 与 4.47 兼容且 PyPI 可达但未安装；结论=自写最小 GRPO 优先，三段式进程（rollout/ref 打分/ZeRO-3 训练）；第一批任务=TG+existence/binary（奖励零成本继承 evaluate_model 规则函数）；B4a 训练已完成、评测当时 29805/30145 进行中。
- 训练优化角度盘点（同目录 `B4DL_训练优化角度分析与策略_20260908.md`）：A 帧身份锚点（TG 输入无帧号信号的根因修复，推荐可学习帧嵌入）、B TG 重加权、C RFT/DPO（RL 的平价前置，训练集采样、测试集 predictions 禁入训练）、D 动态 padding/packing、E DoRA/NEFTune（peft 0.13/4.47 已实测可用）、F 解冻 projector。行动序列=等 B4a 评测→M1 审计→小消融→RFT→组合接 GRPO。
- 文档内已定框架性建议：初始化 = B3/B4a 的 adapter+non_lora（merge 后挂新 LoRA，复用 train.py training_stage==3 路径）；奖励函数直接复用 evaluate_model.py 的 compute_accuracy/extract_time_grounding_frames/compute_miou（与论文指标同构）；transformers 4.31/peft 0.4.0 太老 → 新 TRL 大概率不兼容，倾向自写最小 GRPO；oracle 红线（评测侧 --answer_frames 不进入 RL 训练输入）；里程碑 M1（奖励离线审计，零 GPU）→ M4。
- 环境：共享 5090 单卡（28GB 显存门控），RL 需沿用 run_b4a_pipeline.sh 的门控/断点续训/判据化验收外壳。[[b4dl-training-eval-history]] [[b4dl-per-sequence-refactor]]
