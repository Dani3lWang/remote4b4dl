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
- 2026-09-10 起挂着一个**未完成的实测**：`mllm/_vram_smoke_bs8/run_measure.sh`（v2，setsid 脱离会话）等 ≥28GB 空闲窗口后跑 6 步 bs=8 的 B4a 同构训练（zero3 offload + LoRA r64 + whole_scene/meta2），逐 0.5s 采样并按"非基线 pid"归因自己进程的峰值显存；结果写在同目录 `runner.log` 的 "RESULT" 段（峰值 MB vs 24GB）。踩坑：v1 用 tee 输出到 harness 后台任务 stdout，会话重启后管道断裂把脚本堵死近 10h（后台任务句柄也丢失）——长等待任务必须 setsid + 只落文件 + nvidia-smi 加 timeout。测完记得删 `_vram_smoke_bs8/`（scratch 不入库）。
- 2026-09-09 咨询"新租单张 4090（24GB）做 RL 是否够用"，结论=三段式设计下够用：rollout/ref 进程原按 14GB 门控设计，24GB 宽裕；训练进程 ZeRO-3+CPU offload（LoRA r64/bs8/grad-ckpt）训练日志无峰值记录，估 15-20GB，需 M2 冒烟实测确认；红线=守住三段式（TRL 同进程双 7B 权重 14+14GB 会超）；速度约共享 5090 的 0.55-0.7×（5090 实测 4.9 samples/s、~26 s/step，B4a 纯 GPU 25.6h/3519 步）；共享卡等窗口实测曾 10 轮×10min+ 起，专享 4090 免门控/免抢占，更适配 RL 阶段化小实验流程；迁移成本低（特征 npy 221MB + json ~150MB + ckpt tar 1.4GB + vicuna 基座重下）。
