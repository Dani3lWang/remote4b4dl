---
name: b4dl-qoder-conversation-history
description: Qoder 导出对话的时间线与因果链——复现方案→metatoken 审查→合并训练决策→远端分析→per-seq 设计（2026-08-05~11）
metadata:
  node_type: memory
  type: project
  originSessionId: sess_0638fcf8-2925-4746-ac7e-a2cfe910dda4
---

2026-08-24 导入的 5 份 Qoder（Daniel Windows 端的另一个 AI 助手）对话导出，其中 4 份与本项目相关，串起了 git commit 之间的决策链（原件是临时附件，内容已固化到各专门 memory）：

1. **论文复现具体方案**（最早，当时 HEAD≈82c4377）：解析官方上游仓库（ccho4702/B4DL，本地副本 `D:\tmp\B4DL`）+ 论文 19 页全文 → 产出九章复现方案 `docs/B4DL_复现方案.md`。关键结论：官方仓库六任务评测/GPT Score/Metatoken 均未发布需自建；HF 数据集测试集与论文逐项一致、训练集 148,271 足够但拆成 stage2/stage3 两文件；用户明确"不跑 GPT 数据生成、直接下载 HF 数据"。
2. **metatoken 实现审查**（2026-08-08）：审查 commit e77f816，判定"方向正确但不可直接使用"（3 个设计偏差 + 3 个 bug，详见 [[b4dl-metatoken-implementation]]）→ 在 fix/metatoken-alignment 分支改进为 commit d3af130 并合并 main，同时删除旧 mllm/vtimellm/eval/ 统一评测到 mllm/evaluation/。
3. **训练效果诊断**（2026-08-08~09，问"为什么全是 Yes./from frame 000 to frame 008."）：最关键发现是**官方 stage2.json=3 个简单任务（68,695 条）、stage3.json=3 个复杂任务（79,576 条）**——此前只训 stage2，生成类任务零训练数据，指标崩塌由此而来；另有**回声效应**（99% 帧区间问答的答案区间=问题区间）。建议合并 148k 单阶段混训（方案 B，LLaVA 式主流做法）→ 即后来的 stage2-full 重训；推荐的 lr 2e-5 未被采纳（实际仍 1e-4）。根因链详见 [[b4dl-training-eval-history]] 第 7 条。
4. **远端测试结果分析**（2026-08-11，SSH+paramiko 分析 stage2-full 全量评测）：确认 0 skip、训练/测试 scene 无重叠；给出 P0-P6 建议（统一 pycocoevalcap 口径 > time_grounding 坍缩 > Yes/No 平衡 > 数据清洗 > metatoken per-seq > Stage1 重训 > skip 上报），并给出 per-sequence 改造 5 步设计 → 后来由 [[b4dl-per-sequence-refactor]] 落地。产出报告 `B4DL_远端测试分析报告_20260811.md`（存于 Qoder workspace）。
5. **生成调查问卷图表**：高校"大思政课"问卷（N=824）图表生成，属课程作业，与本项目无关，不记录。

注意：对话中的中间结论可能已被后续推翻（如"须砍到 559-scene/118,722 防泄漏"后被证实是误判，见 [[b4dl-project-overview]]），以各专门 memory 的最终状态为准。

相关：[[b4dl-metatoken-implementation]]、[[b4dl-training-eval-history]]、[[b4dl-per-sequence-refactor]]
