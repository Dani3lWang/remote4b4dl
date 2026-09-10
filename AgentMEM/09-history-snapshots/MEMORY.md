# Memory Index

- [B4DL 项目总览](b4dl-project-overview.md) — mmb4dl 复现架构、6 任务 benchmark、数据规模与关键路径、论文参考值、docs/复现方案.md
- [用户与服务器工作流](b4dl-server-access-workflow.md) — Daniel 的 Windows 本地仓库 + SSH 到 AutoDL 5090 服务器（wqlc 环境、paramiko）
- [Qoder 对话时间线](b4dl-qoder-conversation-history.md) — 复现方案→metatoken 审查→合并训练决策→远端分析的因果链（2026-08-05~11）
- [Metatoken 实现要点](b4dl-metatoken-implementation.md) — Figure 6 四行格式、ego_metadata、special token 注册与 embedding 梯度掩码约束（单卡、wd=0）
- [训练评测时间线](b4dl-training-eval-history.md) — B0-B4a 基线链：B3（整场景+meta2）mIoU 0.3467 超论文 0.311 当前最优；B4a TG 高帧段过采样负结果（mIoU 0.3271 显著回退，病灶=输入无帧号信号）；wiki 已 2026-09-08 全面刷新（2026-08-07~09-08）
- [per-sequence 改造与重训](b4dl-per-sequence-refactor.md) — 论文确认三重错配、seqv2/seqv3/两阶段/mixed 演进；seqv3-mixed 已训完并评测且锁定为基线 B0（acc 0.7629、mIoU 0.2696，ΔmIoU>0.013 才显著）；本环境评测需 HF offline + 显存门控（CoRViD 共存）
- [评测方法学陷阱](b4dl-eval-methodology-caveats.md) — NLTK vs pycocoevalcap 虚高、BERTScore OOM、skip 缩水、ckpt 断点续跑
- [git/文档规范](b4dl-git-commit-conventions.md) — 中文 commit + feat/fix 前缀、改后即 commit、提交后及时 push 远程同步、同步 CLAUDE.md、先 baseline 后新格式
- [Stage1 官方 162K 对齐](b4dl-stage1-official-162k.md) — sample_token 键控数据(161,845)+特征(28,130)，旧 95k frame_id 方案退役保留（2026-08-28）
- [LiDAR-CLIP 权重核对](b4dl-lidarclip-weight-provenance.md) — 本地 ckpt 是原版 ONCE 权重非官方 nuScenes 版；nuScenes 自训已启动；wqlc 环境/去 ddp/向量化 scatter/ckpt 为准的排障经验（2026-08-28）
- [修改边界：仅限项目文件夹](b4dl-modify-only-project-folder.md) — 服务器多项目共用，写操作仅限 /root/autodl-tmp/wql/mmb4dl，跨项目排查只读不写
- [RL 引进计划](b4dl-rl-introduction-plan.md) — 用户拟在 B3/B4a SFT 之后引入 RL/RLVR；流程拆解底稿 + 可行性深度分析（实测环境 4.47/trl 未装/自写 GRPO 结论）已产出（2026-09-07/08），奖励函数可复用 evaluate_model 规则函数
- [Checkpoint 精简](b4dl-checkpoint-cleanup.md) — 2026-09-09 清理日志残留（commit a3f58d1）+ checkpoints 145G→1.7G：仅存 stage1 + B0/B3/B4a 顶层 adapter，旧链整版与中间档全删
