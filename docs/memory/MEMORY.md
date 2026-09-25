# Memory Index

- [B4DL 项目总览](b4dl-project-overview.md) — mmb4dl 复现架构、6 任务 benchmark、数据规模与关键路径、论文参考值、docs/复现方案.md
- [用户与服务器工作流](b4dl-server-access-workflow.md) — Daniel 的 Windows 本地仓库 + SSH 到 AutoDL 5090 服务器（wqlc 环境、paramiko）
- [Qoder 对话时间线](b4dl-qoder-conversation-history.md) — 复现方案→metatoken 审查→合并训练决策→远端分析的因果链（2026-08-05~11）
- [Metatoken 实现要点](b4dl-metatoken-implementation.md) — Figure 6 四行格式、ego_metadata、special token 注册与 embedding 梯度掩码约束（单卡、wd=0）
- [训练评测时间线](b4dl-training-eval-history.md) — B0-B4a 基线链：B3（整场景+meta2）mIoU 0.3467 超论文 0.311 当前最优；B4a TG 高帧段过采样负结果（mIoU 0.3271 显著回退，病灶=输入无帧号信号）；wiki 已 2026-09-08 全面刷新（2026-08-07~09-08）
- [per-sequence 改造与重训](b4dl-per-sequence-refactor.md) — 论文确认三重错配、seqv2/seqv3/两阶段/mixed 演进；seqv3-mixed 已训完并评测且锁定为基线 B0（acc 0.7629、mIoU 0.2696，ΔmIoU>0.013 才显著）；本环境评测需 HF offline + 显存门控（CoRViD 共存）
- [评测方法学陷阱](b4dl-eval-methodology-caveats.md) — NLTK vs pycocoevalcap 虚高、BERTScore OOM、skip 缩水、ckpt 断点续跑
- [git/文档规范](b4dl-git-commit-conventions.md) — 中文 commit + feat/fix 前缀、改后即 commit、提交后及时 push 远程同步、同步 CLAUDE.md、先 baseline 后新格式
- [Stage1 官方 162K 对齐](b4dl-stage1-official-162k.md) — 现行数据 stage1_train.json = 161,629 条（官方 699 scenes），sample_token 键控+特征(28,130)；旧 95k frame_id 方案退役，其 .bak 已于 2026-09-10 清理
- [LiDAR-CLIP 权重核对](b4dl-lidarclip-weight-provenance.md) — 本地 ckpt 是原版 ONCE 权重非官方 nuScenes 版；nuScenes 自训已启动；wqlc 环境/去 ddp/向量化 scatter/ckpt 为准的排障经验（2026-08-28）
- [修改边界：仅限项目文件夹](b4dl-modify-only-project-folder.md) — 服务器多项目共用，写操作仅限 /root/autodl-tmp/wql/mmb4dl，跨项目排查只读不写
- [RL 引进计划](b4dl-rl-introduction-plan.md) — 用户拟在 B3/B4a SFT 之后引入 RL/RLVR；流程拆解底稿 + 可行性深度分析（实测环境 4.47/trl 未装/自写 GRPO 结论）已产出（2026-09-07/08），奖励函数可复用 evaluate_model 规则函数
- [Checkpoint 精简](b4dl-checkpoint-cleanup.md) — 2026-09-09 清理日志残留（commit a3f58d1）+ checkpoints 145G→1.7G：仅存 stage1 + B0/B3/B4a 顶层 adapter，旧链整版与中间档全删；同日整包备份 1.4G tar 至 backups/ + B3 最优方案存档文档（commit 59fed9a）
- [b4dl_dataset 数据清理](b4dl-dataset-cleanup.md) — 2026-09-10：1.5G→501M，只留官方基座+B3 训练数据+评测元数据（15 项）；被删变体全可脚本再生，台账见 docs/learn docs/B4DL_b4dl_dataset数据清理记录_20260910.md
- [AgentMEM 归档的解散](b4dl-agentmem-archive.md) — 仓库根 `AgentMEM/`（整目录被 gitignore、从未进 git）已于 **2026-09-25 删除**：18 条记忆本体 + 4 份 Qoder 会话归档迁入 git 跟踪的 `docs/memory/`，ZCode 65 个会话全文（12M）**移到仓库外** `/root/autodl-tmp/AgentMEM_zcode_sessions_20260925/`，独有的 `RTX5090_DEBUG_LOG.md` 救回 `docs/learn docs/`（**不是** `Claude_record/`，那个目录也被 gitignore）；真正删掉的只有已被取代的旧快照与失效脚本（约 230K），**无不可恢复损失**
- [Codex 导入版项目记忆](b4dl-codex-project-memory.md) — 2026-09-10 对远程项目的全量盘点、目录职责、数据/特征/checkpoint 状态、B0-B4a 结论、运行约束与后续路线
- [ReasonSeg seg 线全过程](b4dl-reasonseg-seg-line-2026-09.md) — 2026-09-18~25 autodl3：编码器换代 6.3×（0.0165→0.1037）后**七条杠杆全判死**；掩码塌缩机制（正例 3e-4）、A2 Tversky 治好尺寸却治不了实例接地（hit 24/330 逐位不动）、B1 切 LOC 先验落未决中间档；**oracle 探针裁定：点特征分辨率是硬约束**（完美 query 下 test AUC 0.9911，干扰负点 ~289 vs top-K 需 ≤12，仍差 24×）；两个度量 bug（分母模型相关 / 前缀当 in-range 掩码）、以及**判据挑错轴这个复发三次的自身错误**；含 How to apply 与产物路径
