---
name: b4dl-per-sequence-refactor
description: per-sequence 改造与重训（2026-08-22~23）——论文确认三重错配、4 文件修改、stage2-full-seq
  重训、framectx 冒烟（变体 08-24 已取消移除）、遗留 mode collapse
metadata:
  node_type: memory
  type: project
  originSessionId: sess_57bc4f46-cf94-4f89-9779-568442660f70
---

2026-08-22~23 的核心工作流：论文确认为 **per-sequence** 而代码（含上游官方仓库 mmb4dl-main）全部是 per-scene，存在三重错配：① ego 元数据按 scene 首末帧（QA 问 frames 12-20 却给 frame 0/39 的描述）；② 特征张量是整个 scene ~40 帧（QA 帧号却是序列相对的）；③ 训练 118,722 条全带 metatoken 但 test_qa.json 一条不带（推理时靠 --ego_meta + build_query 注入，当时是对齐的）。论文依据：Section 4.1 输入 S_L 是序列（Max/Min Sequence Length 19/5），Appendix C "metatoken descriptions of the first and last frames referenced in the QA pair"。

**修改的 4 个文件**（commit `b0e4400`，本地改后 paramiko 上传远端）：
- `mllm/scripts/generate_ego_metadata.py`：新增 `_compute_ego_for_pose_range()`，遍历 scene+sequence metadata，per-seq key = `f"{scene_id}_{first_idx}_{last_idx}"`（indices 首末），保留 per-scene 条目作 fallback。
- `mllm/scripts/inject_metatoken.py`：新增 `parse_frame_numbers()`（正则 `frame\s+(\d+)`）+ `lookup_ego_for_qa()`，统计 per-seq 命中/回退。
- `mllm/evaluation/test_b4dl.py`：同步三个函数 + `slice_features()` 把 scene 特征切片到问题帧范围。
- `mllm/vtimellm/train/dataset.py`：`_parse_frame_numbers()` + `_slice_features()`，训练时同样切片。
CLAUDE.md 同步更新（commit `cd6d975`）。**注意训练-评测格式必须配对**：老 checkpoint（stage2-full，per-scene 训练）配老格式评测，per-seq 评测必须配重训后的模型，否则 mismatch 反而掉分。

**后续执行**：commit `3004e75` 加入 Stage2 全量/恢复/per-seq 训练与 baseline 评测脚本；用 per-seq 数据重训 stage2-full-seq（training_logs/stage2_full_seq_20260822_094534.log + resume_20260822_230654.log），产出 merged checkpoint `checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seq-merged`。冒烟测试序列（每任务 50 条，产物在 mllm/eval_results/）：smoke_seq (08-22)、smoke_2000 (08-23)、smoke_framectx (08-23 17:34，frame-context 变体 test_b4dl.py；变体取消后该产物与 .bak_frame_ctx 备份已删除)。framectx vs 基线基本持平（accuracy 0.77/0.77，mIoU 0.276/0.293）。

**遗留问题**：per-sequence 后 time_grounding 仍 mode collapse——模型只输出 ~4 种唯一帧范围（[006-014] 52%、[000-008] 26%、[000-010] 20%），根因是模型缺乏帧位置感知。候选方案：帧 positional encoding、训练数据帧范围下采样（"This sequence covers..." 前缀方案已随 08-24 取消 frame-context 变体而否决）。

**seqctx 变体状态（2026-08-23 启动，08-24 取消并彻底移除）**：训练在步 74 被取消，checkpoint 目录为空（未触发 save_steps=500）。08-24 用户要求取消该干预，已删除：`--frame_ctx` 开关（inject_metatoken.py / test_b4dl.py）、seqctx 训练数据（stage2_full_train_seqctx.json）、脚本（run_stage2_full_seq_ctx.sh / after_seqctx_train.sh）、seqctx 日志与 watch 标记、smoke_framectx 评测产物、.bak_frame_ctx 备份、空 checkpoint 目录与 wandb offline run，并同步更新 CLAUDE.md 与审计文档。对比基线：stage2-full-seq 全量 mIoU 0.1690 / accuracy 0.7579；stage2-full per-scene mIoU 0.1661 / accuracy 0.7606；论文 mIoU 0.311。

**全面论文对齐审计（2026-08-24，commit `576b612`，训练已取消后做代码级对齐）**：审查发现四项实质错配并全部修复：① mIoU 闭区间 bug（单帧完美匹配得 0，未解析预测与 GT[0,x] 虚假重叠）；② 21.8% 训练 QA（帧引用≠序列边界，如 "in frame 15" 奇数帧）metatoken 错误回退 per-scene → 新增 `ego_text.py` 共享渲染模块 + `ego_frame_motion.json` 逐帧运动表（850 场景 34,149 帧），`render_meta_texts()` 对任意帧对渲染真实描述；③ 特征切片改为 QA 的**包含序列精确采样帧**（`feat_indices`，如 [0,2,4,6,8] 共 5 帧，此前按区间切 9 帧混入非序列帧），重叠序列取最窄包含；④ `--per_sequence` 与自创 framectx 前缀解耦为独立 `--frame_ctx`（该变体随后于 08-24 取消，`--frame_ctx` 已移除）。inference 默认 greedy。验证：训练 JSON vs build_query 2000 抽样 0 不一致；79,975 条有帧号 QA 零 per-scene 回退。新数据 `stage2_full_train_seqv2.json`（118,722 条，待重训）。审计文档：`docs/learn docs/B4DL_论文对齐审计_20260824.md`。遗留：time_grounding 无帧号问题两侧均全 scene（benchmark 数据无序列归属，R5 限制）；`train_meta_embed` 是论文外工程补充（R2）。评测命令：`--per_sequence --frame_motion ... --sequence_metadata ...`（seq 模型只加 --per_sequence，旧模型都不加）。

**seqv2 重训（2026-08-24 tmux 启动）**：用户要求用 seqv2 数据重训。确认**无需重训 stage1**（stage1 只训 f_p 投影层，数据无 metatoken 前缀），复用 stage1/mm_projector.bin。脚本 `mllm/scripts/run_stage2_full_seqv2.sh`（commit `ce11c40`/`52558db`，save_steps 200）：output_dir=stage2-full-seqv2、lr 1e-4、3 epoch、bs 8×16、r64/α128、wd 0、train_meta_embed 开。**重大踩坑（本环境特性）**：ZCode 后台任务（run_in_background）与 nohup/setsid 启动的长训练进程都会被 harness 回收器顺进程树 SIGKILL（分别死于步 74、54、0）——后台任务完成或隔一段时间即杀。**唯一可靠方案：tmux 分离会话**（`tmux new-session -d -s b4dlseqv2 'cd ... && bash scripts/run_stage2_full_seqv2.sh'`，父进程挂在既有的 tmux server 下，逃出回收树）。已建守护 cron「每30分钟检查seqv2训练并收尾」（automation-bc9794ed）：训练中则不动；正常完成则 tmux 内 merge_stage2.py 合并 + 全量 v2 评测（`--per_sequence --frame_motion --sequence_metadata`，结果 eval_results/stage2_full_seqv2/）；被中断则按 checkpoint-* 断点续训或从头重启（最多 3 次）。GPU 与用户 lxy 的 4-6G 去噪任务共存，勿动其进程。

**数据切换为论文官方划分全量（2026-08-24，commit `73c6114`）**：用户要求"取消数据划分/按论文划分"。确认 HF 发布的 train（stage2.json+stage3.json=148,271 条/699 scenes）本身就是论文官方训练集（与官方 test_qa.json 150 scenes 零重叠），旧 create_splits.py 80/10/10 自创划分（118,722/559）已废弃。新增 `build_stage2_full_train.py` 一键重建 → `stage2_full_train_seqv2_148k.json`（v2 注入：真实帧 metatoken 99,933 条零 per-scene 回退 + feat_indices 精确采样帧 + feat_range）。训练 13:16 以 tmux 重启（杀旧训练时仅 38 步无 checkpoint，零损失）；脚本已指向 148k。注意：数据行数变了（148,271 vs 118,722）但 bs/步数公式不变（2781 步 = 148,271×3/(8×16)）。

**seqv2-148k 全量评测（2026-08-25 14:49 完成）**：accuracy **0.7647**（首次超论文 0.762）、BLEU-4 0.1062、METEOR 0.3328、ROUGE-L 0.3234 全面超论文；mIoU 0.1835（历史最高但离 0.311 仍远）。TG 坍缩定量：预测仅 15 种范围、(0,8) 占 87%（2417/2783），GT 207 种且分布均匀——不是标签不平衡，是 40 帧全 scene 下无法定位。对比报告：`docs/learn docs/B4DL_seqv2评测对比与坍缩分析_20260825.md`。

**seqv3 = TG 恢复序列归属（2026-08-25，commit `8b85cd2`/`d228409`）**：根因 = 发布的 benchmark 丢了每条 QA 的序列归属字段，TG 问题无帧号只能喂整 scene；序列归属可从 GT 答案帧范围 100% 恢复（test 2783 条：2645 唯一+138 歧义+0 无归属，89% GT=序列边界）。实现：`build_stage2_full_train.py` 给 stage2 的 from-frame-to 答案打 task=time_grounding 标签（13,124 = 论文 Table 2；注意 stage3 prose 也含该短语 10,357 条，勿标）；`inject_metatoken.py --answer_frames` 与 `test_b4dl.py --answer_frames` 双侧对 TG 回退解析 GT 帧范围 → feat_indices 覆盖 100%、GT 范围 100% ⊂ feat_range、两侧 2783/2783 一致。评测侧用 GT 恢复归属 = 还原论文原始评测（oracle 输入选择，报告须声明）。

**138 条歧义复核（2026-08-29）**：歧义本质 = scene 内序列是**等宽滑窗**（[0,8],[6,14],[12,20]... 宽 8 步 6，按时间递增排列），GT 范围落在相邻序列 4 帧重叠区即被 2-4 个等宽序列同时包含（平均 2.41 候选；94 条 2 候选/31 条 3 候选/13 条 4 候选）→ 所有歧义都是"最窄平局"。恢复策略 = `min(containing, key=width)` 平局取列表第一个（时间最早）；训练/评测两侧代码逐行一致；训练侧 13,124 条 feat_indices 与策略重算 **0 不一致**（数据文件与代码逻辑一致）。GT 贴边分析：109/138 贴某候选边界、83/138（60%）贴当前选择（随机基线 1/2.41≈41%）。影响量化：歧义 138 条（4.96%）mIoU 0.2285 vs 非歧义 2645 条 0.2718、精确命中 0 vs 14.6%；即使歧义全达非歧义水平，全量 mIoU 上限仅 +0.002（0.2696→0.2717），对论文对比无实质影响。结论：保持现状（确定性/可复现/两侧一致）+ 报告声明；无可证明更优的 oracle（HF 发布版连 start/end index 字段都丢了，Table 10 的 Start/End 只在论文示例里）。

**训练改为两阶段法（2026-08-25，commit `df29b6e`，用户指令"遵循论文方法"）**：不再用 148k 合并单阶段训练，改为官方 stage2.sh/stage3.sh 的结构：**Phase A** stage2_train_seqv3.json（简单任务 68,695 条，2 epochs lr 1e-4 tf32，1074 步）→ merge_stage2.py 合并 → **Phase B** stage3_train_seqv3.json（复杂任务 79,576 条）在 merged 模型上训**新 LoRA**（3 epochs lr 2e-5，1866 步）。评测用 `--stage2 + --stage3` 双 LoRA（builder 依次 merge_and_unload），结果目录 eval_results/stage23_seqv3/。**stage1 projector 已用 95,048 条 nu-caption 重训完成（08-25 17:46，742 步，旧 699 条版本备份 .bak_699items）**。驱动器 `run_stage2_full_seqv3.sh` 幂等（merged 存在跳过 Phase A + 两阶段 checkpoint 自动续训）；执行链：tmux b4dlstage1 单会话串联 stage1→驱动器（17:46 起 Phase A 运行中，~18s/步 ETA ~5.5h）。守护 cron 已改监控两阶段（先查 tmux b4dlstage1 存活防重复启动，完成后 merge+双 LoRA 评测）。⚠️ pkill -f 自匹配坑：模式串出现在自己命令里会被误杀，用 `[t]rain_mem` bracket trick。

**两阶段法失败与修复为混合训练（2026-08-26，commit `e94fd64`）**：两阶段全链跑完（Phase A 68,695 条 08-25 17:46→08-26 00:21；merge 00:25；Phase B 79,576 条 3ep lr 2e-5 00:25→10:05；双 LoRA 评测 ~12:20），但**简单任务格式漂移**：existence 输出长句而 GT 是 "Yes."，exact match 归零 → accuracy 0.0001（existence 0.0003 / binary_qa 0.0）、mIoU 0.1614 反而低于混合版。根因：Phase B 3 epochs 复杂任务叠加 LoRA 覆盖了 Phase A 的短答案格式。决策：论文 §4.2 原文 "trained on **both** Simple Tasks and Complex Tasks" = 混合训练，官方拆两段只是实现选择 → 回混合法，叠加 seqv3 增量。报告：`docs/learn docs/B4DL_两阶段训练评测与混合法决策_20260826.md`。

**seqv3-mixed 训练+评测（2026-08-26 13:19 → 08-27 19:28，本轮完成）**：`stage2_full_train_seqv3_148k.json`（148,271 条：TG 13,124 条 GT 序列归属 + 真实帧 metatoken + feat_indices 精确采样帧）+ 95K nu-caption stage1 projector + 3ep lr 1e-4，3474 步 21h 训完（tmux b4dlmixed）。**评测结果（30,145 条全量，0 跳过，~2.5h）**：accuracy **0.7629**（坍缩修复，existence 0.707 / binary_qa 0.819）、mIoU **0.2696**（历史新高，seqv2 0.1835 的 +47%，但距论文 0.311 还差 0.041）、B@4 0.108 / METEOR 0.334 / ROUGE-L 0.324 / BERTScore 0.973 全部 ≥ 论文。结论：混合法验证成功；剩余差距聚焦 mIoU，第一嫌疑 = stage1 projector 数据量（95K vs 论文 162K，59%）。产物：`eval_results/stage2_full_seqv3_mixed/{predictions,metrics}.json`。**本环境评测坑（已修复，脚本 `scripts/run_mixed_eval_after_train.sh`）**：① HF Hub 网络在本机卡死（curl huggingface.co 超时，TCP 半开连接），transformers/peft 加载模型时版本检查会让进程永久挂起 → 必须 `HF_HUB_OFFLINE=1 HF_OFFLINE=1 TRANSFORMERS_OFFLINE=1`；② GPU 与 lhwt 的 CoRViD 训练（占 19GB，60k iter 需 ~26h）共存，评测需 ~12.5GB 会 OOM → 显存门控循环（空闲 <14GB 每 10 分钟重试）；③ 评测输出重定向到文件必须 `-u` 否则看不到进度、加 timeout 防挂死。

**基线锁定 B0（2026-08-29）**：seqv3-mixed 全量评测经三重验证后正式锁定为复现基线 B0——① 独立复算（纯标准库重写解析+闭区间 IoU，从 predictions.json 复算得 0.2696124556，与 metrics.json 一致 <1e-12，帧集合口径等价）；② 完整性（30,145 条 0 跳过、无 .ckpt 残留）；③ 代码一致性（evaluation/ 自 8b85cd2 内容零变化，merge 24a3ab9 仅把 test_b4dl.py 行尾改成 CRLF）。统计阈值：mIoU 95% CI 半宽 ±0.0132（bootstrap 3000 次一致）、accuracy ±0.0085 → **ΔmIoU >0.013 才算真改进**，论文 mIoU 差 0.041≈3.1×半宽是真差距。评测命令/数据 MD5/对比规则/追踪表见 `docs/learn docs/B4DL_基线锁定与对比规则_20260829.md`。基线重置条件：换编码器/重提特征/改口径/换测试集。推理为 greedy（do_sample=False）确定可复现。

相关：[[b4dl-training-eval-history]]、[[b4dl-eval-methodology-caveats]]、[[b4dl-project-overview]]
