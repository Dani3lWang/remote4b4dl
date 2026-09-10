---
name: b4dl-training-eval-history
description: B4DL 训练与全量评测时间线（2026-08-07~09-08）——点号 bug、混合训练、per-sequence 演进、B0-B4a 基线链、B3 mIoU 超论文、B4a 过采样负结果
metadata:
  node_type: memory
  type: project
  originSessionId: sess_57bc4f46-cf94-4f89-9779-568442660f70
---

训练评测史（早期结论保留：点号 bug、数据失衡、stage2-full、四大错误模式、模式坍缩根因链，详见 git 历史与下方演进）。

**基线链与 mIoU 演进（2026-08-25 ~ 09-08）**：

1. **seqv2/seqv3 与 B0 锁定**：per-sequence 改造（三重错配修复）→ seqv3-mixed 训完评测锁定基线 B0（acc 0.7629 / mIoU 0.2696，ΔmIoU>0.013 才显著）。见 [[b4dl-per-sequence-refactor]]。
2. **B1（09-01，162K projector 重训）**：acc 0.7787 / mIoU 0.2653。pipeline 判据 bug（End: 被 /dev/null 吞）与评测打印崩溃（gpt_score None 格式化）先后修复（aa84be6 / af49548）。
3. **mIoU 错位根因（B1 预测分析）**：模型输出"输入窗口内局部帧号"，评测按场景全局解析 → corr(pred−GT, −序列起点)=0.892；对齐后 mIoU 0.62（42% 完全命中）→ 模型窗口内定位强、缺全局位置信息。
4. **B2（09-02，整场景 --whole_scene）反证**：改回官方整场景输入，mIoU 反而 0.1992（61% 塌缩 000-008 纯先验）→ 40 帧无帧号锚点学不会绝对定位。
5. **官方模型评测（决定性）**：官方 repo checkpoint（68695 simple × 2ep，训练数据无 meta 无 <4DLiDAR>）纯视觉评测 acc 0.7542 / mIoU 0.1737（TG 99.8% 塌缩）≈ 论文 Table 4 无 meta 行（0.763/0.161）→ **官方从未发布论文 Table 3 完整模型（0.311）与带 meta 数据**；我们 meta 自渲染方向正确。
6. **meta 渲染 bug（关键修复 commit 4413b6b）**：ego_text.py first frame 恒 "at the starting position"+ 前向差分，违反论文 §4.1 relative-to-previous（Figure 6 样例位置随帧变化）；修复为 spd_prev/yaw_prev/acc_prev + 相对前一帧位移；re_render_meta.py 重注入 148k（113,053 条）。
7. **B3（09-05，整场景 + meta2）mIoU 0.3467 超论文 0.311**（+0.036）：acc 0.7526（existence 0.676 略降）/ bleu4 0.0965 / rouge_l 0.3234 / bertscore 0.8967 / METEOR NLTK-2005 补算 0.3366（09-07 双后端化，jar 0.1747 仅衔接旧表）。TG 预测分布与 GT 对齐（最大单区间 23% vs B2 61%），残余偏早 −3.9 帧、高帧段欠覆盖（GT start≥25 的 413 条仅预测中 13 条 ≈3%）。meta 语义是 mIoU 第一瓶颈（+0.147 > 输入构造 ±0.06）。
8. **B4a（09-08 完结，TG 高帧段过采样负结果）**：B3 + GT start≥25 的 1,951 条 ×2 过采样（150,222 条，09-07 06:00 起训 3,519 步满 3ep）→ acc 0.7775（existence 0.676→0.724）/ **mIoU 0.3271（Δ−0.0196 显著回退）**；高帧段专项更差（Δ−0.0489，命中 13→6 条，top-1 [06-14] 38.9% 向低帧集中）→ 病灶=输入无显式帧号信号而非训练分布；B3 保持当前最优基线。分析工具 `evaluation/analyze_tg_regression.py`（同口径分桶对比）。pipeline 评测阶段曾超时 rc=124，手动补跑 18:18 完结。
9. **共享 GPU 现实**：CoRViD/xmuda 用户实验接龙长期占卡（28GB 门控常等 15-25h），B3 曾中断留下空壳 checkpoint-400 阻塞续训（已清，448ea7a）；断点续训 + trainer_state 判据是可靠兜底。

**环境注意**：官方 peft 0.19.1 checkpoint 需最小化 adapter_config 才能本地加载（alora 等新字段不兼容）；评测/训练必须用 wqlc 环境 python（系统 python 无 numpy，wqlc 在 /root/autodl-tmp/.conda-stuff/envs/wqlc）；`deepspeed -i` CLI 可用但 `python -m deepspeed.launcher.launch --include` 不可用。实测版本：torch 2.8.0+cu128 / transformers 4.47.0 / peft 0.13.2 / accelerate 1.3.0 / flash-attn 未装（SDPA）。

**归档文档**：`docs/learn docs/B4DL_B3整场景meta修复mIoU超论文_20260906.md`（全链复盘 + 产物路径）；wiki/ 已 2026-09-08 全面刷新至 B3/B4a 时代（Reproduction-Log 为状态中枢页）。后续路线：帧身份锚点消融（A.1 文本锚点/A.2 帧嵌入）→ RFT/DPO → GRPO，详见 [[b4dl-rl-introduction-plan]] 与 learn docs 09-08 两份策略文档。

相关：[[b4dl-project-overview]]、[[b4dl-per-sequence-refactor]]、[[b4dl-eval-methodology-caveats]]
