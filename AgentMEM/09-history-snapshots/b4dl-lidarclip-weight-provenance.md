---
name: b4dl-lidarclip-weight-provenance
description: LiDAR-CLIP 编码器权重核对结论——本地 ckpt 是原版 ONCE 权重非官方 nuScenes 版，nuScenes
  自训已启动（2026-08-28）
metadata:
  node_type: memory
  type: project
  originSessionId: sess_c330aa48-61cd-4d72-915c-a767a80c9121
---

2026-08-28 核对（问题清单 #3，报告 `docs/learn docs/B4DL_LiDARCLIP权重核对_20260828.md`）：

- **本地 `encoders/lidarclip/lidarclip/checkpoint/vit_l_14.ckpt` = 原版 LiDARCLIP（atonderski）发布的 ONCE 数据集权重**（MD5 `cd04e5e0...`，与 `/root/autodl-tmp/ljq/mmb4dl-main/encoders/lidarclip/ckpt/lidarclip_vitl14_once.ckpt` 字节级一致；epoch 2 / step 40500 / PL 1.7.2）。README 曾误标"本地训练产物"已修正。
- 官方 B4DL **不发布**编码器权重（README 要求自训），论文 §5.1 要求 nuScenes 预训练（`--dataset-name nuscenes`，官方 train.py CLI 默认值），loader 内置 700 训练场景过滤。现有全部特征（stage1 95K/28K、stage2 28K）均由 ONCE 权重提取。
- **nuScenes 编码器自训已启动**：`ckpt_nuscenes/lidarclip_mm/`（官方配方 batch 32 / Adam 1e-5 + OneCycle 1e-3 / precision 16 / max_epochs 20 / ckpt 每 250 步），CoRViD 共存 ~3-4s/步，一 epoch≈5-6h；ONCE 官方权重停在 epoch 2，计划 2-3 epoch 后早期停止 → 重提全部特征（与问题 #1 的 162K 补数据合并为一次提取）→ 重训 projector/混合。
- 环境与代码要点（复训必读）：**训练用 wqlc 环境**（torch 2.8.0+cu128 支持 5090；b4dl 环境 torch 2.5.1 无 sm_120 内核只能 CPU）；本地 clip.load 不转 fp16 需 `clip.model.convert_weights()`（train.py 已加）；单卡**去掉 strategy="ddp"**（与 fork dataloader 互锁）；`_mmdet3d_compat.py` 的 dynamic scatter 已重写为向量化可微实现（原 Python 循环 batch32 单步 >20 min，且 Function backward 被置 no-op 导致 DynamicVFE point 级层无梯度——对训练是正确性 bug，等价性/梯度/速度验证见 `encoders/lidarclip/validate_scatter.py`）。
- 排障经验：**判断训练是否推进以 ckpt 文件为准**——wandb offline 历史记录刷盘可延迟数十分钟、nohup stdout 缓冲吞进度条，都会造成"卡死"假象（v3/v4 曾因此被误杀）。

**编码器训练定稿（2026-08-30）**：nuScenes 自训实际轨迹 = 高 LR 原日程续训至 step ~31,550（v2 监控硬停，08-29 20:25）→ 链式短程退火（`run_anneal_chain.sh`，tmux b4dlanneal，08-29 20:36 → 08-30 ~10:50 跑完 3 epoch，OneCycle 1e-4→0，seed 0，`--load-only-model` 从 last.ckpt 起步）。**val MSE 探针（`val_mse_probe.py`，15 个 val 场景 3,616 对，train 模式约定）：基线（未退火）0.10609 → epoch1 0.10264 → epoch2/last 0.0992，退火净收益 −6.5%，末段三 ckpt 差 <2e-5 已收敛**。**最终编码器候选 = `ckpt_anneal/lidarclip_mm/last.ckpt`**；train 0.063 vs val 0.099 的泛化差距仍在（过拟合 train 场景），继续训练无益——若下游 mIoU 仍不动，下一嫌疑是编码器数据配方而非时长。下一步（待修改清单阶段 1）：先修提取脚本（eval() 决策 + strict 校验）→ 全量重提 28,130 stage1 帧 + 850 stage2 场景 → stage1 162K → mixed 重训。

**B1 流水线已启动（2026-08-30 12:53，tmux b1pipeline）**：`mllm/scripts/run_b1_pipeline.sh` 链式执行——全量重提特征（定稿编码器 ckpt_anneal last.ckpt，eval 模式+strict 校验）→ stage1 162K projector 重训（stage1.sh）→ mixed-b1 重训（`run_stage2_full_seqv3_mixed_b1.sh`，**独立 output_dir -mixed-b1，B0 checkpoint/评测产物原封不动**；95K projector 备份 .bak_95k，旧特征目录改 `*_once`）→ 同口径评测至 `eval_results/stage2_full_seqv3_mixed_b1/`。任一阶段硬失败即停链（cron 每小时只读报告，automation-a082720a）。判读：对 B0（acc 0.7629/mIoU 0.2696）**ΔmIoU > +0.013 才显著**。清单代码项已全部闭环（0.4/0.6/1.1/1.2/2.1/2.2/2.3 commit 4cd4ada/e8639e2，2.4 本轮 1393b23）。

相关：[[b4dl-project-overview]]、[[b4dl-stage1-official-162k]]、[[b4dl-per-sequence-refactor]]
