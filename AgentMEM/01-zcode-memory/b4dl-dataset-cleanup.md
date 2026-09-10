---
name: b4dl-dataset-cleanup
description: 2026-09-10 b4dl_dataset 数据清理（1.5G→501M）：只留官方基座+B3 数据+评测元数据，被删变体全部可脚本再生
metadata:
  node_type: memory
  type: project
  originSessionId: sess_2187ad8d-b4e1-4f7f-b819-f41ea3e00a7f
---

2026-09-10 清理 `mllm/b4dl_dataset/`：1.5 GiB/34 文件 → 501 MiB/15 文件（删 19 项 ≈ 1.04 GiB）。该目录被 gitignore，删除不可回滚，台账即权威记录。

保留（15 项）：官方来源（`stage1_train.json` 161,629 / `stage1_val.json` 42,597 / `test_qa.json` 30,145 / `test_qa_simple_only.json` 14,078 / `ego_metadata.json` / `ego_frame_motion.json` / 5 个 133B LFS 指针）+ 官方 148k 基座 `stage2_full_train_148k.json`（TG 标签 13,124 完整）+ B3 训练数据 `stage2_full_train_seqv3_meta2_148k.json` + B4a 过采样文件（因 `_vram_smoke_bs8/run_measure.sh` 待跑作业的 `--data_path` 暂留，跑完可删）。

已删：全部 `.bak*`（含 stage1 官方超集 161,845 与旧 95k frame_id 版）、旧 80/10/10 划分链（`stage2_full_train{,_seq,_seqv2}.json`、`stage2/3_{val,test}.json`、`stage2_full_val.json`）、被取代注入代次（`stage2_full_train_148k_seq.json`、`_seqv2_148k`、`_seqv3_148k`（B0/B1/B2 训练数据）、`stage2_train_seqv3.json`/`stage3_train_seqv3.json`（论文两阶段）、`stage2_train.json`/`stage3_train.json`）。

**Why:** 旧变体都能由保留文件经现成脚本再生（`build_stage2_full_train.py` → `inject_metatoken.py` → `re_render_meta.py` → `oversample_tg_highframe.py`）；B 系列评测产物与 adapter 备份未受影响，历史数值照旧可查。

**How to apply:** 台账与逐条再生命令见 `docs/learn docs/B4DL_b4dl_dataset数据清理记录_20260910.md`（commit 12f281e，已 push）；若后续要"用 seqv3_148k 重训 B0"之类操作，先按台账再生再跑。关联 [[b4dl-training-eval-history]]、[[b4dl-checkpoint-cleanup]]、[[b4dl-server-access-workflow]]。
