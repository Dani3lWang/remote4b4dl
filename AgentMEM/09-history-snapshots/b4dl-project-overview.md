---
name: b4dl-project-overview
description: B4DL 复现项目的总体架构、数据规模与关键路径（VTimeLLM + LiDAR-CLIP，6 任务 benchmark）
metadata:
  node_type: memory
  type: project
  originSessionId: sess_57bc4f46-cf94-4f89-9779-568442660f70
---

mmb4dl = B4DL 论文复现项目（ACM MM'25, arXiv 2508.05269，论文 PDF 在 repo `docs/mmb4dl.pdf`）。架构：LiDAR-CLIP(SST) 编码器产出每帧 768 维特征 → mm_projector → VTimeLLM (vicuna-v1.5-7b + LoRA r64)。

- 复现总方案文档：仓库 `docs/B4DL_复现方案.md`（九章：Step0-8 流程、资源清单、官方仓库 12 项坑位清单、附录 B 可运行评测代码）。上游官方仓库 ccho4702/B4DL 的致命坑：vtimellm_arch.py `Linear(128,·)` 与编码器 768 维输出不匹配、generate_description.py 的 break bug、六任务评测/GPT Score/Metatoken 均未随仓库发布——本项目的自建评测与 metatoken 代码即由此而来。

- 6 任务 benchmark：existence / binary_qa / time_grounding(mIoU) / description / temporal_understanding / comprehensive_reasoning(生成类用 BLEU-4/METEOR/ROUGE-L/BERTScore/GPT score)。论文参考值（Table 3）：accuracy 0.762, mIoU 0.311, BLEU-4 0.095, METEOR 0.275, ROUGE-L 0.322, BERTScore 0.897, GPT 59.513。
- 训练数据：HF `ccho4702/nuScenes-B4DL`，stage2.json 68,695 + stage3.json 79,576 = 148,271 ≈ 论文 148K。**官方划分已干净（2026-08-24 实测）**：发布 train 的 699 scenes 与官方 150 test scenes 零重叠，可直接全量 148,271 训练、用 test_qa.json 评测，无泄漏；旧结论"须砍到 559-scene/118,722 防泄漏"是误判（当时把 850 scene 混在一起切内部 test 导致重叠）。原始数据在 `dataset/nuScenes-B4DL/dataset/train/`。
- **Stage1 数据来源（论文 §5）**：stage1 用的是 LiDAR-LLM 论文的 **LiDAR-LLM-Nu-Caption**（162K 静态单帧 caption QA，HF `Senqiao/LiDAR-LLM-Nu-Caption`，GitHub Yangsenqiao/LiDAR-LLM），B4DL HF 仓库不含它。**2026-08-24 已完成转换**：`dataset/LiDAR-LLM-Nu-Caption/`（8 月 2 日已下载，train.json 161,845 条）→ `convert_lidarllm_to_stage1.py` → `mllm/b4dl_dataset/stage1_train.json` **95,048 条 / 16,481 帧，全部落在官方 700 train scenes（0 泄漏）**，文件大小与上游 repo 的 LFS 指针逐字节一致（72,805,495B，即官方数据制备复现成功）。stage1_val.json 42,597 条全是 test scenes，仅可监控不可训练。95K 不是上限（2026-08-27 更正）：本地转换脚本用 sequence_metadata 的 TOKEN_LIDAR_TOP 映射，只覆盖 20,228 个唯一 token（29,862 帧记录含跨序列重复）；而**官方 build_stage1_from_lidarllm.py 根本不做帧映射**——直接按 sample_token→scene 过滤（699 train scenes），train.json 全量 161,845 条全部落在 train scenes，故官方 stage1=162K 即全量。缺的 66,797 条 / 11,513 帧全在 train scenes 且 scene_metadata（850×40 PATH_LIDAR_TOP）可解析，可恢复；瓶颈在特征：现有 29,862 个按 frame_id（官方管线内部 ID，不可本地推导）命名，11,513 帧无特征。补全路径：特征改按 sample_token 命名（scene_metadata 路径 + sample_data.json 得 token），dataset.py 按 id 加载。官方 assets/（sample_token_to_scene.json、train_scene_tokens.json）与 datageneration/tools/ 三个脚本本地缺失，GitHub 外网抓取超时。特征 `encoders/lidarclip/b4dl/stage1_features/`（29,862 帧，frame_id 键控）已就绪。
- 数据关键路径（都在远端 `mllm/b4dl_dataset/`）：stage2_full_train.json、test_qa.json (30,145 条/150 scenes)、ego_metadata.json；特征 `encoders/lidarclip/b4dl/stage2_features/<scene_id>.npy`（scene 级 [~40帧, 768] float16，850 个）；sequence_metadata.json 在 `datageneration/data/metadata/`（5100 序列，每序列 indices 数组）。

相关：[[b4dl-server-access-workflow]]、[[b4dl-training-eval-history]]、[[b4dl-per-sequence-refactor]]、[[b4dl-eval-methodology-caveats]]
