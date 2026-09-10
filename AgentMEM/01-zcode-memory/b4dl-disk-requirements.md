---
name: b4dl-disk-requirements
description: B4DL 磁盘容量需求实测（2026-09-10 盘点）：训练/评测必需 ~30G、含 nuScenes 关键帧子集
  ~60G、验收建议数据盘 200G；sweeps 338G 代码零引用可用不上
metadata:
  node_type: memory
  type: project
  originSessionId: sess_03e0f32b-5fc1-4f0d-9896-2f38649d5be6
---

2026-09-10 为「租云服务器」做的磁盘盘点（实测 du）：

- **本项目真实占用**（与同盘其他项目隔离计算）：mmb4dl 仓库 46G + nuScenes 404G + wqlc 环境 8.4G + CLIP 权重/cache ~2G。
- **仓库 46G 明细**：mllm 17G（base_model/vicuna-v1-5-7b 13G、checkpoints 1.7G、b4dl_dataset 1.6G、wandb 137M、eval_results 87M）、models/roberta-large 15G（BERTScore 用，五种格式重复，只 model.safetensors 1.4G 有效）、encoders/lidarclip 11G（ckpt_nuscenes 3.9G + ckpt_anneal 3.9G 各 4×1.03G、vit_l_14.ckpt 987M、pretrained/ViT-L-14.pt 922M、特征 0.55G）、backups 2.3G、.git 1.5G、.claude 601M（其中 worktrees 452M）。
- **nuScenes 404G 明细**：sweeps 338G、samples 53G（LIDAR_TOP 23G、6 相机 30G、radar 1.1G）、v1.0-trainval 表格 2.5G、lidarseg 1.2G。
- **关键结论（代码验证）**：`sweeps/` 在 datageneration/ 与 mllm/ 全代码零引用；`extract_pc_features_sample_token.py` 经 `NuscenesImageLidarDataset_with_path._setup` 只遍历关键帧 sample，不读 sweeps；lidarseg 也未被引用。所以**训练/评测只需特征，重提特征只需关键帧子集** = LIDAR_TOP 23G + CAM_BACK_RIGHT 5.1G（loader 会 open 该相机图像，虽不参与特征，但缺文件会报错）+ v1.0-trainval 表格 2.5G ≈ 31G。
- **容量建议**：只训练/评测 ~30G（含 env）→ 100G 数据盘够；推荐档 = 加 nuScenes 关键帧子集 ≈ 60-90G → **数据盘 200G**（含训练工作区：merged 完整 7B 14G + save_total_limit 3 的滚动 ckpt ~6G，历史峰值 checkpoints 曾达 145G）；想把 nuScenes 全量放同机则 600G~1T，但 338G sweeps 对本项目是纯浪费。系统盘 ≥30G（环境若装系统盘则 50G+）。
- **迁移注意**：官方 nuScenes blob 按 samples+sweeps 混合打包，无法只下关键帧 → 关键帧子集要从现机拷贝，不要重下 400G。

关联：[[b4dl-project-overview]]、[[b4dl-per-sequence-refactor]]、[[b4dl-checkpoint-cleanup]]、[[b4dl-server-access-workflow]]
