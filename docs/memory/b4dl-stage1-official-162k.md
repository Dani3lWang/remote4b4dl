---
name: b4dl-stage1-official-162k
description: stage1 已对齐官方 162K 方案——sample_token 键控数据与特征（2026-08-28 完成），旧
  frame_id 键控 95k 方案退役保留
metadata:
  node_type: memory
  type: project
  originSessionId: sess_b97f1cf9-261c-4530-8e8b-ee09a10dafbe
---

2026-08-28 完成 stage1 官方对齐（README 确认官方逻辑）：

- 官方：直接读 HF Senqiao/LiDAR-LLM-Nu-Caption 全量，按 assets/{sample_token_to_scene,train_scene_tokens}.json（699 train scenes）过滤，不做帧映射，id=sample_token；特征每帧 {sample_token}.npy
- 本地近似：scene_metadata.json 700 train scenes（0 泄漏实测，161,845 条全命中）
- 产物：`mllm/b4dl_dataset/stage1_train.json`（161,845 条，scene_id=sample_token）、`encoders/lidarclip/b4dl/stage1_features_sample/`（28,130 个 {sample_token}.npy，(1,768) float32）
- 脚本：`datageneration/tools/build_stage1_from_lidarllm.py`（官方逻辑复刻）、`encoders/lidarclip/extract_pc_features_sample_token.py`（复用 with_path loader，点云裁剪到 CAM_BACK_RIGHT 视锥）、`mllm/scripts/verify_stage1_sample_data.py`
- stage1.sh 的 --feat_folder 已指向 stage1_features_sample
- 旧方案（frame_id 键控 stage1_features/）保留给旧 checkpoint（其配对数据 `stage1_train_frameid_95048.json.bak` 已于 2026-09-10 数据清理中删除）；stage1_val.json（42,597 条全在 test scenes）仅监控用
- 验证：161,845 条 0 缺特征，dataset.py 冒烟 300/300 通过（2026-08-28 晚间复核通过）
- 官方 699 scenes 已精确复刻（2026-08-28 晚）：官方仓库 ccho4702/B4DL 的 `datageneration/tools/assets/{sample_token_to_scene,train_scene_tokens}.json` 均已下载（raw 直连下载 2.4MB 需 `--max-time 480 -C -` 续传，默认超时失败）；实测官方 699 ⊂ 本地 700（只差 1 个 scene、216 条样本）；`stage1_train.json` 最终为官方 699 版 **161,629 条**，0 缺特征，dataset 冒烟 300/300 通过；700 版备份 `stage1_train_700scenes_161845.json.bak`（已于 2026-09-10 数据清理删除，可由 `dataset/LiDAR-LLM-Nu-Caption/train.json` + `datageneration/tools/build_stage1_from_lidarllm.py` 再生，见 [[b4dl-dataset-cleanup]]）
- 官方 tools 三脚本已补全：`build_stage1_from_lidarllm_official.py` / `create_metadata.py` / `generate_stage1_caption.py`（官方脚本输出 id 字段 + `<image>` 标记，本地 dataset.py 用 scene_id、自动替换 `<image>`→`<video>`）

**Why:** 论文 stage1 是 162K 全量，本地旧实现只有 95K（frame_id 映射丢 41%）。
**How to apply:** stage1 重训用新数据+新特征目录；提取特征时注意 loader 会遍历 trainval 全量（含 150 val scenes，需按 scene 过滤）；环境有 GPU 共享时提取速度会从 ~1s/it 掉到 ~3s/it。相关 [[b4dl-per-sequence-refactor]] [[b4dl-training-eval-history]]
