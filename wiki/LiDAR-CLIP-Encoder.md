# LiDAR-CLIP 编码器

编码器以 SST 为 backbone，把 nuScenes 点云序列映射为与 CLIP 对齐的 768 维特征。

```bash
cd encoders/lidarclip
conda run -n wqlc python extract_pc_features.py \
  --checkpoint /path/to/lidarclip.ckpt \
  --scene-json-path ./annotations/scene_metadata.json \
  --frame-json-path ./annotations/sequence_metadata.json \
  --data-path /path/to/nuScenes \
  --stage1-save-dir ./b4dl/stage1_features/ \
  --stage2-save-dir ./b4dl/stage2_features/
```

Lightning checkpoint 加载需 `weights_only=False`、`strict=False`。同一次下游训练只能使用同一编码器版本生成的完整特征集，禁止增量混合。

Stage1 的正式方案使用 sample-token 键控数据；Stage2/B3 使用按场景保存的 `(N_frames, 768)` 特征。
