
## 预训练权重下载

请下载 [ViT-L-14.pt](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) 并放置在 `./pretrained/` 目录下。

## 训练 LiDARCLIP

> **注意：** 在 2x RTX 5090 (32GB) 上使用 ViT-L/14 训练时，`--batch-size` 不宜超过 32，否则会触发 OOM。建议配合梯度累积使用，等效保持较大有效 batch size。

```bash
# 从头训练（官方 B4DL 流程：--dataset-name nuscenes，默认即为 nuscenes）
python train.py \
    --name=lidarclip \
    --checkpoint-save-dir=./ckpt \
    --batch-size 32 \
    --workers 4 \
    --data-dir /root/autodl-tmp/Datasets/nuScenes/ \
    --clip-model ViT-L/14 \
    --dataset-name nuscenes

# 从 checkpoint 继续训练（注意：`lidarclip/checkpoint/vit_l_14.ckpt` 是原版
# LiDARCLIP 发布的 **ONCE 数据集**权重，MD5 cd04e5e0eed557cc3073bdf3b8e268b6，
# 并非本地训练产物，也非官方 B4DL 要求的 nuScenes 版本——核对结论见
# docs/learn docs/B4DL_LiDARCLIP权重核对_20260828.md）
python train.py \
    --name=lidarclip \
    --checkpoint-save-dir=./ckpt \
    --checkpoint ./lidarclip/checkpoint/vit_l_14.ckpt \
    --batch-size 32 \
    --workers 4 \
    --data-dir /root/autodl-tmp/Datasets/nuScenes/ \
    --clip-model ViT-L/14 \
    --dataset-name nuscenes
```

### 显存不足时的解决方案

| 方案 | 修改方式 |
|------|---------|
| 减小 batch size | `--batch-size 16` 或 `--batch-size 8` |
| 启用梯度累积 | 在 `train.py` 的 `Trainer` 中添加 `accumulate_grad_batches=4` |
| 换用更小的 CLIP 模型 | `--clip-model ViT-B/32`（约 1/3 显存占用） |

### 为什么不用官方预训练权重？

原始 [LiDARCLIP](https://github.com/atonderski/lidarclip) 官方权重在 **ONCE 数据集**上训练，而本项目使用 **nuScenes 数据集**。两者 LiDAR 传感器不同，存在明显的 domain gap。因此推荐在 nuScenes 上本地训练。

> **重要（2026-08-28 核对）**：官方 B4DL 仓库从未发布训练好的编码器权重，
> 官方 README 要求 "You need to train the model first"；论文 §5.1 明确编码器
> 用 nuScenes 预训练（`--dataset-name nuscenes`，700 训练场景过滤）。
> 本目录 `lidarclip/checkpoint/vit_l_14.ckpt` 实为原版 LiDARCLIP 的 **ONCE 权重**
> （2026-05-26 从 mmb4dl-main 包复制），此前全部特征（stage1 95K/28K、stage2 28K）
> 均由它提取——正式复现应使用 nuScenes 自训权重重新提取。详见
> `docs/learn docs/B4DL_LiDARCLIP权重核对_20260828.md`。
>
> 本地环境注意：① wqlc 环境的 `clip.load()` 不转 fp16，train.py 已加
> `clip.model.convert_weights()`；② `_mmdet3d_compat.py` 的 dynamic scatter
> 已向量化并修复梯度 no-op bug（原实现 batch 32 单步 >20 min 且 point 级层无梯度）。

## 提取 LiDAR 特征

训练完成后，使用训练好的权重提取特征：

```bash
python extract_pc_features.py \
    --checkpoint ./lidarclip/checkpoint/vit_l_14.ckpt \
    --scene-json-path ./annotations/scene_metadata.json \
    --frame-json-path ./annotations/sequence_metadata.json \
    --stage1-save-dir ./b4dl/stage1_features/ \
    --stage2-save-dir ./b4dl/stage2_features/
```

提取的特征将用于 B4DL 模型的 Stage 1/2/3 训练。
