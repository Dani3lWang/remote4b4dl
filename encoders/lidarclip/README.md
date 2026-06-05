
## 预训练权重下载

请下载 [ViT-L-14.pt](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) 并放置在 `./pretrained/` 目录下。

## 训练 LiDARCLIP

> **注意：** 在 2x RTX 5090 (32GB) 上使用 ViT-L/14 训练时，`--batch-size` 不宜超过 32，否则会触发 OOM。建议配合梯度累积使用，等效保持较大有效 batch size。

```bash
# 从头训练
python train.py \
    --name=lidarclip \
    --checkpoint-save-dir=./ckpt \
    --batch-size 32 \
    --workers 4 \
    --data-dir /root/autodl-tmp/Datasets/nuScenes/ \
    --clip-model ViT-L/14

# 从 checkpoint 继续训练（推荐：已有 epoch 2 的本地权重）
python train.py \
    --name=lidarclip \
    --checkpoint-save-dir=./ckpt \
    --checkpoint ./lidarclip/checkpoint/vit_l_14.ckpt \
    --batch-size 32 \
    --workers 4 \
    --data-dir /root/autodl-tmp/Datasets/nuScenes/ \
    --clip-model ViT-L/14
```

### 显存不足时的解决方案

| 方案 | 修改方式 |
|------|---------|
| 减小 batch size | `--batch-size 16` 或 `--batch-size 8` |
| 启用梯度累积 | 在 `train.py` 的 `Trainer` 中添加 `accumulate_grad_batches=4` |
| 换用更小的 CLIP 模型 | `--clip-model ViT-B/32`（约 1/3 显存占用） |

### 为什么不用官方预训练权重？

原始 [LiDARCLIP](https://github.com/atonderski/lidarclip) 官方权重在 **ONCE 数据集**上训练，而本项目使用 **nuScenes 数据集**。两者 LiDAR 传感器不同，存在明显的 domain gap。因此推荐在 nuScenes 上本地训练。

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
