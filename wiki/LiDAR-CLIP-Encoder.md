# LiDAR-CLIP 编码器（encoders/lidarclip/）

基于 SST（Sparse Swin Transformer，CVPR 2022）backbone 的 LiDAR 编码器，训练目标是将点云序列特征对齐到冻结 CLIP 图像编码器（ViT-L/14）的输出空间，为下游 VTimeLLM 提供视觉特征。

## 架构

```
点云帧列表 (每帧 N×4: x,y,z,intensity，已投影到相机坐标系裁剪)
  → voxelization (0.5m voxel, 80×80 网格)
  → DynamicVFE (4→64→128 通道)
  → SSTInputLayerV2 (12×12 窗口) + SSTv2 4 层 Transformer (d_model=128)
  → 特征图 (bs, 128, 80, 80)
  → in_proj (128→768) + AttentionPool2d (CLIP 风格空间注意力池化)
  → (bs, 768)  ← 与 CLIP ViT-L/14 图像特征逐样本计算 MSE
```

- **对齐方式是非对比损失**：CLIP 图像编码器完全冻结，训练目标为 LiDAR 特征与图像特征的 MSE（默认）或负余弦相似度（`--loss-function cosine`）
- 核心类：`LidarEncoderSST` / `SSTEncoder`（`lidarclip/model/sst.py`）、`AttentionPool2d`（`lidarclip/model/attention_pool.py`）；训练 LightningModule 为 train.py 内的 `LidarClip`
- `lidarclip/model/_mmdet3d_compat.py` 兼容层让旧 mmdet3d 0.x 的 SST 代码跑在 mmcv 2.x + mmengine 上，并提供**向量化 dynamic scatter**（修复原实现 point 级层无梯度的 no-op bug，batch 32 单步从 >20min 提速）
- `sst/` 与 `mmdetection3d/` 是外部依赖源码拷贝（SST 官方仓库及其底座），无需改动

## 训练

```bash
cd encoders/lidarclip
python train.py --data-dir /path/to/nuScenes --name lidarclip_nuscenes \
    --nuscenes-datadir /path/to/nuScenes --batch-size 32
```

- **数据**：`build_loader(split="trainval")` 读 `annotations/scene_metadata.json` 排除 150 个 test 场景，强制断言恰好 700 个训练场景（防泄漏）；每样本用全部 6 个相机图像作为 CLIP 目标
- **关键参数**：`--max-epochs`（默认 20）、`--scheduler-max-lr`（OneCycleLR max_lr，默认 1e-3）、`--seed`（默认不设）、`--load-only-model`（只载权重不续训练状态，退火用）、`--clip-model`（默认 ViT-L/14）、`--loss-function`（mse/cosine）、`--resume-wandb-logging`
- **训练配置**：Adam（初始 lr 1e-5）+ OneCycleLR、precision 16、单卡（去掉 ddp，避免与 fork dataloader 互锁）、checkpoint 每 250 步 + epoch 末（save_top_k=3）、wandb offline
- **loss 真值**：wandb offline 不可靠，以 LossCSVCallback 每 50 步写入的 `logs/train_loss.csv` 为准
- `smoke_train.py`：3 batch × 1 epoch 冒烟测试；`early_stop_monitor_v2.py`：loss 连续 500 步降幅 <1% 则 SIGTERM 停训（基于 train_loss.csv 真值）
- `validate_encoder_fit.py`：对比新训 nuScenes 编码器与旧 ONCE 权重的 MSE / 余弦结构 / 特征范数

⚠️ **权重现状**：仓库原有的 `lidarclip/checkpoint/vit_l_14.ckpt` 是原版 LiDAR-CLIP 的 **ONCE 数据集**权重（非 nuScenes），存在 domain gap；官方 B4DL 从未发布编码器权重，必须用 nuScenes 自训。此前所有特征均为旧 ONCE 编码器产物，编码器定稿后需**全量重提**，不可增量混提。

## 特征提取

两个提取脚本，区别在键控方式：

| 脚本 | 键控 | 输出 | 用途 |
|------|------|------|------|
| `extract_pc_features.py`（旧） | frame_id | stage1: `{frame_id}.npy` (1,768)；stage2: `{scene_id}.npy` (N,768) | 配旧版 95K stage1 数据 |
| `extract_pc_features_sample_token.py`（新） | sample_token | stage1: `{sample_token}.npy` (1,768) | 配官方 162K stage1 方案（`build_stage1_from_lidarllm.py` 产出的数据 scene_id=sample_token） |

```bash
# 官方 162K 对齐版
python extract_pc_features_sample_token.py \
    --checkpoint ./lidarclip/checkpoint/vit_l_14.ckpt \
    --scene-metadata /path/to/scene_metadata.json \
    --sample-json /path/to/nuScenes/v1.0-trainval/sample.json \
    --data-path /path/to/nuScenes \
    --save-dir ./b4dl/stage1_features_sample
```

- 只提取 `scene_metadata` 中 `split=='train'` 的 700 场景全部关键帧（28,130 帧）
- stage2 特征按 scene_metadata 中 `PATH_LIDAR_TOP` 路径顺序 `torch.cat` 拼接
- 特征 dtype float16

**提取的硬性约定**：
- checkpoint 加载需 `torch.load(..., weights_only=False)`（PL ckpt 含非张量对象）且 `strict=False`（只为忽略旧版 bbox_head 残留键）；**missing_keys 非空必须 fail-fast**，否则静默产出垃圾特征
- 提特征必须 `eval()` 模式（train 模式下 BN batch 统计 + SST 训练版 token 丢弃会让特征随提取顺序漂移不可复现）

## val_mse_probe.py — 收敛/选型探针

在编码器从未训练过的 val/test 场景（150 个）的确定性子集（`--max-scenes 15`）上，仅 forward 计算 LiDAR→CLIP 对齐 MSE。训练侧只有 train loss、无验证信号，此探针是唯一的泛化判据。

- 编码器保持 **train 模式**（与提特征约定一致，探针分数才与下游特征行为可比）
- 固定 `--seed` + `shuffle=False` 保证跨 checkpoint 可比
- 结果追加写 `logs/val_mse_probe.csv`（时间, tag, ckpt, mean, std）

## run_anneal_chain.sh — 退火链

高 LR 续训被早停监控硬停时 LR 尚未退火，此脚本（tmux 会话 `b4dlanneal`）编排完整退火流程：

1. 轮询等待高 LR 训练进程退出（上限 12h）
2. 对 last.ckpt 跑基线 probe（tag=baseline）
3. 归档 train_loss.csv，另起干净 CSV
4. 短程退火：`train.py --checkpoint <last.ckpt> --load-only-model --max-epochs 3 --scheduler-max-lr 1e-4 --seed 0`，输出 `ckpt_anneal/`
5. 逐 ckpt probe，**以 val MSE 最低者为最终编码器候选**

## annotations/ 元数据

- `scene_metadata.json`：每场景 `{scene_token, scene_id, num_frames, split}`；loader 用它做 700/150 划分，extract 脚本用 `paths.PATH_LIDAR_TOP` 决定 stage2 拼接顺序
- `sequence_metadata.json`：每序列 `{scene_token, sequence_id, frames[{frame_id, sample_token, PATH_*/TOKEN_*}], indices}`

注：该目录在仓库中不落盘（文件在训练机生成），`loader.py` 按相对路径引用；官方发布的镜像版本在 HF 数据集 `metadata/` 下。
