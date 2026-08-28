# LiDAR-CLIP 编码器权重版本核对报告（2026-08-28）

> 对应《B4DL_评测数值对比与问题清单_20260828.md》问题 #3
> 「LiDAR-CLIP 编码器权重版本未核对」。

## 一、结论

**本地 `encoders/lidarclip/lidarclip/checkpoint/vit_l_14.ckpt` = 原版 LiDARCLIP（WACV 2024, atonderski/lidarclip）发布的 ONCE 数据集权重，并非官方 B4DL 流程要求的 nuScenes 预训练版本。** 官方 B4DL 仓库从未发布训练好的编码器权重（只提供训练代码），需要自行在 nuScenes 上训练。已确认后启动 nuScenes 编码器训练（运行中）。

- MD5：`cd04e5e0eed557cc3073bdf3b8e268b6`（1,033,933,185 字节）
- ckpt 元数据：epoch 2 / global_step 40500 / PyTorch Lightning 1.7.2 / wandb run id `lidarclip_mm`
- 影响面：全部任务共用该编码器。现有全部特征（旧 95K frame_id 特征、28,130 stage2 特征、08-28 提取的 28,130 stage1 sample_token 特征）均由该 ONCE 权重提取。

## 二、证据链（六条独立证据）

1. **字节级同一**：本地 ckpt 与 `/root/autodl-tmp/ljq/mmb4dl-main/encoders/lidarclip/ckpt/lidarclip_vitl14_once.ckpt` MD5 完全一致。ljq 目录是官方 B4DL 仓库（mmb4dl-main）的完整解包（README 含官方标题/arXiv 2508.05269 badge），其 ckpt 文件名直接标注 `once`。本地文件（05-26）晚于 ljq 下载（05-22），系复制而来。
2. **原版仓库只发 ONCE 权重**：atonderski/lidarclip（LiDARCLIP 原作）官方 checkpoint 表仅提供 ONCE 数据集的 ViT-L/14 与 ViT-B/32 两个 Google Drive 链接；本 ckpt 的 epoch-2/40500 步 PL 1.7.2 格式与该发布形态吻合。
3. **官方 B4DL 不发布权重**：官方 README（github.com/mmb4dl/mmb4dl，`encoders/lidarclip/README.md`）明文 "You need to train the model first, and with that trained weight, extract the features"；仓库内只有 train.py/extract 脚本，无权重文件，HuggingFace（ccho4702）也无模型权重。
4. **论文 §5.1 要求 nuScenes 预训练**："we pre-train the encoder $E_L$ using nuScenes, which provides 3D point clouds paired with multi-view images. For the image encoder, we adopt the CLIP model, specifically the ViT-L/14 variant." 损失为论文式(1) 的正样本对 MSE（与官方 train.py 实现一致，CLIP 图像塔冻结）。
5. **官方 train.py 默认 nuScenes**：官方 CLI `--dataset-name` 默认 `nuscenes`，loader（`lidarclip/loader.py`）内置 "for 700 scenes only" 过滤（剔除 sequence_metadata.json 末 900 序列对应的 test scenes，恰余 700 训练场景）——与论文 700/150 划分闭环。
6. **本地无训练痕迹**：ckpt 文件时间 05-26 15:56，早于本仓库全部 wandb 记录（06-01/06-05 的 run 全部 `_runtime=0`，为启动即退的失败尝试），排除"本地训练产物"可能。本仓库 README"推荐：已有 epoch 2 的本地权重"的说法是**误标**，已在本次修正。

## 三、影响评估

- ONCE 与 nuScenes 传感器/采集域差异大（README 既有章节也承认 domain gap），特征表示质量受损对 TG 帧级判别影响最大——与"宽度学对、位置错"的 seqv3-mixed 表现相容，但按证据强度排位仍次于 stage1 数据量（问题 #1）。
- 与问题 #1 天然耦合：**编码器重训后需一次性重提全部特征**（162K stage1 + 28K stage2），再训 projector 与混合 stage2。顺序：编码器收敛 → 重提特征 → projector/混合重训 → 评测。

## 四、已执行的动作

### 4.1 启动 nuScenes 编码器训练（运行中）

```bash
# 环境 wqlc（torch 2.8.0+cu128，5090 兼容；b4dl 环境 torch 2.5.1 无 sm_120 内核）
WANDB_MODE=offline python train.py --name lidarclip_nuscenes \
    --checkpoint-save-dir ./ckpt_nuscenes --batch-size 32 --workers 16 \
    --data-dir /root/autodl-tmp/Datasets/nuScenes \
    --clip-model ViT-L/14 --dataset-name nuscenes
```

- 数据：nuScenes trainval 850 scenes → 官方 700-scene 过滤（log 确认 "ok_scene_tokens after filtering: 700"），每帧×6 相机 = 168,780 样本，5,275 步/epoch；
- 配方：官方默认（MSE 对齐、Adam lr 1e-5 + OneCycle max 1e-3、precision 16、max_epochs 20、ckpt 每 250 步 + epoch 末 + last）；
- 与 CoRViD 共存下 ~3-4 s/步（一 epoch ≈ 5-6 h）；ONCE 官方权重停在 epoch 2，**计划 2-3 epoch 后视 loss 平台早期停止**，随后重提特征。

### 4.2 修复两个训练正确性 bug（都发生在本地复现层，不影响官方代码）

| # | bug | 修复 | 验证 |
|---|---|---|---|
| 1 | 本机 `clip.load()` 不把权重转 fp16，而 CLIP `self.dtype` 硬编码 fp16 → 前向 dtype 不匹配 | `train.py` 加 `clip.model.convert_weights(clip_model)` | 冒烟 3 步 loss 0.3186 |
| 2 | `_mmdet3d_compat.py` 的 dynamic scatter 回退是 O(体素×点) 纯 Python 循环（batch 32 单步 >20 min）；且 `_dynamic_scatter(Function)` 的 backward 被 shim 置为 no-op，**梯度无法穿过 scatter，DynamicVFE 的 point 级 vfe_layers 永远收不到梯度**（对训练是正确性 bug） | 重写为向量化可微实现（`torch.unique` + `scatter_reduce`/`index_add`），并替换 `DynamicScatter.forward_single` | ① 数值等价：max 位级一致，sum/mean 差 ≤1e-6，coors/map/count 全等（`validate_scatter.py`）；② 梯度流：vfe_layers 6/6 参数有梯度（原实现为 0）；③ 速度：batch 8 前向+反向 14.7s → 0.6s（~20×），batch 32 单步 ~3-4s |

### 4.3 排障记录（留档）

- v1（含慢 scatter + ddp）：wandb 有零星 history 但主进程 CPU 93% 纯算 Python 体素循环，55 min 未到 step 250 → 杀；
- v2（快 scatter + ddp）：`strategy="ddp"` 在 1 GPU 上派生子进程与 fork dataloader 组合异常 → 去掉 ddp（单卡无需）；
- v3/v4：观察窗口不足（step 250 ckpt 需 ~15 min，而 wandb offline 历史记录刷盘延迟可达数十分钟 + stdout 缓存吞掉进度条，造成"卡死"假象）→ 提前终止，实际可能已在训练；
- v5：全部修复后确认健康——`ckpt_nuscenes/lidarclip_mm/epoch=0-step=250.ckpt` 于启动后 ~16 min 生成。
- 经验：**判断训练是否推进以 ckpt 文件为准，不要依赖 wandb 离线记录或 console 日志。**

## 五、产物路径

- 训练 ckpt：`encoders/lidarclip/ckpt_nuscenes/lidarclip_mm/`（epoch=0-step=250.ckpt 起，含 last.ckpt）
- 训练日志：`encoders/lidarclip/logs/train_nuscenes_20260828_v5.log`（console 有缓冲延迟）；wandb 离线 run：`encoders/lidarclip/wandb/offline-run-20260828_225405-lidarclip_mm`
- 验证脚本：`encoders/lidarclip/validate_scatter.py`（等价性+梯度+速度）、`encoders/lidarclip/smoke_train.py`（启动冒烟）
- 代码改动：`train.py`（fp16 修复、去 ddp）、`lidarclip/model/_mmdet3d_compat.py`（向量化可微 scatter）

## 六、后续（与问题清单联动）

1. 监控 v5，2-3 epoch 后按 loss 平台早期停止（预计 08-29 内）；
2. 用新编码器重提全部特征：28,130 stage2 + 162K stage1（与问题 #1 的补数据合并为一次提取）；
3. 重训 stage1 projector → seqv3-mixed 混合重训 → 评测，目标 mIoU ≥ 0.30；
4. 若 mIoU 仍 <0.25：回到问题 #4/#5（解码策略与评测口径）。
