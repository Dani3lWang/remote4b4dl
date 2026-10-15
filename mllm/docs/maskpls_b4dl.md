# MaskPLS → B4DL 空间实例后端

当前交付是本地代码移植。未运行实际 GPU 推理或训练，未证明优于旧 seg；当前机器没有可用的 wqlc，Conda 启动失败。源码静态检查可在本地完成，下面的单元测试和真实点云冒烟留到隔离的 Linux/CUDA 环境执行。

## 已实现的路径

```text
已有 grounded4d 原始 nuScenes manifest
  → 原始点/标签 SHA-256、点序、真实 scene/sample/sample_data 及官方划分检查
  → 官方 MaskPLS 的 Minkowski 多尺度 backbone、插值、9 层 decoder
  → Thing + Stuff 的全景掩码训练及语义辅助监督
  → 仅 Thing 的帧内实例候选、原始点掩码、256 维 query 特征
  → 后续 B4DL 回答绑定或时序关联模块
```

代码目录是 `mllm/maskpls_b4dl/`；统一入口为 `mllm/scripts/run_maskpls.py`。现有 B3、旧 seg、grounded4d 注意力对照代码均无需变更。

保持 MinkowskiEngine 的官方 backbone，是为了保留官方权重的加载路径。没有把它替换成另一个简化 spconv 模型。上游固定提交、许可证、参数兼容及所有方法差异见 `mllm/maskpls_b4dl/NOTICE.md`。

## 面向单张 RTX 4090 的配置

`mllm/scripts/maskpls_4090_config.json`：

- batch size 1，默认梯度累积 4，等效 batch 4。
- 128 queries、256 维、3 scales × 3 blocks、0.05 m 体素；保留完整模型结构。
- backbone 和解码默认 FP32；不自动将稀疏卷积切换为 FP16。
- decoder activation checkpointing；attention 不生成未使用的权重矩阵。
- 损失采样 16,384 点，匹配 8,192 点；每个实例至少保留一个前景采样点。
- 不下采样模型输入点，也不裁掉小实例来“腾显存”。
- 原始 LiDAR ROI 为 x/y ±51.2 m，z [-5,3)，与当前 grounded4d 数据保持一致。
- KeOps 做 KNN；`knn_backend=torch` 仅适合小样本诊断，可分块但不是大点云的速度优化。

这些是 24 GB 显存目标下的初始配置，不是已经测得的显存承诺。第一步执行 smoke，记录 `peak_cuda_bytes` 和耗时，再定训练预算。如果显存不足，先检查真实点数和编译配置；任何输入采样/结构改动均需独立记录，不能静默丢 GT。

完整 nuScenes 若有超过 128 个 Thing+Stuff 监督掩码，训练预检会停止；须明确增加 query 容量。不会截断实例监督。冻结评测不要求 query 数量覆盖全部 GT，检测不足计入 FN。

## 环境

在独立 Linux 环境中使用名为 `wqlc` 的环境，避免改变已有 B3 环境。至少需要：

- Python 3.10、PyTorch >=2.1 的 CUDA 版本。
- 与该 PyTorch/CUDA toolkit 匹配、带 CUDA 扩展的 MinkowskiEngine 0.5.4。
- KeOps、nuScenes SDK；Python 依赖见 `mllm/requirements-maskpls.txt`。
- RTX 4090 的 CUDA 编译目标需要覆盖其架构。不要直接照搬上游 PyTorch 1.10.1+cu113 的旧环境。
- 配置编译器、CUDA toolkit/nvcc 后先验证导入及实际 forward/backward；本次未验证具体编译组合，不承诺某个现代 Torch wheel 可以无改动编译旧 MinkowskiEngine。

该 requirements 文件故意不自动替换 PyTorch 或安装其不匹配的 MinkowskiEngine 二进制。正式环境确定后再固定版本并记录 `pip freeze`、驱动、CUDA 和 GPU 型号。

## 数据约定

复用已有 `b4dl-grounded4d-v1` 的单帧 panoptic manifest：

- 必须包含完整 10 个 Thing 类实例，不接受 QA 候选、框伪标签或部分对象监督。
- 使用 `min_target_points=1`，含 1 点实例。
- 要求原始点文件 [N,5]（XYZ/intensity/ring）、原始 panoptic、标签哈希及 instance_token 证据。
- 从原始 panoptic 恢复 16 类语义，原始 ignore 不参与掩码/语义损失及指标；从语义标签构建 Stuff 掩码。
- 已有准备器的 ROI 若不同，应先重建 manifest，不能把没有 ROI 内正点的实例转成 NOOBJ。
- 仅接受单帧、identity lidar_to_anchor；多帧关联后续单独实现。
- 此版本沿用已有 manifest 的准入范围：准备器拒绝没有 Thing 的帧。因此不能用它声称完成了无物体帧或无目标语言请求的训练。
- `--metadata-root` 指向含 v1.0-mini/v1.0-trainval 的根目录，也可以直接指向对应版本目录。实际 scene/sample/sample_data 表要与 manifest 相符。
- mini 使用官方 mini_train/mini_val。完整数据训练及内部开发都必须在官方 train 内且场景互斥。
- `--official-val` 仅允许 preflight/evaluate，不允许 train/smoke；完整官方 val 属于 B4DL 保留测试。其 manifest 可用 split=internal_dev 作为存储格式，但结果明确记录 source_split=val，不用于训练或早停。
- 现有准备器并不自动生成官方 val manifest；该路径需要先准备同等审核标准的完整实例 manifest。

如尚无 manifest，先用已有 `prepare_grounded4d.py` 和 `grounded4d_mini_config.json` 准备原始点/标签。两者 ROI 相同，体素尺寸不会改变原始点级 GT；本后端会重新检查全部实例与 Stuff 的 query 容量。

## 运行顺序

以下变量只表示路径；替换为服务器上的实际位置。从仓库根目录执行，每个 output-dir 都使用新目录：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

DATA=/path/to/nuScenes
PANOPTIC=/path/to/nuScenes-panoptic-v1.0-all
LABELS=/path/to/prepared-grounded4d-mini
CONFIG=mllm/scripts/maskpls_4090_config.json
```

1. 先执行契约回归测试。这些测试使用 CPU fixture 验证数学和接口，不等同于真实 MinkowskiEngine/CUDA 验收：

```bash
PYTHONPATH=mllm conda run -n wqlc python -m unittest discover \
  -s mllm/maskpls_b4dl/tests -v
```

2. 预检完整 manifest（不构造 Minkowski 模型）：

```bash
conda run -n wqlc python mllm/scripts/run_maskpls.py preflight \
  --config "$CONFIG" --manifest "$LABELS/train.jsonl" --split train \
  --scene-tokens "$LABELS/train_scene_tokens.json" \
  --nuscenes-root "$DATA" --metadata-root "$DATA" \
  --panoptic-root "$PANOPTIC" --label-root "$LABELS" \
  --output-dir mllm/training_logs/maskpls/preflight-001
```

3. 一帧真实点云的 forward/backward 冒烟：

```bash
conda run -n wqlc python mllm/scripts/run_maskpls.py smoke \
  --config "$CONFIG" --manifest "$LABELS/train.jsonl" --split train \
  --scene-tokens "$LABELS/train_scene_tokens.json" \
  --nuscenes-root "$DATA" --metadata-root "$DATA" \
  --panoptic-root "$PANOPTIC" --label-root "$LABELS" \
  --output-dir mllm/training_logs/maskpls/smoke-001
```

检查退出码、`smoke.json` 的有限损失/梯度及显存。它不是一次完整训练，也不是效果验收。

4. 先完成一个 epoch 的训练链路验证；正式 epoch 预算在看过冒烟与数据规模后确定：

```bash
conda run -n wqlc python mllm/scripts/run_maskpls.py train \
  --config "$CONFIG" --manifest "$LABELS/train.jsonl" --split train \
  --scene-tokens "$LABELS/train_scene_tokens.json" \
  --validation-manifest "$LABELS/internal_dev.jsonl" \
  --dev-scene-tokens "$LABELS/dev_scene_tokens.json" \
  --nuscenes-root "$DATA" --metadata-root "$DATA" \
  --panoptic-root "$PANOPTIC" --label-root "$LABELS" \
  --epochs 1 --accumulate 4 --lr 1e-4 \
  --output-dir mllm/checkpoints/maskpls-mini-001
```

这是从随机权重开始的调用。可加 `--official-checkpoint /path/to/mask_pls_nuscenes.ckpt` 进行迁移初始化：
仅允许已知的 query 行扩展，其余参数严格匹配。扩展明细写入 checkpoint，未做的训练不记作预训练。

官方 Lightning checkpoint 默认使用 `torch.load(weights_only=True)`。如果其中的旧自定义对象导致安全加载失败，应在可信原始环境先导出纯 `state_dict`；入口不自动回退到任意 pickle 执行。只提供官方 nuScenes 权重，不要传 SemanticKITTI 权重。

每轮保存 `last.pt`、优化器、调度器、RNG、配置、源码身份、manifest/元数据身份、训练记录；开发损失仅标记为诊断，不据此宣称分割通过。保留一个 last checkpoint 控制磁盘占用。

恢复时使用同样命令及 `--resume mllm/checkpoints/maskpls-mini-001/last.pt`，output-dir 必须相同；不能同时指定官方初始化。将 `--epochs` 设为大于已完成轮数的累计目标，例如从 1 轮继续至 2 轮时使用 `--epochs 2`。恢复严格检查数据、划分、配方、代码；验证未通过时保留旧运行状态。按完整 epoch 恢复，崩溃发生在 epoch 中途时重放该 epoch；steps 日志可能保留未提交的尝试，应以 checkpoint/progress 的已提交 epoch 为准。

5. 全 manifest 评测：

```bash
conda run -n wqlc python mllm/scripts/run_maskpls.py evaluate \
  --checkpoint mllm/checkpoints/maskpls-mini-001/last.pt \
  --manifest "$LABELS/internal_dev.jsonl" --split internal_dev \
  --scene-tokens "$LABELS/dev_scene_tokens.json" \
  --nuscenes-root "$DATA" --metadata-root "$DATA" \
  --panoptic-root "$PANOPTIC" --label-root "$LABELS" \
  --output-dir mllm/eval_results/maskpls-mini-001
```

评测 checkpoint 自动提供配置。输出包括类别正确的 TP/FP/FN、Recall@0.5、全部 GT 平均 IoU、1–10/11–49/50+ 点分层，以及原始 query 的几何/类别候选覆盖诊断。每帧 NPZ/JSON 与 frame_metrics.jsonl 增量写入；最终才生成 complete=true 的 metrics.json。此版本不支持评测断点恢复，失败后保留部分结果但须新目录重跑，不将 partial 当作全量。

官方冻结权重诊断：把 `--checkpoint` 替换成 `--official-checkpoint`，并加 `--config mllm/scripts/maskpls_official_queries_config.json`。该配置保持 100 queries，不为评测创建随机额外 query。其他部分仍使用 B4DL ROI/评测规则，**不是官方 benchmark 复现**。

官方预训练权重可能看过 B4DL 的 internal dev（它来自官方 train），结果会记录 pretrained_official_train_exposure=true。这些分数仅作候选质量诊断，最终确认需要明确独立的场景；不能拿已见数据的增益作为泛化结论。

## 无标签推理与 B4DL 对象接口

对真实原始点文件直接推理不需要 GT 或 panoptic：

```bash
conda run -n wqlc python mllm/scripts/run_maskpls.py predict \
  --checkpoint mllm/checkpoints/maskpls-mini-001/last.pt \
  --point-file /path/to/LIDAR_TOP.bin \
  --sample-token REAL_SAMPLE_TOKEN --sample-data-token REAL_LIDAR_TOKEN \
  --output-dir mllm/eval_results/maskpls-predict-001
```

每帧输出：

- `masks`：[K,N]，完整原始点轴，ROI 外为 false。
- `point_indices`：0..N-1；`point_sha256`：原始点文件身份。
- `class_ids`：与 B4DL 一致的 0..9 Thing 类。
- `query_features`：[K,256]，对应保留实例；`query_indices` 保留原始 query 行号。
- JSON 对象：`object_id=sample_data_token:q序号`、类别、分数、中心与点数。
- `grounding_verified=false`、`track_verified=false`、`track_id=null`。
- 可选 `--save-all-queries` 保存全部 mask 概率/类别概率/query 特征，便于离线 oracle 诊断，文件会更大。

B4DL 集成时，可将候选对象提供给回答绑定模块，再用 `resolve_answer_objects(records, selected_ids)` 解析明确对象 ID。该函数拒绝同类名称猜测、跨帧 ID 和重复 ID。

**256 维 query 特征不能直接替换原 B3 的 768 维 SST 输入。** 语言到对象的投影/选择器需要独立训练及审核过的“回答短语→唯一实例”监督。当前没有改写 B3 的生成逻辑，也没有训练这条映射；导出对象特征是明确的接入点，不能视为自动完成指代绑定。

## 验收状态

- 源码审查、Ruff 静态检查：本地执行。
- 契约测试：已提供，需在可用 wqlc 环境执行。
- MinkowskiEngine 编译、真实点云 CUDA 前后向、官方权重实际加载：待 GPU 环境验证。
- 训练收敛、与旧 seg 同帧对照：待实际实验。
- 回答绑定、NOOBJ 语言评测、自由跨帧关联：未由本空间移植实现。

验收以 status.json、退出码、checkpoint/数据/代码身份、完整评测范围和独立指标为依据；单个 completion.json 或低训练 loss 不代表模型效果达标。
