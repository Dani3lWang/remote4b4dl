# B4DL 单帧推理分割

该分支在现有 B3 文本问答路径之外增加单帧、点级、多目标分割。模型组合了 MORE3D 的多 `<SEG>` 对象路由、Reason3D 的 `<LOC>` 粗区域软先验，以及独立类别头。原有 B3 加载和评测入口保持不变。

> 当前状态：`seg` 是“实现与离线检查就绪”，不是“实验结果已完成”。在完整 panoptic、spconv 和 CUDA 环境中跑通空间预训练、主训练及全量自由生成评测前，不得把验收阈值、teacher-forcing 指标或代码存在本身写成模型成绩，也不得合入 `main` 取代 B3 基线。

## 模型加载顺序

加载顺序是 checkpoint 兼容约束：

1. 加载 Vicuna 基座及 32000 词表；
2. 注册 `<4DLiDAR>`、`<meta>`，恢复 Stage 1 与 B3，使词表达到 32002；
3. 合并 B3 LoRA；
4. 添加 `<LOC>`、`<SEG>`、`<NOOBJ>`，固定为 32002—32004；
5. 安装独立的输入 token 增量和输出 token 行，再添加分割 LoRA；
6. 恢复空间编码器、场景压缩器、粗/细掩码解码器和分类头。

B3 的 `non_lora_trainables.bin` 保存了完整 `[32002, 4096]` 输入词嵌入。若先把词表扩为 32005，再加载 B3，会发生无法由 `strict=False` 忽略的尺寸错误。

## 数据准备

首版需要 nuScenes trainval 原始点云和 panoptic expansion。lidarseg 只能提供语义类别，不能区分同类实例，因此不能替代 panoptic。

先创建独立环境，避免改变已验证的 B3 `wqlc` 环境：

```bash
cd /root/autodl-tmp/mmb4dl
conda create -p /root/autodl-tmp/.conda-stuff/envs/reasonseg python=3.10 -y
conda run -p /root/autodl-tmp/.conda-stuff/envs/reasonseg \
  pip install -r mllm/requirements-seg.txt
```

```bash
cd /root/autodl-tmp/mmb4dl
export B4DL_SEG_ENV_PREFIX=/root/autodl-tmp/.conda-stuff/envs/reasonseg
"$B4DL_SEG_ENV_PREFIX/bin/python" mllm/scripts/build_reasonseg_nuscenes.py \
  --dataroot /root/autodl-tmp/nuScenes \
  --output-dir ./mllm/reasonseg_data \
  --validation-fraction 0.1 \
  --seed 20260917
```

生成器只使用 nuScenes 官方 700 个 train 场景，再按固定随机种子划出内部 train/val；官方 150 个 val 场景是 B4DL test，始终隔离并写入 `b4dl_test_scenes.json`。不能直接把官方 val 当作 ReasonSeg 验证集，否则会发生测试集调参泄漏。生成结果是 `reasonseg_train.jsonl` 和 `reasonseg_val.jsonl`。问题包含类别、左右/前后/远近、多目标与无目标样本。每条记录只保存实例 ID；训练时才从 panoptic 文件还原点掩码。

同类实例按距离顺序和左右位置生成唯一描述，例如 `nearest left-side car`、`2nd-nearest left-side car`。多目标问题和监督答案复用同一组描述，避免多个 `<SEG>` 查询无法对应到具体实例。

## 环境与预检

先用逐点语义标签监督预训练稀疏空间编码器。默认 `auto`：存在 lidarseg 时优先读取其 `.bin`，否则从已有 panoptic 标签的 `semantic_id * 1000 + instance_id` 中还原完全等价的语义 ID，避免重复保存一套逐点标签。该阶段不加载 Vicuna 或 B3，最佳权重按验证集语义 mIoU 保存：

```bash
cd /root/autodl-tmp/mmb4dl
export B4DL_SEG_ENV_PREFIX=/root/autodl-tmp/.conda-stuff/envs/reasonseg
export B4DL_NUSCENES_ROOT=/root/autodl-tmp/nuScenes
bash mllm/scripts/run_reasonseg_spatial.sh
```

空间启动器只调用隔离环境中的 Python/Accelerate，并在训练前核对输出盘余量、CUDA bf16、真实 spconv CUDA 稀疏卷积、全部点云/语义标签长度和内部场景划分。可用 `B4DL_SEG_SPATIAL_LABEL_SOURCE=lidarseg|panoptic` 强制标签源。空间预训练与 manifest 生成必须使用相同的 `--validation-fraction` 和 `--seed`，确保二者使用完全一致的内部验证场景；如需改动，必须同时设置 `B4DL_SEG_VALIDATION_FRACTION` 和 `B4DL_SEG_SEED` 后重建 manifest。

主训练通过 `--spatial-checkpoint ./checkpoints/reasonseg-spatial/spatial-best` 恢复空间编码器。启动脚本会先检查 GPU、spconv、B3 身份文件、空间 checkpoint、panoptic 数据和 manifest 闭包。预检逐一读取 manifest 引用的全部唯一点云/掩码对，验证点数一致、目标实例实际存在、train/val 场景隔离、保留测试场景排除、目标类别映射和 `<LOC>/<SEG>/<NOOBJ>` 数量契约。首次完整预检会产生顺序磁盘读取，这是启动前的预期行为。任何一项缺失都会停止，不会启动不完整训练。

```bash
export B4DL_SEG_ENV_PREFIX=/root/autodl-tmp/.conda-stuff/envs/reasonseg
export B4DL_NUSCENES_ROOT=/root/autodl-tmp/nuScenes
bash mllm/scripts/run_reasonseg.sh
```

默认训练 20 epochs、micro batch 1、梯度累积 16。新增空间与掩码模块学习率为 `1e-4`，语言 LoRA 和特殊 token 为 `2e-5`。损失是文本 CE、精掩码 BCE+Dice、0.5 倍粗区域 BCE+Dice、类别 CE。每个 epoch 在验证清单前 32 条固定样本上计算 teacher-forcing mean/global IoU；`checkpoint-best` 按 mean IoU 保存，连续 5 个 epoch 无提升时早停。该指标只验证掩码头是否能拟合正确的 `<LOC>/<SEG>` 路由，自由生成 token 成功率仍由下述独立评测统计。

## 评测

评测必须使用自由生成，而不是从真值答案提取 `<SEG>`：

```bash
cd /root/autodl-tmp/mmb4dl
export B4DL_SEG_ENV_PREFIX=/root/autodl-tmp/.conda-stuff/envs/reasonseg
export B4DL_NUSCENES_ROOT=/root/autodl-tmp/nuScenes
export B4DL_SEG_CHECKPOINT=/root/autodl-tmp/mmb4dl/mllm/checkpoints/reasonseg-b3-single-frame/checkpoint-best
bash mllm/scripts/run_reasonseg_eval.sh
```

评测启动器先验证 B3/ReasonSeg checkpoint 结构和验证 manifest 的完整数据闭包，再执行不带 `--max-samples` 的全量自由生成。输出包括 cIoU、gIoU、实例精确率/召回率、类别宏 F1、数量准确率、无目标误报率、特殊 token 失败率、平均/P95 推理延迟和 CUDA 峰值显存。结果同时记录完整数据量、实际评测量与是否为抽样评测，避免把 `--max-samples` 冒烟结果误写成全量成绩。cIoU 的累计 union 会加入未匹配预测与未匹配真值，gIoU 会为每个漏检或额外实例补零；低于 0.5 IoU 的已配对实例只记一次 FP/FN。这样漏检、额外掩码与 token 配对失败都不会被指标静默剔除。

## 主线晋级条件

ReasonSeg 只有同时满足以下条件后才可从实验分支申请进入主线：

1. 预检打印完整文件闭包，且 train/val 与 `b4dl_test_scenes.json` 均无交叉；
2. 空间预训练保存可恢复的 `spatial-best`，主训练 checkpoint 能在新进程中恢复；
3. 32 条 teacher-forcing 验证仅作为路由调试门，最终结果来自全量验证集自由生成；
4. 至少三个随机种子，并报告类别名称基线、仅 `<SEG>`、完整 `<LOC>/<SEG>` 三组消融；
5. 同时回归 B3 文本任务，任何带 `--answer_frames` 的 TG 数值必须标为 oracle 输入口径。

## Gradio

在原查看器参数上增加分割 checkpoint：

```bash
python -m vtimellm.demo_gradio \
  --nuscenes-root /root/autodl-tmp/nuScenes \
  --model-base ./base_model/vicuna-v1-5-7b \
  --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
  --seg-checkpoint ./checkpoints/reasonseg-b3-single-frame/checkpoint-N
```

分割始终读取当前帧完整点云；查看器下采样仅影响显示。预测掩码通过保存的原始点行号映射回 3D 和 BEV 视图，切换帧后旧预测自动清除。
