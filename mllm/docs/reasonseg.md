# B4DL 单帧推理分割

该分支在现有 B3 文本问答路径之外增加单帧、点级、多目标分割。模型组合了 MORE3D 的多 `<SEG>` 对象路由、Reason3D 的 `<LOC>` 粗区域软先验，以及独立类别头。原有 B3 加载和评测入口保持不变。

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

```bash
cd /root/autodl-tmp/mmb4dl/mllm
python scripts/build_reasonseg_nuscenes.py \
  --dataroot /root/autodl-tmp/nuScenes \
  --output-dir ./reasonseg_data \
  --exclude-scenes ./reasonseg_data/b4dl_test_scenes.json
```

生成器按 nuScenes 官方场景划分输出 `reasonseg_train.jsonl` 和 `reasonseg_val.jsonl`。问题包含类别、左右/前后/远近、多目标与无目标样本。每条记录只保存实例 ID；训练时才从 panoptic 文件还原点掩码。

同类实例按距离顺序和左右位置生成唯一描述，例如 `nearest left-side car`、`2nd-nearest left-side car`。多目标问题和监督答案复用同一组描述，避免多个 `<SEG>` 查询无法对应到具体实例。

## 环境与预检

分割依赖放在独立环境，避免改变已验证的 `wqlc`：

```bash
conda create -p /root/autodl-tmp/.conda-stuff/envs/reasonseg python=3.10 -y
conda run -p /root/autodl-tmp/.conda-stuff/envs/reasonseg \
  pip install -r mllm/requirements-seg.txt
```

先用 lidarseg 监督预训练稀疏空间编码器。该阶段不加载 Vicuna 或 B3，最佳权重按验证集语义 mIoU 保存：

```bash
cd /root/autodl-tmp/mmb4dl/mllm
accelerate launch scripts/train_reasonseg_spatial.py \
  --dataroot /root/autodl-tmp/nuScenes \
  --exclude-scenes ./reasonseg_data/b4dl_test_scenes.json \
  --output-dir ./checkpoints/reasonseg-spatial \
  --num-train-epochs 20 \
  --mixed-precision bf16
```

主训练通过 `--spatial-checkpoint ./checkpoints/reasonseg-spatial/spatial-best` 恢复空间编码器。启动脚本会先检查 GPU、spconv、B3 身份文件、空间 checkpoint、panoptic 数据和 manifest 闭包。任何一项缺失都会停止，不会启动不完整训练。

```bash
export B4DL_SEG_ENV_PREFIX=/root/autodl-tmp/.conda-stuff/envs/reasonseg
export B4DL_NUSCENES_ROOT=/root/autodl-tmp/nuScenes
bash mllm/scripts/run_reasonseg.sh
```

默认训练 20 epochs、micro batch 1、梯度累积 16。新增空间与掩码模块学习率为 `1e-4`，语言 LoRA 和特殊 token 为 `2e-5`。损失是文本 CE、精掩码 BCE+Dice、0.5 倍粗区域 BCE+Dice、类别 CE。每个 epoch 在验证清单前 32 条固定样本上计算 teacher-forcing mean/global IoU；`checkpoint-best` 按 mean IoU 保存，连续 5 个 epoch 无提升时早停。该指标只验证掩码头是否能拟合正确的 `<LOC>/<SEG>` 路由，自由生成 token 成功率仍由下述独立评测统计。

## 评测

评测必须使用自由生成，而不是从真值答案提取 `<SEG>`：

```bash
cd mllm
python evaluation/evaluate_reasonseg.py \
  --model-base ./base_model/vicuna-v1-5-7b \
  --pretrain-mm-mlp-adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --b3-checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 \
  --seg-checkpoint ./checkpoints/reasonseg-b3-single-frame/checkpoint-N \
  --manifest ./reasonseg_data/reasonseg_val.jsonl \
  --dataroot /root/autodl-tmp/nuScenes \
  --output-dir ./eval_results/reasonseg-b3
```

输出包括 cIoU、gIoU、实例精确率/召回率、类别宏 F1、数量准确率、无目标误报率和特殊 token 失败率。漏检、额外掩码与 token 配对失败都会计入指标。

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
