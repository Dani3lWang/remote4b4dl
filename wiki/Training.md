# 模型训练（mllm/）

基于 VTimeLLM 架构（Vicuna-7B backbone）的多模态 LLM 训练，输入为预提取的 LiDAR-CLIP 特征 + QA 对话数据。

## vtimellm/ 包核心文件

| 文件 | 职责 |
|------|------|
| `train/train.py` | 训练主程序：参数、LoRA 装配、stage3 merge 逻辑、metatoken 梯度掩码、断点续训（校验 trainer_state.json，缺失 fail-fast） |
| `train/train_mem.py` | 脚本实际入口（先打 flash-attn monkey-patch 再调 train.py） |
| `train/dataset.py` | `LazySupervisedDataset`：特征加载、`_select_features` 帧切片（feat_indices > feat_range > 问题文本帧号）、对话预处理；启动时预校验特征文件存在（fail-fast） |
| `train/vtimellm_trainer.py` | 自定义 Trainer，Stage1 只按步落盘 `mm_projector.bin`（规避 ZeRO-3 全量 checkpoint 冲突） |
| `model/vtimellm_llama.py` / `vtimellm_arch.py` | 模型与多模态融合（见 [[Architecture]]） |
| `model/builder.py` | `load_pretrained_model` 加载链：base → 注册 `<4DLiDAR>`/`<meta>` 特殊 token → stage1 projector → stage2 LoRA merge → stage3 LoRA merge |
| `conversation.py` / `mm_utils.py` | 对话模板 / `<video>` tokenizer 与视频抽帧 |
| `train/llama_flash_attn_monkey_patch.py` | Llama attention 替换为 flash-attn |

## 三阶段标准训练

```bash
cd mllm

# 一键 Stage1 + Stage2（不跑 Stage3）
bash run_stages.sh \
    --s1_data ./b4dl_dataset/stage1_lidarllm_mm.json \
    --s1_feat ../encoders/lidarclip/b4dl/stage1_features \
    --s2_data ./b4dl_dataset/stage2.json \
    --s2_feat ../encoders/lidarclip/b4dl/stage2_features \
    --model_name_or_path ./base_model/vicuna-v1-5-7b

# 单独跑某一阶段（末尾可追加任意覆盖参数）
bash scripts/stage1.sh [--额外参数]
bash scripts/stage2.sh [--output_dir ./checkpoints/custom --num_train_epochs 3]
bash scripts/stage3.sh [...]
```

| 阶段 | 数据/模板 | 训练内容 | 超参 |
|------|----------|---------|------|
| Stage1 | plain 模板，human 含 `<video>` | 仅 mm_projector，LLM 全冻结 | 1 epoch，lr 1e-3，bs 16×accum 8，bf16 |
| Stage2 | v1 模板多轮 QA | LoRA（r64/α128/dropout 0.05，target=全部 Linear 除 lm_head），projector 冻结 | 2 epochs，lr 1e-4 |
| Stage3 | v1 模板 | `--model_name_or_path` 指向 **stage2-merged 全量模型**，先 merge 旧 LoRA 再加新 LoRA | 3 epochs，lr 2e-5 |

所有阶段均为 `deepspeed --include localhost:0 vtimellm/train/train_mem.py --deepspeed ./scripts/zero3.json ...`。`zero3.json`：ZeRO-3 + optimizer/param offload CPU（pin_memory）+ 保存时 gather 16bit 权重；另有 `zero2.json`、`zero3_offload.json` 变体。

**monkey-patch**：`train.py` 头部把 `DeepSpeedEngine.no_sync` 替换为 ZeRO-3 下直接 `yield` 的 no-op（原版与 accelerate `no_sync` 在 ZeRO-3 梯度分区下会崩溃）。

## 训练数据格式

顶层 JSON 数组，每条：

```json
{"scene_id": "...", "scene_token": "...", "task": "time_grounding",
 "feat_range": [0, 8], "feat_indices": [0, 2, 4, 6, 8],
 "conversations": [
   {"from": "human", "value": "<4DLiDAR>\n<video>\n{question}\n<meta> ..."},
   {"from": "gpt",   "value": "{answer}"}]}
```

- **特征查找约定**：`{feat_folder}/{scene_id}.npy` —— 数据 JSON 的 `scene_id` 必须与特征文件名严格一致（stage1 新方案 scene_id=sample_token）
- `feat_indices` 是 QA 所属序列的精确采样帧下标（论文输入 S_L），`feat_range` 是兼容闭区间；`<video>` token 位于 human 首条消息开头；**B2 起整场景代际传 `--whole_scene`，跳过帧切片逻辑，视觉输入=该 scene 全部帧**（39/40/41 帧，与特征文件行数一致）
- 数据源：HF `ccho4702/nuScenes-B4DL` 的 train 目录（stage2.json 68,695 + stage3.json 79,576 = **148,271 条**，本身就是论文官方训练集，700/150 划分，与官方 test 零重叠），由 `scripts/build_stage2_full_train.py` 转换并对 TG 答案打 `task="time_grounding"` 标签（13,124 条）

## Metatoken 注入

```bash
python scripts/inject_metatoken.py --input stage2_train.json --output stage2_train_seqv3.json \
    --frame_motion ./b4dl_dataset/ego_frame_motion.json \
    --sequence_metadata ../encoders/lidarclip/annotations/sequence_metadata.json \
    --answer_frames
```

- `generate_ego_metadata.py`：从 nuScenes LIDAR_TOP ego_pose 计算每帧 `{x,y,z,yaw,spd,yaw_next/prev,acc_next/prev}`，产出 `ego_metadata.json`（per-scene/per-sequence 描述）与 `ego_frame_motion.json`（逐帧运动表）
- `ego_text.py`：自然语言模板**单一来源**（`render_meta_texts()` 对任意 (first,last) 帧对渲染），generate/inject/评测三方共用，保证训练与推理注入逐字符一致
- `inject_metatoken.py`：注入 `<4DLiDAR>`/`<meta>` 前缀 + 写 `feat_indices`/`feat_range`；`--answer_frames` 仅对 time_grounding 类生效（从 GT 答案解析帧号恢复序列归属）；`--no_4dlidar/--no_meta` 用于消融

## 复现采用的训练方案

### 帧位置 3-Epoch 实验

当前直接复用已训练完成的 B3 作为无帧位置对照，只训练 `scripts/run_stage2_full_seqv3_mixed_framepos3ep.sh`。完整迁移验收、训练冒烟、断点续训与评测由 `scripts/run_framepos3ep_migration_chain.sh` 编排。原两套 2-Epoch 脚本已删除，历史参数见 `mllm/docs/train_2ep_frame_position.md`。

### 混合法（当前路线：B3）

148,271 条全任务混合、**单 LoRA** 3 epochs lr 1e-4（r64/α128）、`<4DLiDAR>`/`<meta>` 两个可训练 embedding 行的框架自 B0 沿用至今（stage1 projector 数据 B1 起升级为官方 162K 方案，见下）。B 系列每次只改一个上游变量：

| 代号 | 数据/输入构造 | 唯一上游变量 | acc / mIoU |
|------|--------------|--------------|-----------|
| B0 | seqv3 切片 + 95K projector + 旧特征 | 基线锁定 | 0.7629 / 0.2696 |
| B1 | 切片 + 162K projector + 新编码器特征 | projector 数据 + 特征 | 0.7787 / 0.2653 |
| B2 | `--whole_scene` 整场景 + 旧 meta 语义 | 输入构造（对齐官方） | 0.7649 / 0.1992（TG 坍缩） |
| B3 | 整场景 + **meta2**（relative-to-previous 语义） | meta 渲染语义 | 0.7526 / **0.3467** |
| B4a | B3 配方 + TG 高帧段过采样 150,222 条 | 训练分布 | 0.7775 / 0.3271 |

- **整场景输入**（B2+）：`dataset.py`/`test_b4dl.py` 加 `--whole_scene` 门控，视觉输入不再按 QA 序列切片、直接喂入整场景 39/40/41 帧（官方 dataset.py 同款），meta 锚定与 `--answer_frames` 归属逻辑不变
- **meta2 数据**（B3）：`ego_text.py` 渲染改为论文 §4.1 relative-to-previous 语义；`re_render_meta.py` 对已注入 JSON 批量重渲染（113,053/148,271 条，35,218 条无帧号样本保留），产出 `stage2_full_train_seqv3_meta2_148k.json`
- **B4a 历史实验**：曾将 TG 高帧段样本过采样到 150,222 条，结果 mIoU 0.3271 低于 B3；数据构建与训练脚本已清理，结果保留在 [[Reproduction-Log]]
- 当前驱动：`run_stage2_full_seqv3_mixed_b3.sh`；需要显存门控、断点续训和自动评测时使用 `run_b3_pipeline.sh`

### Stage1 数据（官方 162K 方案）

`datageneration/tools/build_stage1_from_lidarllm.py` 把 LiDAR-LLM-Nu-Caption（161,845 条）过滤到训练 scene，转成 `{scene_id: sample_token, conversations}` 格式；配对特征由 `extract_pc_features_sample_token.py` 产出（28,130 帧）。

## 其他脚本

- `run_b3_pipeline.sh`：当前 B3 训练→评测链
- `run_framepos3ep_migration_chain.sh` / `smoke_framepos_train.sh`：帧位置实验迁移验收、冒烟与编排
- `re_render_meta.py`：meta2 批量重渲染（B3 用，见 [[Architecture]] 的 meta 语义）
- `build_stage2_full_train.py`：使用 HF 官方训练划分构建 Stage2 全量数据，替代已移除的自定义划分脚本
- `eval_stage1_ppl.py` / `verify_stage1_sample_data.py` / `verify_stage2.py` / `verify_frame_position.py`：评测与数据校验
- `stage1.sh` / `stage2.sh` / `stage3.sh`：标准训练驱动

训练日志统一 tee 到 `mllm/training_logs/`（含 loss 曲线 PNG）。
