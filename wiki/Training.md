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
- `feat_indices` 是 QA 所属序列的精确采样帧下标（论文输入 S_L），`feat_range` 是兼容闭区间；`<video>` token 位于 human 首条消息开头
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

### 两阶段法（论文/官方 stage2.sh+stage3.sh 流程）

`scripts/run_stage2_full_seqv3.sh` 幂等驱动器：

- **Phase A**：`stage2_train_seqv3.json`（简单任务）训 LoRA —— 2 epochs，lr 1e-4，bs 8×accum 16，r64/α128，tf32
- **Merge**：`scripts/merge_stage2.py` 用 `load_pretrained_model` 加载 base+projector+LoRA 后 `merge_and_unload`，保存全量模型（并把 stage1 的 mm_projector.bin 复制进去）
- **Phase B**：`stage3_train_seqv3.json`（复杂任务）在 merged 模型上训**新 LoRA** —— 3 epochs，lr 2e-5
- 评测时 `--stage2 <stage2-seqv3> --stage3 <stage3-seqv3>` 双 LoRA 依次 merge

⚠️ **实测两阶段法失败**（2026-08-26）：简单任务格式漂移、exact match 归零（acc 0.0001），已回退混合法（见 [[Reproduction-Log]]）。

### 混合法（当前基线 B0 采用）

148,271 条全部任务混合，**单 LoRA** 3 epochs lr 1e-4（r64/α128），`<4DLiDAR>`/`<meta>` 两个可训练 embedding 行，stage1 projector 用 95K nu-caption 数据重训。驱动脚本 `run_stage2_full_seqv3_mixed.sh`（B1 变体 `run_stage2_full_seqv3_mixed_b1.sh`，独立 output_dir 保 B0 可比）。

### Stage1 数据（官方 162K 方案）

`datageneration/tools/build_stage1_from_lidarllm.py` 把 LiDAR-LLM-Nu-Caption（161,845 条）过滤到训练 scene，转成 `{scene_id: sample_token, conversations}` 格式；配对特征由 `extract_pc_features_sample_token.py` 产出（28,130 帧）。

## 其他脚本

- `run_b1_pipeline.sh`：B1 全流水线驱动（重提特征 → stage1 162K → mixed-b1 → 同口径评测）
- `create_splits.py`：**已废弃**（80/10/10 自创划分会与官方测试集冲突），被 build_stage2_full_train.py 取代
- `convert_lidarllm_to_stage1.py`：LiDAR-LLM 数据转 stage1 格式的旧版映射（frame_id 键控）
- `eval_stage1_ppl.py` / `verify*.sh` / `verify_stage1_sample_data.py` / `verify_stage2.py`：评测与数据校验
- `run_metatoken.sh` / `run_stage2_full*.sh` / `resume_stage2_full.sh`：各代际数据版本的训练驱动

训练日志统一 tee 到 `mllm/training_logs/`。
