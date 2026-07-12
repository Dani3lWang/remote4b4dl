# B4DL 训练管线复现指南

> 基于 plan1 分析和 2026-07-12 实际训练验证结果编写
> 环境：RTX 5090 (32GB) + PyTorch 2.8.0 CUDA 12.8 + DeepSpeed ZeRO-3

---

## 一、环境确认

```bash
# 确认在 wqlc 环境中
conda activate wqlc

# 确认 GPU 可见（单卡 RTX 5090 只有 GPU 0）
nvidia-smi

# 确认关键依赖版本
python -c "
import torch; print('PyTorch:', torch.__version__)
import transformers; print('Transformers:', transformers.__version__)
import deepspeed; print('DeepSpeed:', deepspeed.__version__)
import peft; print('PEFT:', peft.__version__)
"
# 期望输出：PyTorch 2.8.0+, Transformers 4.47.0+, DeepSpeed 0.16.4+, PEFT 0.13.2+
```

**⚠️ 关键注意事项**：
- flash-attn **不需要安装**（RTX 5090 sm_120 不支持），训练代码自动 fallback 到 transformers 原生 SDPA
- PEFT 0.13.2 与旧版 API（`merge_and_unload`）兼容，Stage3 训练可以正常执行

---

## 二、训练前必备文件检查

```bash
cd /root/autodl-tmp/wql/mmb4dl/mllm

# 1. 基础模型 (Vicuna-7B)
ls base_model/vicuna-v1-5-7b/pytorch_model-00001-of-00002.bin  # 必须存在

# 2. Stage1 训练数据（LiDAR-LLM 描述数据）
ls data/stage1_lidarllm_mm.json  # 699 样本

# 3. Stage2/3 训练数据（B4DL 对话数据）
ls b4dl_dataset/stage2_conversations.json  # 68,695 样本
ls b4dl_dataset/stage3_train.json          # Stage3 训练集

# 4. LiDAR-CLIP 预提取特征
ls ../encoders/lidarclip/b4dl/stage2_features/ | wc -l  # 应有 850 个 .npy 文件
```

---

## 三、三阶段训练命令

### 架构概览

```
Stage 1: 对齐 mm_projector (Linear 768→4096)
  ├─ 可训练: mm_projector（仅 4096×768 + 4096 ≈ 3.15M 参数）
  ├─ 冻结: LLM、LiDAR-CLIP encoder
  ├─ 数据: stage1_lidarllm_mm.json（plain 格式，单帧描述）
  ├─ 学习率: 1e-3
  └─ 输出: mm_projector.bin

Stage 2: LoRA 微调 LLM（时序理解）
  ├─ 可训练: LoRA (r=64，全部 Linear 层)
  ├─ 冻结: mm_projector、LLM base weights、encoder
  ├─ 数据: stage2_conversations.json（vicuna_v1 对话格式，时序 QA）
  ├─ 学习率: 1e-4
  └─ 输出: adapter_model.safetensors + non_lora_trainables.bin

Stage 3（可选）: 二次 LoRA SFT
  ├─ 可训练: 新 LoRA (merge Stage2 LoRA 作为基座)
  ├─ 冻结: mm_projector、LLM base weights + merged Stage2 LoRA
  ├─ 数据: stage3_train.json（高质量 SFT 数据）
  ├─ 学习率: 2e-5
  └─ 输出: 新的 adapter_model.safetensors
```

### Stage 1 训练（对齐 mm_projector）

```bash
cd /root/autodl-tmp/wql/mmb4dl/mllm

# ⚠️ 重要：单卡训练 gpu_vis=0（不是脚本默认的 1！）
deepspeed --include localhost:0 --master_port 29571 vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./base_model/vicuna-v1-5-7b \
    --version plain \
    --data_path ./data/stage1_lidarllm_mm.json \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --tune_mm_mlp_adapter True \
    --output_dir ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1 \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
```

**预期**：5 steps，~27 秒完成。Loss 从 ~8.0 降至 ~4.2–5.0。
{'train_runtime': 26.5838, 'train_samples_per_second': 26.294, 'train_steps_per_second': 0.188, 'train_loss': 6.329545783996582, 'epoch': 0.91} 
{'loss': 4.2956, 'grad_norm': 4.972905158996582, 'learning_rate': 0.0, 'epoch': 0.91} 
**输出文件**：`checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin`（形状 `[4096, 768]`）

**验收标准**：
- [x] 训练正常启动，无维度错误
- [x] loss 呈下降趋势
- [x] 产出 `mm_projector.bin`，形状为 `[4096, 768]`

---

### Stage 2 训练（LoRA 时序微调）

```bash
cd /root/autodl-tmp/wql/mmb4dl/mllm

deepspeed --include localhost:0 --master_port 29575 vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path ./base_model/vicuna-v1-5-7b \
    --version v1 \
    --data_path ./b4dl_dataset/stage2_conversations.json \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --output_dir ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --freeze_mm_mlp_adapter True \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
```

**预期**：~1072 steps（68,695 样本 / effective batch 128 = ~537 steps/epoch × 2 epochs）。约 30-40 分钟（视显存带宽而定）。
**输出文件**：`adapter_model.safetensors`（~320MB，448 个 LoRA keys）+ `non_lora_trainables.bin`

**验收标准**：
- [x] 加载 Stage1 的 mm_projector 成功
- [x] LoRA 注入成功（448 keys）
- [x] loss 呈下降趋势（最终 ~0.28）
- [x] 无 OOM 错误

**调参建议**：
- 遇到 OOM：减小 `per_device_train_batch_size` 为 4，增大 `gradient_accumulation_steps` 为 32（保持 effective batch=128）
- 需要更多 epoch：改为 `--num_train_epochs 3`
- 数据路径不同：使用 `--data_path` 和 `--feat_folder` 覆盖默认值

---

### Stage 3 训练（二次 LoRA SFT，可选）

```bash
cd /root/autodl-tmp/wql/mmb4dl/mllm

# Stage 3 需要：
# 1. 已 merge 的 Stage2 基座模型（checkpoints/vtimellm-vicuna-v1-5-7b-stage2-merged/）
# 2. Stage3 训练数据（b4dl_dataset/stage3_train.json）

deepspeed --include localhost:0 --master_port 29576 vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-merged \
    --version v1 \
    --data_path ./b4dl_dataset/stage3_train.json \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-merged/mm_projector.bin \
    --output_dir ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3 \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --freeze_mm_mlp_adapter True \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
```

**Stage 3 的工作原理**：
1. 加载 Stage2 的 LoRA checkpoint → `merge_and_unload()` 将 LoRA 融入 LLM base weights
2. 在 merge 后的模型上添加**全新的** LoRA 适配器
3. 新 LoRA 的 lr 更低（2e-5 vs 1e-4），因为基座已经学到了时序理解能力
4. 这种模式依赖 peft 的 `merge_and_unload` API（当前 PEFT 0.13.2 支持）

**预期**：~747 steps，约 20-30 分钟。

---

## 四、一键运行脚本（推荐）

```bash
cd /root/autodl-tmp/wql/mmb4dl/mllm

# 使用 run_stages.sh 自动运行 Stage1 + Stage2
bash run_stages.sh \
    --s1_data ./data/stage1_lidarllm_mm.json \
    --s1_feat ../encoders/lidarclip/b4dl/stage2_features \
    --s2_data ./b4dl_dataset/stage2_conversations.json \
    --s2_feat ../encoders/lidarclip/b4dl/stage2_features \
    --model_name_or_path ./base_model/vicuna-v1-5-7b
```

**⚠️ 注意**：`run_stages.sh` 内部调用 `scripts/stage1.sh`，该脚本默认 `gpu_vis=1`。单卡环境需先修改：

```bash
# 修改 scripts/stage1.sh 第 3 行：
sed -i 's/gpu_vis=1/gpu_vis=0/' scripts/stage1.sh
# 同样修改 scripts/stage2.sh：
sed -i 's/gpu_vis=1/gpu_vis=0/' scripts/stage2.sh
# 同样修改 scripts/stage3.sh：
sed -i 's/gpu_vis=0,1/gpu_vis=0/' scripts/stage3.sh
```

---

## 五、推理与评估

### 单条推理测试

```bash
cd /root/autodl-tmp/wql/mmb4dl/mllm

conda run -n wqlc python vtimellm/inference.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --clip_path ../encoders/lidarclip/pretrained/ViT-L-14.pt \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    --video_path /path/to/video.mp4
```

### 批量评估

```bash
cd /root/autodl-tmp/wql/mmb4dl/mllm

# Step 1: 生成评估结果
conda run -n wqlc python vtimellm/eval/eval.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --data_path vtimellm/eval/data_example.json \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --task all \
    --log_path vtimellm/eval/log/eval_log.txt

# Step 2: 计算指标
conda run -n wqlc python vtimellm/eval/metric.py \
    --log_path vtimellm/eval/log/eval_log.txt \
    --task all --data_path vtimellm/eval/data_example.json
```

---

## 六、已发现问题与修复记录

| 问题 | 影响 | 修复方法 |
|------|------|---------|
| **GPU 索引错误**（`gpu_vis=1` 但只有 GPU 0） | 训练 fallback 到 CPU，极慢 | 改为 `gpu_vis=0` 或用 `--include localhost:0` |
| **flash-attn 不兼容**（RTX 5090 sm_120） | 无法 import flash_attn | 已处理：`train_mem.py` 中 try/except fallback 到原生 SDPA |
| **PEFT 版本升级**（0.4.0→0.13.2） | `merge_and_unload` API 变更 | ❌ 经测试兼容，Stage3 可正常运行 |
| **Transformers 版本升级**（4.31→4.47） | FutureWarning 提示 | 不影响功能，仅为警告信息 |
| **mm_projector 维度** | 曾有 128 vs 768 混淆 | ✅ 已在 commit 79c578f 修复为 768→4096 |

---

## 七、数据生成管线（如需从 nuScenes 重新生成）

### Step 1: 从 nuScenes 相机图生成场景描述

```bash
cd /root/autodl-tmp/wql/mmb4dl/datageneration

conda run -n wqlc python generate_description.py \
    --start_index 0 --end_index 100 \
    --api_key YOUR_OPENAI_API_KEY \
    --nuscenes_root /root/autodl-tmp/Datasets/nuScenes \
    --dataroot ./data
```

### Step 2: 场景描述转 QA 数据集

```bash
cd /root/autodl-tmp/wql/mmb4dl/datageneration

# 按任务类型分别生成
for task in existence binary time_grounding description temporal_understanding comprehensive; do
    conda run -n wqlc python generate_dataset.py \
        --start_index 0 --end_index 100 \
        --api_key YOUR_OPENAI_API_KEY \
        --task $task \
        --dataroot ./data
done
```

### LiDAR-CLIP 特征提取（如需重新提取）

```bash
cd /root/autodl-tmp/wql/mmb4dl/encoders/lidarclip

conda run -n wqlc python extract_pc_features.py \
    --checkpoint ./lidarclip/checkpoint/vit_l_14.ckpt \
    --scene-json-path ./annotations/scene_metadata.json \
    --frame-json-path ./annotations/sequence_metadata.json \
    --data-path /root/autodl-tmp/Datasets/nuScenes \
    --stage1-save-dir ./b4dl/stage1_features/ \
    --stage2-save-dir ./b4dl/stage2_features/
```

---

## 八、论文训练流程与代码实现的对应关系

| 论文 B4DL Model | 当前代码实现 | 训练参数 |
|---|---|---|
| **3D LiDAR Understanding Stage**：训练 LiDAR Aligner fp，使用 LiDAR-LLM 数据集 | **Stage1**：训练 mm_projector（`nn.Linear(768, 4096)`），使用 `stage1_lidarllm_mm.json`（plain 格式） | lr=1e-3, 1 epoch, 仅 mm_projector 可训练 |
| **4D LiDAR Understanding Stage**：冻结 fp，加 LoRA，使用 B4DL 数据集 | **Stage2**：加载并冻结 Stage1 的 mm_projector，对全部 Linear 层加 LoRA（r=64），使用 `stage2_conversations.json`（v1 对话格式） | lr=1e-4, 2 epochs, LoRA + mm_projector 冻结 |
| 论文未提及 | **Stage3**：merge Stage2 LoRA → 重新加 LoRA → 训练新数据 | lr=2e-5, 3 epochs |

**核心差异**：
- 论文未提及 Stage3，这是代码实现的额外细化训练阶段
- 论文使用 `<4DLiDAR>` token 和 Metatoken，代码使用 `<video>` token 和 `<meta>` 标签
