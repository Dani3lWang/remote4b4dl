# 架构总览

## 数据流全景

```
nuScenes 数据集（相机图像 + LiDAR 点云）
  │
  ├─ [datageneration/] 相机图像 → GPT-4o → 场景描述 JSON → QA 数据集 JSON
  │     数据格式：{"from": "human/gpt", "value": "..."}
  │
  ├─ [encoders/lidarclip/] LiDAR 点云 → SST编码器 → CLIP 特征 (.npy)
  │     Stage1 特征：每帧独立 .npy，shape (1, 768)
  │       - stage1_features/（旧，frame_id 键控，95K 数据用）
  │       - stage1_features_sample/（新，sample_token 键控，对齐官方 162K 方案）
  │     Stage2 特征：每场景拼接 .npy，shape (N_frames, 768)
  │
  └─ [mllm/] 预提取特征 + QA 数据 → VTimeLLM → 训练 / 推理 / 评估
```

三个模块之间只通过**文件**耦合：特征文件名 = 数据 JSON 中的 `scene_id`（详见 [[Training]] 的特征查找约定）。

## 三阶段训练流程

| 阶段 | 目标 | 关键配置 |
|------|------|----------|
| **Stage 1** | 对齐 mm_projector | 只训练 `nn.Linear(768, 4096)`，LLM 冻结。`--tune_mm_mlp_adapter True`，`--version plain` |
| **Stage 2** | LoRA 微调 LLM | 加载 Stage1 的 mm_projector 并冻结，对全部 Linear 层加 LoRA。`--lora_enable True`，`--freeze_mm_mlp_adapter True` |
| **Stage 3**（可选） | 二次 LoRA | 先 merge Stage2 的 LoRA 权重作为基座，再加新 LoRA，解冻 mm_projector |

实际复现采用 **Stage2+Stage3 混合单 LoRA** 或 **两阶段法（Phase A 简单任务 → merge → Phase B 复杂任务）**，详见 [[Training]]。

## 多模态融合机制

训练数据中的 `<video>` token 在前向传播时被替换为投影后的视觉特征，核心逻辑在 `vtimellm_arch.py` 的 `prepare_inputs_labels_for_multimodal()`：

1. 找到 `<video>` token 位置（tokenizer 阶段已被 `tokenizer_image_token` 替换为 `IMAGE_TOKEN_INDEX=-200`）；
2. CLIP 特征（N, 768）经 `mm_projector` 投影到 LLM hidden size（Vicuna-7B 为 4096）；
3. 投影后的 embedding 插入对应位置，labels 对应位置设为 `IGNORE_INDEX`（不参与 loss）；
4. 序列填充/截断到统一长度，生成对应 attention mask。

## 关键模型组件

| 组件 | 位置 | 说明 |
|------|------|------|
| `VTimeLLMLlamaForCausalLM` | `vtimellm/model/vtimellm_llama.py` | 继承 `LlamaForCausalLM` + `VTimeLLMMetaForCausalLM`，重写 `forward()` 在调用父类前先做多模态融合 |
| `VTimeLLMMetaForCausalLM` | `vtimellm/model/vtimellm_arch.py` | 多模态融合核心，实现图像 token 替换、序列填充、attention mask 生成 |
| `mm_projector` | 同上 | 单层 `nn.Linear(768, hidden_size)`，将 CLIP ViT-L/14 输出映射到 LLM hidden space |
| `VTimeLLMTrainer` | `vtimellm/train/vtimellm_trainer.py` | 继承 HF Trainer；Stage1 时只保存 mm_projector 权重（非全量 checkpoint），规避 ZeRO-3 冲突 |
| `load_pretrained_model` | `vtimellm/model/builder.py` | 推理/评测的完整加载链：base → stage1 projector → stage2 LoRA merge → stage3 LoRA merge |
| ChatGLM backbone | `vtimellm/model/vtimellm_chatglm.py` | 支持 ChatGLM3-6b，通过模型名含 "chatglm" 自动选择；对应 `stage1_glm.sh` / `stage2_glm.sh` |

## 对话模板系统

`conversation.py` 通过 `conv_templates` dict 管理模板，训练时由 `--version` 参数选择，不同模板的 separator style 决定 `preprocess()` 的分支：

| 模板 | 用途 | 格式 |
|------|------|------|
| `plain` | Stage1 | 无角色标记，`<video>\n` + 描述文本，只对 gpt 部分计算 loss |
| `v1` / `vicuna_v1` | Stage2/3 与评测 | `USER: ... ASSISTANT: ...</s>`，loss 只算 ASSISTANT 部分 |
| `llama_2` | 备选 | `[INST] ... [/INST]` |

## 六种 QA 任务类型

在 `datageneration/config.py` 中定义，分简单/复杂两组：

- **简单任务**：`existence`（物体存在性）、`binary`（二值问答）、`time_grounding`（时间定位，答案为 `from frame X to frame Y.` 时间跨度）
- **复杂任务**：`description`（整体描述）、`temporal_understanding`（时序理解）、`comprehensive`（综合推理）

简单任务用 Accuracy / mIoU 评测，复杂任务用文本生成指标评测（详见 [[Inference-and-Evaluation]]）。

## Metatoken 机制（论文 Appendix C）

评测与训练的 human 消息最终格式为：

```
<4DLiDAR>\n<video>\n{question}\n<meta> The metadata of the first frame is '...' and the metadata of the last frame is '...'
```

- `<4DLiDAR>`：任务标记特殊 token（可训练 embedding 行）
- `<video>`：视觉特征插入占位符
- `<meta>`：后接 QA 所引用**首帧与末帧**的 ego 车辆运动状态自然语言描述（位置/地形/速度/转向/加速度），由 `mllm/scripts/ego_text.py` 统一渲染，训练与推理逐字符一致

详见 [[Training]] 与 [[Inference-and-Evaluation]]。
