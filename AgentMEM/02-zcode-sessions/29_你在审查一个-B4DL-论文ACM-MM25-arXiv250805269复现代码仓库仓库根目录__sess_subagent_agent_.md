# 你在审查一个 B4DL 论文（ACM MM'25, arXiv:2508.05269）复现代码仓库，仓库根目录：/...

| 项 | 值 |
|---|---|
| 会话 ID | `sess_subagent_agent_759addcf-a44f-4201-aafc-ad8ec320d54e` |
| 工作目录 | `/root/autodl-tmp/wql/mmb4dl` |
| 父会话 | `sess_5c6e22af-4ead-4c47-a935-c6ba054b8d75` |
| 时间 | 2026-08-29 19:20 → 2026-08-29 19:20 |
| 模型 | GLM-5.3-Flash |
| 消息数 | 3（文本块 1，工具调用 0） |

---

### ASSISTANT  ·  `GLM-5.3-Flash`

> 🔀 模型切换：None → builtin:bigmodel-start-plan/GLM-5.3-Flash


### USER

你在审查一个 B4DL 论文（ACM MM'25, arXiv:2508.05269）复现代码仓库，仓库根目录：/root/autodl-tmp/wql/mmb4dl。你的任务：审查 `mllm/` 模块的模型架构与训练实现，逐项对照论文 §4（B4DL Model and Training Pipeline）。搜索广度：very thorough。

论文基准事实（用于对齐）：
1. §5.1：骨干 LLM 为 Vicuna-7b-v1.5，训练与推理全程冻结。
2. §4.1 三模块：E_L（冻结的编码器，768 维/帧）；LiDAR Aligner f_p 是「a single linear layer」，把编码器嵌入投影到 LLM 文本嵌入空间；Metatoken '<meta>' 前置于 ego 车辆文本描述，内容是把传感器数值转成「相对前一帧的方向、位置、速度、加速度」的自然语言，且只对 QA 引用的「首帧与末帧」写描述（附录 C / Figure 6）。
3. §4.2 两阶段：Stage 1（3D LiDAR Understanding）只用静态 3D 点云的 LiDAR-LLM-Nu-Caption 数据（162K QA），只训 f_p，其余全冻结；f_p 在拼接输入上也能工作（为 4D 做准备）。Stage 2（4D LiDAR Understanding）用 B4DL 数据（178K QA），冻结 E_L 与 f_p，只引入 LoRA；所有 QA 输入前加 '<4DLiDAR>' token；训练同时用 Simple Tasks 和 Complex Tasks（"The model is trained on both Simple Tasks and Complex Tasks"）；损失为因果语言建模交叉熵。
4. Figure 6 输入格式：<4DLiDAR> + 问题文本 + <meta> + "The metadata of the first frame is '...' and the metadata of the last frame is '...'"（红字连接词）。
5. 论文未披露：lr/epochs/batch/LoRA rank/α（➖论文未规定，按官方仓库惯例）。

审查对象（仓库根目录下 mllm/）：
- mllm/train.py：可训练参数控制逻辑——tune_mm_mlp_adapter（stage1 只训 projector）、lora_enable、freeze_mm_mlp_adapter（stage2 冻结 projector）、special token embedding 的梯度处理（train_meta_embed / gradient mask / requires_grad）、断点续训逻辑（train.py:444-451 附近的 glob 是否排序——已知问题清单说没排序，确认现状）、DeepSpeed 配置。
- mllm/vtimellm/model/vtimellm_arch.py：prepare_inputs_labels_for_multimodal——<video> token 替换、特征切片用法（feat_indices/feat_range）、labels IGNORE_INDEX。
- mllm/vtimellm/model/vtimellm_llama.py 与 builder.py：mm_projector 定义（是否 nn.Linear(768, hidden) 单层）、LoRA target_modules（是否含 mm_projector——应不含）、special token 注册（<4DLiDAR>/<meta> tokenizer add_tokens？embedding resize？）。
- mllm/vtimellm/train/dataset.py：LazySupervisedDataset 特征切片逻辑（feat_indices 优先、feat_range、回退帧号解析）、异常兜底 random.choice（已知问题，确认现状）、metatoken 在数据中的位置。
- mllm/scripts/run_stage2_full_seqv3_mixed.sh 与 merge_stage2.py：当前混合训练配置（epochs、lr、batch、LoRA r/α、tf32、ZeRO-3）。
- mllm/scripts/stage1.sh / run_stages.sh：stage1 配置（tune_mm_mlp_adapter、plain 模板）。
- mllm/scripts/inject_metatoken.py 与 mllm/scripts/ego_text.py：metatoken 文本格式——是否 <4DLiDAR> 前缀 + 问题 + <meta> + "The metadata of the first frame is '...' and the metadata of the last frame is '...'"；ego 文本是否含方向/位置/速度/加速度四要素且相对前一帧；是否只写首末帧。
- mllm/b4dl_dataset/stage2_full_train_seqv3_148k.json 或 stage2_train_seqv3.json：读前 3 条样本，确认人侧 value 的实际格式（<4DLiDAR> <video> <meta> 顺序？）。
- mllm/scripts/run_stage1_ppl / stage1 数据：stage1_lidarllm_mm.json 之类的 stage1 训练文件条数（当前是 95,048 还是 161,845？看 stage1_train.json / stage1_train_700scenes 文件，用 python3 统计）。

输出要求（中文）：逐项判定 ✅一致 / ⚠️偏差 / ❌不一致 / ➖论文未规定，每项附 file:line 证据与关键代码摘录。特别回答：①stage1 只训 f_p 的实现路径是否严密（有无其它参数泄漏可训练）；②stage2 LoRA-only + f_p 冻结是否严密；③<4DLiDAR>/<meta> 两 token 与 Figure 6 格式一致性（逐字符比对格式模板）；④LLM 冻结 + embedding 梯度掩码实现；⑤stage1 数据当前条数（162K 对齐进度）；⑥输入序列中 <video> 特征占位与 4D embedding（多帧 CLS 拼接）如何对应。最后给总体结论。


### ASSISTANT  ·  `GLM-5.3-Flash`

