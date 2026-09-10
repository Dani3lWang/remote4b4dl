---
name: b4dl-metatoken-implementation
description: metatoken 实现要点与约束——Figure 6 四行格式、ego_metadata 结构、special token
  注册、embedding 梯度掩码（单卡 ZeRO-3、weight_decay=0）
metadata:
  node_type: memory
  type: project
  originSessionId: sess_0638fcf8-2925-4746-ac7e-a2cfe910dda4
---

Metatoken 实现演进：e77f816（初版，审查判定不可直接用）→ d3af130（对齐论文，2026-08-08 合并 main）→ b0e4400（per-sequence 改造，见 [[b4dl-per-sequence-refactor]]）。

- **注入格式（论文 Figure 6/Appendix C）**：`<4DLiDAR>\n<video>\n<question>\n<meta> The metadata of the first frame is '...' and the metadata of the last frame is '...'`。tag 剥离必须**先 `split("<meta>")[0]` 再剥 `<video>`/`<4DLiDAR>`**——顺序反了会让元数据正文残留为孤立文本，造成训练/推理格式错配（修过一次的 bug）。
- **ego_metadata.json**：`{scene_id: {"first_frame","last_frame"}}`。初版 e77f816 生成"单段合并运动概括"，被审查否定——论文 Table 4 中 Metatoken 对 mIoU 的贡献（0.161→0.311）依赖首末帧**各自**的瞬时状态。计算细节：首帧用帧0→1算速度、前3帧算加速度；末帧位置/朝向相对首帧并按首帧朝向重新分解横纵向；yaw 正=左转（初版 abs() 导致恒"左转"的 bug）；加速度 Δt 用区间中点差 `(t2-t0)/2`（直接用 t2-t0 会差因子 2）。
- **special token 注册**：train.py 与 builder.py 把 `<4DLiDAR>`/`<meta>` 加入 additional_special_tokens + resize_token_embeddings（均值初始化）；**注册必须在 get_peft_model 之前**（stage3 load_lora 需要已扩容的基座才能恢复 embedding 行）。
- **embedding 梯度掩码（--train_meta_embed 默认 True）**：get_peft_model 冻结全部后对 embed_tokens.weight 重开 requires_grad + register_hook 掩码，只训 2 行 metatoken embedding，其余 32000 行 bit-identical；保存复用 non_lora_trainables.bin（键名重映射 `base_model.model.model.embed_tokens.weight` → `model.embed_tokens.weight`）。**硬约束**：仅单卡 ZeRO-3 可靠（多卡 reduce-scatter 分片下 hook 不可靠）；weight_decay 必须=0（AdamW 解耦衰减绕过梯度会侵蚀冻结行，stage2.sh 恰好是 0）；代价 ~1GB 额外 Adam 状态。曾否决 modules_to_save=["embed_tokens"] 方案（peft 0.4.0 wrap 后 resize 会 TypeError）。
- **评测管线统一（d3af130）**：删除 mllm/vtimellm/eval/ 全目录（18 文件 1570 行——旧评测按答案文本猜任务、不注入 metatoken、手写 LCS/加 1 帧的 mIoU），统一到 mllm/evaluation/ 一套；GPT_EVAL_PROMPT 补齐论文 Table 9 全部 4 个打分示例（90/100/50/10）。
- 验证：44 项数据管线测试 + 12 项 torch 模拟测试全过（Figure 6 布局逐行、注入幂等、训练/评测 round-trip、梯度掩码优化器 step 后冻结行逐位不变等）。

相关：[[b4dl-per-sequence-refactor]]、[[b4dl-training-eval-history]]、[[b4dl-qoder-conversation-history]]
