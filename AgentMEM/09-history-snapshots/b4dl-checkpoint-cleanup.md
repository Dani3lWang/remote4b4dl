---
name: b4dl-checkpoint-cleanup
description: 2026-09-09 checkpoints 精简至标志性版本：145G→1.7G，仅存 stage1+B0/B3/B4a 顶层 adapter
metadata:
  node_type: memory
  type: project
  originSessionId: sess_5a5b4be3-9fb0-410d-8a91-b144c7085a8d
---

2026-09-09 按"只保留标志性 checkpoint"清理 `mllm/checkpoints/`（145G → 1.7G，释放约 143G，不可恢复无备份）：

- **保留**：`vtimellm-vicuna-v1-5-7b-stage1/`（61M，含 mm_projector.bin，B 系 stage2 重训依赖，评测脚本加载它）、`-stage2-full-seqv3-mixed/`（B0 基线顶层，556M）、`-mixed-b3/`、`-mixed-b4a/`（各 556M，仅顶层 README/adapter_config/adapter_model.safetensors/non_lora_trainables.bin/config/trainer_state.json）
- **整版删除**：stage2、stage2-full、stage2-full-seq(+merged)、stage2-full-seqv2(+merged)、stage2-seqv3(+merged)、stage3-seqv3、mixed-b1、mixed-b2
- **删除 B3/B4a 内 checkpoint-3200/-3400/-3474(或3519)**：中间档仅服务断点续训；评测与训练实际加载的是目录顶层 adapter + stage1/mm_projector.bin（已 grep eval_log 与 run_b4a_pipeline.sh 核实）

**Why:** 用户原话"只保留带有标志性的 checkpoint"，AskUserQuestion 两次未答复故取推荐口径执行；B3/B4a 顶层 adapter+non_lora 即 [[b4dl-rl-introduction-plan]] 的 RL 初始化物，不受影响；各删除版本评测产物（predictions/metrics）均已在 mllm/eval_results/<run>/ 归档。

**How to apply:** 后续引用旧链权重（seq/seqv2/seqv3/stage3/B1/B2）需重训或从 git/评测 JSON 恢复指标；磁盘现状可查 `df -h /root/autodl-tmp`（2026-09-09 时 939G 可用）。此前同期还清理了 mllm/training_logs 与 eval_results 冒烟残留（commit a3f58d1）。
