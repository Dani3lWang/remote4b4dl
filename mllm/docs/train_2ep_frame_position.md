# Frame-Position Experiment Notes

> 原两套 2-Epoch 对照脚本已于 2026-09-13 删除：它们需要重复训练无帧位置与有帧位置两组，合计约 24 小时，且论文没有明确披露 epoch 数。需要复核旧参数时请查看 Git 历史。

当前实验直接复用已训练完成的 B3（整场景 + meta2，3 epochs）作为无帧位置对照，只训练一组零初始化绝对场景帧位置 embedding：

```bash
cd mllm

# CPU 结构校验
python scripts/verify_frame_position.py

# 完整迁移验收、训练冒烟、3ep 训练与同口径评测
bash scripts/run_framepos3ep_migration_chain.sh
```

训练入口为 `scripts/run_stage2_full_seqv3_mixed_framepos3ep.sh`。它与 B3 的 148,271 条混合 QA、whole-scene 输入、BF16、LoRA `r=64/alpha=128/dropout=0.05`、batch/梯度累积、学习率、调度器和 ZeRO-3 配置一致，唯一实验变量是 `--use_frame_position_embedding True`。

该分支会把 `use_frame_position_embedding=true` 与 `frame_position_max=64` 写入 checkpoint config，并将额外的非 LoRA 参数保存到 `non_lora_trainables.bin`；评测加载器会自动恢复这些参数。
