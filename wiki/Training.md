# 训练

## B3 基线

B3 是 `main` 上唯一维护的实验配方：

| 配置 | 值 |
|---|---|
| 数据 | `stage2_full_train_seqv3_meta2_148k.json` |
| 输入 | 整场景 `--whole_scene True` |
| Epoch | **2** |
| 学习率 | 1e-4，cosine |
| LoRA | r=64，alpha=128 |
| Batch | 8 × gradient accumulation 16 |
| 并行 | BF16 + ZeRO-3 |

```bash
cd mllm
bash scripts/run_b3.sh
```

训练完成判据为最终 adapter 存在且 `trainer_state.json` 中 `epoch >= 1.99`。脚本会从编号最大的 checkpoint 恢复。

历史 mIoU 0.3467 来自旧 3-epoch B3 checkpoint，新 2-epoch 配方需重新评测后再发布数值。

## 通用入口

`scripts/stage1.sh`、`stage2.sh`、`stage3.sh` 及 ChatGLM 版本保留官方能力。B3 复现应使用上面的专用流水线。
