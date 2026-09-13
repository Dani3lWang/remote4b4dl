# 仓库结构

```text
datageneration/       数据生成
encoders/lidarclip/   LiDAR-CLIP 与 SST 随仓依赖
mllm/
  scripts/            官方入口、B3 构建/训练/校验脚本
  vtimellm/           模型、训练与推理实现
  evaluation/         B3/通用评测与分析
docs/learn docs/      当前 B3、编码器与指标说明
wiki/                 精简使用文档
```

关键入口：

- `mllm/scripts/run_b3_pipeline.sh`：B3 2-epoch 训练与评测。
- `mllm/scripts/run_stage2_full_seqv3_mixed_b3.sh`：仅训练 B3。
- `mllm/run_baseline_eval.sh`：仅评测最新 B3。
- `mllm/evaluation/analyze_tg_regression.py`：B3 与更新模型的 TG 回归分析。
