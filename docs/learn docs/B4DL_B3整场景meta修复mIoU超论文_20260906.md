# B4DL B3 整场景 + meta2 结果记录

> 本文只保留当前 B3 基线的关键结论。数值来自历史 3-epoch checkpoint；当前训练脚本已统一为 2 epochs。

## 方法

B3 使用：

- 148,271 条全任务训练数据；
- 整场景 39/40/41 帧 LiDAR 特征；
- relative-to-previous 语义的 meta2；
- 162K Stage1 projector；
- LoRA r=64、alpha=128、lr=1e-4。

meta2 修复了旧元数据把首帧恒定渲染为 starting position、并错误使用前向差分的问题。当前渲染与论文“相对前一帧”的描述一致。

## 历史 3-epoch 结果

| 指标 | 论文 | B3 |
|---|---:|---:|
| accuracy | 0.762 | 0.7526 |
| mIoU | 0.311 | **0.3467** |
| BLEU-4 | 0.095 | 0.0965 |
| ROUGE-L | 0.322 | 0.3234 |
| BERTScore | 0.897 | 0.8967 |
| METEOR（NLTK-2005） | 0.275 | 0.3366 |

## 当前使用方式

```bash
cd mllm
bash scripts/run_b3.sh
```

当前脚本训练 2 epochs，以 `epoch >= 1.99` 为完成判据。新 2-epoch checkpoint 必须重新进行 30,145 条全量评测后才能形成正式结果，不能直接继承上表。

评测固定使用 `--whole_scene --per_sequence --answer_frames`；其中 `--answer_frames` 属于 oracle 输入选择，报告时必须声明。
