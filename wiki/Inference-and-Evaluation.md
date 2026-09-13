# 推理与评测

最新基线评测入口：

```bash
cd mllm
bash run_baseline_eval.sh
```

该脚本固定加载 B3 adapter，使用：

```text
--whole_scene --per_sequence --answer_frames
```

预测和指标保存到 `mllm/eval_results/b3/`。评测可断点续跑。`--answer_frames` 使用答案恢复输入序列归属，属于 oracle 输入选择，报告结果时必须声明。

captioning 任务报告 SODA_c、METEOR、CIDEr；grounding 任务报告 mIoU 和 R@n。METEOR 与论文比较使用 NLTK-2005 主口径。
