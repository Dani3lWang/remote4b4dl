# B4DL 结构化几何推理与 GRPO

本实验在已有 B3 和 SST 特征上增加独立版本的结构化推理、规则奖励及 GRPO 训练入口。它是尚待完整 GPU 与任务效果验证的候选方案；不修改旧 `geometry_json_v1`、冻结实验配置或历史检查点。

## 数据与输出约定

`b4dl_core.reasoning build-data` 以既有几何数据集为父集，重新从 nuScenes 原始表、ego 位姿和输入裁剪支持记录核验每条监督；源表和支持文件允许换路径，但内容哈希必须与父数据清单一致。输出独立的 `requests.jsonl`、`supervision.jsonl`、`exclusions.jsonl` 和 `manifest.json`，模型输入不包含对象框、标注实例、几何真值及答案。场景划分沿用父集。对已知无法确认不存在的对象记录排除原因；来源、身份或标注真值不一致时停止构建。

每个回答严格以 `steps` 开头、`answer` 结尾。步骤分别使用 `observe`、`exists`、`sector`、`argmin`、`delta`、`predicate` 中该任务规定的操作，逐步引用前一项。存在性、方位、距离和相对运动分别使用固定模板；`motion` 估计两端距离和横向坐标，再计算差值。与原几何数据一样，运动量是两个时刻各自 ego 坐标中的相对变化，不是世界坐标速度。对象支持仅指历史 `CAM_BACK_RIGHT` 裁剪输入内的标注框点支持，不代表点级真值或完整全景。

观测数值按 1 m 容差连续计分，数量精确匹配；推导步骤还需与前序生成数值一致，得分不超过其依赖步骤。答案正确为 1、错误为 0；非法结构、禁止的特殊 token、未产生 EOS 的截断为 −1，步骤项不计分。缺失或矛盾的私有真值是错误，不能转成模型的失败奖励。评测的步骤通过率定义为对应分数大于 0，即米制误差严格小于 1 m 且依赖链通过；另报告连续步骤分数和有效数值的绝对误差。

## 准备与运行

使用 `mllm/docs/upcore_runtime.md` 的 Python 3.10 依赖栈。真实实验需要独立 CUDA 环境、支持 BF16 且空闲的单卡，以及已锁定的 B3 权重、SST 特征、nuScenes 表、裁剪支持记录和几何父集。启动时核验基底及 SST 绑定、训练数据身份、生成上下文长度；默认 GPU 锁沿用旧本体作业的锁。24 GB 是否足够必须由短跑峰值显存确认。先用独立输出目录完成工程短跑，不接管任何历史任务。

以下示例在仓库根目录运行，路径请按已核验资产替换；`BASE`、`PREP` 是已有本体清单，`DATA` 是旧几何父集。每个 `--output` 都必须是新路径。

```bash
export PYTHONPATH="$PWD/mllm"
PY=/path/to/wqlc/bin/python
DATA=/path/to/trainval-v1
TABLES=/path/to/nuScenes/v1.0-trainval
SUPPORT=/path/to/input-support.jsonl
BASE=mllm/configs/core_geometry_baseline_original_b3_trainval.json
PREP=mllm/configs/core_geometry_preparation.json
RECIPE=mllm/configs/reasoning_grpo_v1.json
RUN=/path/to/new-reasoning-run

"$PY" -m b4dl_core.runtime --require-cuda
"$PY" -m b4dl_core.reasoning build-data --geometry-dataset "$DATA" --tables "$TABLES" --support "$SUPPORT" --output "$RUN/data"
"$PY" -m b4dl_core.reasoning audit --dataset "$RUN/data" --geometry-dataset "$DATA" --tables "$TABLES" --support "$SUPPORT"

# 先进行有界工程检查；工程产物不视为完整训练或模型效果。
"$PY" -m b4dl_core.reasoning sft --dataset "$RUN/data" --geometry-dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --recipe "$RECIPE" --engineering --max-updates 50 --output "$RUN/sft-smoke"
"$PY" -m b4dl_core.reasoning grpo --dataset "$RUN/data" --geometry-dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --recipe "$RECIPE" --engineering --max-updates 50 --sft-checkpoint "$RUN/sft-smoke/step-50.pt" --output "$RUN/grpo-smoke"
```

工程 SFT 和 GRPO 的 `latest.json` 记录检查点及哈希，`training.jsonl` 记录损失、梯度、奖励、合法率、生成长度、旧策略概率误差、组内奖励零方差比例及 CUDA 峰值显存；`completion.json` 标注工程预算及累计峰值，不代表有效的全量模型。异常会写入 `failure.json`。记录身份不符、非有限数值、采样与重算概率不符或显存不足时停止，不自动更改批量或配方。

短跑确认后，正式 SFT 使用同一条命令去掉 `--engineering --max-updates 50`，以训练集单轮、固定末步为终点。正式 GRPO 必须加载有完整预算完成回执的 SFT 最终检查点；选择 `--mode outcome`、`--mode scalar` 或 `--mode process`，各组使用相同数据、配置、种子及 SFT 终点。工程 GRPO 只接受有工程完成回执的工程 SFT 终点或已完成的正式 SFT。训练续跑使用相同数据、源码、配方和原总预算，指定 `--resume 旧检查点 --output 新目录`；只能从已保存更新恢复，不能把诊断或不同身份的检查点当作续跑输入。

默认配方见 `mllm/configs/reasoning_grpo_v1.json`：SFT 更新投影器、A3 时序适配器和 rank 16 的语言 LoRA；GRPO 冻结前两者，只更新策略 LoRA。参考 LoRA 是 SFT 终点的副本，与策略共享冻结基座。每题 4 条候选，逐条生成并反传，每 4 个问题更新一次。old-policy 概率取自采样时的策略，reference 概率来自固定的 SFT 参考策略。有效 token 的优势由同题答案标准化奖励及对应步骤的后续标准化奖励构成；`scalar` 对照把相同步骤项广播到整个回答，`outcome` 不使用步骤项。

## 几何评测与六任务

完成训练后，对每个检查点分别运行 `infer`、`evaluate`；`--condition` 支持 `identity`、`zero`、`repeat_first`、`shuffle_features` 和 `cross_scene`，缺少合适供体的项目保留在分母。`infer --resume` 只恢复相同身份的同一输出目录，预测尾行先校验再修复。默认评测需要完整预测回执和哈希；`--allow-incomplete` 只用于明确标记的中间诊断。`compare` 对两份完整评测结果按同一数据、输入干预和生成设置作配对比较。

```bash
CKPT=/path/to/completed/final-checkpoint.pt
"$PY" -m b4dl_core.reasoning infer --dataset "$RUN/data" --geometry-dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --checkpoint "$CKPT" --split test --condition identity --output "$RUN/test-identity"
"$PY" -m b4dl_core.reasoning evaluate --dataset "$RUN/data" --geometry-dataset "$DATA" --split test --predictions "$RUN/test-identity/predictions.jsonl" --output "$RUN/test-identity-report.json"
"$PY" -m b4dl_core.reasoning compare --base /path/to/sft-report.json --candidate /path/to/grpo-report.json --output "$RUN/comparison.json"
```

报告包含每类任务的准确率、步骤得分/通过率、整条链通过率、格式失败、有效数值误差、生成 token 和耗时、场景级 bootstrap，以及相反标签配对和选项轮换。缺失、截断与不可用预测不从分母中删除。几何任务改善只有结合完整对照和输入干预结果，才能判断是否使用了 LiDAR 证据。

自由六任务协议使用历史协议所绑定的同一测试集和指标资源，从问题文本及 ego 元数据重新构建输入，绝不把答案传入输入构建。它与历史答案辅助协议分开保存和报告：

```bash
HISTORICAL=/path/to/six-task-manifest-final.json
"$PY" -m b4dl_core.reasoning prepare-six-task --project-root "$PWD" --geometry-dataset "$DATA" --historical-protocol "$HISTORICAL" --output "$RUN/six-free-protocol"
"$PY" -m b4dl_core.reasoning six-task-infer --dataset "$RUN/data" --geometry-dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --original-b3 --protocol "$RUN/six-free-protocol/manifest.json" --output "$RUN/six-b3"
"$PY" -m b4dl_core.reasoning six-task-infer --dataset "$RUN/data" --geometry-dataset "$DATA" --baseline "$BASE" --preparation "$PREP" --checkpoint "$CKPT" --protocol "$RUN/six-free-protocol/manifest.json" --output "$RUN/six-candidate"
"$PY" -m b4dl_core.reasoning six-task-metrics --dataset "$RUN/data" --geometry-dataset "$DATA" --protocol "$RUN/six-free-protocol/manifest.json" --predictions "$RUN/six-b3/predictions.jsonl" --output "$RUN/six-b3-metrics"
"$PY" -m b4dl_core.reasoning six-task-metrics --dataset "$RUN/data" --geometry-dataset "$DATA" --protocol "$RUN/six-free-protocol/manifest.json" --predictions "$RUN/six-candidate/predictions.jsonl" --output "$RUN/six-candidate-metrics"
"$PY" -m b4dl_core.reasoning six-task-compare --base "$RUN/six-b3-metrics" --candidate "$RUN/six-candidate-metrics" --output "$RUN/six-comparison.json"
```

指标命令需要 `requirements-eval.txt` 和历史协议锁定的 NLTK、Java/METEOR、RoBERTa 等资产；缺少后端或某项指标失败时拒绝生成完成回执，不以降级回退数字冒充完整结果。内部 dev 可能被历史基底见过，应在实验报告中披露。几何 SFT、规则奖励和 CPU 测试均不能单独证明六任务整体提升。
