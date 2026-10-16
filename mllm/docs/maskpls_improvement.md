# MaskPLS 自车掩码问题：改进方案与执行约定

2026-10-16 的旧运行完成 1 epoch、81 次更新，但 81 帧 mini_val 的 3,503 个真实实例全部漏检；每帧唯一输出覆盖自车忽略标签。此次改进首先验证空间掩码学习，不引入语言或跟踪结论。

## 改动依据

上游固定提交 `84881ba83f2f2e846cbded677544993d0c8ecd76` 的
[semantic_dataset.py](https://github.com/PRBonn/MaskPLS/blob/84881ba83f2f2e846cbded677544993d0c8ecd76/mask_pls/datasets/semantic_dataset.py#L277-L324)
将每个 Thing/Stuff 目标构造成完整点轴上的二值掩码，非该实例的点为 0；
[loss.py](https://github.com/PRBonn/MaskPLS/blob/84881ba83f2f2e846cbded677544993d0c8ecd76/mask_pls/models/loss.py#L140-L172)
对采样的二值掩码计算 BCE/Dice。原移植将语义 ignore 同时用于排除掩码匹配与损失，导致输入中保留的自车点没有实例掩码约束。

新增 `mask_ignore_policy=background`：掩码匹配与 BCE/Dice 将 ROI 内语义 ignore 点视为负样本；语义 CE/Lovasz 和评测仍忽略它们。保留全部原始点、全部一点评测目标、原始 ROI 和输出阈值。模型前向及候选提取没有读取 GT。默认值仍是 `exclude`，旧配方的含义不被静默改变。

## 先对照，再扩大预算

1. 从已完成的同一份 1-epoch 权重开始；按照训练场景 token 排序，每个场景固定选时间戳居中的一帧，共 8 帧、255 个 Thing，其中 42 个有至少 50 个有效点。选择不使用 mini_val 结果。
2. 三组各训练 160 步，batch=1、AdamW lr=1e-4、相同帧顺序与逐步采样随机种子。A 保留旧监督和 masked attention；B 仅将第一层 cross-attention 放开；C 只修正掩码负监督。每 40 步评估这 8 帧。
3. 仅 C 继续进行有上限的小样本收敛诊断：最多累计 800 步，每 160 步检查训练集门槛；这部分不能再声称与 160 步 A/B 等预算。
4. 预设扩大训练门槛：50+ 点目标的原始 query 几何召回至少 50%，最终类别正确召回至少 20%，至少 6/8 帧有正确实例，忽略区域占输出前景不超过 10%。所有大小目标始终计入完整主指标。
5. 小样本诊断结束后，对全部 81 帧 mini_val 评估一次，并独立复算保存的预测。若训练门槛失败，停止直接扩大完整数据训练，记录失败项；不把消除自车误检等同于实例分割可用。
6. 门槛通过后，使用相同完整 323 帧训练 manifest 启动修正配方的有限预算 pilot，保留 81 帧开发集和完整评测链路。训练集过拟合指标与开发集指标分别报告。

## 可复现入口

修正配置：`mllm/scripts/maskpls_background_config.json`。`train` 新增显式 `--warm-start` 和 `--warm-start-source-sha256`：只迁移已审核源代码的模型权重，训练和开发数据身份必须一致，模型结构/ROI/阈值不允许随迁移变化。优化器和轮数重置，来源训练量单列；`--resume` 保持原来的精确恢复限制。

脚本及运行证据不提交：本地 `training_logs/maskpls_autodl3_20261016/`；旧服务器运行目录 `/root/autodl-tmp/mmb4dl-maskpls-20261016` 下的 `training_logs/improvement-001`、`improvement-002`、`improvement-003`。修正代码单独部署在 `/root/autodl-tmp/mmb4dl-maskpls-improved-20261016`，原代码与原始权重保留。

阶段结果和最终是否满足门槛，以本地运行目录中的《改进执行报告.md》和对应 JSON/退出码为准。本文件描述方案，不把正在执行的阶段当成完成结果。

## 边界

指标是原始点空间、类别正确、一对一实例 Recall/Precision@0.5，以及分层与原始 query 诊断，不是官方 PQ，也不是 B3 语言指代、NOOBJ 或跨帧跟踪结果。若以后使用官方预训练权重，应逐场景核对训练暴露；官方 mini_train 与完整 train/val 的成员并非简单子集关系，不能仅根据 mini 名称推断泄漏或独立性。

## 小实例 BCE 修正与有限预算筛选

`mask_bce_reduction` 默认仍为 `point_mean`，旧配置按该值解析。显式配置
`mllm/scripts/maskpls_balanced_bce_config.json` 使用 `instance_balanced`：对每个匹配实例的正、负采样点分别求 BCE 均值，再平均两组和所有实例。全正目标只使用正组；缺少正点会报错。自车等 ROI 内忽略点继续作为掩码负监督，语义 ignore、ROI、匹配代价、Dice、分类损失和输出阈值不变。该配方需要实验验证，不默认视为性能修复。

受控 `--warm-start` 现在仅允许显式更改上述两项掩码损失策略，仍要求源实现摘要、相同训练/开发数据、相同模型结构及推理配置；精确 `--resume` 的身份约束不放宽。修改代码后的旧权重须经过显式迁移，不能伪装成原运行的续训。

`mllm/scripts/run_maskpls_loss_probe.py` 提供有上限的两臂筛选：核实父权重及全部源训练/开发数据身份，每个训练场景取时间戳居中的一帧作为冻结探针；从中均匀选择最多 16 帧作为优化子集。先评估冻结训练探针和全部内部开发帧，然后从相同父权重、相同随机流、相同帧序列分别训练原 BCE 与实例平衡 BCE。默认每臂 160 次更新、batch=1、恒定 lr=1e-4、梯度裁剪 1.0，优化器均重置；中点和终点权重分别保留。

筛选只使用父训练场景与内部开发集，不读取 mini_val。计划与工程门槛在推理、训练前落盘；按小实例几何 TP、最终 TP、精度、忽略区域占比及自车预测点数判定是否值得扩大验证。工程门槛不等于统计显著或模型验收。子集权重使用独立 `b4dl-maskpls-loss-probe-v1` schema，防止被生产 runner 误当完整数据训练 checkpoint。

`audit` 模式独立重算保存的原始 query 和最终掩码，核对全部点数桶、ROI、忽略区域、自车点、原文件哈希与 checkpoint 哈希。测试和筛选结果应与源码提交身份一同报告；未通过时保留默认配方，不直接扩大训练。

首轮 `balanced-bce-001` 的 50/50 配方未通过筛选：96 帧开发集的 1–10 点几何 TP 从冻结的 161 降到 0，总 TP 从 161 降到 24。训练初期梯度及候选覆盖率明显增加，因此该结果不支持启用完整平衡配方。

后续 `sqrt_positive` 使用更温和的正点权重 `min(20, max(1, sqrt(N_negative/N_positive)))`，每个实例按权重和归一化后再平均；全部负点仍计入。显式配置为 `maskpls_sqrt_positive_config.json`。该配方同样保持默认关闭，需单独验证。

`continuation --reference-dir <首轮 results>` 可在核对首轮独立审计与数据身份后，从相同父权重、帧序列和预算运行 C 臂；它不是从失败 B 权重续训。`audit --audit-arm C --reference-dir ...` 独立复核新增 C 产物并验证引用证据哈希。首轮门槛保留，后续结果不冒充第一轮预注册的三臂实验。
