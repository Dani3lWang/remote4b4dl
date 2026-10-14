# mini 空间实例命中改进：计划与代码体系

本计划先解决单帧实例掩码可学习性，再进入多帧关联和语言绑定。已完成的旧诊断为：完整 323 帧训练集 Recall@0.5=1.2066%；8 帧追加 50 轮后仍为 2/255 命中。以下新实验的状态以 `study_state.json` 和实际权重、预测产物为准，不能将本计划当作已完成的效果提升。

## 不变条件

- 用户指定服务器 `connect.bjb1.seetacloud.com:21358`；独立实验目录，复用官方 mini 原始数据，不修改既有 ReasonSeg/B3 运行。
- 10 个官方 Thing 类；全部实例保留；原始点序、panoptic、官方 ignore、有效点和类别一致性逐帧预检。
- 每个训练场景固定取中间帧，共 8 帧 / 255 实例；未按模型表现选取。迁移时核对旧帧身份、原始点文件哈希及逐点 GT，不依赖旧服务器绝对路径。
- 新准备器的 source_panoptic_file 使用相对 panoptic_root 的路径，避免再次产生服务器绝对路径绑定；源文件哈希和准入验证不变。
- 每个 pilot 均从同一个旧 5 轮 checkpoint 加载权重，新优化器；seed=2026、lr=1e-4、50 epochs / 400 steps、梯度累积 1。
- 固定 score=0.25、mask=0.5、IoU=0.5。模型预测只输入点云和时间戳，不输入 GT、语言、轨迹身份，不使用 GT 选择 query 或 memory。
- 标准 TP/FP/FN 及平均 GT mask IoU 不改变；诊断和内部开发成绩单列。

## 顺序与预算

| 阶段 | 实际工作 | 对照关系 / 进入条件 |
| --- | --- | --- |
| P0 | 环境、代码隔离、mini 标签重建与迁移等价核对 | 原始点云和 GT 一致后才能训练 |
| E0 | 初始权重的同一 8 帧全 query 诊断；原 mean BCE 重新训练 50 轮 | 建立新 GPU / 新环境上的配对基线 |
| E1 | foreground_balanced BCE，同一 8 帧 50 轮 | 对比 E0，只改变训练 BCE；Dice 和 Hungarian 匹配代价不变 |
| E2 | E1 未通过时，balanced BCE + memory=8192，再训练同一 8 帧 50 轮 | 与 E1 比较 memory 单项作用；与 E0 比较整个配方。GT 不参与 memory 采样 |
| E3 | 有配方通过 pilot 筛查后，完整 323 帧上重新训练 mean 与候选各 5 轮 / 405 步 | 同一初始权重、新优化器、梯度累积 4；分别完整评测 323 帧 train 和 81 帧 internal_dev |
| P1 | 独立复算、分层、预测图、报告、权重及配置打包下载 | 用真实预测结果决定是否可继续空间训练 |

本次最多执行 3 个 pilot，共 1,200 步；扩展阶段最多两个 5 轮运行。失败时保留退出码、错误及已完成产物，不反复无限训练。不会自动启动完整 trainval、时序或语言模型训练。

pilot 扩展门槛在启动前固定：相对 mean 基线 Recall 至少增加 5 个百分点、全部 GT 平均 mask IoU 至少增加 0.01，且 F1 不下降。它只决定是否值得扩大实验，不是官方验收标准，也不表示已经充分过拟合。8 帧上两项指标的一点波动不能视为成功。

## 代码边界

- `criterion.py`：`mask_losses(..., bce_mode=...)`，默认 `mean` 保留原 BCE；`foreground_balanced` 使用正、负组均值等权，只有一组时使用该组均值，忽略 -1 和 ROI 外点。Dice、匹配代价、负实例抑制保持原有定义。记录 BCE、Dice、classification、前景/背景 BCE 分量。
- `train.py`：新增 `--mask-bce-mode`，写入 checkpoint 的 training_config；resume 时不允许跨损失模式。`--allow-memory-points-change` 仅在 init-from 中明确允许 memory_points 这一不改变权重形状的配置变化，并记录前后值。其他配置仍严格匹配。
- `diagnostics.py` / `evaluate.py --query-diagnostics`：保存过滤前全部 query 的二值掩码和类别概率；GT 只用于评分。分开计算全 query 几何命中、全 query 类别正确命中、正式保留预测命中，全部采用一对一匹配。单 GT 的 best-IoU 可复用 query，仅作诊断，不称为召回率。
- 漏检原因分为：掩码 IoU 不足、无对象、类别错误、分数过滤、query 竞争。统计点数分层与实际 forward 设备上的 memory 直接前景覆盖；覆盖关联不是因果结论。
- `study.py` / `run_grounded4d_spatial_study.py`：串行执行预检、训练、评测和 gate；每条真实命令、起止时间、退出码、初始权重/manifest 哈希记录到 study_state.json，不覆盖旧运行。
- `tests/test_spatial_study.py`：验证小目标正负梯度总权重、背景复制不稀释正例、ignore 零梯度、原 mean/Dice 兼容性、单组边界，以及类别/无对象/阈值/query 竞争的独立记账。仅在云端执行，遵循用户不做本地测试的要求。

## 云端入口

```bash
conda run -n wqlc python mllm/scripts/run_grounded4d_spatial_study.py \
  --study-dir training_logs/grounded4d-spatial-study-001 \
  --config mllm/scripts/grounded4d_mini_config.json \
  --subset-root /path/to/equivalent-overfit8 \
  --label-root /path/to/prepared-mini \
  --nuscenes-root /path/to/nuScenes \
  --panoptic-root /path/to/nuScenes-panoptic-v1.0-all \
  --init-from /path/to/original-5epoch/last.pt
```

运行目录必须是新的；完整 epoch 的断点续训继续使用 trainer 的 `--resume`，不能用旧优化器跨损失配方。失败的 study 不冒充完成，应核对原因后新建 study，或按记录的单条 trainer 命令恢复其自身运行。

## 当前阶段之后

若所有 pilot 均未通过，依据全 query 的分层结果选择下一项单变量实验：掩码几何始终低时检查早期 masked attention 与 backbone；有高 IoU 掩码却被过滤时检查类别/无对象头；归一化问题另做固定条件对照。每次只有一个因素改变，不把 BatchNorm 只读 probe 当作训练成绩。

空间阶段通过后才生成 mini 多帧 manifest，使用 sample_annotation.instance_token 审核跨帧身份、坐标、遮挡与不可见状态，再训练/评测纯预测关联；语言绑定需独立的回答—唯一实例审核。当前实现不承诺这些阶段已完成。

## 第一层注意力对照（独立后续运行）

三组已完成 pilot 中，mean 命中 4/255、平衡 BCE 命中 4/255、平衡 BCE + 8192 memory 命中 3/255；候选均未通过原扩展门槛。下一项仅改变第一层的硬注意力排除规则，继续使用 mean BCE、4096 memory、同一初始权重、8 帧、50 轮 / 400 步与固定阈值。

`train_grounded4d.py --first-layer-attention full` 通过第一层的 forward pre-hook 将 attention bias 中的 -inf 改为 0，保留有限的位置 bias；后续层不变。默认 masked 不安装 hook，不修改原模型源码、参数名、权重形状或初始化。该 full 选项仅用于 spatial 阶段。checkpoint 记录 training_config.first_layer_attention、attention_behavior 及实现哈希；评测按该记录恢复模式并核对实现身份。旧 checkpoint 只有默认 masked 兼容路径，原 model/backbone 源码哈希检查继续保留。

`run_grounded4d_attention_study.py` 先复评上轮 mean checkpoint，要求完整标准 metrics 与原报告完全一致，再从原初始权重训练 full 第一层。复用的 mean 对照不算本轮重新训练。门槛仍是 Recall 增加至少 5 个百分点、全部 GT 平均 mask IoU 增加至少 0.01、F1 不下降，即相对本轮基线至少命中 17/255、IoU≥0.02717166。通过后才从相同初始权重进行完整 mini 的 masked/full 配对训练，各 5 轮 / 405 步，并分别评测 323 帧 train 和 81 帧 internal_dev。未通过时不扩大训练，进一步只读检查编码器空间特征。

注意力实现和边界检查分别位于 `attention.py` 与 `tests/test_attention.py`。检查默认行为、仅第一层变化、state_dict 兼容、重复配置及恢复、旧 checkpoint 和缺失/错误行为哈希。所有检查在云端执行。
