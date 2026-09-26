# ReasonSeg 训练可靠性修复（2026-09-26）

基于 AutoDL3 `37909b5`，在独立工作树 `/root/autodl-tmp/mmb4dl-seg-audit-fixes` 修改。
原仓库 `/root/autodl-tmp/mmb4dl` 的多任务训练保持原样；修复不会自动应用到已有 Python 进程。
不修改 B3 训练入口、原始清单或已有权重。分支为 `codex/seg-training-audit-fixes`。

## 改动与目的

1. **空间语义验证**：保留最近增加的线性探针协议；新预训练档额外保存分类头、标签映射、验证样本指纹与精度。新增 `--validate-only --restore-semantic-head` 只接受完整且协议一致的档。历史编码器档仍只可做同协议线性探针对照，不能复现旧 mIoU。
2. **恢复位置**：新 trainer_state v2 记录下一个 batch、轮次是否完成、数据摘要和训练预算；采用独立的逐轮确定性采样。步中恢复继续剩余 batch；最后一个 batch 后、验证前恢复仍执行该轮验证。旧档缺少可靠游标时明确拒绝完整恢复，可用 `--eval-checkpoint` 开启新实验。
3. **恢复配置**：先继承档内 ReasonSeg 配置，再处理显式参数。完整续训不允许更改损失、架构或已记录的数据/调度契约；仅权重初始化需单独命名新实验。
4. **不可达目标**：训练数据入口拒绝完全越界的正目标；准备工具按整条记录过滤，不改变剩余问题、答案与目标数量。损失也屏蔽不可达对象的 SEG/LOC/分类项，作为防御。部分越界目标保留并单列数量。
5. **负例**：准备工具从各自场景内的完整 panoptic 标签验证类别缺失，包含微小目标、范围外目标及 instance ID 为零的点，避免伪负例；默认约 10% 负例是可调整的数据配方，不是经实验优化的比例。生成时核对 train/dev/test 场景互斥。自由生成评测单独报告正例指标、负例 NOOBJ 准确率和掩码误报率；无负例时返回 null，不伪装为 0 误报。
6. **收尾脚本**：v4 使用单一报告路径，汇总失败传播非零退出码；相关启动脚本从自身目录定位，并要求显式设置 `REASONSEG_MANIFEST_DIR`。A2 扩跑默认保留恢复状态。
7. **结论边界**：移除 oracle 探针中的 AUC→top-K 可行性硬门槛；报告逐对象、分类别 top-K IoU/recall。GT 尺寸 top-K 是诊断量，不是可部署预测。单次有限训练不能证明单帧特征的普遍上限，先前日志中对此的裁定应收紧。

## 数据与使用

最终独立数据目录：`/root/autodl-tmp/reasonseg-audit-data-20260926-final`。
完整制作参数、输入/输出 SHA-256 和过滤数量见该目录 `preparation_report.json` 及本目录同名审查报告。
清单不提交到 Git；可用下面命令在一个**不存在的新目录**重建：

```bash
cd /root/autodl-tmp/mmb4dl-seg-audit-fixes/mllm
E=/root/autodl-tmp/.conda-stuff/envs/reasonseg
V=/root/autodl-tmp/mmb4dl/mllm/reasonseg_data_trainval
"$E/bin/python" scripts/reasonseg_experiments/prepare_audited_manifests.py \
  --train-manifest "$V/reasonseg_train_internal679x11.jsonl" \
  --validation-manifest "$V/reasonseg_val_internal_es.jsonl" \
  --test-manifest "$V/reasonseg_val_thin.jsonl" \
  --dataroot /root/autodl-tmp/nuScenes \
  --output-dir /root/autodl-tmp/reasonseg-audit-data-NEW --negative-fraction 0.1
```

生成的验证/测试集属于新的可达目标加负例诊断口径，不能与原 `val_thin` 指标直接比较，也不能称为官方全量结果。
新训练需显式传入新清单，模型与初始检查点使用原目录的绝对路径，输出必须使用新目录。
禁止将整个原 `checkpoints` 目录链接为隔离工作树的可写输出目录。
当前训练结束后再运行 GPU 冒烟、自由生成和同预算对照；本次没有启动新训练。

## 验证与限制

CPU：原有 25 项测试及新增恢复/配置/数据/语义头/负例回归测试。
恢复测试使用真实 Accelerate、梯度累积、dropout、AdamW 和调度器，比较中断与连续运行的数据顺序、权重、学习率和 RNG。
另检查 Shell 语法、Python 语法、Git diff 及新清单完整性。
当前验证覆盖单进程 CPU 恢复；未证明多 GPU 数值一致性，未运行新的 spconv/7B GPU 训练、B3 能力回归或性能提升实验。
新格式只保证相同代码、数据、精度、批量和调度下的恢复契约；不能把旧无游标或 `--no-resume-state` 产物宣称为精确续训档。
