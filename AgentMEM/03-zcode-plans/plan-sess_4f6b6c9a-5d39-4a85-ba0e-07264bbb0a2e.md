# 方案 A：stage2 视觉输入改回整场景（对齐论文/官方代码）— 实施计划

## 背景与目标
根因已证实：seqv3 按序列切片使 TG 模型学会"序列局部帧编号"，评测按场景全局编号解析，mIoU 被错位压低（0.265 vs 论文 0.311）。官方实现（ccho4702/B4DL）训练/评测输入均为整场景 (40,768) 特征、无切片。本计划把 stage2 的视觉输入改回整场景，重训 + 评测出 B2，目标 mIoU 对齐论文（≥0.30，论文 0.311）。

## 改动 1：训练端（1 个文件，门控式、可回滚）
`mllm/vtimellm/train/dataset.py`
- DataArguments（L65-70）新增字段 `whole_scene: bool = False`
- L470 改为 `if not self.data_args.whole_scene: image = _select_features(image, source)`
- 默认 False，B0/B1 复现不受影响；B2 脚本传 `--whole_scene True` 即回到官方无切片行为（官方 dataset.py 对 np.load 结果直接用）
- `_slice_features/_select_features` 保留为兼容死代码，不删

## 改动 2：评测端（1 个文件，门控式）
`mllm/evaluation/test_b4dl.py`
- argparse（L440-457 区域）新增 `--whole_scene`（store_true）
- L583-589 的 if/else 改为：`if getattr(args, 'whole_scene', False): feat_used = feat else: <原切片逻辑原样保留>`
- L591-602 meta 构建**不动**：`--per_sequence` + `--answer_frames` 继续锚定 meta（TG 用 GT 答案帧、其余用问题引用帧），视频输入与 meta 解耦

## 改动 3：新增 B2 脚本（2 个）
1. `mllm/scripts/run_stage2_full_seqv3_mixed_b2.sh`：复制 `run_stage2_full_seqv3_mixed_b1.sh`，仅改三处——`OUT=...-mixed-b2`、`master_port 29582`、追加 `--whole_scene True`；数据（148k JSON 原样复用，feat_indices 被忽略）、162K projector、3 epochs、lr 1e-4、LoRA r64/α128、ZeRO-3 等超参**全部与 B1 一致**（唯一变量=输入构造）；保留 `sort -V` 断点续训与 pipefail/End 日志修复
2. `mllm/scripts/run_b2_pipeline.sh`：两阶段链（训练→评测），复用 b1 pipeline 的成熟机制：
   - 训练阶段：显存门控（≥28GB，12 次×10 分钟）+ 失败重试 3 次，成功判据沿用已修复的 `trainer_state.json epoch≥2.99`（不重蹈 End: 判据 bug）
   - 评测阶段：显存门控（≥14GB，5 次）+ timeout 18000，命令与 b1 阶段4 相同但加 `--whole_scene`，输出 `eval_results/stage2_full_seqv3_mixed_b2/`

## 执行顺序
1. 改 `dataset.py` + `test_b4dl.py`（门控开关）
2. 冒烟验证①：用 wqlc 环境实例化 `LazySupervisedDataset(whole_scene=True)`，抽样打印 image shape 应为 39/40/41（确认不再被切到 3-10）
3. 新增两个 b2 脚本；`bash -n` 语法检查
4. git commit（中文，fix/feat 前缀）+ push
5. 冒烟验证②：`--max_steps 3 --output_dir /tmp/b2_smoke` 跑 3 步确认 argparse/collator/整场景变长 39/40/41 全通（模型 arch 对变长已兼容，探索已确认）
6. tmux 会话 `b2pipeline` 后台启动 `run_b2_pipeline.sh`（GPU 被 xmuda 占用时门控自动等待；断点续训兜底），确认前几步 loss 正常后交后台
7. 训练完成后评测产出 `eval_results/stage2_full_seqv3_mixed_b2/metrics.json`

## 预期与验收
- mIoU 目标 ≥0.30（论文 0.311；B1 为 0.2653 错位值），TG 预测恢复全局帧编号
- acc 预期从 0.7787 回落至论文水平 ~0.76（整场景是论文原始设置，属预期代价）；bleu4/meteor/rouge_l/bertscore 保持论文附近
- 验收：B2 vs B1 vs 论文三方对比表；若 mIoU<0.30，启动备选方案 C（meta 注入帧号线索）另行讨论
- 时间成本：训练约 24-30h（整场景多 ~35 token/样本，比 B1 慢约 30-50%）+ 评测约 5h