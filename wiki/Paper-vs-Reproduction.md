# 与论文的对照（差异清单）

复现 ≠ 复刻。本页列出复现实现与论文（arXiv:2508.05269）的全部已知一致项与偏差项，依据 2026-08-24 与 2026-08-29 两轮全面审计（详见 `docs/learn docs/B4DL_论文对齐审计_20260824.md`、`B4DL_全仓库论文对齐审查_20260829.md`）。

## 结果对比（B0 vs 论文）

| 指标 | 复现 B0 | 论文 | 状态 |
|------|---------|------|------|
| accuracy（existence + binary 平均） | 0.7629 | 0.762 | ✅ 持平 |
| mIoU（time_grounding） | 0.2696 | 0.311 | ❌ 差 0.041（≈3.1× 显著性半宽） |
| BLEU-4（语料级） | 0.0973 | 0.095 | ✅ |
| METEOR（Meteor-1.5 jar） | 0.1729 | 0.275 | ❌ 口径一致仍偏低，唯一遗留考证项 |
| ROUGE-L | 0.3244 | — | 口径一致 |
| BERTScore（roberta-large L17） | 0.8973 | 0.897 | ✅ 精确命中 |

分任务 accuracy：existence 0.7072、binary 0.8187。

## 一致项

- **数据集产物 100% 对齐**：官方发布 train 148,271 条 / test 30,145 条，六任务条数与论文 Table 2 的 14 个数字精确吻合；官方 train 与 test 零重叠（700/150 scene 划分）
- **模型架构**：VTimeLLM 范式一致（Vicuna-7B + mm_projector 单线性投影 + `<video>` embedding 注入 + LoRA），与 LiDAR-LLM 的 Q-Former 路线区别清楚
- **metatoken 机制**：`<4DLiDAR>`/`<meta>`/`<video>` 格式与论文 Figure 6 / Appendix C 一致，训练与评测注入逐字符一致（`ego_text.py` 单一来源）
- **评测口径（修正后冻结）**：pycocoevalcap 语料级 BLEU-4、Meteor-1.5 jar、roberta-large 第 17 层 BERTScore、greedy 解码

## 偏差项

| # | 偏差 | 说明 | 状态 |
|---|------|------|------|
| 1 | **编码器权重自训** | 官方从未发布 LiDAR-CLIP 权重（"You need to train the model first"）；本地旧 ckpt 是原版 LiDAR-CLIP 的 ONCE 权重（domain gap），已改用 nuScenes 自训（SST + AttentionPool2d，MSE 对齐冻结 CLIP） | 自训进行中（B2），将触发基线重置 |
| 2 | **训练方法为混合单 LoRA** | 论文 §4.2 描述两阶段（stage2 简单任务 → stage3 复杂任务）；实测两阶段法简单任务格式漂移（exact match 归零，acc 0.0001）。审计后确认论文描述的即混合训练，B0 采用 148K 混合、单 LoRA、3 epochs lr 1e-4 | 已决策，见 [[Reproduction-Log]] |
| 3 | **`--answer_frames` 属 oracle** | 发布的 benchmark 丢失每条 QA 的序列归属字段，TG 问题文本无帧号；复现用 GT 答案帧范围恢复包含序列（训练/评测两侧 2783/2783 一致） | 属"还原论文原始评测设置"，报告中必须声明 |
| 4 | **METEOR 偏低** | 修正口径（Meteor-1.5 jar）后 0.1729 vs 论文 0.275，仍未定位到口径差异来源 | 唯一遗留考证项 |
| 5 | **mIoU 差距** | TG 时间定位是最大差距项。曾出现答案坍缩（预测 (0,8) 占 87%）；seqv3（真实帧 metatoken + feat_indices + answer_frames）后缓解至 0.2696 | 待编码器（B2）与 B1 改进 |
| 6 | **stage1 数据量** | B0 的 projector 用 95K nu-caption 数据训练，论文/官方方案为 162K | B1 修复中（162K 数据已构建） |
| 7 | **GPT-4o Score 未复现** | 论文 Table 9 的 GPT-4o 评分未纳入 B0 对比（缺失记 null 而非 0） | 可选 |

## 复现过程中定位并修复的关键 bug

| 时间 | 问题 | 影响 |
|------|------|------|
| 08-07 | `<4DLiDAR>` token 未注册到训练 | 模型对所有任务输出连续点号 |
| 08-24 | mIoU 半开区间计算 bug、per-scene metatoken 回退错配、单帧伪造 stationary | 评测口径与论文不一致 |
| 08-28 | BERTScore 按 config 取第 24 层（应取第 17 层，虚高 ~0.07）、BLEU 句级（偏高 ~17%）、METEOR 用 NLTK 后端（不可比） | 指标口径全面修正并冻结 |
| 08-29 | datageneration 三缺口：time_grounding 生成缺失、人工标注（HA）注入断链、任务名别名未归一 | 管线修复，用户实测 API 链路跑通 |

## 对比规则（冻结）

- 基线 **B0 四位一体锁定**：6 个数据/权重组件 MD5 + 冻结评测命令（`--per_sequence --frame_motion --sequence_metadata --answer_frames`，greedy、fp16）+ 评测代码版本 + 特征目录——详见 [[Reproduction-Log]]
- **基线重置条件**：换编码器/重提特征、换 projector 数据版本、改 mIoU 指标逻辑或 answer_frames 口径、测试集变化
- 后续实验统一用冻结口径对比，禁止混用 NLTK 后端等旧口径数值
