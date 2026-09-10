# 与论文的对照（差异清单）

复现 ≠ 复刻。本页列出复现实现与论文（arXiv:2508.05269）的全部已知一致项与偏差项，依据 2026-08-24 / 08-29 两轮全面审计（`docs/learn docs/B4DL_论文对齐审计_20260824.md`、`B4DL_全仓库论文对齐审查_20260829.md`），并持续更新至 2026-09-08（B3 超论文、METEOR 溯源闭环、B4a 负结果）。

## 结果对比（当前基线 B3 vs 论文，B0/B4a 参照）

主对比线自 2026-09-06 起切到 **B3**（整场景 + meta2 relative-to-previous 语义）；B0 保留为历史参照；B4a 是 B3 之上的训练分布干预实验（2026-09-08 完结，负结果，见 [[Reproduction-Log]]）。

| 指标 | 论文 | B3（当前最优） | B4a（TG 过采样） | B0（历史基线） |
|------|------|------|------|------|
| accuracy（existence+binary） | 0.762 | 0.7526 | **0.7775** ✅ | 0.7629 |
| mIoU（time_grounding） | 0.311 | **0.3467** ✅ +0.036 | 0.3271（Δ-0.0196 显著回退） | 0.2696 |
| BLEU-4（语料级） | 0.095 | 0.0965 ✅ | 0.0976 | 0.0973 |
| METEOR（NLTK-2005 主口径） | 0.275 | **0.3366** ✅ +0.062 | 0.3378 ✅ | 0.3344 |
| ROUGE-L | 0.322 | 0.3234 ✅ | 0.3269 | 0.3244 |
| BERTScore（roberta-large L17） | 0.897 | 0.8967 ✅ | 0.8977 | 0.8973 |

- B3 分任务：existence 0.6761 / binary 0.8291 / TG mIoU 0.3467；METEOR 为 09-07 离线补算的 dual 主口径（jar 参考 0.1747，仅用于衔接旧表）
- **官方对照**（2026-09-02，决定性）：官方发布 checkpoint + 官方特征 → acc 0.7542 / mIoU 0.1737，与论文 Table 4"无 Metatoken"行（0.763/0.161）吻合 → **论文 Table 3 的完整模型从未发布**
- B4a 的 acc 提升主要来自 existence（0.6761→0.7244）；METEOR 口径论证见 `docs/learn docs/B4DL_METEOR双口径溯源与评测规则_20260907.md`

## 一致项

- **数据集产物 100% 对齐**：官方发布 train 148,271 条 / test 30,145 条，六任务条数与论文 Table 2 的 14 个数字精确吻合；官方 train 与 test 零重叠（700/150 scene 划分）
- **模型架构**：VTimeLLM 范式一致（Vicuna-7B + mm_projector 单线性投影 + `<video>` embedding 注入 + LoRA），与 LiDAR-LLM 的 Q-Former 路线区别清楚
- **metatoken 机制**：`<4DLiDAR>`/`<meta>`/`<video>` 格式与论文 Figure 6 / Appendix C 一致；2026-09-02 起渲染语义对齐论文 §4.1 relative-to-previous（meta2），训练与评测注入逐字符一致（`ego_text.py` 单一来源）
- **评测口径（冻结，METEOR 2026-09-07 双后端化）**：pycocoevalcap 语料级 BLEU-4、METEOR 主口径 NLTK-2005（jar 参考）、roberta-large 第 17 层 BERTScore、greedy 解码

## 偏差项

| # | 偏差 | 说明 | 状态 |
|---|------|------|------|
| 1 | **编码器权重自训** | 官方从未发布 LiDAR-CLIP 权重（"You need to train the model first"）；本地旧 ckpt 是原版 LiDAR-CLIP 的 ONCE 权重（domain gap） | ✅ 已定稿（2026-09-06）：nuScenes 自训 + 退火 3ep，val MSE 0.0992；B1 起特征全量重提，见 [[LiDAR-CLIP-Encoder]] |
| 2 | **训练方法为混合单 LoRA** | 论文 §4.2 描述两阶段（stage2 简单任务 → stage3 复杂任务）；实测两阶段法简单任务格式漂移（exact match 归零，acc 0.0001）。审计后确认论文描述的即混合训练 | 已决策；B0-B4a 保持混合单 LoRA（B2+ 输入改整场景），见 [[Training]] |
| 3 | **`--answer_frames` 属 oracle** | 发布的 benchmark 丢失每条 QA 的序列归属字段，TG 问题文本无帧号；复现用 GT 答案帧范围恢复包含序列（训练/评测两侧一致） | 属"还原论文原始评测设置"，报告中必须声明 |
| 4 | **METEOR 口径** | 旧冻结口径 pycocoevalcap Meteor-1.5 jar 是 2014 年 WMT 重调参变体，**不是**论文引用 [2]（Banerjee & Lavie 2005）；NLTK-2005 才是其忠实实现 | ✅ 2026-09-07 溯源闭环：NLTK 口径 B0-B3/B4a = 0.3344~0.3378 全超论文 0.275；论文 0.275 的精确实现不可考（变体扫描区间 0.14~0.38） |
| 5 | **mIoU 差距** | TG 时间定位曾是最大差距项（B0 0.2696，含 (0,8) 坍缩史） | ✅ 已闭环并反超：B1 特征/projector → B2 整场景反证 → meta2 语义修复 → **B3 0.3467**。残余病灶转 #8 |
| 6 | **stage1 数据量** | B0 的 projector 用 95K nu-caption 数据训练，论文/官方方案为 162K | ✅ B1 已完成（162K projector 已用并重训） |
| 7 | **GPT-4o Score 未复现** | 论文 Table 9 的 GPT-4o 评分未纳入对比（缺失记 null 而非 0） | 可选：predictions 已落盘，配 API key 可离线补算，无需重跑推理 |
| 8 | **TG 输入无显式帧号信号（当前主差距）** | 问题文本无帧号、meta 只有首末帧状态、视觉特征只是 (40,768) 顺序堆叠——"数帧"只能靠位置记忆：B2 坍缩 (000,008)（61%）、B3 系统性 −3.9 帧偏早且 GT start≥25 的 413 条测试样本只预测到 13 条（≈3%）、B4a 高帧段过采样反而更差（Δ-0.0489）三连证据 | 工作假设成立，方案队列待实验：帧身份锚点（A.1 文本锚点 / A.2 帧嵌入，推荐）→ TG 重加权 / DoRA / NEFTune / RFT-DPO，见 `docs/learn docs/B4DL_训练优化角度分析与策略_20260908.md` 与 RL 可行性分析 |

## 复现过程中定位并修复的关键 bug

| 时间 | 问题 | 影响 |
|------|------|------|
| 08-07 | `<4DLiDAR>` token 未注册到训练 | 模型对所有任务输出连续点号 |
| 08-24 | mIoU 半开区间计算 bug、per-scene metatoken 回退错配、单帧伪造 stationary | 评测口径与论文不一致 |
| 08-28 | BERTScore 按 config 取第 24 层（应取第 17 层，虚高 ~0.07）、BLEU 句级（偏高 ~17%）、METEOR 用 NLTK 后端（不可比） | 指标口径全面修正并冻结 |
| 08-29 | datageneration 三缺口：time_grounding 生成缺失、人工标注（HA）注入断链、任务名别名未归一 | 管线修复，用户实测 API 链路跑通 |

| 09-02 | ego_text.py metatoken 渲染语义与论文不符：首帧恒 `at the starting position`、运动用前向（next）差分；论文 §4.1/Figure 6 为每帧相对**前一帧**（relative-to-previous） | mIoU 最大单项杠杆：B2→B3 +0.147；`re_render_meta.py` 重渲染出 meta2 数据 |
| 09-07 | METEOR 口径错配：冻结的 Meteor-1.5 jar（2014 WMT 重调参）不是论文引用 [2]（2005）的实现 | 评测双后端化：NLTK-2005 主口径下 B0-B3 全部超论文 0.275，"唯一遗留考证项"闭环 |

## 对比规则（冻结）

- **当前对照基线 = B3**（2026-09-06 起）：整场景配方；B0 四位一体锁定（6 组件 MD5 + 冻结命令 + 评测代码版本 + 特征目录）为历史参照，见 [[Reproduction-Log]]
- **冻结评测命令**（B2+ 代际）：`--whole_scene --per_sequence --answer_frames`，greedy、fp16；B0/B1 代际为 `--per_sequence --frame_motion --sequence_metadata --answer_frames`
- **显著性阈值**：ΔmIoU > +0.013、Δaccuracy > ±0.009、文本指标 ≥ 0.01 才判真改进
- **基线重置条件**：换编码器/重提特征、换 projector 数据版本、改 mIoU 指标逻辑或 answer_frames 口径、测试集变化
- 后续实验统一用冻结口径对比；METEOR 与论文对比一律用 `meteor`（NLTK-2005），`meteor_pycocoevalcap`（jar）仅用于衔接旧文档数值
