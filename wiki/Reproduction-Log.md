# 复现记录与基线

本页汇总 `docs/` 与 `docs/learn docs/` 中的复现历程、当前基线锁定与文档索引。**这些文档是排查"为什么这样做"的第一手资料**。

## 复现时间线（2026-07 ~ 2026-08）

| 阶段 | 内容 |
|------|------|
| 07 初 | plan1：打通训练管线（修复 mm_projector 768 维、验证特征链路） |
| 07-12 | VTimeLLM 源码分析；时序能力弱的根因分析（metatoken 缺失致命等 7 项） |
| 08-07~10 | 首次全量训练评测：定位 `<4DLiDAR>` token 未训练 bug；发现数据失衡（Yes 45.5%）导致模板化输出 |
| 08-24 | 论文对齐审计（A1-A9/B1-B6/R1-R4）：mIoU 半开区间 bug、per-scene 回退错配等修复；seqv3 数据格式引入 |
| 08-25 | seqv2 评测 acc 0.7647 首超论文但 TG 坍缩（(0,8) 占 87%）；两阶段法实测失败（acc 0.0001），决策回混合法 |
| 08-27 | seqv3-mixed 全量评测：acc 0.7629 达标、mIoU 0.2696 未达 |
| 08-28 | 权重核对：本地 ckpt 实为 ONCE 权重（domain gap）；评测数值口径考证（BERTScore 层号 bug 等）；启动 nuScenes 编码器自训 |
| 08-29 | 全仓库论文对齐审查；**基线 B0 正式锁定**；编码器退火链与早停监控上线；B1 全流水线启动 |

## 基线 B0（seqv3-mixed）锁定

2026-08-29 正式将 08-27 的 seqv3-mixed 全量评测锁为复现基线 **B0**，四位一体：

1. **6 个数据/权重组件 MD5**：test_qa（43959740…）、ego_metadata（c3f152b6…）、ego_frame_motion（91ecf31a…）、sequence_metadata（73857e61…）、stage1 projector（af218814…）、stage2 adapter（7565ce2d…）
2. **冻结的逐字评测命令**：`--per_sequence --frame_motion --sequence_metadata --answer_frames`，greedy、fp16
3. **评测代码版本**：8b85cd2 起 `evaluation/` 零变化
4. **特征目录**：重提特征 = 破坏可比性

### B0 数值

| 指标 | B0 | 论文 | 状态 |
|------|-----|------|------|
| accuracy（existence+binary） | 0.7629 | 0.762 | ✅ 持平 |
| mIoU（time_grounding） | 0.2696 | 0.311 | ❌ Δ-0.041 |
| BLEU-4 | 0.0973（修正口径） | 0.095 | ✅ |
| BERTScore | 0.8973（修正口径） | 0.897 | ✅ 精确命中 |
| ROUGE-L | 0.3244 | — | 口径一致 |
| METEOR | 0.1729（修正口径） | 0.275 | ❌ 唯一遗留口径考证项 |

分任务：existence 0.7072、binary 0.8187。

**显著性阈值**（对比实验判真改进的门槛）：ΔmIoU > +0.013（95% CI 半宽 ±0.0132）；Δaccuracy > ±0.009；文本指标 ≥ 0.01。

**指标口径冻结**：pycocoevalcap 语料级 BLEU-4 + Meteor-1.5 jar + roberta-large 第 17 层 BERTScore。

### B0/B1/B2 命名

- **B0**：当前基线（旧编码器特征 + 95K projector + 148K 混合单 LoRA）
- **B1**：stage1 补至 ~162K → projector 重训 → 混合重训（`run_b1_pipeline.sh` 已启动）
- **B2**：nuScenes 自训编码器 → 新特征（触发**基线重置**）

**基线重置条件**：换编码器/重提特征、换 projector 数据版本、改 mIoU 指标逻辑或 answer_frames 口径、测试集变化。

## 已冻结的对比规则

- 修正口径 B0 参考值：BERTScore 0.8973 / BLEU-4 0.0973 / METEOR 0.1729 / ROUGE-L 0.3244；后续对比统一后端，禁止混用 NLTK 后端数值
- `--answer_frames` 用 GT 恢复归属属"还原论文原始评测设置"（oracle），报告中必须声明
- GPT-4o Score 缺失记 null 而非 0

## 全仓库论文对齐审查结论（08-29）

- **数据集产物 100% 对齐**：Table 2 的 14 个数字精确吻合
- **模型架构 ~90%** 对齐
- **评测指标实现**：发现并修正 3 处偏差（BERTScore 层号、METEOR 换库、BLEU 句级/语料级）
- **datageneration 管线缺口**（已修复）：TG 生成缺失、HA（人工标注）注入断链、任务名别名归一

## 文档索引

### docs/（顶层）

| 文档 | 内容 |
|------|------|
| `B4DL_复现方案.md` | 论文+官方仓库逐行解析的完整复现方案：资源清单、流水线、命令超参、坑位清单、目标数值 |
| `B4DL_训练评测报告_20260810.md` | 首次训练评测报告（`<4DLiDAR>` bug、数据失衡分析） |
| `mmb4dl.pdf` | 论文原文 |
| `参考/LiDAR LLM.py` | LiDAR-LLM（arXiv:2312.14074）单文件参考源码 |
| `参考/记录.md` | stage1/2 训练过程手记 |

### docs/learn docs/（16 篇开发记录）

| 文档 | 主题 |
|------|------|
| `B4DL_全仓库论文对齐审查_20260829.md` | ★ 四路深查的全面对齐审查与指标口径修正 |
| `B4DL_基线锁定与对比规则_20260829.md` | ★ B0 四位一体锁定、显著性阈值、B1/B2 规划 |
| `B4DL_待修改清单_20260829.md` | 按执行顺序的修复清单与完成进度 |
| `B4DL_评测数值对比与问题清单_20260828.md` | seqv3-mixed 分任务对比表 + 10 项问题清单 |
| `B4DL_LiDARCLIP权重核对_20260828.md` | ONCE 权重 domain gap 发现与编码器自训启动 |
| `B4DL_训练方法对比_论文vs复现_20260828.md` | seqv3-mixed 训练法与论文逐项对比 |
| `B4DL_seqv3混合训练评测分析_20260827.md` | seqv3 混合方案与两阶段法失败记录 |
| `B4DL_两阶段训练评测与混合法决策_20260826.md` | 两阶段法失败（acc 0.0001）与回退决策 |
| `B4DL_seqv2评测对比与坍缩分析_20260825.md` | TG 坍缩（(0,8) 占 87%）分析 |
| `B4DL_论文对齐审计_20260824.md` | 首次全面审计（B1-B6 修复项、greedy 解码） |
| `B4DL_temporal_weakness_analysis.md` | 时序能力弱根因分析（7 项）与 metatoken 方案设计 |
| `B4DL_vs_LiDAR-LLM_comparison.md` | 与 LiDAR-LLM（Q-Former 架构）逐维对比 |
| `VTimeLLM_analysis.md` | 源项目 VTimeLLM 分析与借鉴 |
| `training_reproduction_guide.md` | 训练管线复现指南 |
| `plan1.md` / `plan1_learning_results.md` | 早期阶段化计划与执行结果 |
