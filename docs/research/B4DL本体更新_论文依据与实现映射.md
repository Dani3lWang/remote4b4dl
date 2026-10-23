# B4DL 本体更新：论文依据与实现映射

更新日期：2026-10-23。本文专门维护论文使用、方法改写与可写作边界；代码注释不承担文献说明。对应工程说明：[独立模块、数据与评测准备](../../mllm/docs/core_geometry_preparation.md)。

## 一、两份材料如何用于本次更新

用户提供的 `E:/Download/4DMLLM.md` 是 LiDAR-Hallu 论文与仓库的研究报告，适合用于识别答案先验、输入干预和评测分母问题。报告里的模型分数属于外部研究，不能移作当前 B3 的结果。报告中的命令和建议视为待评估材料，不视为用户的额外执行指令。

`E:/Download/MaskPLS 仅译文.pdf` 对应 Marcuzzi 等人的 MaskPLS。其方法在点/体素特征上学习掩码，并结合空间位置与语义信息。对本轮最有用的是“必须明确输入的空间信息是什么、是否在实际计算中被使用”这一设计启发。帧级 SST 不再包含可直接访问的逐点位置，因此不能把 MaskPLS 的位置嵌入或掩码解码器照搬到这里。

本轮落地为自车相对位姿与时间条件化的帧注意力；这是本项目改写，**不实现 MaskPLS 分割、不宣称复现其局部位置编码或达到其 PQ**。原分割路线的结果和方案独立保留。

## 二、论文—实现—证据对照

| 来源 | 本次使用的机制 | 对应实现 | 与原工作的差别 | 当前验证与论文表述边界 |
|---|---|---|---|---|
| Vaswani et al., *Attention Is All You Need*, 2017 | 多头自注意力、前馈层和残差组成的序列建模 | `mllm/b4dl_core/temporal.py` 的 `TemporalBlock` | 输入为整帧 SST，2 层/768 维/8 头，采用 pre-LN 和零初始化输出残差；这些配置为项目选择 | 已检查有效帧等价、补齐一致性、梯度传递；可写“采用 Transformer 帧时序适配器”，不能写“已提升时空理解” |
| Shaw, Uszkoreit & Vaswani, *Self-Attention with Relative Position Representations*, 2018 | 在注意力中表达相对关系，而非只依赖绝对序号 | `relative_geometry`、`TemporalBlock.bias` | 原文的相对位置表示与本项目的连续时间/自车位姿 MLP 加性偏置不是同一公式；raw/sin/cos、尺度均为改写 | 已检查全局刚体变换不变性及位姿变化对激活后输出的影响；可写“受相对位置建模启发”，不能写“复现 Shaw 方法” |
| Caesar et al., *nuScenes: A Multimodal Dataset for Autonomous Driving*, CVPR 2020 | 同步传感器、姿态、标注身份及官方场景划分 | `data.py`、`geometry.py`、`audit.py` | 使用标注框中心形成 QA 真值；关键帧同一实例跨端点，度量是各端点自车相对关系 | 404 帧/26,168 个数值原始表复算通过；可写“由 nuScenes 标注构造可核验监督”，不能写“LiDAR 观测无误差” |
| Marcuzzi et al., *Mask-Based Panoptic LiDAR Segmentation for Autonomous Driving*, RA-L 2023 | 几何位置与语义共同参与表示的设计启发 | 只映射到帧位姿接口的设计动机；没有引入 MaskPLS 分割代码 | 不使用 mask queries、mask loss、稀疏卷积或点级位置编码；自车帧位姿不等于物体/点级几何 | 本轮无 MaskPLS 算法复现；宜放相关工作或设计讨论，不宜写成本方法核心公式的直接来源 |
| LiDAR-Hallu / `4DMLLM_Hallucination_Bench` | 严格解析、几何关系题、固定先验、相反答案配对、输入干预、场景 bootstrap、修复与退化 | `data.py`、`rewards.py`、`evaluation.py`、`inputs.py` | 独立实现；增加显式关键帧/位姿校验、证据 JSON、train-only 先验、四次选项轮换；不生成有跨度泄漏风险的 temporal QA；不实现 TSCD | 25 项测试及真实 mini 准备通过；mini 开发/测试无严格相反答案对，尚不能报告关系辨别性能 |
| Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, 2024 | 后续 GRPO 的算法背景 | `rewards.py` 只实现离线 reward；准备配置记录未来对照 | 0.7/0.3、1m 容差、证据自洽检查及冻结时序块是项目设计，不来自该论文的已验证结论 | 未实现/运行 GRPO；可写“拟以 GRPO 优化并对比奖励设计”，不能写“GRPO 实验已完成” |

本轮没有用论文训练出来的权重，也没有将外部 benchmark 归档预测混入当前数据。参考文献支撑方法来源；实验收益需要当前模型的独立对照。

## 三、上游代码学习与出处

固定参考仓库：[Awesome4D/4DMLLM_Hallucination_Bench](https://github.com/Awesome4D/4DMLLM_Hallucination_Bench/tree/2ba623dabe2c59ea164e761320d537cdc09a6583)，commit `2ba623dabe2c59ea164e761320d537cdc09a6583`。

| 上游位置 | 采用/改写内容 |
|---|---|
| [legacy/mllm/prepare_b4dl_hallucination_v2.py](https://github.com/Awesome4D/4DMLLM_Hallucination_Bench/blob/2ba623dabe2c59ea164e761320d537cdc09a6583/legacy/mllm/prepare_b4dl_hallucination_v2.py) | 核对类别、八扇区、最近起点实例和运动阈值；本项目补全 keyframe 限制与 B4DL 帧行对齐，真值另存，不搬运参考回答 |
| [scripts/relational_audit/analyze_prompt_matched_pairs.py](https://github.com/Awesome4D/4DMLLM_Hallucination_Bench/blob/2ba623dabe2c59ea164e761320d537cdc09a6583/scripts/relational_audit/analyze_prompt_matched_pairs.py) | 同完整 prompt/长度、异场景、相反标签的最大匹配约束；本项目使用迭代增广路径，配对与预测解耦 |
| `lidar_hallu/metrics.py`、`lidar_hallu/priors.py`、README 的 scientific scope | 保留解析失败分母、场景级统计与先验诊断的原则；本项目 prior 只拟合 train，未复现上游使用其他评测场景标签的诊断设定 |
| [NOTICE.md](https://github.com/Awesome4D/4DMLLM_Hallucination_Bench/blob/2ba623dabe2c59ea164e761320d537cdc09a6583/NOTICE.md) | 上游未给继承代码/数据/权重统一新增许可；本轮没有整库复制、重新授权或公开发布，若后续发布数据或衍生材料须核对相应权利 |

特别区分：本项目的 shuffled-only feature control 是输入消融；上游 TSCD 是解码干预，两者不是同一个实验。把 SST 置零而保留位姿也不是完全 sensor-free。

## 四、可用于论文初稿的表述

**方法段落草稿（只描述实现）**：

> 我们在场景级 LiDAR 特征与原有语言投影层之间引入一个可关闭的帧时序适配器。该模块通过相对时间、自车平移和相对朝向构造注意力偏置，并以零初始化残差保持初始帧表示不变。监督数据由带身份标注的关键帧构成，分别覆盖存在性、相对方位、距离排序和同一实例的自车相对运动。物体中心及实例身份仅用于构造和核验监督，不作为模型输入。

**实验设置草稿（仍需补充真实实验信息）**：

> 我们以场景划分训练、开发和测试数据，并在完整请求集合上评测答案与数值证据。解析失败和缺失输出保留在分母中。诊断包含固定答案先验、选项轮换、特征干预以及同提示相反标签配对，并按场景重采样估计指标差值区间。

第二段只有在完整数据覆盖及真实模型评测落实后才能用作正式实验设置。当前 mini 数据不能支撑“严格配对测试已完成”或统计显著性结论。若论文列结果，需写真实模型身份、参数/训练预算、匹配 SFT 数据、全部随机种子、有效配对数、供体覆盖以及六任务回归结果。

**不宜写入当前成果**：优于 B3、缓解幻觉、改善左右运动、小物体性能提升、端到端复现 MaskPLS/TSCD/GRPO、达到论文分数。现有证据只支持代码正确性与数据准备完成。

## 五、引用信息与核验状态

1. Vaswani et al. **Attention Is All You Need**. NeurIPS, 2017. [原文](https://arxiv.org/abs/1706.03762)。用于 Transformer 序列模块背景。
2. Peter Shaw, Jakob Uszkoreit, Ashish Vaswani. **Self-Attention with Relative Position Representations**. NAACL-HLT, 2018. [ACL 官方条目](https://aclanthology.org/N18-2074/)。用于相对关系建模背景。
3. Holger Caesar et al. **nuScenes: A Multimodal Dataset for Autonomous Driving**. CVPR, 2020. [原文](https://arxiv.org/abs/1903.11027)、[官方 devkit](https://github.com/nutonomy/nuscenes-devkit)。数据几何与官方 split 为实际使用依据。
4. Rodrigo Marcuzzi, Lucas Nunes, Louis Wiesmann, Jens Behley, Cyrill Stachniss. **Mask-Based Panoptic LiDAR Segmentation for Autonomous Driving**. *IEEE Robotics and Automation Letters*, 8(2):1141–1148, 2023. DOI [10.1109/LRA.2023.3236568](https://doi.org/10.1109/LRA.2023.3236568)。作者、卷期、页码及 DOI 已由 [官方仓库 Citation](https://github.com/PRBonn/MaskPLS#citation) 核对；对应 [官方 PDF](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/marcuzzi2023ral.pdf)。不要把此次核对得到的 RA-L 条目误写为 ICRA 论文条目。
5. Zhihong Shao et al. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**. 2024. [原文](https://arxiv.org/abs/2402.03300)。只用于未来 GRPO 背景。
6. **LiDAR-Hallu / 4DMLLM_Hallucination_Bench**，上述固定 commit。用户研究报告将其论文题名记为 *Do LiDAR Language Models Really Understand Spatio-temporal Relationships?*，arXiv 记为 `2609.24452`。本轮已核对仓库，但未独立核验这条 arXiv 编号/完整作者/正式发表状态；正式文献库暂按软件来源记录，定稿前核实论文条目，不把报告中的检索标记当成可用引用。

本轮基础论文官方网页访问遇到网络超时，1/2/3/5 使用稳定的题名、作者与标识，不补填未经核对的卷期、页码或 DOI。MaskPLS 引用由成功读取的官方 README 和用户 PDF 交叉核对。后续如改用其他位置编码或优化算法，应同步更新此映射，不把仅作为相关工作的论文列成直接复现。

## 六、落实记录

- 2026-10-23 用户选定原始 B3：固定此前 mini FP16 使用的历史三轮 `_protected_b3_3ep_final_20260910`；语言底座、projector、B3 的当前远端指纹与历史采集记录全部一致。后续 A0–A3 使用同一初始模型身份；这个选择不构成性能比较结果，也不改变上文论文引用关系。
- 已完成：帧适配器、真实 mini 数据、数值审计、严格评测器及 CPU 检查。详见工程说明与 `docs/项目方法与落实情况.md`。
- 待验证：基底真实接入、完整 trainval 覆盖、A0–A3 公平消融、SFT/RL、原六任务回归、实际传感信息利用程度。
- 每次功能改动同步维护：对应机制、代码位置、验证证据、论文表述边界。不要将工程 smoke 自动升级成实验结论。
