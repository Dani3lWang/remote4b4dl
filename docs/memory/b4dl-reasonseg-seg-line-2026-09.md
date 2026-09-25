---
name: b4dl-reasonseg-seg-line-2026-09
description: ReasonSeg（seg 分支）单帧点级推理分割的 09-18~09-25 全过程——七条杠杆逐条判死、掩码塌缩机制、两个度量 bug、oracle 探针天花板，以及"判据挑错轴"这个复发三次的自身错误
metadata:
  node_type: memory
  type: project
  originSessionId: f554c11b-93d7-4687-a2ad-c0a5d4dec88a
---

ReasonSeg 在 `seg` 分支、autodl3（4090 24G）上跑，conda 环境是独立的 `reasonseg`（`/root/autodl-tmp/.conda-stuff/envs/reasonseg`，torch 2.8.0+cu128 / spconv 2.3.6），**不复用 `wqlc`**。原始对话见 `docs/memory/qoder-sessions/`（4 份，含会话 ID）；实时状态见 `docs/learn docs/B4DL_ReasonSeg训练现状_20260925.md`。运行口径（`--validation-samples 0`、`--dtype fp16 --encoder-dtype fp32`、`cIoU`=面积加权 / `gIoU`=逐对均值键名反直觉、模型选择只能用 internal 划分）已固化在仓库根 `CLAUDE.md`，此处不重复。

## 一、指标演进主线

| 阶段 | run | 关键数 | 判读 |
|---|---|---|---|
| 环境（09-18） | `reasonseg-mini-4090` | epoch 1 崩于 ENOSPC（单档 16G 无轮转） | `--no-resume-state` + `--keep-last` 修（`2d4b8ec`），单档 1.2G |
| 口径纠偏（09-19） | mini（10 场景） | best 0.0075，训练 loss 全 0 | 完全记忆化、零泛化；`--validation-samples` 默认 32 造成的"val mIoU≈0"是单场景探针假象 |
| 数据规模（09-19~20） | thin（200 场景 / 7910 条 / 495 step） | 20ep best mean IoU **0.01652 @ ep14** | 较 mini 抬 9×，但排除"数据规模是唯一瓶颈"；lr 5e-4 与 head dropout 0 两个旋钮均否决 |
| 编码器（09-20~21） | `reasonseg-spatial-tv` | lidarseg mIoU **0.2876 @ ep3**（旧线上编码器 0.1244 且只训过 1 epoch） | fp32 训 12 epoch；bf16 会触发 spconv NaN |
| 编码器（09-21） | `reasonseg-tvenc` | TF 全量 val best mean IoU **0.1037 @ ep13** = 基线 **6.3×** | 判据 PASS，台阶式提升成立。自由生成 cIoU 0.2922 / recall@0.5 0.0534 |
| 编码器 A/B（09-21） | 同 300 条自由生成 | cIoU 2.68× / gIoU 1.82× / recall@0.5 **18.5×**（0.00289→0.0534） | 两侧 token_failure 均 0 ⇒ 同质可比；换编码器收益在自由生成口径同样成立 |
| 复核（09-22 00:14） | `--validate-only` 独立进程 | **0.10370878** vs 训练记录 0.10366375，objects 2322 逐位一致，2248 batch 零 non-finite | 0.1037 是干净可引用的结果，训练期 TF 未踩评测侧 fp16 NaN |
| 场景多样性（09-22） | `reasonseg-tvenc680`（700 场景 × 11 帧 / 7666 条） | 20ep best **0.11280 @ ep9** | 落在预注册"不显著"档（<0.114）⇒ **场景多样性不是有效杠杆**。⚠ 该轮用 `val_thin` 早停（测试集泄漏进选模型）+ 修复前的模型相关分母（objects 2320），故此数不再可引用，也是后来改用 internal 协议的原因 |
| 无泄漏协议 | `reasonseg-internal679`（对照） | internal_es 20ep best **0.1301 @ ep15**；val_thin TF **0.10888** | 此后所有对比的对照组 |
| 损失（09-24~25） | `reasonseg-lossA2-plain-tversky-ext20` | internal_es **0.14437 @ ep15**（+0.0143，看着显著）；val_thin mean IoU 0.11555（+0.0067，**不显著**）、recall@0.5 0.08029（**更差**）、global IoU 0.32795（**更差**） | **A2-ext20 不是新基线** |

## 二、七条杠杆全部判死

lr/dropout、场景数、语言侧、二值化阈值、数据协议、编码器、**损失函数**（09-25 由 A2-ext20 判死）。每条都有机制解释，不是简单的"没涨"：

- **掩码塌缩的成因**（Phase 2.0 接地诊断）：掩码正例只占帧内点的 **3.2e-4**，逐点平均的 BCE 把正例梯度稀释约 3000 倍，"全不触发"与"在所有同类候选上对冲"的损失面几乎持平（下坡只占沉默损失的 **4.9%**）→ 模型学会对小物体沉默，预测掩码中位数 **3 点** vs GT **11 点**。
- **balanced BCE 判死**：确实让模型敢触发（prob_pos 0.165→0.82），但兑现方式是无差别触发，尺寸冲到 GT 的 136–155 倍。
- **Tversky(α=0.3, β=0.7) 治好症状、治不了病**：下坡 4.9%→8.5%，尺寸校准到 **11/11 = 1.00×**（对照 3/11 = 0.27×），空掩码 140→99。但自由生成 headline `instance recall@0.5` 与对照**逐位相同（都是 24/330 = 0.072727）**；失败分解显示那 41 个空掩码只转化成 29 个 `near_miss_below_0.5` + 15 个 `right_class_wrong_place`，后者升为**第一大失败模式**（87→102），`hit` 一个没多。
- 结论：**模型知道类别、知道大致在哪一片，就是不知道是哪一个实例** —— 这是实例接地失败，不是分割失败，任何损失重加权都治不了它。
- 代价：A2 从 4ep 到 20ep，AUC 0.9167→**0.8371**，即后 16 个 epoch 在拿排序质量换尺寸校准。

## 三、瓶颈定位：oracle-query 探针（Phase 2.3 B0）

`scripts/reasonseg_experiments/probe_oracle_query.py`：**完全不加载 7B LM**，用 GT 中心（按 `point_cloud_range` 归一化）+ one-hot 类别构造 query，只训 `OracleQueryNet` + `QueryMaskDecoder`，编码器冻结，其余与真实头同构（逐物体 memory 展开）。

首跑（4 epoch / lr 3e-4 / 600 条评测，`eval_results/_oracle_probe/report.json`）：

| | dev (internal_es) | test (val_thin) |
|---|---|---|
| AUC | 0.9735 | **0.9776** |
| recall@0.5 | 0.0471 | 0.0449 |
| iou_mean | 0.1071 | 0.0917 |
| 尺寸 pred/target | 57 / 15 | **37 / 11**（3.4× 过触发） |
| objects（另跳过不可达） | 573（+29） | 623（+30） |

三个读法：① **点特征够用，编码器不是瓶颈**（oracle AUC 0.978 vs 完整模型 0.858）；② **LM query 通路吃掉约 0.12 AUC**；③ **即使给了完美定位，解码器+损失也校不准尺寸**。

**先验推导的量化读法**（非事后拟合）：排在正点之上的负点数 = (1−AUC)×N，N≈32530 个框内点。完整模型 0.858 → ~4619；oracle 0.978 → ~716；而 top-K（K≈11）要成立需 ≤11，即 **AUC ≳ 1−K/N ≈ 0.99966**（这个门槛 Phase 2.0 就从 K/N 算出，与观测无关）。⇒ oracle query 带来 6.5× 改善但**仍差 65 倍**。方向含义：**query 侧再怎么改都有硬天花板**（更多 scene query / 实例对比监督 / CoT 查询都填不上这 65 倍），真正的约束在**点特征的分辨率与感受野**（`voxel_size` 0.1 m、单尺度、U-Net 感受野）——一个 11 点的物体在 32530 点里靠现有特征无法被锐利分出。

**首跑的两个缺陷**（都已修，`057cf1e`）：末轮失稳（train_loss 0.9233→0.8146→0.7784→**1.0274**；嫌疑是每轮固定遍历顺序 + lr 3e-4），而评的正是这个最差末轮且无逐轮存档，ep2 指标已取不回；修正为每轮洗牌、lr 1e-4、逐轮 dev 评测并按 dev `iou_mean` 选轮次（recall@0.5 在几百个物体上太粗）、test 只评一次、补 `recall_top_k`（免幅值轴）。**失稳只会压低指标 ⇒ AUC 0.978 是下界 ⇒ "LM query 通路吃掉 ≥0.12 AUC"这个结论更稳。**

### 修正版探针（v2，09-25 20:26:31 `rc=0`）：裁定 = **点特征是硬约束**

6 epoch / lr 1e-4 / 7436 训练记录 / dev 每轮 400 条 / test 只在选定轮评一次。train_loss 单调下降 0.8067→0.6105，**首跑的末轮失稳没有复现**。按 dev `iou_mean` **选定 ep4**：

| | dev (internal_es, 381 obj +22 不可达) | **test (val_thin, 416 obj +23 不可达)** |
|---|---|---|
| AUC | 0.99272 | **0.99110** |
| recall@0.5 | 0.16010 | **0.09615** |
| recall_top_k（免幅值） | 0.20735 | **0.11058** |
| iou_mean / iou_top_k_mean | 0.20359 / 0.23680 | 0.15856 / 0.18386 |
| 尺寸 pred/target | 33 / 12 | 29 / 12 |

按先验门槛（top-K 可用需 AUC ≥ 1−K/N，K=GT 尺寸中位 **12**、N≈**32530** 框内点 ⇒ **≈0.99963**）换算成"压在正点之上的负点数"：

| | AUC | 正点上方的负点数 | 需要 | 差距 |
|---|---|---|---|---|
| 完整模型（诊断口径，有偏） | 0.8576 | ~4630 | ≤12 | ~386× |
| oracle 首跑 | 0.9776 | ~729 | ≤12 | ~61× |
| **oracle v2（ep4 test）** | **0.99110** | **~289** | ≤12 | **~24×** |
| oracle v2（ep5 dev，AUC 最高轮） | 0.99395 | ~197 | ≤12 | ~16× |

⇒ **给了完美定位，干扰负点从 ~4630 降到 ~289（≥16×），仍差约 24×。裁定：点特征的分辨率/感受野是硬约束**（`voxel_size` 0.1 m、单尺度、U-Net 感受野）—— 12 点的物体在 32530 点里靠现有特征无法被锐利分出。**query 侧任何改进都有天花板**（scene query 容量 / 实例对比监督 / CoT 查询都填不上 24×），下一步应投 `voxel_size` / 多尺度 / 感受野。"继续训就能到"可排除：(1−AUC) 逐轮 0.0156→0.0095→0.0076→0.0075→0.0073→0.0061，已在 0.006–0.007 平台，达标需 0.00037。

交叉印证：探针 test `recall@0.5` 仅 0.09615、`recall_top_k` 0.11058 —— query 完美 + 免幅值解码，命中率也只从完整模型的 ~0.086 抬到 ~0.11，与"289 个负点压在正点上 ⇒ top-12 被污染"自洽。**修正版把两个轴（AUC 与 top-K）统一到同一档，首跑那个"AUC 高但 recall 低"的矛盾消失。**

三处必须一起引用的局限：① **选轮的量（dev `iou_mean`）与裁定的量（AUC/top-K）不是同一轴** —— ep5 的 AUC 更高（0.99395）、尺寸校准最好（11/12 = 0.92×），但 `iou_mean` 更低故未选中，**ep5 的 test 数不存在**；按 AUC 轴选轮只会让差距从 24× 收到 16×，裁定方向不变，但不在看过数据后改选轮规则。② **dev 参与选轮且是 in-distribution**（官方 train 场景内划分），test 是 unseen 官方 val，两者差一大截（top_k 0.2073 vs 0.1106），**对外只引用 test**。③ **与接地诊断不可直接相减**：诊断脚本仍带前缀掩码 bug（故意没改，待办 #20），它**抬高**完整模型的 0.8576 ⇒ "~4630 / ≥16×"是低估，裁定更稳。

### 体素并列天花板：改 `voxel_size` 的前提被实测推翻（09-25 20:55，纯 CPU 23 秒）

`SparseUNetPointEncoder` 体素化 → 稀疏 U-Net → 用 `inverse` 把体素特征散射回**原始点**（`spatial_encoder.py:128-134`），故同一体素内的点特征逐位相同、logit 必然并列（AUC 里算 0.5）。设 T = 与 GT 点共享体素的非 GT 框内点数，则并列单独造成的上限 `AUC_max = 1 − 0.5·T/(K·N)`，与判据 `AUC ≥ 1−K/N` 联立得 **T ≤ 2K² ≈ 288**。

实测（`scripts/reasonseg_experiments/probe_voxel_tie_ceiling.py`，val_thin 前 150 条 / 155 物体，K 中位 11、N 中位 32503）：voxel 0.2/0.1/0.05/0.025 m 下 **T 中位全为 0**、均值 1.09/0.30/0.05/0.00，AUC 上限 ≥0.99999984。⇒ 体素栅格极稀疏（32503 点 vs 8400 万候选体素），**并列不解释任何缺口，那 ~289 个负点全是真实的特征/排序失败**，改 `voxel_size` 换不回 AUC。

两个由同一份代码事实推出、别再走回头路的点：① `voxel_size` **不改变 N 与 K**（打分对象是原始 LiDAR 点），故判据门槛 1−K/N 对 voxel_size 是**不变量**；② U-Net 感受野以**体素**为单位固定 ⇒ 减小 voxel_size 会同比**缩小以米计的感受野**，方向上有害；要动这个旋钮只有"增大"才有 RF 理由（代价是掩码边界变粗，而并列成本实测可忽略）。

⇒ **§三 的裁定要收紧**：探针的编码器是**冻结**的，量到的是"现有语义预训练特征 + 完美 query"的天花板，**不是"点特征本身"的天花板**。编码器只在 lidarseg（语义）上训过，特征编码"这是车的点"而非"这是哪一辆车"，实例判别性从未被任何监督要求过。**下一个最便宜且真正决定性的实验是在 oracle 探针里解冻编码器。**

### B1 臂（切除 LOC 先验 + Tversky，4 epoch）：未决中间档

09-25 18:21:57 主链五步全 `rc=0` 收尾。B1 四个数（`checkpoints/reasonseg-lossB1-noprior-tversky`）：

| 口径 | B1（4ep） | 对照 | A2 |
|---|---|---|---|
| 接地诊断 AUC（val_thin 1000 条，n=1068） | 0.9048 | 0.8576 | 0.9167（4ep）/ 0.8371（20ep） |
| 诊断 recall@0.5 | 0.0796 | **0.0861** | 0.0777（4ep）/ 0.0805（20ep） |
| 尺寸 pred/target | 7 / 11（**0.64×**） | 3/11（0.27×） | 14/11（1.27×）4ep、11/11（1.00×）20ep |
| LOC 包含率 | **0.372**（四者最高） | 0.252 | 0.352 / 0.306 |
| `internal_es` ep3 mean IoU（早停 manifest） | 0.12175 | 0.1301（20ep best） | 0.1255（4ep）、0.14437（20ep best） |
| 过滤后 val_thin 2108 条 mean IoU（held-out） | **0.10542** | 0.11047 | 0.11722 |

预注册判据（PASS = AUC≥0.93 **且** 尺寸比∈[0.7,1.5] **且** ep3 mIoU>0.1255 **且** 诊断 recall@0.5>0.0861；FAIL = AUC<0.90 **或** 尺寸比∉[0.5,2.0]）：**PASS 四条全不中、FAIL 两条都未触发 ⇒ 未决中间档**。可以确定的是：先验切除**没有与 Tversky 复合**成期望效果（两者单独各有一项达标，合起来 AUC 低于 A2@4ep、尺寸反向欠触发到 0.64×），且 **headline `recall@0.5` 第四次没动**（对照 0.0861、A2@4ep 0.0777、A2@20ep 0.0805、B1 0.0796 全部低于对照）。⚠ B1 只训 4 epoch、对照/A2 是 20 epoch，held-out 那张表不能读成"B1 比 A2 差"。

**乐观偏差又兑现一次**：同一个 B1 档，`internal_es` 0.12175 vs held-out 0.10542（差 0.0163），与 A2-ext20 的模式（0.14437 vs 0.11555）一致。

## 四、两个度量 bug（都已修，且都被实测验证）

1. **TF 分母模型相关**（`train_reasonseg.py:437`，修于 `a684511`）：原为 `present = object_valid & union.gt(0)`，`union = ((predicted | target) & valid)`。全越界物体 `target & valid` 为空，但模型只要在框内任何地方触发过 → `union>0` → 被计入分母且 IoU 必为 0。**触发越多的模型被塞进越多"不可能物体"**（对照 28 个、A2 32 个，A2 因敢触发多背 4 个必零项 —— 这正是当初"2325 vs 2329"这个 4 物体差异的真相）。改为 `scorable = object_valid & (target & valid).any(-1)`（纯数据口径，同时接替防除零），`intersection`/`union` 的全局求和也一并限制在 scorable 上，否则 mean IoU 修好而 global IoU 仍随触发量漂移。实测验证：两侧 objects 都精确变成 **2297 = 2440−143**，mean IoU 0.11020 / 0.11714，与事前手算 0.11021 / 0.11718 吻合到 1e-5。
2. **诊断脚本把"框内点数"当"前缀长度"**（`point_count = point_valid.sum()` 后切 `[:point_count]`）：实测越界点占帧内 **6.18%**（中位 ~2035 点），但**尾部集中度只有 0.0585** ⇒ 越界点是散布的、不堆在文件尾，该近似不成立。后果：前缀内混入约 2035 个越界点（logit 被 mask 成 −1e4、概率恒 0），同时丢掉尾部约 2035 个框内点，**8/330 = 2.42% 的物体 GT 被截成空**；而 `diagnose_reasonseg_grounding.py` 里有 `if target_size == 0: continue`，这些物体被**静默跳过**（1000 条本应 1091 个物体、实报 1068，差的 23 个就是它们），也因此报告里看不到 NaN。
   - **影响判定**：失真在所有 run 上完全相同（同 manifest / 同 1000 条 / 同点集）⇒ **对照 / A1 / A2 / A3 / prior-off / B1 之间的比较全部成立**；但**绝对值有偏** —— AUC 被约 2035 个"必为 0 的简单负例"抬高，真实值比 0.858 更低 ⇒ 只会**加强**"top-K 不可行"的结论。
   - **处置**：探针（新代码）改用正确的 in-range 掩码并把跳过的不可达物体计数上报（`skipped_unreachable`）；**诊断脚本暂不改**，因为 Phase 2.3 的预注册判据是按既有口径定的，裁定前换口径等于移动球门。偏差单独量化，记为待办 #20。探针侧修复见 `aa8437e`。

另有 **5.86% 不可达 target**（2440 中 143 个全部点落在 `point_cloud_range=[-51.2,-51.2,-5, 51.2,51.2,3]` 外，物理上不可分割；car 71 / truck 53 / bus 8 / barrier 5 / construction_vehicle 4 / trailer 2，点数中位 5 vs 保留物体 13），另 38 个部分越界（掩码被截断但仍可达）。用户定的方向是**过滤**（V3，`a9993b3` + `6bf33da`）：`reasonseg_val_thin_reachable.jsonl`，2248 → **2108** 条记录。过滤粒度必须是**记录**而非单个 target —— query 指名一个具体物体、answer 每个 target 带一个 `<SEG>`，从多目标记录里摘掉一个会让 query/answer/targets 三者失配、连带弄坏语言侧监督。实测：过滤几乎不改结论（mean IoU Δ 从 +0.00694 变 +0.00675，差 0.0002；140 条被丢记录里只有 19 个可达 target 是附带损失）⇒ **结论对数据口径稳健**。⚠ 过滤后的数与所有历史 val_thin 数**不可比**，对照必须一起重算。

## 五、复发三次的自身错误：判据挑错轴

**这是本条记忆里最该被复用的部分。** Phase 2.0 的四条判据、Phase 2.2 的尺寸护栏、Phase 2.3 oracle 探针的分档判据 —— 三次预注册阈值都没能把结论区分开，**复发原因完全相同：一直拿 `recall@0.5` 当判别量，而真实信号在 AUC（排序质量）与幅值（尺寸/阈值校准）的分离上**。`recall@0.5` 把"排序够不够好"和"尺寸/阈值校不校得准"混成一个数，在正例占 3e-4 的稀疏掩码问题上天然无效。

探针首跑就是活例：判据写"recall@0.5 ≤ 0.20 → 编码器/点特征是瓶颈"，实测 0.045 落进这一档，但同一份数据的 **AUC 0.978 直接反驳该裁定**（点特征若撑不起实例掩码，AUC 不可能这么高）。该判据已被明确撤回。

**换轴后的正确做法**：判别量用 AUC 与 `recall_top_k`（免幅值），阈值用**先验推导**（如 1−K/N）而非拟合观测值；若必须在看过数据后改判据，要显式声明"是看过首跑数据后改的"并给出与观测无关的推导来源。

## 六、How to apply（下次直接照做）

- **绝不用早停 manifest 裁定结论。** 实测乐观偏差量级：同一个 A2-ext20 在 `internal_es` 上是 +0.0143（看着显著胜出），换到未参与选模型的 `val_thin` 就变成 +0.0067（不显著）且 recall 转负、global IoU 转负。早停 manifest 只用于选模型，判定一律走 held-out。
- **报任何 ReasonSeg 数必须写明 manifest 名 + 记录数 + 分母口径**（是否 `scorable` 修复后、是否 `reachable` 过滤后）。TF mean IoU 的绝对值只在同一 manifest 内可比（query 构成主导：同一份权重在 val_thin 2248 / internal_fresh 19450 / oov500 1991 上分别是 0.113 / 0.082 / 0.138）。
- **`val_thin` 的 20 个场景是 nuScenes 官方 val 即 B4DL 测试集**，只能作最终测试报告，不得用于早停（tvenc680 踩过）。
- **新增消融开关要区分"架构字段"与"纯损失字段"**：`--no-loc-prior` 是架构字段，`validate-only` 复算时也必须带上，否则 loader 拒绝加载；纯损失/权重字段必须同步加进 `config.py:LOSS_ONLY_FIELDS`，否则用非默认损失训出的档无法被默认配置的评测入口加载。改损失前先跑 `tests/test_reasonseg.py::MaskLossTests`。
- **消融测试要防空洞通过**：LOC 先验切除的测试是一对——关闭态扰动先验权重对 `mask_logits` **零影响**，开启态同样扰动**必须改变它**。只有前者会被"根本没接线"的实现骗过。
- **全新代码进串行链前必须挂冒烟关卡（中止门）**。Phase 2.3 的 S0/S0b/S0c 三个冒烟救了两次：一次是 `QueryMaskDecoder` 返回 `[K,1,N]` 被当 `[K,N]` 用（训练侧 BCE 形状不匹配会崩，但**评测侧 `probabilities[row][:point_count]` 会去切那个大小为 1 的维度、静默返回错误结果**——比崩溃危险得多；已把 squeeze 收进 `decode_per_object` 并加形状断言）；一次牵出上面第四节的前缀掩码 bug。
- **`tar cf - .` 同步会 chown 目标目录**：它把 `./` 这个目录条目本身也存进归档，以 root 解包时 tar 保留归档属主，仓库根被 chown 成 Windows 侧 uid（197612）而 `.git` 仍是 root → git 报 `detected dubious ownership`。修法是 `chown root:root`，**不要**去加 `git config --add safe.directory`（那是掩盖问题）。
- **服务器 ref 前移不能用 `update-ref` 抄近路**：服务器没有那个对象时 git 会拒绝（`nonexistent object`，自我保护生效）。autodl3 无 GitHub 直连且是 blob 过滤的 partial clone，同步一律走区间 bundle（`git bundle create x.bundle <base>..seg`；全历史 bundle 会因缺 blob 反复回源卡死）。
- **per-arm 新建的 DataLoader 必须过 `accelerator.prepare`**，否则 batch 留在 CPU，spconv 直接 `AssertionError: implicit gemm only support cuda`（修于 `531664c`）。

## 七、关键提交与产物路径

- 提交：`2d4b8ec`（ckpt 轮转）、`053fcb6`（balanced BCE + Tversky）、`3bd8991`（接地诊断脚本）、`04ec539`（LOC 先验置零 + 相对阈值两臂）、`78cf410`（Phase 2 过夜链）、`590a0f4`（判据无歧义时自动扩跑 20ep 的 watcher）、`a684511`（分母修复）、`a9993b3` + `6bf33da`（不可达过滤 + 缓存 bug）、`617a90d`（训练期 `--no-loc-prior`）、`5a34323`（oracle 探针 + Phase 2.3 链）、`aa8437e`（探针改 in-range 掩码）、`057cf1e`（探针逐轮选优 + top-K + 先验判据）、`6b65cef`（修 docstring 里算错的量级说法）。
- 脚本：`mllm/scripts/reasonseg_experiments/`（`chain_phase23.sh`、`probe_oracle_query.py`、`diagnose_reasonseg_grounding.py`、`watch_then_rerun_oracle_probe.sh`、`repartition_reasonseg_val.py`）。
- 日志：`mllm/training_logs/phase23/`（driver.log + s0~s4）、`mllm/training_logs/verify_v123/`（s1~s7 + filter_report.json）、`mllm/training_logs/tvenc680.log`。
- 结果：`mllm/eval_results/_oracle_probe/report.json`、`mllm/eval_results/_grounding_reasonseg-loss*/report.json`。
- ⚠ `mllm/reasonseg_data_trainval/` 与 `mllm/training_logs/` 整体被 gitignore，脚本必须放进前者才入库。

相关：[[b4dl-project-overview]]、[[b4dl-eval-methodology-caveats]]、[[b4dl-git-commit-conventions]]、[[b4dl-server-access-workflow]]
