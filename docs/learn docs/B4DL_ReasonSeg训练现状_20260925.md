# B4DL ReasonSeg 训练现状（2026-09-25 20:50 CST 快照）

**采集方式**：在 autodl3（`/root/autodl-tmp/mmb4dl`）上直接读 tmux / driver 日志 / 训练日志 / `eval_results` 报告得到，**不是引用任何对话记录**。快照会过期，复核命令见文末。

**一句话现状**：**Phase 2.3 全部跑完，GPU 已空、无 tmux 会话。** 两个结论：① B1 臂（切除 LOC 先验 + Tversky）落在预注册判据的**未决中间档**，held-out mean IoU 低于对照；② 修正版 oracle 探针 20:26:31 出数（`rc=0`，选 ep4），test AUC **0.99110**，按先验判据（需 ≳0.99963）**仍差约 24×** ⇒ **点特征分辨率是硬约束，query 侧改进有天花板**。下一步该投向 `voxel_size` / 多尺度 / 感受野，而不是 scene query 容量。

---

## 1. 跑完了什么

| 会话 | 内容 | 结果 |
|---|---|---|
| tmux `p23` | Phase 2.3 串行链 `scripts/reasonseg_experiments/chain_phase23.sh`（12:58:25 启动） | **18:21:57 `PHASE2.3 CHAIN DONE`**，S0/S0b/S0c/S1/S2/S3/S4 全 `rc=0`，会话已退出 |
| tmux `probe2` | watcher `watch_then_rerun_oracle_probe.sh`（14:37:04 启动） | 18:22:06 检测到主链完成并自动接手，**20:26:31 `probe_v2 rc=0`、watcher 退出** |
| GPU | 4090 24G | **空闲**（0% / 1 MiB，无计算进程） |

代码状态：分支 `seg`，HEAD `6bc7208`，工作树全净。（服务器本地 `origin/seg` 远端跟踪 ref 停在 `ea39c94`，因为推送是从 Windows 侧完成的、服务器未 fetch —— 只是 ref 陈旧，不代表未同步。）

### 主链五步

| 步 | 内容 | 时间 | rc |
|---|---|---|---|
| S0 / S0b / S0c | 三个冒烟中止门：oracle 探针（40 训 / 20 评）、`--no-loc-prior` 通路（60 条 / 1ep）、该档能否被同旗标 `validate-only` 重载 | 12:58:25 → 13:07:59 | 0 / 0 / 0 |
| S1 | B0 oracle-query 探针首跑，4 epoch / lr 3e-4 / 600 条评测 | 13:07:59 → 14:30:50 | 0 |
| S2 | **B1 = 切除 LOC 先验 + Tversky(0.3/0.7)**，4 epoch（465 step/ep × 4 = 1860 step） | 14:30:50 → 18:07:17 | 0 |
| S3 | B1 接地诊断（val_thin 1000 条，与对照/A2 同口径） | 18:07:17 → 18:14:08 | 0 |
| S4 | B1 在**过滤后** `reasonseg_val_thin_reachable.jsonl`（2108 条）上的 TF，与 verify 链的 s4/s5 同口径 | 18:14:08 → 18:21:57 | 0 |
| — | 修正版 oracle 探针（`probe2`，6 epoch / lr 1e-4 / 400 条评测） | 18:22:06 → 20:26:31 | 0 |

---

## 2. B1 已出数的结果

### 2.1 训练侧 TF（`internal_es`，objects 2161，token_failure 全程 0.0）

| epoch | mean IoU | global IoU | recall@0.5 | hits |
|---|---|---|---|---|
| 0 | 0.08174 | 0.46293 | 0.08191 | 177 |
| 1 | 0.10688 | 0.38487 | 0.09671 | 209 |
| 2 | 0.11931 | 0.46355 | 0.10273 | 222 |
| **3（best）** | **0.12175** | 0.45695 | 0.10273 | 222 |

ep2 与 ep3 的 recall 逐位相同是因为 `recall = hits/objects = 222/2161`，两轮命中**数**相同（不必然同一批物体）。

⚠ 这是**早停 manifest**，同时用于选模型，天然乐观偏差 —— 已知量级：A2-ext20 在 `internal_es` 上 +0.0143（看着显著），换到 `val_thin` 只剩 +0.0067（不显著）且 recall 转负。**不能用这张表裁定 B1。**

### 2.2 接地诊断（S3，val_thin 1000 条，n=1068）

| | AUC | prob_pos | recall@0.5 | 尺寸 pred/target | LOC 包含 | recall_top_k |
|---|---|---|---|---|---|---|
| 对照 internal679（20ep） | 0.8576 | 0.165 | **0.0861** | 3 / 11（0.27×） | 0.252 | — |
| A2 plain+Tversky（4ep） | **0.9167** | 0.222 | 0.0777 | 14 / 11（1.27×） | 0.352 | — |
| A2 plain+Tversky（20ep） | 0.8371 | 0.214 | 0.0805 | **11 / 11（1.00×）** | 0.306 | — |
| **B1 no-prior+Tversky（4ep）** | 0.9048 | 0.199 | 0.0796 | **7 / 11（0.64×）** | **0.372** | 0.1021 |

口径说明：诊断脚本**故意没改**前缀掩码那个 bug（Phase 2.3 判据是按既有口径预注册的，裁定前换口径等于移动球门）。该失真对所有 run 完全相同 ⇒ **上表的横向比较成立**，但 AUC 绝对值被约 2035 个"必为 0 的简单负例"抬高。

### 2.3 预注册判据裁定

`chain_phase23.sh` 头注释里写死的判据（对照 = A2@4ep：AUC 0.9167 / 尺寸 1.27× / internal_es ep3 mIoU 0.1255）：

| 条件 | 阈值 | B1 实测 | 结果 |
|---|---|---|---|
| AUC | ≥ 0.93 | 0.9048 | ✗ |
| 尺寸比 | ∈ [0.7, 1.5] | 7/11 = **0.636** | ✗（欠触发，比 A2 的 1.00× 反向偏了） |
| internal_es ep3 mIoU | > 0.1255 | 0.12175 | ✗（低 0.0038） |
| 诊断 recall@0.5 | > 0.0861 | 0.0796 | ✗（hit 数仍未动） |
| **PASS** | 四条全中 | — | **不成立** |
| **FAIL** | AUC < 0.90 **或** 尺寸比 ∉ [0.5, 2.0] | AUC 0.9048 ≥ 0.90；0.636 ∈ [0.5,2.0] | **两条都未触发** |

⇒ **落在未决中间档：既没 PASS 也没 FAIL。**

可以确定的读法（不过度外推）：
- 先验切除**没有与 Tversky 复合成期望的效果**。两者单独时各有一项达标（Phase 2.1 先验置零 AUC 0.917、A2 尺寸 1.00×），合起来 AUC 0.9048（低于 A2@4ep 的 0.9167）、尺寸 0.64×（欠触发）。
- **headline 指标 `recall@0.5` 第四次没有动**：对照 0.0861、A2@4ep 0.0777、A2@20ep 0.0805、B1 0.0796 —— 全部低于对照。这与 V1 失败分解的结论一致（`right_class_wrong_place` 是第一大失败模式，hit 恒为 24/330）。
- LOC 包含率 0.372 是四个 run 里最高的，但**在没有 hit 数变化的前提下它不构成收益**，此处只记数、不做机制解释。

### 2.4 held-out 口径（S4，过滤后 val_thin 2108 条，objects 2278）

同一份过滤后 manifest、同一个 `scorable` 分母，三个模型并排：

| | mean IoU | global IoU | recall@0.5 | hits |
|---|---|---|---|---|
| 对照 internal679（20ep） | 0.11047 | 0.33911 | 0.09087 | 207 |
| A2-ext20 plain+Tversky（20ep） | **0.11722** | 0.31701 | 0.08165 | 186 |
| **B1 no-prior+Tversky（4ep）** | **0.10542** | 0.27830 | 0.08253 | 188 |

- B1 的 held-out mean IoU **比对照低 0.00505、比 A2 低 0.01180**；global IoU 是三者最低。
- ⚠ **口径不对等**：B1 只训了 4 epoch，对照与 A2 都是 20 epoch；A2@4ep 没在这份过滤后 manifest 上测过。所以这张表**不能**读成"B1 比 A2 差"，只能读成"B1 的 4-epoch 档在 held-out 上没有超过对照的 20-epoch 档"。B1 的正式裁定仍按 §2.3 的预注册判据（对照取 A2@4ep），结论是未决中间档。
- **早停 manifest 的乐观偏差又兑现了一次**：同一个 B1 档，`internal_es` ep3 是 0.12175，held-out 过滤后 val_thin 只有 0.10542（差 0.0163）。这与 A2-ext20 的模式一致（`internal_es` 0.14437 vs val_thin 0.11555）。**再次印证：不能用早停 manifest 裁定任何结论。**

---

## 3. oracle 探针：修正版已出数，**裁定 = 点特征是硬约束**

修正版（`057cf1e`：每轮洗牌、lr 1e-4、逐轮 dev 评测并按 dev `iou_mean` 选轮、test 只评一次、补 `recall_top_k` 免幅值轴）18:22:06 → **20:26:31 `rc=0`**，报告 `eval_results/_oracle_probe_v2/report.json`，日志 `training_logs/phase23/probe_v2.log`。配置：6 epoch / 7436 训练记录 / 每轮 dev 400 条 / test 只在选定轮评一次 / Tversky(0.3,0.7) + plain BCE / 编码器冻结 / **完全不加载 7B LM**。

**选定 ep4**（dev `iou_mean` 0.2036 最高）：

| | dev (internal_es, 381 obj +22 不可达) | **test (val_thin, 416 obj +23 不可达)** |
|---|---|---|
| **AUC** | 0.99272 | **0.99110** |
| recall@0.5 | 0.16010 | **0.09615** |
| **recall_top_k**（免幅值） | 0.20735 | **0.11058** |
| iou_mean / iou_top_k_mean | 0.20359 / 0.23680 | 0.15856 / 0.18386 |
| mean_prob_positive | 0.4732 | 0.3971 |
| 尺寸 pred/target | 33 / 12（2.75×） | 29 / 12（2.42×） |

逐轮 dev（train_loss 单调下降，**首跑的末轮失稳没有复现**）：

| epoch | train_loss | AUC | recall@0.5 | top_k | iou_mean | 尺寸 |
|---|---|---|---|---|---|---|
| 0 | 0.8067 | 0.98440 | 0.0866 | 0.1102 | 0.1060 | 0 / 12 |
| 1 | 0.7133 | 0.99052 | 0.1155 | 0.1417 | 0.1538 | **12 / 12** |
| 2 | 0.6611 | 0.99241 | 0.1024 | 0.1549 | 0.1669 | 50 / 12 |
| 3 | 0.6446 | 0.99254 | 0.1024 | 0.1444 | 0.1638 | 28 / 12 |
| **4（选定）** | 0.6252 | 0.99272 | **0.1601** | **0.2073** | **0.2036** | 33 / 12 |
| 5 | 0.6105 | **0.99395** | 0.1207 | 0.1732 | 0.1622 | **11 / 12** |

### 3.1 裁定：按先验判据仍差约 24×

判据是**先验推导**的、与本次观测无关（不构成事后拟合）：top-K 解码要能命中，需 AUC ≥ 1 − K/N，K = GT 尺寸中位 **12**、N ≈ **32530** 框内点 ⇒ **门槛 ≈ 0.99963**。用"压在某个正点之上的负点数" = (1−AUC)×N 换算：

| | AUC | 正点上方的负点数 | 需要 | 差距 |
|---|---|---|---|---|
| 完整模型（对照，诊断口径） | 0.8576 | ~4630 | ≤ 12 | ~386× |
| oracle 首跑（4ep / lr 3e-4 / 600 条） | 0.9776 | ~729 | ≤ 12 | ~61× |
| **oracle 修正版（ep4 test）** | **0.99110** | **~289** | ≤ 12 | **~24×** |
| oracle 修正版（ep5 dev，AUC 最高轮） | 0.99395 | ~197 | ≤ 12 | ~16× |

⇒ **给了完美定位（GT 中心 + one-hot 类别，完全绕过 7B LM），排序质量从 ~4630 个干扰负点改善到 ~289 个（≥16×），但仍差约 24× 才够 top-K 命中。**

**裁定：点特征的分辨率/感受野是硬约束**（`voxel_size` 0.1 m、单尺度、U-Net 感受野）—— 一个 12 点的物体在 32530 个框内点里，靠现有特征无法被锐利分出来。**query 侧的任何改进都有天花板**：更多 scene query、实例对比监督、CoT 查询都填不上这 24×。

而且"继续训就能到"这条路可以排除：(1−AUC) 逐轮是 0.0156 → 0.0095 → 0.0076 → 0.0075 → 0.0073 → 0.0061，**已在 0.006–0.007 平台**，而达标需要 0.00037，差 16–20 倍且趋势已平。

顺带一个交叉印证：探针 test 的 `recall@0.5` 只有 0.09615、`recall_top_k` 0.11058 —— 即便 query 完美、即便用免幅值的 top-K 解码，命中率也只是从完整模型的 ~0.086 抬到 ~0.11。这与"289 个负点压在正点之上 ⇒ top-12 被污染 ⇒ IoU 上不去"的机制自洽。

### 3.2 三处必须诚实标注的局限

1. **选轮的量与裁定的量不是同一个轴。** 选轮用 dev `iou_mean` → 选了 ep4；但判据轴是 AUC / top-K，而 **ep5 的 AUC 更高（0.99395 vs 0.99272）、尺寸校准最好（11/12 = 0.92× vs ep4 的 2.75×）**，只是 `iou_mean` 更低（0.1622 vs 0.2036）。test 只在 ep4 评了一次，**ep5 的 test 数没有**。若按 AUC 轴选轮，结论只会更强（差距从 24× 收到 16×，仍远不达标），所以裁定方向不变；但要引用"最好那一轮"的 test 数，需重跑（约 20 min/epoch + 评测）。**不在看过数据后改选轮规则**，这条按原样报。
2. **dev 与 test 差一大截，且 dev 参与选轮有乐观偏差。** top_k 0.2073(dev) vs 0.1106(test)、recall@0.5 0.1601 vs 0.0962、iou_mean 0.2036 vs 0.1586。dev 是官方 train 场景内划分（in-distribution），test 是官方 val 场景（unseen）。**对外只引用 test 列。**
3. **与接地诊断的数不可直接相减。** 探针用正确的 in-range 掩码 + 400 条 / 416 物体；`diagnose_reasonseg_grounding.py` 仍带前缀掩码 bug（故意没改，见记忆待办 #20）+ 1000 条 / 1068 物体。前缀 bug 会把约 2035 个"必为 0 的简单负例"塞进 AUC，**抬高**完整模型的 0.8576 ⇒ 上表"完整模型 ~4630"是**低估**，oracle 的改善倍数 ≥16× 是下界，裁定因此更稳、不是更弱。

### 3.3 首跑留档（已被修正版取代，不要再引用）

首跑（S1，`eval_results/_oracle_probe/report.json`；4 epoch / lr 3e-4 / 600 条）：dev AUC 0.9735 / recall@0.5 0.0471 / iou_mean 0.1071 / 尺寸 57:15；test AUC 0.9776 / recall@0.5 0.0449 / iou_mean 0.0917 / 尺寸 37:11（3.4× 过触发）。train_loss 0.9233 → 0.8146 → 0.7784 → **1.0274**，末轮失稳而评的正是这个最差末轮。

它的预注册判据（按 `recall@0.5` 分档）**已被明确撤回** —— 那是**判据挑错轴的第三次复发**（前两次：Phase 2.0 四条判据、Phase 2.2 尺寸护栏），根因是 `recall@0.5` 把"排序够不够好"与"尺寸/阈值校不校得准"混成一个数。修正版换成先验 AUC 阈值 + `recall_top_k` 后，**两个轴指向同一档，首跑那个"AUC 高但 recall 低"的矛盾消失了**。

### 3.4 体素并列天花板实测：**改 `voxel_size` 的前提被推翻**（20:55，纯 CPU，23 秒）

`SparseUNetPointEncoder` 先体素化、跑稀疏 U-Net，再用 `inverse` 把体素特征散射回原始点（`spatial_encoder.py:128-134`），所以**同一体素内的点拿到逐位相同的特征、logit 必然并列**，并列对在 AUC 里只算 0.5。设某物体 GT 框内点数 K、帧内框内点数 N、与 GT 点共享体素的非 GT 框内点数 T，则并列单独造成的上限是 `AUC_max = 1 − 0.5·T/(K·N)`；与判据 `AUC ≥ 1 − K/N` 联立得 **T ≤ 2K² ≈ 288**（与 voxel_size 无关的硬条件）。

实测（`scripts/reasonseg_experiments/probe_voxel_tie_ceiling.py`，val_thin 前 150 条 / 155 个物体，K 中位 11、N 中位 32503）：

| voxel_size | T 中位 / 均值 / p90 | 并列造成的 AUC 上限 | 落在 T ≤ 2K² 预算内的比例 |
|---|---|---|---|
| 0.2 m | 0 / 1.09 / 1 | 0.99999984 | 100% |
| **0.1 m（现状）** | **0 / 0.30 / 0** | **0.99999996** | 100% |
| 0.05 m | 0 / 0.05 / 0 | ~1.0 | 100% |
| 0.025 m | 0 / 0.00 / 0 | 1.0 | 100% |

**T 基本是 0**：体素栅格极稀疏（32503 个点 vs 1024×1024×80 ≈ 8400 万候选体素），GT 点几乎从不与其他点撞进同一体素。⇒ **并列不解释任何缺口，那 289 个排在正点之上的负点全部是真实的特征/排序失败**；改 `voxel_size` 换不回 AUC，原计划的"重训空间编码器"前提不成立，**没有执行**。

另外两点由同一份代码事实推出，别再走回头路：① `voxel_size` **不改变** N 与 K（打分对象是原始 LiDAR 点，`point_features` 按 `points.shape[0]` 展开），所以判据门槛 1−K/N 对 voxel_size 是不变量；② U-Net 的感受野以**体素**为单位固定，故减小 voxel_size 会**同比缩小以米计的感受野** —— 方向上是有害的，若真要动这个旋钮，增大而非减小才有 RF 上的理由（代价是掩码边界分辨率变粗，而并列成本实测仍可忽略）。

**因此 §3.1 裁定的正确读法要收紧**：探针的编码器是**冻结**的，它量到的是"现有语义预训练特征 + 完美 query"的天花板，**不是"点特征本身"的天花板**。编码器在 lidarseg（语义）目标上预训练，特征编码"这是车的点"而非"这是哪一辆车"；实例判别性从未被任何监督要求过。⇒ 下一个最便宜且真正决定性的实验是**在 oracle 探针里解冻编码器**，看特征在完美 query 监督下能否被改造成实例判别的。

---

## 4. 今天已经完成、可作为基线引用的数

verify 链（`training_logs/verify_v123/`，11:25:02 → 12:07:48，七步全 `rc=0`）：

| 项 | 结果 |
|---|---|
| **V2 分母修复**（`a684511`） | 未过滤 val_thin 上两侧 objects 都精确变成 **2297 = 2440−143**；mean IoU 对照 **0.11020** / A2 **0.11714**（事前手算 0.11021 / 0.11718，吻合到 1e-5） |
| **V3 不可达过滤**（`a9993b3` + `6bf33da`） | 2248 → **2108** 条；143 个不可达 target 丢弃、38 个部分越界保留；过滤后两侧 objects **2278**，mean IoU 0.11047 / 0.11722 ⇒ Δ 从 +0.00694 变 +0.00675，**结论对数据口径稳健** |
| **V1 A2-ext20 自由生成**（300 条） | `cIoU 0.22492 / gIoU 0.09954 / inst P@0.5 = inst R@0.5 = 0.072727 / token_failure 0.0 / count_accuracy 1.0`；对照是 `0.23690 / 0.09069 / 0.072727`。**instance recall@0.5 = 24/330 两边逐位相同** |
| **失败分解**（S7） | `predicted_mask_empty` 140→99（−41）、`near_miss_below_0.5` 71→100（+29）、`right_class_wrong_place` 87→**102**（升为第一大失败模式）、`hit` **24→24**（0）、`wrong_class_wrong_place` 7→5、`overlap_stolen` 1→0 |

⇒ **A2-ext20 不是新基线，损失函数这条杠杆判死；至此七条全判死**（lr/dropout、场景数、语言侧、阈值、协议、编码器、损失函数）。

---

## 5. 下一步（GPU 已空，方向由 §3 裁定给出）

| 优先 | 事项 | 说明 / 状态 |
|---|---|---|
| ~~1~~ | ~~读 S4 的过滤后 val_thin TF~~ | **已完成**，见 §2.4 |
| ~~2~~ | ~~等修正版探针裁定"LM query 通路 vs 点特征"~~ | **已完成**，裁定 = 点特征是硬约束，见 §3.1 |
| 1 | **在 oracle 探针里解冻编码器**（而非改 `voxel_size` —— 该前提已被 §3.4 实测推翻） | §3.1 量到的是**冻结**语义特征的天花板，不是点特征本身的。编码器只在 lidarseg（语义）目标上训过，从未被要求实例判别性。用完美 query 监督解冻重训，看 AUC 能否从 0.9911 往 0.99963 走；约 2–3 h，仍是判方向而非出成品 |
| 1b | 若 1 也停在 0.99 量级 ⇒ 单帧点特征确实撑不起实例接地，此时才谈**换问题设定**（输出粗区域而非实例掩码 / 引入时序多帧 / 把这条作为负结果如实写进论文） | 取决于第 1 项 |
| 2 | B1 的处置：预注册判据落在**未决中间档**（PASS 四条全不中、FAIL 两条都不触发） | 要么按判据原文报"未决"，要么补一个 4-epoch 的 A2 对照在同一过滤后 manifest 上重测以对齐口径。**不要在看过数据后临时改判据阈值**（已复发三次）。注意：既然裁定指向点特征，B1 这条 query/损失侧的路线优先级已下降 |
| 3 | 待办 #20：单独量化诊断脚本的前缀掩码偏差 | Phase 2.3 已裁定完，**现在可以做**（此前刻意压后，避免裁定前移动球门）。做完才能把完整模型的 AUC 0.8576 换成无偏值，§3.1 表里"~4630 / ≥16×"也才能从下界变成点估计 |
| 4 | 若要引用"最好那一轮"的 test 数 | 选轮用的 dev `iou_mean` 选了 ep4，但 AUC 轴上 ep5 更好（0.99395、尺寸 11/12）而 test 未评。补测约 20 min + 评测。**裁定方向不受影响**（24× → 16×，都远不达标） |
| — | V4（追 AUC 随训练退化 0.9167→0.8371 的轨迹） | 仍**先记下不做**：中间档被 `keep-last 2` 轮转掉，要拿轨迹得重训并保留每 4 轮的档 |

---

## 6. 复核命令

```bash
cd /root/autodl-tmp/mmb4dl/mllm
tmux ls; tail -5 training_logs/phase23/driver.log          # 链在哪一步（现已 DONE）
tail -3 training_logs/phase23/watcher_probe2.log           # probe2 是否已启动/退出
cat training_logs/phase23/probe2_decision.txt 2>/dev/null   # GO / NOGO
grep teacher_forcing_mean_iou training_logs/phase23/s2_train_b1.log   # B1 逐轮 TF
grep teacher_forcing_mean_iou training_logs/phase23/s4_tfval_b1.log   # B1 过滤后 val_thin
tr '\r' '\n' < training_logs/phase23/probe_v2.log | grep -E '^epoch|selected'  # 探针逐轮 dev + 选定轮
# 接地诊断（n / AUC / 尺寸 / recall）：
/root/autodl-tmp/.conda-stuff/envs/reasonseg/bin/python -c "import json;d=json.load(open('eval_results/_grounding_reasonseg-lossB1-noprior-tversky/report.json'))['overall'];print({k:d[k] for k in ('n','auc_mean','recall_thresholded','recall_top_k','predicted_size_median','target_size_median','loc_containment_mean')})"
# oracle 探针修正版（选定轮 + dev/test）：
/root/autodl-tmp/.conda-stuff/envs/reasonseg/bin/python -c "import json;d=json.load(open('eval_results/_oracle_probe_v2/report.json'));print('selected',d['selected_epoch']);print({k:d['test'][k] for k in ('objects','auc_mean','recall_at_0.5','recall_top_k','iou_mean','predicted_size_median','target_size_median')})"
```

> 注意：服务器系统 `python3` 不可用，一律走 `/root/autodl-tmp/.conda-stuff/envs/reasonseg/bin/python`。
