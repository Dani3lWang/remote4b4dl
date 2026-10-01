# ReasonSeg 单帧点级推理分割：瓶颈定位与负结果报告（2026-09-18 ~ 10-01，autodl3）

**性质**：方案 D 收口。数据一律取自落盘产物（`report.json` / `history.json` / 探针 JSON / bootstrap JSON），
不引对话记忆；未复核项显式标注。逐时刻快照见 `B4DL_ReasonSeg训练现状_20260925.md`，实验弧线与被撤回判断见
`B4DL_ReasonSeg实验总结_20260927.md`，选项 A 判据见 `B4DL_ReasonSeg多帧A_预注册_20260927.md`。本文只留
**结论、机制、区间与边界**。

## Abstract (English)

On the ReasonSeg task built over B4DL/nuScenes (single-frame point-level reasoning segmentation), the binding
constraint is **instance grounding**, and its mechanism is a **per-point ranking ceiling**: even when a perfect
query (GT centre + one-hot class) is fed straight to the mask decoder, bypassing the 7B language model, point-level
AUC saturates at 0.995-0.998, i.e. 10-30x away from the `1 - K/N ~ 0.9996` required for a top-K hit, where a typical
target has K~13 positives among N~32,400 in-range points. Eight candidate levers, three training regimes and two
epoch budgets, and a three-arm multi-frame control all fail or are negative. The one positive architectural finding
is that **semantic (lidarseg) and instance objectives coexist** in a single encoder (linear-probe mIoU +7.5% to
+8.2% over the pretrained baseline at both 20 and 30 epochs). We also report measured **noise floors**
(dev +-0.002, test +-0.003 AUC; paired bootstrap CI over objects) that bound what any single-evaluation ablation in
this setting can resolve.

## 1. 主张与不主张

- **主张**：瓶颈在点级实例接地；不是分割质量、不是损失、不是数据规模、不是语言侧、不是体素栅格。
- **主张**：语义与实例可共存于同一份特征（本文唯一的正面结论）。
- **不主张**：不声称"时序/多帧无用"。A1 检验的是"体素均值 + 无时间通道"的聚合方式，见 §5。
- **不主张**：不外推到带 3D detector 先验或多帧融合 pipeline 的 Reason3D / MORE3D 设定。

## 2. 口径（引用任何数字都必须带齐）

- 掩码 = nuScenes panoptic 逐点 `instance_index == panoptic_id`（`segmentation/data.py:104`）；**不是 3D 框内点**，
  manifest 连 `center` 都不落盘。
- K = 该实例在 range 内的正例点数中位；N = 每帧落在 `point_cloud_range = [-51.2,-51.2,-5, 51.2,51.2,3]` 内的点数中位。
  冒烟门实测（`val_thin`）：单帧 K 13 / N 32,412；a1/a3 K 41 / N 97,287；a2 K 37.5 / N 97,236。
- 先验门槛 = `1 - K/N`，**按臂实测**：单帧与 a2 = 0.999599；a1/a3 = 0.999579。
- 测试集 = `reasonseg_val_thin.jsonl`（官方 val 的 20 个场景 = B4DL 测试集），**不得用于早停**；选模型只能用
  `internal_es`。⚠ 该 dev 清单 94% 的记录落在各自场景首帧 ⇒ 多帧臂在 dev 上只拼得到未来一帧（N≈64k），
  与 test（N≈97k）的绝对值**不可横比**。
- 编码器全程 fp32（spconv 内部转 fp16 会 NaN/硬崩）；BN 钉 eval（`dW stats` 全程 0.000000 为证）。

## 3. 完整模型侧：接地失败是端到端低分的直接来源

| 口径 | manifest / 分母 | 对照 internal679 | plain+Tversky（A2-ext20） |
|---|---|---|---|
| TF mean IoU / global IoU / recall@0.5 | `val_thin` 2,248 条 / 2,297 可评物体 | 0.11020 / 0.35490 / 0.09055 | 0.11714 / 0.33151 / 0.08185 |
| 无偏接地诊断 AUC / recall@0.5 / recall_top_k | 1,000 条 / 1,038 物体 / 53 不可达 | 0.8755 / 0.0886 / 0.1127 | 0.8540 / 0.0829 / 0.1156 |
| 自由生成 cIoU / gIoU / **instance recall@0.5** | 同 300 条，`--dtype fp16 --encoder-dtype fp32` | 0.23690 / 0.09069 / **0.0727 (= 24/330)** | 0.22492 / 0.09954 / **0.0727 (= 24/330)** |

两跑的 instance recall@0.5 **逐位相同 = 24/330**，这是"掩码损失这条杠杆判死"最硬的一条证据。

⚠ 总结报告 §3.2 中 tvenc / tvenc680 的自由生成行（0.2922 / 0.2796 / 0.0534 等）是**旧 `metrics.py` 修正前**的存档值
（gIoU 曾被系统性高估约 3 倍），文档尚未改写；本文不引用它们。修正后的重算值目前只存在于迁移前的记忆快照
（cIoU 0.2288 / gIoU 0.0881 / recall 0.0667），**标记为未复核**，需要重跑自由生成评测才能正式引用。

## 4. oracle-query 探针：天花板在哪里（五跑）

| 跑 | 编码器状态 | test AUC | R@0.5 | topK_R | IoU | 尺寸 pred/target | `dW conv` |
|---|---|---|---|---|---|---|---|
| v2 | 冻结 | 0.99110 | 0.09615 | 0.11058 | 0.15856 | 29/12 | 0% |
| v3 | 解冻 lr1e-5（假微调） | 0.99510 | 0.14423 | 0.18990 | 0.19197 | 17/12 | 3.52% |
| v4 | 真微调 lr5e-5 20ep、纯实例 | 0.99038 | 0.24760 | 0.33890 | 0.28710 | 23/12 | 19.97% |
| 多任务 20ep | 语义+实例 | 0.99575 | 0.26440 | 0.31970 | 0.29910 | 22/12 | 21.81% |
| 多任务 30ep | 同上，预算 x1.5 | 0.99377 | 0.26923 | 0.35817 | 0.31219 | 27.5/12 | 26.97% |

- 给了完美定位 + 真改造过的特征，**仍只有约 1/4 的物体达到 IoU>=0.5**（三跑落在同一位置：0.2476 / 0.2644 / 0.2692）。
- 平台站得住的依据：20ep 的 dev AUC 在 ep9-19 连续十一轮稳在 0.9963-0.9976，同期 `dW conv` 14.77%→22.41%、
  实例 loss 0.4052→0.3055；30ep 的 ep22-29 极差 0.00138、每轮斜率 +5.4e-5，且**从未越过前 21 轮最高值 0.99748**，
  同期 `dW conv` 21.79%→26.97% ⇒ **损失在降、特征在变、排序不动**，两次独立复现。
- ⚠ 事后修正：30ep 用来认定平台的"极差 <= 0.0015"容差**本身落在噪声内**（空对照 a2 的 dev 极差是 0.00260）。
  平台的成立依据是"AUC 停在 0.995-0.998 这条带、距门槛差一个量级以上"这个量级判断，不是极差阈值。

## 5. 选项 A：三臂多帧对照（F=3、20 epoch、串行 87 h）

| 臂 | test AUC [95% CI] | 距本臂门槛 | IoU [CI] | R@0.5 [CI] | 尺寸 pred/target | `dW conv` | 选定轮 | 墙钟 |
|---|---|---|---|---|---|---|---|---|
| a2_repeat（空对照，复制 3 份） | 0.99296 [0.98943, 0.99581] | 17.6x | 0.2818 [0.2534, 0.3103] | 0.2404 [0.1995, 0.2812] | 79.5/36 | 21.99% | ep19 | 22.5 h |
| a3_naive（不对齐拼接） | 0.99088 [0.98720, 0.99400] | 21.7x | 0.2569 [0.2320, 0.2818] | 0.1842 [0.1483, 0.2225] | 61/37 | 19.57% | ep14 | 32.5 h |
| a1_compensated（ego 补偿拼接） | **0.98730 [0.98199, 0.99173]** | **30.1x** | 0.2928 [0.2662, 0.3204] | 0.2206 [0.1823, 0.2614] | 63/37 | 23.53% | ep19 | 32.4 h |

**配对 bootstrap（同一批物体、键交集，10,000 次重采样，seed 20261001）**

| 比较 | ΔAUC | 95% CI | n | 逐物体同向 | 判定 |
|---|---|---|---|---|---|
| a1 - a2 | **-0.00569** | **[-0.00991, -0.00215]** | 416 | 39.9% | **可分辨** |
| a1 - a3 | -0.00356 | [-0.00855, +0.00106] | 417 | 53.5% | 与噪声不可分 |
| a3 - a2 | -0.00212 | [-0.00648, +0.00190] | 416 | 37.7% | 与噪声不可分 |

**裁定：落预注册 A-e（劣于预期）。** 三臂中 a1 最差，且相对空对照 a2 的恶化越过噪声地板（CI 不含 0）。

两处必须更正的自我判断：

1. A-e 事前写的机制"0.5 s 错位代价高于新信息收益"被数据推翻 —— 不对齐的 a3 点估计反而优于对齐的 a1。
2. 我随后据此提出的"对齐比不对齐更差"**也不成立为可分辨**（a1-a3 的 CI 含 0）。

因此保留的结论只有一条：**在当前契约下，多帧拼接没有带来任何可分辨的排序改善，且 ego 补偿臂相对空对照显著更差。**
另注：a1-a2 的逐物体同向只有 39.9%，说明该均值差由一部分物体拉动 —— 效应是偏斜的，与 §4 里"排序误差偏斜"的观察一致。

- **候选机制（未证）**：编码器对同体素点取均值且无时间/帧号通道。ego 补偿使静止背景的三帧点叠入同一体素（均值不变），
  而运动物体不同时刻的点也叠入同一体素 ⇒ **物体自身的时变信号被均值抹平**。
- **边界**：A1 检验的是"无时间通道的体素均值聚合"，**不是"时序建模"**。要真正检验后者必须先给编码器加时间通道，
  而那会改 `input_dim` 并撞上 `checkpoint.py:169-182` 的 strict + voxel 契约校验 ⇒ **需重做空间预训练**；
  且盘上只有 2 Hz 关键帧（`sweeps` 未解包，补 10 Hz 约需 230 G，实例剩 70 G）。
- **副作用记录**：a3 的选定轮落在 ep14（中间），a1/a2 仍是 ep19（末轮）；a3 的 dev 末 8 轮极差只有 0.00092（三臂最平）。
  尺寸轴跨臂只能读相对位移：a2 的 79.5/36 与单帧的 28/12 是同一次过触发的 3x 恒等像，不是"更过触发"。

## 6. 正面结论：语义与实例可共存

线性探针（冻结编码器永久 eval + 从零训头 2 轮，fp32，25,324 / 2,806；输出硬写 `comparable_to_recorded_miou: false`，
只能同协议横比）：预训练 0.30785 → v3 0.30560（-0.7%，`dW conv` 3.52%）→ **v4 0.22073（-28.3%，19.97%）**
→ 多任务 20ep **0.33109（+7.5%，21.81%）** → 多任务 30ep 的 ep23 / ep27 / ep29 = **0.33159 / 0.33301 / 0.33099
（+7.7% / +8.2% / +7.5%，此段 conv 24.00%→26.97%）**。

⇒ v4 的语义坍塌不是必然代价，而是"只有实例目标时特征被单向拉走"；加语义锚后同样的改造幅度反而小幅提升语义，
且 30 轮尺度上仍成立。**被证伪的是"它能补上那 10-30 倍的排序缺口"，不是"让编码器适应实例目标"这条路本身。**

## 7. 八条杠杆判死（每条带机制）

| 杠杆 | 证据 | 机制 |
|---|---|---|
| lr 5e-4 / head dropout=0 | 均否决 | 不是优化不足 |
| 场景数 200→680（等算力） | best 0.11280@ep9，落预注册"不显著"档 | 场景多样性不是杠杆 |
| 语言侧 | 无偏诊断与 AUC 无差异 | 语言条件已饱和 |
| 解码端阈值 / 幅值校正 | 全负 | 幅值轴与排序轴分离 |
| 协议（无泄漏 internal679） | 判据 PASS，泄漏未虚增成绩 | 换协议不改变结论 |
| 空间编码器换代 | TF 0.0165 → 0.1037（6.3x） | **唯一台阶**，但天花板仍在 |
| 掩码损失（balanced BCE / Tversky / 切 LOC 先验） | headline recall@0.5 四次不动；B1 落未决中间档 | 3e-4 正例率下"沉默"与"同类对冲"的下坡仅占 4.9% |
| `voxel_size` | 同体素并列 T 中位 0、AUC 上限 0.99999984 | 打分对象是原始点 ⇒ N、K 与该旋钮无关；减小还会同比缩小以米计感受野 |

## 8. 方法学贡献（可独立引用）

1. **判别量必须按轴拆开**：`recall@0.5` 把"排序够不够好"与"阈值/幅值校得准不准"混成一个数，在正例占 3e-4 的
   稀疏掩码问题上天然无效。预注册五次落空的完整记录见总结 §8（挑错轴 x3、缺"优于预期"出口、缺"劣于预期"出口）。
2. **噪声地板实测值**：dev +-0.002（空对照与单帧同数据不同排列的逐轮 dev AUC 差，典型 0.0005、最大 0.0018）；
   **test +-0.003**（a2 与单帧 20ep 携带完全相同信息，test AUC 却差 0.0028）。⇒ 任何"单轮训练 + 单次 test 评测"
   的消融，**小于这两个量级的差异不应被叙述为效应**。
3. **test 只评一次是本类程序的结构性弱点**：dev 有 20 个读数可估噪声，test 只有 1 个 ⇒ 必须用逐物体 bootstrap 配对
   区间（§5）而不是点估计下结论。
4. **空对照是可复用的设计**：把 anchor 帧的点复制 F 份 ⇒ 同体素均值恒等（严格零新信息）、`1-K/N` 逐位不变、
   尺寸轴恰为 F 倍。它同时充当三件事的门：(a) 拼接未改变度量、(b) 噪声地板的量具、(c) 幅值轴跨臂比较的标尺。
5. **度量 bug 的两个教训**（详见总结 §7）：TF 分母不得依赖模型触发量；**预测一个 bug 的偏差方向要把分子分母两侧
   都算全** —— 曾断言"AUC 被白送的简单负例抬高、真值更低"，修好后三个档的 AUC 全部**升高**（真机制是越界的 GT 点
   被当前缀里的正例，概率恒 0，把逐物体 AUC 拽向 0.5）。

## 9. 复现索引与未尽事项

- 提交：`c05ba2d` 起（本文所在提交在其后），本地 / 服务器 / GitHub 三方一致。关键实现：
  `mllm/vtimellm/segmentation/multiframe.py`（三臂装配；框半尺寸须按 `(length, width, height)/2`，
  且点云在 **LiDAR 传感器系**，需复合 `calibrated_sensor` —— 这两处都由冒烟门抓出）；
  `mllm/scripts/reasonseg_experiments/{smoke_multiframe.py, bootstrap_temporal_arms.py, run_temporal_arm.sh, chain_three_arms.sh}`。
- 产物：`mllm/eval_results/_temporal_{a2_repeat,a1_compensated,a3_naive}_f3/report.json`、
  `_multiframe_smoke*.json`（按臂实测门槛）、`_temporal_bootstrap_ci.json`（CI 与配对差）、
  `mllm/training_logs/temporal/`（各臂 driver、`chain_driver.log`、`a2_verdict.txt`）。
- ⚠ 这些产物都在 gitignore 目录里，**只有服务器那一份**；代码与文档已三方同步，读数尚未备份。
- **未尽事项（诚实列出）**：
  1. 单帧 20ep / 30ep 的 `report.json` 未落盘逐物体指标 ⇒ §5 中与单帧参照的比较**只有点估计、没有区间**；
     20ep 那跑的实例头当年没保存（这正是后来加 `state_*.pt` 的原因）⇒ **无法补算**；30ep 可用
     `state_selected.pt` 做一次 eval-only 重放（约 20-30 min GPU）以拿到"多帧 vs 单帧"的配对 CI。
  2. A'（加时间通道 + 重做空间预训练）未做；10 Hz 稠密采样受盘容量限制。
  3. **LM query 通路本身未被隔离测量**（oracle 探针刻意绕过 7B），完整模型 recall 0.0727 与 oracle 天花板 0.269
     之间的差距**未分解**为"query 通路损失"与"接地损失"。
  4. 总结报告 §3.2 的 tvenc 系自由生成数字仍是修正前值，需要按新 `metrics.py` 重跑后改写。
