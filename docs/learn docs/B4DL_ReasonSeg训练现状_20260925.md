# B4DL ReasonSeg 训练现状（2026-09-25 18:25 CST 快照）

**采集方式**：在 autodl3（`/root/autodl-tmp/mmb4dl`）上直接读 tmux / driver 日志 / 训练日志 / `eval_results` 报告得到，**不是引用任何对话记录**。快照会过期，复核命令见文末。

**一句话现状**：Phase 2.3 串行链（tmux `p23`）**五步全 `rc=0`，18:21:57 `PHASE2.3 CHAIN DONE`**；B1 臂（切除 LOC 先验 + Tversky）四个数全部出齐，**预注册 PASS 判据四条全部未达、FAIL 判据两条都未触发，落在未决中间档**，且 held-out 口径上 mean IoU 低于对照；`probe2` watcher 已于 **18:22:06** 自动接手，修正版 oracle 探针正在跑（约 2.6 h，预计 21:00 前后出数）。

---

## 1. 在跑什么

| 会话 | 内容 | 状态（18:25） |
|---|---|---|
| tmux `p23` | Phase 2.3 串行链 `scripts/reasonseg_experiments/chain_phase23.sh`（12:58:25 启动） | **已完成**：S0/S0b/S0c/S1/S2/S3/S4 全 `rc=0`，18:21:57 打出 `PHASE2.3 CHAIN DONE` |
| tmux `probe2` | watcher `watch_then_rerun_oracle_probe.sh`（14:37:04 启动） | **18:22:06 已启动修正版探针**（轮次 225 检测到主链完成，`probe2_decision.txt` = `GO probe_rerun`），6 epoch / lr 1e-4 / 400 条评测 |
| GPU | 4090 24G | 由修正版探针占用（不加载 7B LM，显存远低于全模型） |

代码状态：分支 `seg`，HEAD `6b65cef`，工作树全净。（服务器本地 `origin/seg` 远端跟踪 ref 停在 `ea39c94`，因为推送是从 Windows 侧完成的、服务器未 fetch —— 只是 ref 陈旧，不代表未同步。）

### 主链五步

| 步 | 内容 | 时间 | rc |
|---|---|---|---|
| S0 / S0b / S0c | 三个冒烟中止门：oracle 探针（40 训 / 20 评）、`--no-loc-prior` 通路（60 条 / 1ep）、该档能否被同旗标 `validate-only` 重载 | 12:58:25 → 13:07:59 | 0 / 0 / 0 |
| S1 | B0 oracle-query 探针，4 epoch / lr 3e-4 / 600 条评测 | 13:07:59 → 14:30:50 | 0 |
| S2 | **B1 = 切除 LOC 先验 + Tversky(0.3/0.7)**，4 epoch（465 step/ep × 4 = 1860 step） | 14:30:50 → 18:07:17 | 0 |
| S3 | B1 接地诊断（val_thin 1000 条，与对照/A2 同口径） | 18:07:17 → 18:14:08 | 0 |
| S4 | B1 在**过滤后** `reasonseg_val_thin_reachable.jsonl`（2108 条）上的 TF，与 verify 链的 s4/s5 同口径 | 18:14:08 → 18:21:57 | 0 |

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

## 3. oracle 探针：首跑已出数，修正版待跑

**首跑**（S1，`eval_results/_oracle_probe/report.json`；4 epoch / lr 3e-4 / 600 条 / 编码器冻结 / 完全不加载 7B LM）：

| | dev (internal_es) | test (val_thin) |
|---|---|---|
| AUC | 0.9735 | **0.9776** |
| recall@0.5 | 0.0471 | 0.0449 |
| iou_mean | 0.1071 | 0.0917 |
| 尺寸 pred/target | 57 / 15 | 37 / 11（3.4× 过触发） |
| objects（+ 跳过的不可达） | 573（+29） | 623（+30） |

train_loss 逐轮 0.9233 → 0.8146 → 0.7784 → **1.0274**（末轮失稳在盘上可见），而评的正是这个最差末轮。

**首跑的预注册判据已被明确撤回**：它按 `recall@0.5` 分档，0.045 落进"≤0.20 → 编码器/点特征是瓶颈"，但同一份数据的 AUC 0.978 直接反驳该裁定。这是**判据挑错轴的第三次复发**（前两次：Phase 2.0 四条判据、Phase 2.2 尺寸护栏），根因相同 —— `recall@0.5` 把"排序够不够好"与"尺寸/阈值校不校得准"混成一个数。

**改用先验推导的读法**（阈值与观测无关，不构成事后拟合）：正点上方的负点数 = (1−AUC)×N，N≈32530 框内点。

| | AUC | 正点上方的负点数 | top-K（K≈11）需要 |
|---|---|---|---|
| 完整模型 | 0.858 | ~4619 | ≤ 11 |
| oracle query | 0.978 | ~716 | ≤ 11 |

⇒ 给了完美定位，改善 6.5×，**仍差 65 倍**（门槛 AUC ≳ 1−K/N ≈ 0.99966）。方向含义：**query 侧有硬天花板，约束在点特征的分辨率/感受野**（`voxel_size` 0.1 m、单尺度、U-Net 感受野）。失稳只会压低指标 ⇒ 0.978 是下界 ⇒ "LM query 通路吃掉 ≥0.12 AUC"这条更稳。**但这仍建立在一次有缺陷的运行上，只当方向、不当定论。**

**修正版**（`057cf1e`：每轮洗牌、lr 1e-4、逐轮 dev 评测并按 dev `iou_mean` 选轮、test 只评一次、补 `recall_top_k` 免幅值轴）**已于 18:22:06 由 `probe2` watcher 自动启动**（轮次 225 检测到 `PHASE2.3 CHAIN DONE`，GPU 无残留进程，`probe2_decision.txt` = `GO probe_rerun`）：6 epoch / 400 条评测，输出 `eval_results/_oracle_probe_v2/report.json`，日志 `training_logs/phase23/probe_v2.log`。按首跑 0.183 s/步外推（6 epoch 训练 + 6 次 dev 评测 + 1 次 test），**约 2.6 h，预计 21:00 前后出数**。

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

## 5. 下一步（等数，不要提前动）

| 优先 | 事项 | 触发条件 / 状态 |
|---|---|---|
| ~~1~~ | ~~读 S4 的过滤后 val_thin TF~~ | **已完成**，见 §2.4 |
| 1 | 等修正版探针出数，用**先验 AUC 阈值（≳0.99966）+ `recall_top_k`** 裁定"LM query 通路 vs 点特征"。注意按 dev `iou_mean` 选的轮次，不要再看末轮 | 探针 18:22:06 起跑，约 21:00 出数 |
| 2 | 探针裁定后再定架构方向：若确认点特征是硬约束，投向 `voxel_size` / 多尺度 / 感受野；若确认 query 通路，才考虑 scene query 容量 / 实例对比监督 / CoT 查询 | 第 1 项出数 |
| 3 | B1 的处置：预注册判据落在未决中间档（PASS 四条全不中、FAIL 两条都不触发）。要么按判据原文报"未决"，要么补一个 4-epoch 的 A2 对照在同一过滤后 manifest 上重测以对齐口径 —— **不要在看过数据后临时改判据阈值**（这是已经复发三次的错误） | 需要用户定方向 |
| 4 | 待办 #20：单独量化诊断脚本前缀掩码偏差（**裁定完 Phase 2.3 之后再做**，避免移动球门） | 第 1 项收口后 |
| — | V4（追 AUC 随训练退化 0.9167→0.8371 的轨迹）已明确**先记下不做**：中间档被 `keep-last 2` 轮转掉，要拿轨迹得重训并保留每 4 轮的档 | — |

---

## 6. 复核命令

```bash
cd /root/autodl-tmp/mmb4dl/mllm
tmux ls; tail -5 training_logs/phase23/driver.log          # 链在哪一步
tail -3 training_logs/phase23/watcher_probe2.log           # probe2 是否已启动
cat training_logs/phase23/probe2_decision.txt 2>/dev/null   # GO / NOGO
grep teacher_forcing_mean_iou training_logs/phase23/s2_train_b1.log   # B1 逐轮 TF
grep teacher_forcing_mean_iou training_logs/phase23/s4_tfval_b1.log   # B1 过滤后 val_thin
# 接地诊断（n / AUC / 尺寸 / recall）：
/root/autodl-tmp/.conda-stuff/envs/reasonseg/bin/python -c "import json;d=json.load(open('eval_results/_grounding_reasonseg-lossB1-noprior-tversky/report.json'))['overall'];print({k:d[k] for k in ('n','auc_mean','recall_thresholded','recall_top_k','predicted_size_median','target_size_median','loc_containment_mean')})"
```

> 注意：服务器系统 `python3` 不可用，一律走 `/root/autodl-tmp/.conda-stuff/envs/reasonseg/bin/python`。
