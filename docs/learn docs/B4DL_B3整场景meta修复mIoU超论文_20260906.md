# B4DL mIoU 对齐论文：官方模型评测 → meta 渲染 bug 定位 → B3 超论文（2026-09-06）

## 一、结论速览

**B3（整场景输入 + 修复版 metatoken）mIoU 0.3467，超过论文 0.311（+0.036）**，是"复现论文整场景真实定位能力"目标的闭环。决定性改动只有一处：`ego_text.py` 的 metatoken 渲染从"first frame 恒 at the starting position + 前向差分"修复为论文 §4.1 的 **relative-to-previous（相对前一帧）** 语义。同配置下 mIoU：B2（旧 meta）0.1992 → B3（修复 meta）0.3467，**+0.147**。

## 二、完整调查链（2026-08-31 ~ 09-05）

### 2.1 起点：B1 的 mIoU 错位分析（08-31）

- B1（seqv3 序列切片输入）TG 预测形态正常（99.9% 可解析、区间长度 11.0 vs GT 10.4），但 49.3% 与 GT 零重叠；
- **根因**：预测用"输入窗口内从 0 数起的局部帧号"，评测按场景全局帧号解析 → 系统性错位。`corr(pred_start − gt_start, −序列起点) = 0.892`；预测帧号加回序列起点后 mIoU 0.2655 → 0.6202（42% 完全命中）——模型窗口内定位很强，缺"窗口全局位置"信息。

### 2.2 方案 A（B2）：整场景输入 → 反证（09-01 ~ 09-02）

- 假设：官方实现（ccho4702/B4DL）训练/评测输入均为整场景 (40,768) 特征、无切片 → 切片是偏离论文之源 → 改回整场景（`--whole_scene` 门控，commit `21a5d4d`）重训 B2；
- **结果反证假设**：B2 mIoU **0.1992**（比 B1 0.2653 更差）——61% 预测塌缩到 "from frame 000 to frame 008"（训练集该区间仅占 9.4%）= 纯先验拟合。40 帧无帧号锚点输入下模型学不会绝对定位。
- 附带验证：模型其实具备帧号能力（description 任务 50.3% 答案正确跟随问题帧号："Describe at frame 008" → "At frame 008..."），缺的是 TG 需要的"视觉内容→绝对帧号"反向定位。

### 2.3 官方模型评测：论文完整模型从未发布（09-02，决定性）

用官方仓库自带 checkpoint（stage1 projector + stage2 LoRA，68695 simple 数据 × 2 epochs = 1074 步，与官方脚本完全吻合）+ 官方特征直接评测：

- **官方 stage2 训练数据无 meta、无 <4DLiDAR>**（stage1/2/3 全部 JSON 均只有 `<video>/<image>` + 问题）→ 官方发布模型 = 纯视觉训练；
- 纯视觉输入评测：**acc 0.7542 / mIoU 0.1737**，TG 预测 99.8% 塌缩 "from frame 000 to frame 008"；
- 对应论文 Table 4 **"无 Metatoken"行**（acc 0.763 / mIoU 0.161）→ **官方从未发布 Table 3 的完整模型（HA+Metatoken，mIoU 0.311），也从未发布带 meta 的训练数据**；
- 推论：论文 0.311 依赖训练数据中的 meta 注入（Table 4 证明 meta 值 +0.15 mIoU），我们自研 meta（B0/B1 0.27）已超官方发布模型，方向正确但渲染有 bug。

### 2.4 meta 渲染 bug 定位与修复（09-02，commit `4413b6b`）

- **Bug**：`ego_text.py render_meta_texts` 对 first frame 恒输出 "at the starting position"（3000/3000 TG 样本位置短语雷同，实锤）+ 运动用前向差分（spd_next/yaw_next）；
- **论文定义**（§4.1）：metatoken = "relative direction, position, velocity, and acceleration **with respect to previous frames**"；Figure 6 样例位置是 "slightly ahead and to the right"（随帧变化，非恒定）；
- **修复**：first>0 时改用 spd_prev/yaw_prev/acc_prev + 相对前一帧位移方向（帧 0 保留 fallback）；last frame 同改相对前一帧（原为相对范围首帧）。训练/评测双侧共享 `ego_text.py`，改一处全部生效；
- **重注入**：`scripts/re_render_meta.py` 按 inject_metatoken 同规则（问题帧号 / TG 用 GT 答案帧）重渲染 148k 中 113,053 条可定位样本的 meta 段（35,218 条无帧号 fallback 保留）→ `stage2_full_train_seqv3_meta2_148k.json`；
- 验证：同一场景不同帧对渲染出 "slightly ahead / nearly at the same position / ahead" 等随帧变化的真实运动描述（与论文 Figure 6 语义一致）。

### 2.5 特征对照探针（辅助证据）

- 官方与我们特征同场景逐帧余弦相似度仅 0.07-0.11（几乎正交，两编码器特征空间本质不同）；
- 相邻帧距离：官方 0.057 vs 我们 0.187（3.3 倍）——我们 anneal 编码器特征更"跳"；GT 起点处两套特征均无显著事件边界信号（z≈0）——事件不以"边界突变"编码；
- 结论：特征差异真实存在但非 mIoU 首要瓶颈（B3 用同特征仅修 meta 即超论文，证明 meta 语义才是关键变量）。

## 三、B3 实验（整场景 + meta2）

### 3.1 配方

- 与 B2 逐项一致，唯一变量 = 训练数据 meta 换修复版（`stage2_full_train_seqv3_meta2_148k.json`）：整场景 39/40/41 帧输入（`--whole_scene`）、148,271 条、3 epochs / 3,474 步、lr 1e-4、LoRA r64/α128、bs8×16、ZeRO-3；
- 流程：9-3 18:54 pipeline 重启（此前 9-3 曾放行训练至 checkpoint-200 后被 CoRViD/xmuda 抢占中断，空壳 checkpoint-400 阻塞续训已清除，commit `448ea7a`）；门控等待 25h 后 9-4 19:44 从 checkpoint-200 续训，一次成功跑完（~18h）；9-5 13:59 训练完成 → 评测 rc=0 → 17:28 全链完成。

### 3.2 结果 vs 论文 vs B0/B1/B2

| 指标 | 论文 | B0 | B1 | B2 | **B3** |
|---|---|---|---|---|---|
| accuracy | 0.762 | 0.7629 | 0.7787 | 0.7649 | **0.7526** |
| mIoU | 0.311 | 0.2696 | 0.2653 | 0.1992 | **0.3467** (+0.036 vs 论文) |
| bleu4 | 0.095 | 0.0973 | 0.0994 | 0.0959 | 0.0965 |
| rouge_l | 0.322 | 0.3244 | 0.3272 | 0.3229 | 0.3234 |
| meteor | 0.275 | 0.1729 | 0.1750 | 0.1747 | 0.1747 |
| bertscore | 0.897 | 0.8973 | 0.8979 | 0.8967 | 0.8967 |

per-task（B3）：existence 0.6761 / binary_qa 0.8291 / TG mIoU 0.3467 / description bleu4 0.1055 / temporal bleu4 0.0853 / comprehensive bleu4 0.0987。

### 3.3 TG 预测形态（真实定位确认）

- 解析率 99.9%；最大单一预测占比 B2 61% → B3 23%（012-020，训练集第三常见区间）；
- 预测 start 分布与 GT 分布形状对齐（0-4: 28% vs 21%，5-9: 24% vs 18%，10-14: 26% vs 17%…），模型学会全局帧定位；
- 残余偏置：整体偏早 mean −3.9 帧（std 10.2）；高帧段覆盖不足（GT 25-38 占 14%，预测仅 4%，30+ 为 0）——先验残留而非塌缩，仍有提升空间。

## 四、结论与归档

1. **mIoU 0.3467 超论文 0.311**：整场景 + 论文语义 metatoken（relative-to-previous）是论文完整配置的最忠实复现，已达并超过论文水平；
2. **meta 渲染语义是 mIoU 的第一瓶颈**（+0.147），大于切片/整场景输入构造的影响（B1 vs B2 仅 ±0.06）；
3. **官方仓库不可复现论文 Table 3**：官方数据无 meta、发布 checkpoint 是无 meta 弱配置（0.161-0.174 级）——我们的 meta 自渲染方案是必要且正确的；
4. 已知残余问题：
   - existence acc 0.676 比 B0/B1 低 ~0.05（整场景输入下简单任务略降，论文 acc 0.762 同为此设置，差距 ~0.01 内）；binary_qa 0.829 正常；
   - meteor 0.175 vs 论文 0.275 为 B0-B3 一致的口径问题（Meteor-1.5 实现口径存疑，与 meta/输入无关）；
   - TG 预测仍有 −3.9 帧偏早与高帧段欠覆盖（可作后续优化方向：如 meta 措辞、数据重平衡）。

### 关键 commit

- `21a5d4d` B2 整场景门控（dataset.py / test_b4dl.py --whole_scene）
- `4413b6b` **ego_text.py relative-to-previous 修复 + re_render_meta.py**
- `adb0d8f` B3 训练脚本（meta2 数据）
- `448ea7a` b3 LOG 名修复 + 清空壳 checkpoint

### 产物路径

- 模型：`mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3/`
- 训练日志：`mllm/training_logs/stage2_full_seqv3_mixed_b3_*.log`、`b3_pipeline.log`
- 评测：`mllm/eval_results/stage2_full_seqv3_mixed_b3/`（predictions/metrics/eval_log）
- 数据：`mllm/b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json`（gitignore）
- 官方模型评测对照：`mllm/eval_results/official_ckpt_nometa/`（纯视觉 0.1737）
