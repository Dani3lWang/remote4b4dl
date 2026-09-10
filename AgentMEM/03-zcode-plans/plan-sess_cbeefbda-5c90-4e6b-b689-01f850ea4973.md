# 并行执行：Phase 1（B4a 过采样实验）+ 全量工作进度总结

> **B4a 是什么**：B 系列实验的下一个（B0 基线→B1 切片→B2 整场景→B3 修复 meta=当前最优 0.3467）。B4a = 把训练集中 GT 起始帧≥25 的 1,951 条 TG 样本复制一份（频率×2，148k→150k），其余与 B3 完全一致，针对 B3 的 TG 高帧段欠覆盖（GT 占 15% vs 预测仅 4%）与 −3.9 帧偏早。成本：20 行脚本 + 1 次重训（~18h，等 GPU 窗口）+ 自动评测；判定：mIoU Δ>0.013 且高帧段覆盖改善、六任务无回退。

## 执行顺序（两件事并行，B4a 启动后即挂门控等待，总结文档填等待期）

1. **写并运行过采样脚本** `mllm/scripts/oversample_tg_highframe.py`（CPU，几分钟）：过滤 `task=time_grounding` 且 GT start≥25（断言 1,951 条），deepcopy 追加 → `stage2_full_train_seqv3_meta2_oversampled_150k.json`（150,222 条）；抽查验证。
2. **写 B4a 训练脚本** `mllm/scripts/run_stage2_full_seqv3_mixed_b4a.sh`：复制 B3 仅改数据路径、OUT=b4a、LOG 名；grep 查无 b3 残留。
3. **写 B4a 管线** `mllm/scripts/run_b4a_pipeline.sh`：28GB 门控 72h（432×10min）、成功判据 trainer_state epoch≥2.99、三次断点续训、评测冻结命令 `--whole_scene --per_sequence --answer_frames`、输出 `eval_results/stage2_full_seqv3_mixed_b4a/`。
4. **tmux `b4apipeline` 后台启动**（门控等待 GPU，不干预 xmuda）→ 转 B 部分。
5. **写进度总结** `docs/learn docs/B4DL_工作进度总结_20260906.md`：B0→B3 复现闭环时间线与六指标总表（B3 mIoU 0.3467 超论文）；编码器侧（LiDAR-CLIP nuScenes 自训退火定稿 + 28,130+850 特征重提）；评测方法学现状（BERTScore/BLEU 已对齐、METEOR 0.175 遗留、GPT-4o 待补）；会议待办逐条对照；残余问题与 B4 路线图（B4a 已启动 → B4b 措辞 → RL 阶梯）。
6. **git**：中文 commit（feat: B4a 脚本与数据；docs: 进度总结）+ push。

## 产出

- 过采样脚本 + 150,222 条数据（gitignore）；B4a 训练/管线脚本；tmux 后台 B4a 全链
- 进度总结文档（归档 docs/learn docs/）
- 聊天报告：全量进度总结 + B4a 启动状态