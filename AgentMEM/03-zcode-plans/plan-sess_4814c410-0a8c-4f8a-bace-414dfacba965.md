## 目标
让新训练（B4 系列起）的评测在 METEOR 上采用**双后端并存**：主口径为 NLTK-2005（忠实论文引用 [2] Banerjee & Lavie 2005：α0.9/β3.0/γ0.5、exact+porter+WordNet 同义、无 paraphrase、逐样本平均），参考口径为 Meteor-1.5 jar（与 B0-B3 既有数值衔接）。BLEU-4/ROUGE-L/BERTScore/accuracy/mIoU 口径冻结不动。

## 1. 改造 mllm/evaluation/evaluate_model.py
- `__init__`（:210-248）：新增参数 `meteor_backend='dual'`（可选 'dual'|'pycocoevalcap'|'nltk'；dual 下主口径=nltk-2005）。
- `metric_backend['meteor']`（:247）：单字符串改为结构化记录——主口径标识、NLTK 参数与模块（alpha/beta/gamma=0.9/3.0/0.5、[exact, porter, wordnet-synonymy]、无 paraphrase、聚合=逐样本均值、nltk 版本、出处注明 "Banerjee & Lavie 2005 = B4DL 论文引用 [2]"）、jar 参数（rank 0.85/0.2/0.6/0.75、含 paraphrase、EVAL corpus 聚合），旧字符串值保留在子字段兼容。
- `compute_meteor`（:373-401）重构为 `_meteor_pycocoevalcap`（现 jar 逻辑原样迁入）与 `_meteor_nltk_2005`（现 NLTK fallback 逻辑原样迁入：word_tokenize(lower)+逐样本 meteor_score 平均，docstring 注明出处），`compute_meteor` 按模式返回主值。
- `evaluate_task`（:488-496）：complex 任务 dual 模式下写 `meteor`（NLTK-2005 主值）+ `meteor_pycocoevalcap`（jar 参考）。
- `compute_final_score`（:522-546）：**零改动**——per_metric 自动平均所有键，final_scores 自动多出 `meteor_pycocoevalcap`。
- CLI `main()`（:683-698）：加 `--meteor_backend`（默认 dual）并透传。

## 2. mllm/evaluation/test_b4dl.py：零改动
它只调 `evaluate_all + save_results`，默认 dual 自动生效；评测命令与历史逐字一致。

## 3. 新增离线补算脚本 mllm/evaluation/recompute_dual_meteor.py（零 GPU）
- 对 eval_results/stage2_full_seqv3_mixed{,_b1,_b2,_b3} 的 predictions.json 用 `B4DLEvaluator(meteor_backend='dual')` 重算，写 `metrics_recomputed_20260907_dual.json`（不动原 metrics.json）。
- 断言：B0 的 NLTK final ≈0.3344±0.001（复现旧值）、jar ≈0.1729±0.001（复现 8/29 重算值），不符则告警退出非零。
- 打印 B0-B3 双口径汇总表。

## 4. 文档更新（补算结果出来后）
- docs/learn docs/B4DL_工作进度总结_20260906.md：METEOR 行与口径段改为双口径叙述（NLTK-2005 0.33x 高于论文 0.275；jar 0.17x 为参考），写入 [2] 溯源依据。
- 同目录新增/更新口径规则：B4 起 JSON 自带 meteor（主，NLTK-2005）与 meteor_pycocoevalcap；论文对比用 NLTK-2005，与 B0-B3 衔接用 jar 值。
- CLAUDE.md 评测节补一句双口径说明；同步更新项目记忆文件。

## 5. 验证
- evaluate_model CLI `--demo --meteor_backend dual` 冒烟（零 GPU）。
- 补算脚本断言通过。
- 完成后按仓库惯例中文 commit + push（feat: METEOR 双后端…）。

## 风险与备注
- wqlc 离线环境需有 wordnet/omw-1.4 语料缓存；若 LookupError 则补 nltk_data 路径（evaluate_model 有自动下载逻辑兜底）。
- dual 模式每次评测多约 5-10 分钟（3 个 complex 任务 NLTK 逐样本含 WordNet 查询），可接受。
- 不改历史 metrics.json 原文件、不动推理与训练代码。