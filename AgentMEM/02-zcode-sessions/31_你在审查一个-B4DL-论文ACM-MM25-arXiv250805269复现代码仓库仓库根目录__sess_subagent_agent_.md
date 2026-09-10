# 你在审查一个 B4DL 论文（ACM MM'25, arXiv:2508.05269）复现代码仓库，仓库根目录：/...

| 项 | 值 |
|---|---|
| 会话 ID | `sess_subagent_agent_c89b0232-4771-44fa-b039-a06e41bef415` |
| 工作目录 | `/root/autodl-tmp/wql/mmb4dl` |
| 父会话 | `sess_5c6e22af-4ead-4c47-a935-c6ba054b8d75` |
| 时间 | 2026-08-29 19:20 → 2026-08-29 19:20 |
| 模型 | GLM-5.3-Flash |
| 消息数 | 3（文本块 1，工具调用 0） |

---

### ASSISTANT  ·  `GLM-5.3-Flash`

> 🔀 模型切换：None → builtin:bigmodel-start-plan/GLM-5.3-Flash


### USER

你在审查一个 B4DL 论文（ACM MM'25, arXiv:2508.05269）复现代码仓库，仓库根目录：/root/autodl-tmp/wql/mmb4dl。你的任务：审查 `mllm/evaluation/` 评测实现，逐项对照论文的评测协议。搜索广度：very thorough。

论文基准事实（用于对齐）：
1. §3.1 指标定义：Simple Tasks 用 accuracy（Existence 与 Binary QA），基于「exact answer matching」；Time Grounding 用 mIoU（"measure overlap between predicted and ground-truth temporal segments"）；Complex Tasks（Description / Temporal Understanding / Comprehensive Reasoning）用 BLEU-4 (B@4)、METEOR、ROUGE-L、BERTScore；另用 GPT-4o 做 reference-free 评估，0-100 分（coherence/relevance/correctness）。
2. §5.1 聚合方式："each metric is computed per task and then averaged to obtain the final score. Specifically, the accuracy of Simple Tasks is the average of Existence and Binary QA scores. mIoU is based on the Time Grounding task, while the remaining four metrics come from Complex Tasks."
3. 答案格式：Existence 答案是大写类别词（"MOTORCYCLE"/"CAR"）；Binary 是 "Yes."/"No."；Time Grounding 是 "from frame 018 to frame 020."。
4. Table 9 的 GPT 评分 prompt（原文关键句）："You are an expert evaluator for semantic answer quality. You will be given a set of question-answer-ground truth (Q/A/GT) triplets..." 评分档位 100/80-99/60-79/40-59/20-39/0-19，带 4 个 few-shot 示例，结尾 "Question: {q} GT: {ref} Answer: {pred} Please provide a score between 0 and 100... Only provide the score without any additional text."
5. 论文参考结果（Table 3，B4DL(Ours)）：Accuracy 0.762 / mIoU 0.311 / B@4 0.095 / ROUGE-L 0.322 / METEOR 0.275 / BERTScore 0.897 / GPT Score 59.513。
6. 复现已报告结果（B0 基线 seqv3-mixed）：accuracy 0.7629 / mIoU 0.2696 / BLEU-4 0.1080 / METEOR 0.3344 / ROUGE-L 0.3244 / BERTScore 0.9734 / GPT 缺失。注意 METEOR 和 BERTScore 与论文差异较大（0.3344 vs 0.275；0.9734 vs 0.897），需要从实现上找原因（不同库/参数会导致不可比）。

审查对象（仓库根目录下）：
- mllm/evaluation/test_b4dl.py：推理流程——prompt 构造（<4DLiDAR>/<meta> 注入）、--per_sequence 特征切片、--frame_motion、--answer_frames（TG 用 GT 恢复序列归属，oracle 声明）、解码策略（greedy?）、6 任务如何路由与答案抽取（exact match 前后处理：大小写、句号、大写类别词匹配？binary 的 yes/no 解析？TG 帧段解析？）。
- mllm/evaluation/evaluate_model.py：各指标实现——mIoU（闭区间修正是否在：inter = min(pe,ge)-max(ps,gs)+1）；BLEU-4 用什么库（nltk? pycocoevalcap? tokenize 方式）；METEOR 用什么库；ROUGE-L 用什么库；BERTScore 用什么包、什么模型（bert-base-uncased? roberta-large? rescale_with_baseline? language 参数）；compute_final_score 聚合（accuracy=mean(existence,binary)? 其余 4 指标对复杂任务平均?）；GPT_EVAL_PROMPT 是否与 Table 9 一致（逐句比对，摘录差异）。
- mllm/eval_results/stage2_full_seqv3_mixed/metrics.json（若存在）：读出实际各任务分项指标（existence acc、binary acc、TG mIoU、各复杂任务 4 指标），验证聚合公式与最终数。
- 检查 requirements（requirements_sum/requirements_b4dl.txt 或 mllm 相关）里 bert-score / nltk / pycocoevalcap / rouge 的版本。
- mllm/evaluation/README.md 若有，看评测口径说明。

输出要求（中文）：逐项判定 ✅一致 / ⚠️偏差 / ❌不一致 / ➖论文未规定，每项附 file:line 证据与代码摘录。特别回答：①exact match 的实现细节与论文「exact answer matching」的贴合度（宽松/严格程度、会否虚高或虚低）；②mIoU 语义（闭区间、无解析处置）；③BERTScore/METEOR 的实现与论文可比性（这是重点——解释 0.9734 vs 0.897 与 0.3344 vs 0.275 的差异来源，给出实现参数）；④GPT 评分是否缺失（已知 P1 缺失，确认现状）；⑤聚合公式与论文一致性（用 metrics.json 实测数验算一遍）。最后给总体结论。


### ASSISTANT  ·  `GLM-5.3-Flash`

