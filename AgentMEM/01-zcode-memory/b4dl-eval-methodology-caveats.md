---
name: b4dl-eval-methodology-caveats
description: 评测口径的坑——NLTK vs pycocoevalcap、BERTScore OOM、skip 不计分母、随机替换样本、ckpt 断点续跑
metadata:
  node_type: memory
  type: project
  originSessionId: sess_57bc4f46-cf94-4f89-9779-568442660f70
---

B4DL 评测（`mllm/evaluation/evaluate_model.py` + `test_b4dl.py`）的方法学陷阱，对比论文 Table 3 前必查：

- **BLEU/METEOR 口径虚高**：代码用 NLTK `sentence_bleu`(method1 smoothing) + `meteor_score` 逐样本算再平均，而论文标准是 pycocoevalcap corpus-level。**2026-08-29 晚已切换后端并重算 B0（commit 3d2e9db）**：BERTScore 层号 24→17 → **0.8973 精确命中论文 0.897**；BLEU-4 语料级 → **0.0973≈论文 0.095**；METEOR Meteor-1.5 jar → **0.1729 反而显著低于论文 0.275**（NLTK 0.334 虚高、jar 0.173 偏低，论文所用 METEOR 变体未知，是唯一遗留口径考证项）；accuracy/mIoU/ROUGE-L 逐位不变。结果 JSON 现自带 `metric_backend` 字段，B0/B1 对比必须同后端。重算产物 `mllm/eval_results/stage2_full_seqv3_mixed/metrics_recomputed_20260829.json`（eval_results 在 gitignore 内，仅存盘）。全仓审查报告：`docs/learn docs/B4DL_全仓库论文对齐审查_20260829.md`（commit 8e4353e）。
- **METEOR 口径悬案已溯源解决（2026-09-07）**：B4DL 论文引用 [2]=Banerjee & Lavie 2005（ACL W05-0909，PDF 在 /tmp/meteor2005.pdf）——其参数 α0.9(Fmean=10PR/(R+9P))/β3.0/γ0.5、模块 exact+Porter+WordNet 同义、无 paraphrase，**正是 NLTK meteor_score 的实现**（数学逐项等价）；Meteor-1.5 jar rank（0.85/0.2/0.6/0.75+paraphrase）是 2014 WMT 重调参，非 [2] 内容。NLTK-2005 口径补算 B0-B3=0.3344/0.3371/0.3362/0.3366 **全部超论文 0.275**（B3 +0.062），jar 口径 0.1729-0.1750 保留作参考——METEOR 不再孤立偏低，与 BLEU/ROUGE/BERTScore 命中论文自洽。**评测代码已双后端化**（evaluate_model.py `B4DLEvaluator(meteor_backend='dual')` 默认）：`meteor`=NLTK-2005 主口径（论文对比用），`meteor_pycocoevalcap`=jar（与 B0-B3 旧表衔接）；离线补算脚本 `mllm/evaluation/recompute_dual_meteor.py`（零 GPU，断言 d=0.0000 通过）；规则文档 `docs/learn docs/B4DL_METEOR双口径溯源与评测规则_20260907.md`。
- **METEOR 变体敏感性已量化（2026-09-07 B0 预测实测，16,067 文本段）**：同一份预测不同变体 → **0.141~0.381**，论文 0.275 落在区间内但无标准变体精确命中：jar rank 默认（冻结口径）stdio 0.1729 / batch 0.1761 / 逐段均值 0.1801；jar adq 调参 0.3172；jar hter 0.3810；去 paraphrase 0.1686；仅 exact 0.1564；无 -norm 0.1409；NLTK 式参数 (0.9/3.0/0.5/0.75) 全模块 0.3463、去 paraphrase 0.3227（≈NLTK 0.334，验证参数映射正确）；NLTK+分词小写逐样本 0.3331。主因：rank 调参 beta=0.2 的 fragmentation 惩罚重，B4DL 答案多为短句/类别词/帧号，措辞稍异即多 chunk，惩罚被放大；NLTK beta=3.0 惩罚几乎消失故虚高。扫表脚本 `mllm/eval_results/_meteor_sweep/{sweep,sweep2}.py`（含 hyp/ref 导出）。BLEU-4/BERTScore/ROUGE-L 同预测精确命中论文 → **METEOR 差距确证为口径非模型能力**。
- **skip 不计分母**：test_b4dl.py 遇缺 .npy 特征或推理异常直接 skip，不写入 predictions，测试集静默缩水；查日志 "Skipped N items" 才知道。
- **训练 loader 静默替换**：`dataset.py` 缺 .npy 时 `random.choice(self)` 换随机样本，batch 不变但分布被悄悄改变、无任何记录。
- **断点续跑**：预测每 50 条写 `<output>.ckpt`，中断后用相同 `--output` 重跑即 resume（日志会打印 "resuming from N samples"）；正常结束自动删 ckpt。冒烟测试用 `--max_samples 50`。
- **BERTScore 易 OOM**：GPU 被其他进程占用时报 "BERTScore computation failed: CUDA out of memory" 并置 0（日志里 bertscore 0.0 但 metrics json 可能存了重算后的 0.97，见到不一致先查 OOM）。
- GPT score 需 `--use_gpt` + API key，默认 0。
- **推理解码随机性**：inference/评测用 do_sample=True + temperature=0.05（VTimeLLM 默认），近贪心会放大模式坍缩且引入不可复现的采样噪声；对比实验建议 do_sample=False。
- GPT 评分 prompt 已含论文 Table 9 全部 4 个打分示例（90/100/50/10，commit d3af130 补齐低分锚点，缺低分锚点会令 GPT Score 偏高）；bert-score `lang='en'` 默认模型就是 roberta-large（README 写的 deberta-xlarge-mnli 是早期默认，不是 BERTScore 偏差来源）。
- 无数据平衡：loader 纯顺序索引，无 WeightedRandomSampler/group_by_length，任何"平衡采样"方案都要改 `make_supervised_data_module`。

**TG mIoU 口径锁定（2026-08-28 审查确认）**：官方 B4DL 仓库（ccho4702/B4DL, commit a0104ca）**没有 B4DL 六任务评测代码**——`mllm/vtimellm/eval/{eval,metric}.py` 是 VTimeLLM 视频基准（ActivityNet/DVC/SODA，`iou()` 半开区间+归一化 0-1+逐条 round2，与 B4DL 帧号无关）；HF 数据集（ccho4702/nuScenes-B4DL）也只有数据。论文仅 §5.1 "mIoU is based on the Time Grounding task"、Table 8 帧格式 "from frame 000 to frame 000"（三位数整数帧）、Table 10 证实每条 QA 本有 start/end index 序列归属（发布版丢失）。因此 0.311 无法与官方实现直接比对，只能口径声明+敏感性量化。实测敏感性（2783 条预测）：闭区间整数帧 IoU 0.2696（当前实现）、半开区间 0.2547（差 −0.015）、逐条 round2 0.2698（≈0）、解析失败 2/2783。论文 Table 6 数据缩放旁证 TG 对数据最敏感（10%→100%: 0.168→0.311）。注意：闭区间 IoU 对"完全不相交区间"必须归零（`inter<=0` 跳过），否则负 IoU 会把均值从 0.2696 拉到 0.1143（复算陷阱，已踩）。

**本机 clip.load 实测（2026-08-28）**：`clip.load('ViT-L/14', jit=False)` 返回的权重**已是 fp16**（conv1.weight.dtype=float16），且模型无 `visual.dtype` 属性（非官方 CLIP 结构）——`clip.model.convert_weights()` 是幂等 no-op。train.py 里"本机 clip.load 不转 fp16"的注释与实测不符（无实质危害，但会误导）。

相关：[[b4dl-training-eval-history]]、[[b4dl-per-sequence-refactor]]
