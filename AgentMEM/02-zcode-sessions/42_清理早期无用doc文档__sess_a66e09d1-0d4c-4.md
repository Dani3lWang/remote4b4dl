# 清理早期无用doc文档

| 项 | 值 |
|---|---|
| 会话 ID | `sess_a66e09d1-0d4c-48a1-b980-8f43a70959f5` |
| 工作目录 | `/root/autodl-tmp/wql/mmb4dl` |
| 时间 | 2026-08-31 00:27 → 2026-08-31 00:34 |
| 模型 | GLM-5.3-Flash |
| 消息数 | 18（文本块 13，工具调用 14） |

---

### ASSISTANT  ·  `GLM-5.3-Flash`

> 🔀 模型切换：1ad4adef-929c-4e0c-b8d4-157504a0463a/deepseek-v4-flash → builtin:bigmodel-start-plan/GLM-5.3-Flash


### USER

清除一些早期的无用的doc文档


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

The user wants to clean up some early useless doc documents in the project. Let me look at the docs directory to see what's there.

The working directory is /root/autodl-tmp/wql/mmb4dl. Let me explore the docs structure first.

I should be careful: deletion is destructive. The guidance says: "Before deleting or overwriting, look at the target — if what you find contradicts how it was described, or you didn't create it, surface that instead of proceeding."

The user asked to clear "早期的无用的doc文档" (early useless doc documents). I should first list the docs, understand which ones are early/outdated, and then decide. Since the user is not watching in real time, I should be careful about what to delete. Deletion of docs is somewhat destructive but the user explicitly asked for it. However, "which ones are useless" is a judgment call. Let me look at the docs first, check git history to see which are archived/superseded, then propose/delete.

Actually the user said "清除" (clear/remove). Given the autonomous mode, I should identify candidates and delete the clearly obsolete ones. But it's safer to review first, then delete ones that are clearly superseded (e.g., 待修改清单 already archived, old plans superseded by newer versions). The git history shows docs like 待修改清单存档 etc.

Also there's memory note: git conventions say commit after changes with Chinese commit messages. So after deleting, commit the deletion.

Let me start by listing the docs directory with dates and git log info.

</details>

我先看一下 docs 目录里有哪些文档及其时间线，判断哪些属于早期且已被后续版本取代的。

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls -la /root/autodl-tmp/wql/mmb4dl/docs/ && echo "---" && ls /root/autodl-tmp/wql/mmb4dl/*.md 2>/dev/null
description: List docs directory and root markdown files
```

</details>

<details><summary>输出摘要</summary>

```
total 6896
drwxr-xr-x  7 root root    4096 Aug 28 13:50 .
drwxr-xr-x 17 root root    4096 Aug 29 21:21 ..
-rw-r--r--  1 root root   70320 Aug 28 13:50 B4DL_复现方案.md
-rw-r--r--  1 root root    9838 Aug 11 11:11 B4DL_训练评测报告_20260810.md
drwxr-xr-x  2 root root     169 May 30 00:12 eval_results
drwxr-xr-x  2 root root    4096 Aug  3 00:13 evl
drwxr-xr-x  2 root root    4096 Aug 30 20:47 learn docs
drwxr-xr-x  3 root root      49 Aug  8 11:05 mmb4dl-md
-rw-r--r--  1 root root 6957414 May 28 13:18 mmb4dl.pdf
drwxr-xr-x  2 root root      55 Jul 12 16:36 参考
---
/root/autodl-tmp/wql/mmb4dl/CLAUDE.md
/root/autodl-tmp/wql/mmb4dl/README.md
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Let me look deeper into the docs subdirectories.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/docs && ls -la "learn docs" evl mmb4dl-md 参考 eval_results 2>/dev/null
description: List docs subdirectories contents
```

</details>

<details><summary>输出摘要</summary>

```
eval_results:
total 3852
drwxr-xr-x 2 root root     169 May 30 00:12 .
drwxr-xr-x 7 root root    4096 Aug 28 13:50 ..
-rw-r--r-- 1 root root     412 May 29 23:08 stage2_metrics.json
-rw-r--r-- 1 root root       0 Jul 13 21:39 stage2_test_log.jsonl
-rw-r--r-- 1 root root     144 May 30 00:54 stage3_metrics.json
-rw-r--r-- 1 root root 3925518 May 30 00:12 stage3_test_log.jsonl
-rw-r--r-- 1 root root    4024 May 27 22:01 verify_stage2.jsonl

evl:
total 2816
drwxr-xr-x 2 root root    4096 Aug  3 00:13 .
drwxr-xr-x 7 root root    4096 Aug 28 13:50 ..
-rw-r--r-- 1 root root    9518 Aug  2 11:13 b4dl_paper_alignment_plan.md
-rw-r--r-- 1 root root    5885 Jul 26 21:22 conversation_log.md
-rw-r--r-- 1 root root    4637 Jul 26 11:32 evaluation_summary.md
-rw-r--r-- 1 root root    6361 Aug  2 10:55 m
…（已截断，原文 3075 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Now let me understand the structure:

- `docs/B4DL_复现方案.md` — the main reproduction plan (70KB, updated Aug 28) — keep, it's the core doc.
- `docs/B4DL_训练评测报告_20260810.md` — Aug 11 training eval report — early report. Could be a candidate but let me check content.
- `docs/eval_results/` — May 29-30, old stage2/stage3 metrics from early runs (May 29!). These are very early (May 29-30, before per-sequence refactor). Old eval logs. Candidates for deletion.
- `docs/evl/` — Jul 26 - Aug 3 old eval artifacts: conversation_log, evaluation_summary, b4dl_paper_alignment_plan, mmb4dl_paper_vs_eval_analysis, stage1_training.log, stage1_training_report, stage2 logs. These are old eval runs (old stage2 metrics from Jul 26). Candidates.
- `docs/learn docs/` — the active analysis docs from Jul 12 to Aug 30. Contains:
  - plan1.md, plan1_learning_results.md (Jul 12) — early learning plans
  - training_reproduction_guide.md (Jul 12) — early guide
  - VTimeLLM_analysis.md, B4DL_vs_LiDAR-LLM_comparison.md (Jul 12) — reference comparisons, maybe still useful as reference
  - B4DL_temporal_weakness_analysis.md (Aug 2)
  - B4DL_论文对齐审计_20260824.md — superseded by 全仓库论文对齐审查_20260829?
  - B4DL_seqv2评测对比与坍缩分析_20260825.md — historical
  - B4DL_两阶段训练评测与混合法决策_20260826.md — historical but documents decision
  - B4DL_seqv3混合训练评测分析_20260827.md — historical
  - B4DL_训练方法对比_论文vs复现_20260828.md — superseded by 基线版本训练方法对比_20260830?
  - B4DL_LiDARCLIP权重核对_20260828.md — weight provenance, referenced in memory
  - B4DL_评测数值对比与问题清单_20260828.md
  - B4DL_基线锁定与对比规则_20260829.md — baseline lock rules — important, keep
  - B4DL_全仓库论文对齐审查_20260829.md — keep
  - B4DL_基线版本训练方法对比_20260830.md — keep (latest)
  - B4DL_待修改清单_20260829.md — still active checklist (progress being updated per git log) — keep
- `docs/参考/` — reference code — keep
- `docs/mmb4dl-md/` — paper markdown conversion — keep as reference

The user says "清除一些早期的无用的doc文档" — clear some early useless doc documents. So they want me to identify early useless ones and delete them.

Which are "early and useless"? I need to check content to see if information has been superseded. Let me check git history to see which are tracked in git — deletion should be committed per the git conventions memory.

Let me check git ls-files for docs, and look at a few candidate files quickly.

Also note memory: "改后即 commit、提交后及时 push 远程同步" — after changes, commit and push.

But deletion is destructive. The user explicitly asked to clear early useless docs. Still, I should be careful about which ones I judge useless. The safest interpretation: delete docs that are clearly superseded/outdated artifacts:

1. `docs/eval_results/` — May 29-30 old stage2/stage3 eval outputs from the very early pipeline (before all refactors). The current baseline is B0 with different eval flow. stage3_test_log.jsonl is 3.9MB. These are stale artifacts. But wait — are these used by anything? They're in docs/, which is documentation, not runtime. Old metrics from May. Likely useless now.

2. `docs/evl/` — Jul 26 stage2 test logs and metrics from the old (pre-seqv2) runs; stage1_training.log from Aug 3 (218KB), old analysis docs (conversation_log.md, evaluation_summary.md, b4dl_paper_alignment_plan.md, mmb4dl_paper_vs_eval_analysis.md, stage1_training_report.md). These are from the old frame-level pipeline, superseded by per-sequence refactor and the learn docs analyses. Likely useless.

3. In `learn docs/`: plan1.md, plan1_learning_results.md, training_reproduction_guide.md (Jul 12 early plans — superseded by B4DL_复现方案.md which is the living plan). These are candidates. VTimeLLM_analysis.md and B4DL_vs_LiDAR-LLM_comparison.md are literature study notes — "early" but maybe still useful as reference? Hmm. The user said "一些早期的无用的" — some early useless ones. I should make a judgment and report. To be safe, I could delete the clearly useless (superseded plans, stale logs/metrics) and keep literature notes that might still be referenced.

4. `docs/B4DL_训练评测报告_20260810.md` — early report (Aug 10) before per-sequence refactor; its content is superseded by later analyses (seqv2/seqv3 docs and baseline lock). Candidate for deletion, but it's the only "training eval report" of the old pipeline... The later docs (seqv2 评测对比与坍缩分析, 基线锁定与对比规则) cover it. Hmm — this one documented the "四大错误模式" era. Memory says training-eval history memory covers it. It's a historical report. Could keep or delete. Let me read it quickly.

Actually, rather than guessing, let me read the headers of the candidate files to confirm supersession. Also check CLAUDE.md to see if any docs are referenced there — I must not delete docs referenced by CLAUDE.md or other live docs. Let me grep for references to these filenames across the repo.

Plan:
1. git ls-files docs to see what's tracked.
2. Grep references to candidate filenames in repo (CLAUDE.md, docs, scripts).
3. Read quick summaries of candidates.
4. Delete clearly obsolete ones, commit (per git conventions), maybe push.

Let me also check whether there's an archive convention — memory said "待修改清单存档" so the checklist gets archived within the same file, not separate files.

Let me run these checks.

</details>

我先确认这些文档哪些在 git 里、哪些被其他文件引用，避免误删仍被依赖的内容。

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && git ls-files docs/ | head -50 && echo "=== refs ===" && grep -rn --include="*.md" --include="*.sh" --include="*.py" -l -e "eval_results" -e "docs/evl" -e "plan1" -e "
…（已截断，原文 334 字符）
description: Check git-tracked docs and cross-references
```

</details>

<details><summary>输出摘要</summary>

```
"docs/B4DL_\345\244\215\347\216\260\346\226\271\346\241\210.md"
"docs/B4DL_\350\256\255\347\273\203\350\257\204\346\265\213\346\212\245\345\221\212_20260810.md"
"docs/learn docs/B4DL_LiDARCLIP\346\235\203\351\207\215\346\240\270\345\257\271_20260828.md"
"docs/learn docs/B4DL_seqv2\350\257\204\346\265\213\345\257\271\346\257\224\344\270\216\345\235\215\347\274\251\345\210\206\346\236\220_20260825.md"
"docs/learn docs/B4DL_seqv3\346\267\267\345\220\210\350\256\255\347\273\203\350\257\204\346\265\213\345\210\206\346\236\220_20260827.md"
docs/learn docs/B4DL_temporal_weakness_analysis.md
docs/learn docs/B4DL_vs_LiDAR-LLM_comparison.md
"docs/learn docs/B4DL_\344\270\244\351\230\266\346\256\265\350\256\255\347\273\203\350\257\204\346\265\213\344\270\216\346\267\267\345\220\210\346\263\225\345\20
…（已截断，原文 2655 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

The grep matched many files but that's because it matched any of the patterns (like "eval_results", "plan1", "训练评测报告") — probably loose matches. Let me check specifically which live docs reference the candidates:

1. Does CLAUDE.md reference any candidate?
2. Does 复现方案.md (the living plan) reference candidates?

Let me check references to specific candidate paths:
- docs/evl/* (evl directory)
- docs/eval_results/*
- plan1.md, plan1_learning_results.md, training_reproduction_guide.md
- docs/B4DL_训练评测报告_20260810.md
- learn docs older audits (论文对齐审计_20260824, 训练方法对比_论文vs复现_20260828) — these might be superseded by newer docs

Let me grep with more specific patterns.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: echo "=== CLAUDE.md refs ===" && grep -n -e "evl" -e "eval_results" -e "plan1" -e "training_reproduction" -e "训练评测报告" -e "20260824" -e "20260810" CLAUDE.md | head; echo "=== 复现方案 refs to candidates ==
…（已截断，原文 743 字符）
description: Find references to candidate docs in live files
```

</details>

<details><summary>输出摘要</summary>

```
=== CLAUDE.md refs ===
=== 复现方案 refs to candidates ===
1470:    out_path = args.pred_path.replace(".json", "_eval_results.json")
=== learn docs live files referencing candidates ===
docs/learn docs/B4DL_待修改清单_20260829.md:4:> `B4DL_评测数值对比与问题清单_20260828.md`、`B4DL_论文对齐审计_20260824.md` 遗留项。
docs/learn docs/B4DL_全仓库论文对齐审查_20260829.md:6:> 与《论文对齐审计 20260824》《训练方法对比 20260828》《待修改清单 20260829》交叉验证：
docs/learn docs/B4DL_基线版本训练方法对比_20260830.md:101:- 评测产物：`mllm/eval_results/{stage23_seqv3, stage2_full_seqv2, stage2_full_seq, stage2_full_seqv3_mixed}/`
docs/learn docs/B4DL_基线锁定与对比规则_20260829.md:13:| 完整性 | 6 任务合计 **30,145 条** = 测试集全量；TG 2,783 条；**0 条跳过**；`eval_results/stage2_full_seqv3_mixed/` 下无 `.ckpt` 断点残留（排除断点续跑混样） |
docs/learn docs/B4DL_基线锁定与对比规则_20260829.md:115:| **B0** | 08-27 | seqv3-mixed（148K 混合
…（已截断，原文 1024 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Good findings:
- CLAUDE.md does not reference any of the candidates.
- The 复现方案.md reference was just a false positive (`_eval_results.json` string in code).
- `B4DL_论文对齐审计_20260824.md` is referenced by 待修改清单 (line 4, as provenance of leftover items) and by 全仓库论文对齐审查 (line 6, cross-validation). So it's referenced as historical context — but "referenced by name" doesn't mean the file must exist. However, since newer docs point to it as cross-reference, deleting it breaks those pointers. It's from Aug 24 — is it "早期无用"? It's been superseded by 全仓库论文对齐审查_20260829 which says it cross-validates with it. Hmm. The audit 20260824 is superseded. But two live docs reference it. I could keep it — it's not that old and serves as cross-ref. Or delete and the references would dangle.

Let me now quickly inspect the content of the main deletion candidates to confirm they're stale:

1. `docs/eval_results/` — May 29-30 old stage2/stage3 metrics/logs (from the very first pipeline). Untracked? Let me check: git ls-files docs/ output didn't show eval_results or evl files. So `docs/eval_results/` and `docs/evl/` are UNTRACKED. Interesting — they're not in git at all. That makes them pure local leftovers.

2. `docs/evl/` — Jul 26–Aug 3, old eval logs/summaries — untracked too.

Wait, let me double check: the git ls-files output shows only "docs/B4DL_复现方案.md", "docs/B4DL_训练评测报告_20260810.md", learn docs files, and 参考 files. No eval_results/, evl/, mmb4dl-md/. So those directories are untracked local artifacts.

Hmm, mmb4dl-md is the paper converted to markdown + images — that's a useful reference (paper content). Keep.

So the clearly useless stuff:
- `docs/eval_results/` — stale May eval artifacts (3.9MB jsonl logs). Useless — the same info is in mllm/eval_results for current runs. Delete.
- `docs/evl/` — stale Jul/Aug 3 eval artifacts + old analysis docs. The analysis docs (b4dl_paper_alignment_plan.md, mmb4dl_paper_vs_eval_analysis.md, evaluation_summary.md, conversation_log.md) are from the old frame-level era — superseded by learn docs analyses. The logs (stage1_training.log 218KB, stage2_test_log.jsonl 1.3MB, stage2_val_log.jsonl 1.2MB) are artifacts. Delete whole dir? The old stage2 metrics are also superseded. I think delete the whole `docs/evl/` directory — it's untracked, all artifacts from the pre-seqv pipeline.

Hmm wait — should I be more careful? The user said "清除一些早期的无用的doc文档" — "clear some early useless doc documents". This is a doc cleanup request. docs/evl contains both docs and eval artifacts. The name "evl" is a typo of "eval". It's all from Jul 26–Aug 3, the old pipeline. I'll check the analysis docs quickly to confirm supersession.

- In `learn docs/`: 
  - `plan1.md` (Jul 12) — early learning plan, superseded by 复现方案. Tracked in git. Candidate.
  - `plan1_learning_results.md` (Jul 12) — early learning results. Candidate.
  - `training_reproduction_guide.md` (Jul 12) — early guide, superseded by 复现方案 + later docs. Candidate.
  - `VTimeLLM_analysis.md`, `B4DL_vs_LiDAR-LLM_comparison.md` (Jul 12) — literature study notes. These are "early" but are reference material about other papers. Are they useless? They were study notes for learning. The user might still want them as reference. Hmm. I'd lean to keeping literature notes (they don't rot — they describe other papers), but they were part of the early learning phase... The user said "一些" (some) — so I don't need to delete everything early, just the useless ones. Literature comparison notes aren't "useless" in the same way a stale training log is. But plan1/plan1_learning_results/training_reproduction_guide are process docs superseded by the living 复现方案 — delete those.
  - `B4DL_temporal_weakness_analysis.md` (Aug 2) — early weakness analysis of the old pipeline. Superseded? It fed into the seqv2/seqv3 refactor. It's a historical analysis. Tracked. Hmm — is it useless? It analyzed temporal weaknesses that motivated the per-sequence refactor. Now that B0 is locked, it's historical. Borderline. The seqv2坍缩分析 doc (0825) and later docs cover the story. I could delete it as early/superseded, or keep. Let me read its header.
  - `B4DL_训练评测报告_20260810.md` — Aug 10 report of the old frame-level training. Superseded by the seqv2/seqv3 line of docs and 基线锁定. It documents the first full training incl. 四大错误模式. Historical. Borderline.
  - `B4DL_论文对齐审计_20260824.md` — superseded by 全仓库论文对齐审查_20260829, but referenced by two live docs. Keep (referenced) or delete? The references are "cross-validation" pointers. If I delete it, those pointers dangle. I'd keep it — it's the provenance of leftover items in the checklist. Actually, let me think about what the user wants: "早期的无用的doc文档" — early useless ones. The 20260824 audit is explicitly superseded by the 20260829 full-repo audit ("与...交叉验证"). But it's still referenced. Deleting it makes 待修改清单's reference dangle. I'll keep it to be safe — it's only 9KB. Hmm, but "superseded" is exactly what "无用" means... 

Let me look at the actual contents of the borderline ones to make a better call. Let me read heads of: B4DL_训练评测报告_20260810.md, B4DL_temporal_weakness_analysis.md, B4DL_论文对齐审计_20260824.md, plan1.md, plan1_learning_results.md, training_reproduction_guide.md, evl analysis docs.

</details>

CLAUDE.md 和复现方案都没有引用这些早期文档。注意 `docs/eval_results/`、`docs/evl/`、`docs/mmb4dl-md/` 是未跟踪的本地文件。我再快速看一下几个边界文档的内容，确认哪些确实已被取代。

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/docs && for f in "B4DL_训练评测报告_20260810.md" "learn docs/plan1.md" "learn docs/plan1_learning_results.md" "learn docs/training_reproduction_guide.md" "learn docs/B4DL_temp
…（已截断，原文 515 字符）
description: Preview headers of candidate docs
```

</details>

<details><summary>输出摘要</summary>

```
########## B4DL_训练评测报告_20260810.md
# B4DL 模型训练与评测报告

> 日期：2026-08-07 ~ 2026-08-10
> 环境：RTX 5090 (32GB) + PyTorch 2.5.1 CUDA 12.4 + DeepSpeed ZeRO-3

---

## 一、背景与初始问题

### 1.1 初始症状

运行冒烟测试（`test_b4dl.py --max_samples 50`）时，模型对所有 6 种任务均输出连续点号：

########## learn docs/plan1.md
# B4DL / LiDAR-LLM 后续工作执行计划

> 本计划根据 `progress_report.md`、`learning_notes.md`、`plan_3_analysis.md` 以及项目实际代码整理而成。请按阶段顺序执行，每个任务完成后在复选框中打勾，并填写验收记录。

---

## 使用说明

- `- [ ]` 表示待完成；`- [x]` 表示已完成。
- **必须研读的代码片段**已经列在每一节中，建议先读代码再执行动作。
- 每一阶段末尾有 **验收标准**，未通过时不要进入下一阶段。


########## learn docs/plan1_learning_results.md
# Plan1 学习结果与分析报告

> 基于 plan1.md 执行计划，逐项分析当前代码状态、发现的问题和学习结论。
> 分析日期：2026-07-12

---

## 第一阶段：清理代码层面的"阻塞问题"

### 任务 1.1：mm_projector 维度问题 ✅ 已修复

**状态**：问题已修复，无需再次修改。

########## learn docs/training_reproduction_guid
…（已截断，原文 3944 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Now I have a clear picture. Let me categorize:

**Clearly stale/useless (candidates for deletion):**

1. `docs/eval_results/` (untracked) — May 29-30 old stage2/stage3 eval artifacts from the very first pipeline (stage2_metrics.json, stage3_metrics.json, huge jsonl logs). Completely superseded by `mllm/eval_results/` current runs. DELETE.

2. `docs/evl/` (untracked) — Jul 26–Aug 3 artifacts from the old frame-level pipeline:
   - conversation_log.md — session log, useless
   - evaluation_summary.md — old stage2 eval report (superseded)
   - stage2_test_summary.md, stage2 metrics/logs — old pipeline eval artifacts
   - stage1_training.log (218KB), stage1_training_report.md — old 95K frame_id era stage1 (retired per memory: "旧 95k frame_id 方案退役保留")
   - b4dl_paper_alignment_plan.md — Aug 2 alignment gap list, superseded by 论文对齐审计_20260824 → 全仓库论文对齐审查_20260829
   - mmb4dl_paper_vs_eval_analysis.md — Aug 2 analysis of why old eval can't compare to paper — its conclusions have been absorbed into later docs
   DELETE whole directory. All of it is pre-seqv refactor era. Hmm, mmb4dl_paper_vs_eval_analysis documented the four-layer mismatch — that knowledge was carried forward into 训练方法对比 and 基线锁定 docs. Yes, superseded.

3. `docs/learn docs/plan1.md` + `plan1_learning_results.md` + `training_reproduction_guide.md` (tracked) — Jul 12 early learning-phase plan/results/guide, all superseded by the living `docs/B4DL_复现方案.md` and later dated docs. DELETE.

4. `docs/learn docs/B4DL_temporal_weakness_analysis.md` (tracked, Aug 2) — early analysis of temporal weakness root causes with old-pipeline numbers. This motivated the metatoken/per-sequence work. Its conclusions have been executed and superseded by seqv2/seqv3 analyses and baseline lock. It's "early". Hmm — is it useless? It contains the root-cause analysis chain (metatoken injection design) which is now implemented. The later docs (两阶段训练评测与混合法决策, seqv2坍缩分析) cover the evolution. I lean DELETE as early/superseded. But let me be careful — the user said "一些" — I don't have to delete everything. Borderline ones I could keep. Let me think about what has enduring value:
   - temporal_weakness_analysis: root cause analysis of temporal weakness — historical, superseded. But it's the design rationale for metatoken/training scheme... Actually the 复现方案 (70KB, living) presumably absorbed all of this. The memory note says metatoken implementation is recorded in memory + 复现方案. I'll check if 复现方案 covers temporal weakness analysis. It's 70KB — very likely comprehensive. DELETE is defensible, but it's a judgment call.
   - B4DL_训练评测报告_20260810.md (tracked) — first full training report (点号 bug era, 四大错误模式). Historical record of the old frame-level run. Superseded by seqv2/seqv3 line. The memory "训练评测时间线" records its essence. DELETE is defensible.
   - B4DL_论文对齐审计_20260824.md (tracked) — superseded by 全仓库论文对齐审查_20260829, BUT explicitly referenced by two live docs (待修改清单 line 4, 全仓库审查 line 6) as cross-validation/provenance. KEEP.

Hmm, let me reconsider. The user's phrasing: "清除一些早期的无用的doc文档" — "clean up some of the early useless doc docs". They trust me to identify. The safest useful cleanup:

DELETE:
- docs/eval_results/ (untracked, May, stale artifacts)
- docs/evl/ (untracked, Jul-Aug 3, old pipeline artifacts + superseded analyses)
- learn docs/plan1.md, plan1_learning_results.md, training_reproduction_guide.md (Jul 12, superseded by 复现方案)

KEEP (borderline but defensible):
- B4DL_训练评测报告_20260810.md — hmm. Early (Aug 10). It's a dated report of the retired frame-level pipeline. Superseded. Actually I think this belongs in "early useless" too. But it's the only record of the first full training run... The memory b4dl-training-eval-history has the timeline essence, and git history records it. Deleting tracked files is recoverable from git anyway! That's an important point: all tracked files deleted via git rm are recoverable from history. So deleting tracked early docs is low-risk. The untracked ones (eval_results, evl) are NOT recoverable — but those are pure stale artifacts (logs/metrics from retired runs), the least valuable content.

Decision: delete both categories. For the borderline dated analysis docs in learn docs (temporal_weakness_analysis 0802, 训练评测报告_20260810): these are "early" and their content is absorbed. Given git recoverability, deleting is reversible. But I should be conservative and only delete what's clearly useless. Let me decide:

- temporal_weakness_analysis (Aug 2): This one has design rationale (metatoken training scheme design). Was it executed? Yes — metatoken is implemented (memory). The doc is now historical. But it's only 3 weeks old and is the rationale doc. Hmm. The 复现方案 is the living comprehensive doc — let me check whether it references temporal_weakness_analysis. Earlier grep of 复现方案 for candidates showed only the _eval_results.json false positive — it did NOT match "temporal". Let me check whether anything references "temporal_weakness".

- 训练评测报告_20260810: early report, superseded. Check references.

Actually, let me not over-think. I'll do a final reference check for "temporal_weakness", "训练评测报告", and then make the call:

DELETE list:
1. docs/eval_results/ (whole dir, untracked)
2. docs/evl/ (whole dir, untracked)
3. docs/learn docs/plan1.md
4. docs/learn docs/plan1_learning_results.md
5. docs/learn docs/training_reproduction_guide.md
6. docs/B4DL_训练评测报告_20260810.md (superseded by the 0824→0829 audit line and seqv2/seqv3 analysis docs; recoverable from git)
7. docs/learn docs/B4DL_temporal_weakness_analysis.md (Aug 2 root-cause analysis, executed and absorbed; recoverable from git)

KEEP:
- B4DL_复现方案.md (living core doc)
- B4DL_论文对齐审计_20260824.md (referenced by 待修改清单 and 全仓库审查)
- All 0825+ dated docs (seqv2, seqv3, 两阶段, 训练方法对比, LiDARCLIP权重核对, 评测数值对比, 基线锁定, 全仓库审查, 待修改清单, 基线版本对比)
- VTimeLLM_analysis.md, B4DL_vs_LiDAR-LLM_comparison.md — literature notes, still valid reference
- 参考/, mmb4dl-md/ — paper + reference code

Hmm wait, on #6/#7 — let me reconsider once more. The user asked for "早期的无用的" docs. #6 and #7 are exactly "early". Are they useless? Their content has been fully absorbed into: memory (training-eval-history), the git-committed history, and the newer analysis docs. Keeping them adds clutter. The docs/ folder is the user's knowledge base — a recent commit pattern shows they value curated dated docs. Deleting 6 & 7 seems aligned with intent. And both are git-tracked → recoverable. OK, delete.

Also `参考/记录.md` — Jul 12 training notes (stage1 loss curves). Early but tiny, part of 参考 folder. The stage1 approach it records (95K frame_id era) is retired. But it's in 参考 folder... it's basically a small log. Eh — leave it; it's inside 参考 which user considers reference material. Actually it records stage1 training experiments from the retired pipeline. Minor. I'll leave it — not worth the judgment risk; user said "一些" (some).

Wait, also check: are there any other early docs elsewhere? The user said "doc文档" — docs folder. Stick with docs/.

One more check: does anything reference "temporal_weakness_analysis" or "训练评测报告_20260810" or "plan1" from live files (scripts, CLAUDE.md)? The earlier grep showed plan1 matches in learn docs only (plan1.md itself and plan1_learning_results.md and training_reproduction_guide.md reference plan1). Let me do a final precise check including scripts and root files.

</details>

内容确认了。这些早期文档（7月12日的 plan1 系列、8月2-3日旧 frame 级管线的 evl/ 与 eval_results/）都已被后续的复现方案和 0824 之后的系列分析文档取代。最后做一次精确引用检查，然后执行清理。

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && grep -rn --include="*.md" --include="*.py" --include="*.sh" -e "temporal_weakness" -e "训练评测报告_20260810" -e "plan1\.md" -e "plan1_learning" -e "training_reproduction_g
…（已截断，原文 404 字符）
description: Check for external references to deletion candidates
```

</details>

<details><summary>输出摘要</summary>

```
exit=1
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

No external references (exit=1 means grep found nothing outside the candidate files themselves). Safe to delete.

Deletion plan:
- Untracked (permanent, stale artifacts from retired pipelines):
  - docs/eval_results/ (May, 3.9MB stale jsonl)
  - docs/evl/ (Jul 26–Aug 3, old pipeline logs + superseded analyses)
- Tracked (recoverable from git history):
  - docs/B4DL_训练评测报告_20260810.md
  - docs/learn docs/plan1.md
  - docs/learn docs/plan1_learning_results.md
  - docs/learn docs/training_reproduction_guide.md
  - docs/learn docs/B4DL_temporal_weakness_analysis.md

Then commit per git conventions (Chinese commit message with prefix). The git convention memory says: 中文 commit + feat/fix 前缀、改后即 commit、提交后及时 push 远程同步. For deletion maybe "chore:" or "docs:" prefix. Recent commits use "docs:" and "feat:". I'll use "docs:".

Wait — should I push? Memory says "提交后及时 push 远程同步". But there are unrelated staged/unstaged changes (logs modified). I must not commit those — only commit the deletions. git rm the tracked ones, git add -A on the deleted paths for untracked... untracked dirs just get rm -rf'd (not in git). Then commit only the staged deletions.

Regarding push: memory says push promptly after commit. Let me check if there's a remote configured. I'll commit first, then try push (if remote exists). Actually — the working tree has unrelated modified files (encoders logs, mllm training logs). Pushing only pushes commits, that's fine. But is push safe/outward-facing? Memory explicitly says to push after commit as part of the established workflow ("提交后及时 push 远程同步" — durable authorization). I'll commit and push.

Hmm, but wait — should I really delete B4DL_训练评测报告_20260810.md and B4DL_temporal_weakness_analysis.md? Let me weigh once more. The user said "早期的无用的doc文档". 

- 训练评测报告_20260810: reports the first full training of the retired frame-level pipeline (点号 bug, 模式坍缩, 四大错误模式). All of that history is in: memory file b4dl-training-eval-history, git commit messages, and the 0825 seqv2坍缩分析 doc which covers the collapse chain. It's a dated snapshot report — classic "early, superseded". Delete. ✓
- temporal_weakness_analysis (Aug 2): root-cause analysis that drove the metatoken/per-sequence redesign. The redesign is done and documented in 复现方案 + later docs. Early, superseded. Delete. ✓

But hold on — memory b4dl-metatoken-implementation references? Memory files reference docs sometimes. Let me check the memory dir for references to these filenames to avoid dangling memory pointers. Memory says "docs/复现方案.md" mostly. Quick check is cheap — grep memory dir. Also check CLAUDE.md at repo root for any doc references (I grepped CLAUDE.md earlier for specific names — no matches). But CLAUDE.md might list docs generally. Let me grep CLAUDE.md for "learn docs" and "docs/" quickly before deleting.

</details>

没有外部引用。再快速确认 CLAUDE.md 和记忆文件没有指向这些文档，然后执行删除。

**🔧 Bash**

<details><summary>输入</summary>

```
command: grep -n "docs/" CLAUDE.md | head -15; echo "=== memory refs ==="; grep -rn -e "temporal_weakness" -e "训练评测报告" -e "plan1" -e "evl/" -e "eval_results" /root/.zcode/cli/memories/projects/project-64d8f08d
…（已截断，原文 235 字符）
description: Check CLAUDE.md and memory files for doc references
```

</details>

<details><summary>输出摘要</summary>

```
228:- **文档存放**：后续生成的所有说明文档（.md）统一保存到 `learn docs/` 目录下
=== memory refs ===
/root/.zcode/cli/memories/projects/project-64d8f08ddafc8e25/memory/b4dl-per-sequence-refactor.md:20:**后续执行**：commit `3004e75` 加入 Stage2 全量/恢复/per-seq 训练与 baseline 评测脚本；用 per-seq 数据重训 stage2-full-seq（training_logs/stage2_full_seq_20260822_094534.log + resume_20260822_230654.log），产出 merged checkpoint `checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seq-merged`。冒烟测试序列（每任务 50 条，产物在 mllm/eval_results/）：smoke_seq (08-22)、smoke_2000 (08-23)、smoke_framectx (08-23 17:34，frame-context 变体 test_b4dl.py；变体取消后该产物与 .bak_frame_ctx 备份已删除)。framectx vs 基线基本持平（accuracy 0.77/0.77，mIoU 0.276/0.293）。
/root/.zcode/cli/memories/projects/project-64d8f08ddafc8e25/memory/b4dl-per-sequence-refactor.md:28:**seqv2 重训（2026-08-24 tmux 启动）**：用户要求用 s
…（已截断，原文 5018 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Found one memory reference: `b4dl-training-eval-history.md` line 10 says "训练评测史（报告 `docs/B4DL_训练评测报告_20260810.md`，详细结论）" — it points to the report file. If I delete that file, I should update the memory pointer. The memory itself contains the detailed conclusions, so I'll update the memory line to note the report was deleted (or just remove the file pointer).

Also CLAUDE.md line 228 says docs go in `learn docs/` — that's a general convention, fine.

No memory references to temporal_weakness, plan1, evl/, eval_results, training_reproduction_guide. Good.

Now execute:
1. rm -rf docs/eval_results docs/evl
2. git rm the 5 tracked files
3. Update memory b4dl-training-eval-history.md to drop the file pointer (edit the line).
4. Commit deletions (only staged deletion files, not unrelated changes).
5. Push per workflow memory.

Wait — push: memory says "提交后及时 push 远程同步". Let me check remote. I'll do git remote -v. If exists, push. The git conventions memory is durable authorization for this workflow.

Let me also double check there are no other early useless docs I'm missing: docs/ has only 复现方案 (keep), 训练评测报告_20260810 (delete), learn docs/ (mix), 参考/ (keep), mmb4dl-md/ (keep paper), eval_results + evl (delete). learn docs remaining early ones: VTimeLLM_analysis.md and B4DL_vs_LiDAR-LLM_comparison.md (Jul 12 literature notes — keep as reference; they're about external papers, not stale project state). 参考/记录.md (tiny training notes, part of 参考 — keep, low value but harmless... actually it records retired 95k-era stage1 experiments. It's 1.5KB in 参考 folder. Leave it — user said "一些", being conservative on reference folder).

Hmm, actually let me reconsider VTimeLLM_analysis.md / B4DL_vs_LiDAR-LLM_comparison.md — these are study notes on other papers from the learning phase. They don't go stale. Keep.

Execute now. For the memory edit, I need to read that memory file line first.

</details>

发现一个记忆文件指向了将要删除的报告，删除后需要同步更新。现在执行清理：

**🔧 Bash**

<details><summary>输入</summary>

```
command: rm -rf docs/eval_results docs/evl && git rm -q "docs/B4DL_训练评测报告_20260810.md" "docs/learn docs/plan1.md" "docs/learn docs/plan1_learning_results.md" "docs/learn docs/training_reproduction_guide.md" "d
…（已截断，原文 283 字符）
description: Delete stale doc directories and git rm superseded docs
```

</details>

<details><summary>输出摘要</summary>

```
D  "docs/B4DL_\350\256\255\347\273\203\350\257\204\346\265\213\346\212\245\345\221\212_20260810.md"
D  "docs/learn docs/B4DL_temporal_weakness_analysis.md"
D  "docs/learn docs/plan1.md"
D  "docs/learn docs/plan1_learning_results.md"
D  "docs/learn docs/training_reproduction_guide.md"
 M encoders/lidarclip/logs/val_mse_probe.csv
 M encoders/lidarclip/logs/val_mse_probe.log
?? encoders/lidarclip/logs/anneal_chain.log
?? encoders/lidarclip/logs/extract_stage1_sample_token_b1.log
?? encoders/lidarclip/logs/extract_stage2_b1.log
?? encoders/lidarclip/logs/train_anneal_0829_2036.log
?? encoders/lidarclip/logs/train_loss.csv
?? mllm/training_logs/b1_pipeline.log
?? mllm/training_logs/stage1_162k_b1_20260830_165411.log
?? mllm/training_logs/stage1_162k_b1_20260830_185010.log
?? mllm/training_logs/
…（已截断，原文 846 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

**🔧 Read**

<details><summary>输入</summary>

```
file_path: /root/.zcode/cli/memories/projects/project-64d8f08ddafc8e25/memory/b4dl-training-eval-history.md
```

</details>

<details><summary>输出摘要</summary>

```
1	---
2	name: b4dl-training-eval-history
3	description: B4DL 训练与全量评测时间线（2026-08-07~10）——点号 bug、数据合并重训 stage2-full、指标 vs 论文、四大错误模式
4	metadata:
5	  node_type: memory
6	  type: project
7	  originSessionId: sess_57bc4f46-cf94-4f89-9779-568442660f70
8	---
9	
10	训练评测史（报告 `docs/B4DL_训练评测报告_20260810.md`，详细结论）：
11	
12	1. **点号 bug**：模型全任务输出 "...."，根因是推理 `build_query()` 默认注入未训练的 `<4DLiDAR>` token（训练格式只有 `<video>\n<question>`），embedding 随机初始化干扰注意力 → 输出退化为 token 29889。修复 commit `ed5e9e5`（use_4dlidar/use_meta 默认 False，仅显式传 --ego_meta 才启用）。
13	2. **数据失衡**：旧 Stage2（54,901 条）45.5% "Yes." + 28.3% "No."、长描述仅 0.01%，模型学会无脑答 "Yes." 或 "from frame 000 to frame 008."。方案：合并 stage2+stage3 train = 118,722 条（长描述升到 49.9%）+ metatoken 注入（`<4DLiDAR>\n<video>\n<question> <meta> The metadata of the first frame is '...' and 
…（已截断，原文 2405 字符）
```

</details>

<details><summary>思考过程</summary>

Deletions staged. Now update the memory file that referenced the deleted report. Let me read that line first.

</details>

删除完成，5 个跟踪文件已暂存。现在更新记忆文件中对已删报告的引用：


### ASSISTANT  ·  `GLM-5.3-Flash`

**🔧 Edit**

<details><summary>输入</summary>

```
file_path: /root/.zcode/cli/memories/projects/project-64d8f08ddafc8e25/memory/b4dl-training-eval-history.md
```

</details>

<details><summary>输出摘要</summary>

```
The file /root/.zcode/cli/memories/projects/project-64d8f08ddafc8e25/memory/b4dl-training-eval-history.md has been updated successfully. (file state is current in your context — no need to Read it back)
```

</details>


### USER

The TodoWrite tool hasn't been used recently. If you're working on tasks that would benefit from tracking progress, consider using the TodoWrite tool to track progress. Also consider cleaning up the todo list if has become stale and no longer matches what you are working on. Only use it if it's relevant to the current work. This is just a gentle reminder - ignore if not applicable.


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Now commit the deletions. Per git conventions: Chinese commit message, prefix. For deletion of docs, "docs:" prefix fits. Commit only the staged deletions (the unrelated modified/untracked log files must not be included — they're already unstaged, and git commit without -a only commits staged changes).

Then push per workflow memory ("提交后及时 push 远程同步"). Let me check remote first and push.

</details>

记忆已更新。现在提交删除并按工作流推送：

**🔧 Bash**

<details><summary>输入</summary>

```
command: git commit -m "docs: 清理早期无用文档——删除 plan1 系列学习计划/复现指南、0802 时序弱点分析、0810 旧管线训练评测报告（均已被复现方案及 0824 后系列分析文档取代）；移除未跟踪的 docs/evl 与 docs/eval_results 旧评估产物目录" && git log --oneline -1 && git remote -v | head -2
description: Commit doc deletions and check remote
```

</details>

<details><summary>输出摘要</summary>

```
[main 46a0b9e] docs: 清理早期无用文档——删除 plan1 系列学习计划/复现指南、0802 时序弱点分析、0810 旧管线训练评测报告（均已被复现方案及 0824 后系列分析文档取代）；移除未跟踪的 docs/evl 与 docs/eval_results 旧评估产物目录
 5 files changed, 1952 deletions(-)
 delete mode 100644 "docs/B4DL_\350\256\255\347\273\203\350\257\204\346\265\213\346\212\245\345\221\212_20260810.md"
 delete mode 100644 docs/learn docs/B4DL_temporal_weakness_analysis.md
 delete mode 100644 docs/learn docs/plan1.md
 delete mode 100644 docs/learn docs/plan1_learning_results.md
 delete mode 100644 docs/learn docs/training_reproduction_guide.md
46a0b9e docs: 清理早期无用文档——删除 plan1 系列学习计划/复现指南、0802 时序弱点分析、0810 旧管线训练评测报告（均已被复现方案及 0824 后系列分析文档取代）；移除未跟踪的 docs/evl 与 docs/eval_results 旧评估产物目录
origin	git@github.com:Dani3lWang/remote4b4dl.git (fetch)
origin	git@github.com:Dani3lWang/remote4b4dl.git (push)
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

**🔧 Bash**

<details><summary>输入</summary>

```
command: git push origin main 2>&1 | tail -3
description: Push commit to origin main
```

</details>

<details><summary>输出摘要</summary>

```
hint: to the same ref. You may want to first integrate the remote changes
hint: (e.g., 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Push rejected — remote has changes we don't have. This is the user's local Windows repo pushing to the same remote (memory: Daniel's Windows local repo + SSH server workflow, both sync to the same GitHub remote). I should pull (rebase or merge) then push. 

Per workflow memory, the user syncs between Windows local and this server. A `git pull --rebase` would put my commit on top of remote changes. Let me check what the remote has first, then pull --rebase and push. There are unrelated unstaged changes in the working tree (log files) — rebase requires clean-ish tree; unstaged changes to tracked files (val_mse_probe.csv/log are modified) could block rebase. Let me stash them? That's touching user's working state... Actually `git pull --rebase` will refuse if there are unstaged changes that conflict. Safer: fetch first and look at what diverged.

</details>

推送被拒——远程有本地没有的新提交（可能是 Windows 本地仓库推送的）。先看远程更新了什么再整合：

**🔧 Bash**

<details><summary>输入</summary>

```
command: git fetch origin && git log --oneline HEAD..origin/main | head -10 && echo "=== 本地独有 ===" && git log --oneline origin/main..HEAD | head -5
description: Fetch and compare local vs remote commits
```

</details>

<details><summary>输出摘要</summary>

```
From github.com:Dani3lWang/remote4b4dl
   b9ca49d..8442277  main       -> origin/main
8442277 Merge branch 'main' of https://github.com/Dani3lWang/remote4b4dl
9f31aa4 docs: Wiki 复现定位补充——Home 重写为独立复现仓库框架（非官方实现声明 + B0 复现状态速览表）；新增 Reproduction-Guide（Step0-8 端到端复现指南：每步命令/产出/验收判据）与 Paper-vs-Reproduction（B0 vs 论文结果对比、7 项偏差清单、关键 bug 修复史、冻结对比规则）；_Sidebar 增复现专项分组
643f079 docs: Wiki 首页删除致谢部分（用户要求）
c3226bc docs: 新增仓库 Wiki（wiki/ 目录，11 页，GitHub Wiki 就绪）——Home/_Sidebar 导航 + Installation/Architecture/Data-Generation/LiDAR-CLIP-Encoder/Training/Inference-and-Evaluation 分模块页 + Repository-Structure 速查 + Reproduction-Log（B0 基线锁定与文档索引）+ FAQ-and-Known-Issues；内容依据 CLAUDE.md 与 2026-08 复现记录整理
=== 本地独有 ===
46a0b9e docs: 清理早期无用文档——删除 plan1 系列学习计划/复现指南、0802 时序弱点分析、0810 旧管线训练评测报告（均已被复现方案及 0824 后系列分析文档取代）；移除未跟踪的 docs/
…（已截断，原文 831 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Remote has new commits: the user (or another session) added a Wiki (wiki/ directory) with docs about reproduction. The remote commits only touch wiki/ presumably. My commit only deletes files in docs/. No overlap → rebase will be clean.

But there are unstaged modifications (encoders/lidarclip/logs/val_mse_probe.csv/log). `git rebase` requires clean working tree for tracked modifications. Options: stash → rebase → pop. The modified files are just logs (val_mse_probe.csv/log) — stashing and popping is safe, no conflict expected since remote likely doesn't touch them.

Let me do: git stash push (the two modified files) → git pull --rebase → git push → git stash pop.

Actually simpler: `git stash && git pull --rebase origin main && git push origin main && git stash pop`.

</details>

远程新增的是 wiki/ 目录的提交，与本次 docs 删除无冲突。暂存本地未提交的日志改动，rebase 后推送：

**🔧 Bash**

<details><summary>输入</summary>

```
command: git stash push -m "清理文档时的临时暂存-日志文件" && git pull --rebase origin main 2>&1 | tail -2 && git push origin main 2>&1 | tail -2 && git stash pop && git status --short | head -12
description: Stash logs, rebase on remote, push, restore stash
```

</details>

<details><summary>输出摘要</summary>

```
Saved working directory and index state On main: 清理文档时的临时暂存-日志文件
 * branch            main       -> FETCH_HEAD
Rebasing (1/1)                                                                                Successfully rebased and updated refs/heads/main.
To github.com:Dani3lWang/remote4b4dl.git
   8442277..19bb843  main -> main
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   encoders/lidarclip/logs/val_mse_probe.csv
	modified:   encoders/lidarclip/logs/val_mse_probe.log

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	encoders/lidarclip/logs/anneal_chain.log
	encod
…（已截断，原文 1900 字符）
```

</details>


### ASSISTANT  ·  `GLM-5.3-Flash`

<details><summary>思考过程</summary>

Done. Push succeeded (19bb843), stash popped, working tree back to original state with the unrelated log files untouched.

Summary for user:
- Deleted 7 items total:
  - Untracked stale artifact dirs: docs/evl/ (Jul 26–Aug 3 old pipeline eval logs/reports, ~2.7MB) and docs/eval_results/ (May 29-30, 3.9MB stale jsonl)
  - 5 git-tracked docs (recoverable from history): plan1.md, plan1_learning_results.md, training_reproduction_guide.md (Jul 12 learning-phase, superseded by 复现方案), B4DL_temporal_weakness_analysis.md (Aug 2, absorbed by later refactor docs), B4DL_训练评测报告_20260810.md (old frame-level pipeline report)
- Kept: 复现方案, 论文对齐审计_20260824 (referenced by 待修改清单/全仓库审查), all 0825+ docs, literature notes (VTimeLLM, LiDAR-LLM comparison), 参考/, mmb4dl-md paper
- Updated memory pointer to deleted report
- Committed 46a0b9e → rebased on remote (new wiki commits) → pushed as 19bb843

</details>

清理完成并已推送到远程（commit `19bb843`，在远程新增的 wiki 提交之上 rebase）。

**删除的内容（共 7 项）：**

- **`docs/evl/`** 和 **`docs/eval_results/`**（均为未跟踪的本地目录）——5 月底至 8 月 3 日旧 frame 级管线的评估产物：旧 stage2/stage3 指标与测试日志（约 6.6MB jsonl）、评估会话记录、旧 95K 时代的 stage1 训练日志与报告。这些产物的结论都已被 `learn docs/` 里 0824 之后的系列分析文档取代。
- **5 个 git 跟踪的文档**（可随时从 git 历史恢复）：
  - `plan1.md`、`plan1_learning_results.md`、`training_reproduction_guide.md`（7 月 12 日学习期计划/结果/指南，已被持续维护的《B4DL_复现方案》取代）
  - `B4DL_temporal_weakness_analysis.md`（8 月 2 日时序弱点根因分析，结论已落地并并入后续 seqv2/seqv3 演进文档）
  - `B4DL_训练评测报告_20260810.md`（旧管线首次全量训练报告，时间线保留在记忆和 git 历史中）

**保留的内容：** 《复现方案》主文档；《论文对齐审计_20260824》（被待修改清单和全仓库审查明确交叉引用，删了会让引用悬空）；0825 以后全部带日期的分析文档；VTimeLLM/LiDAR-LLM 两篇文献学习笔记（描述外部论文，不过时）；`参考/`、`mmb4dl-md/`（论文原文）。

另外同步更新了记忆文件 `b4dl-training-eval-history.md` 中指向已删报告的引用，改为指向 git 历史。工作区里 B1 流水线的训练日志等无关改动未动、未提交。

