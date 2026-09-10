# 在 /root/autodl-tmp/wql/mmb4dl 项目内做只读核查（禁止任何修改/删除），帮助判断哪些日...

| 项 | 值 |
|---|---|
| 会话 ID | `sess_subagent_agent_47e1ce38-7b3e-4a3d-a2f2-ebc1fbed0d41` |
| 工作目录 | `/root/autodl-tmp/wql/mmb4dl` |
| 父会话 | `sess_5a5b4be3-9fb0-410d-8a91-b144c7085a8d` |
| 时间 | 2026-09-08 23:23 → 2026-09-08 23:31 |
| 模型 | deepseek-v4-flash |
| 消息数 | 24（文本块 8，工具调用 33） |

## 会话待办

- [x] Judge per-file outcome for all listed training logs  _(completed, high)_
- [ ] Disk usage measurements (du) for eval_results/training_logs/wandb/checkpoints etc.  _(in_progress, high)_
- [ ] Analyze eval_results root scattered files vs run subdirs  _(pending, high)_
- [ ] Grep docs/README/CLAUDE.md for references to cleanup candidates  _(pending, high)_
- [ ] Compile final safe-delete vs keep lists  _(pending, high)_

---

### ASSISTANT  ·  `deepseek-v4-flash`

> 🔀 模型切换：None → 2801678c-7b5c-456f-ad57-1e9cb4a76bba/deepseek-v4-flash


### USER

在 /root/autodl-tmp/wql/mmb4dl 项目内做只读核查（禁止任何修改/删除），帮助判断哪些日志与产物可以清理。搜索广度：medium。

背景：该项目多版本训练（B0-B4a/seq/seqv2/seqv3/stage1 等），主训练日志在 mllm/training_logs/，评测产物在 mllm/eval_results/<run名>/（gitignore）。现在要清理"无用"的日志与产物，需要你判定候选文件的"有用性"。

请完成以下四件事：

1. **逐文件判定下面这批日志的结局**（对每个文件：看头部 5 行左右 + 尾部 15 行左右 + 时间戳，判断是：正常完成 / 中途崩溃(OOM/报错) / 刚启动即退出(近乎空) / 与其他文件重复(同一次 run 的拷贝或 nohup 双写) / 疑似失败后的重试前奏）。文件清单：
   - /root/autodl-tmp/wql/mmb4dl/training_logs/stage2_20260807_200248.log 与 stage2_final.log（两者是否同一 run 的重复？）
   - /root/autodl-tmp/wql/mmb4dl/mllm/training_logs/stage2_full_20260809_170938.log、_171216.log、_171708.log、_184921.log
   - /root/autodl-tmp/wql/mmb4dl/mllm/training_logs/stage2_full_resume_20260810_092622.log、_093210.log、_093455.log
   - /root/autodl-tmp/wql/mmb4dl/mllm/training_logs/stage1_95k_20260825_161616.log 与 _162055.log（同 run 重试？哪个是成功的）
   - /root/autodl-tmp/wql/mmb4dl/mllm/training_logs/stage2_full_seq_20260812_171413.log、_171640.log（08-12 与 08-22/23 的 seq 日志是否同一 run？08-12 两个文件结局如何）
   - /root/autodl-tmp/wql/mmb4dl/mllm/training_logs/stage2_full_seqv2_20260824_*.log 全部（_114030/_121607/_124310/_125007/_131353/_131601/nohup）——哪个是成功完成的大 run，其余是否失败尝试
   - /root/autodl-tmp/wql/mmb4dl/mllm/training_logs/stage2_seqv3_merge_20260826_002121.log（1.6KB，merge 步骤？正常否）
   - /root/autodl-tmp/wql/mmb4dl/mllm/training_logs/stage2_full_seqv3_mixed_b1_20260830_213612.log、_20260831_135828.log、_20260831_164030.log——哪个是完整成功 run，其余是什么
   - /root/autodl-tmp/wql/mmb4dl/mllm/training_logs/stage2_full_seqv3_mixed_b2_20260901_112252.log、_150156.log、_205231.log、_20260903_142959.log、_171433.log、_172716.log——同上
   - /root/autodl-tmp/wql/mmb4dl/mllm/training_logs/b1_pipeline.log(749B)、b1_eval_pipeline.log(357B)、b2_pipeline.log(638B)、b3_pipeline.log(13.8KB)——内容分别覆盖哪些阶段，是否被后续 b4a_pipeline.log 或 eval_results 的正式产物取代
   - /root/autodl-tmp/wql/mmb4dl/smoke_test.log（根目录 20KB，Aug 9 推理 smoke）——有无引用价值
   输出每个文件一行：结论短语 + 一两句证据（如"尾部 CUDA OOM"、"尾部显示 3 epoch 完成"、"与 xxx 同时间同大小疑为重复"）。

2. **磁盘占用实测**：用 du -sh 报：mllm/eval_results/ 总量及各 run 子目录、mllm/training_logs/ 总量、mllm/evaluation/ 总量、mllm/wandb/ 总量与 offline-run 个数、根 wandb/、mllm/checkpoints/ 总量（只报数不删）、backups/（若存在）总量、mllm/b4dl_dataset 总量（只报数）。以及项目总占用 du -sh 一级子目录排序前 10。

3. **检查 mllm/eval_results/ 根目录散落文件**（非 run 子目录内的）：eval_log.txt(479KB, Aug22)、seq_predictions.json.ckpt、smoke_predictions.json.ckpt、smoke_2000_*、smoke_meta_*、smoke_seq_*、_meteor_sweep/（含 hyp.txt/ref.txt 等，Sep 7）——这些是否已被正式 run 目录产物取代（判断依据：mllm/eval_results/stage2_full_seq/ 等子目录是否有同性质且更新的产物；metrics_recomputed_* 是否已把 _meteor_sweep 的成果固化）。

4. **引用关系检查**：在 docs/（含 docs/learn docs/ 子目录）、README*.md、CLAUDE.md、*.md 根文件里 grep 上面提到的文件名（如 smoke_、stage1_95k、seqv2、stage2_20260807、b1_pipeline、predictions_full、evaluation_results.json 等），报告哪些 md 文档明确引用了哪些将被清理的文件路径——只报告"文档在引用将被清理对象"的情况，帮助判断删除是否会破坏文档中的路径指引。

最后汇总：列出你认为【可安全删除】（被正式产物取代/空跑/崩溃残留/重复拷贝，且无文档引用）与【建议保留】（文档引用、pipeline 关键记录、唯一留档）两组清单，每组按目录列出绝对路径。


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Let me start by exploring the project structure and understanding what we're working with. This is a read-only investigation task with four parts:

1. Judge each log file's ending (normal completion / crash / empty / duplicate / retry)
2. Disk usage measurements
3. Check scattered files in mllm/eval_results/
4. Reference checks in docs/markdown files

Let me start with parallel exploration commands.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls -la /root/autodl-tmp/wql/mmb4dl/ && echo "---" && ls -la /root/autodl-tmp/wql/mmb4dl/mllm/training_logs/ | head -100
description: List project root and training_logs directory
```

</details>

<details><summary>输出摘要</summary>

```
total 104
drwxr-xr-x 19 root root  4096 Sep  8 12:49 .
drwxr-xr-x  5 root root    66 May 18 10:59 ..
drwxr-xr-x 25 root root  4096 Aug 12 17:37 .claude
drwxr-xr-x 10 root root  4096 Sep  8 23:19 .git
-rw-r--r--  1 root root   114 Aug  7 22:25 .gitattributes
-rw-r--r--  1 root root   848 Sep  6 16:07 .gitignore
drwxr-xr-x  2 root root   120 Aug  7 22:13 .ssh
drwxr-xr-x  2 root root    35 May 27 20:58 .vscode
drwxr-xr-x  3 root root    27 Sep  1 11:12 .zcode
-rw-r--r--  1 root root 18847 Sep  8 12:49 CLAUDE.md
drwxr-xr-x  2 root root  4096 Jun  5 16:28 Claude_record
-rw-r--r--  1 root root  4301 May 12 13:02 README.md
drwxr-xr-x  2 root root  4096 May 12 13:02 assets
drwxr-xr-x  2 root root   153 Sep  7 00:25 backups
drwxr-xr-x  7 root root  4096 Aug 29 21:17 datageneration
drwxr-xr-x  4 roo
…（已截断，原文 5338 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls -la /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/ && echo "---" && ls -la /root/autodl-tmp/wql/mmb4dl/ | grep -iE "wandb|backup|training_logs|smoke" 
description: List eval_results and check for wandb/backup dirs
```

</details>

<details><summary>输出摘要</summary>

```
total 1424
drwxr-xr-x 13 root root   4096 Sep  8 07:40 .
drwxr-xr-x 13 root root   4096 Aug 29 21:20 ..
drwxr-xr-x  2 root root     89 Sep  7 00:23 _meteor_sweep
-rw-r--r--  1 root root 479487 Aug 22 21:07 eval_log.txt
drwxr-xr-x  2 root root     86 Sep  2 16:34 official_ckpt_nometa
drwxr-xr-x  2 root root    113 Sep  2 16:05 official_model_stage2
-rw-r--r--  1 root root 692905 Aug 22 21:07 seq_predictions.json.ckpt
-rw-r--r--  1 root root  30254 Aug 23 13:21 smoke_2000_log.txt
-rw-r--r--  1 root root    991 Aug 23 13:21 smoke_2000_metrics.json
-rw-r--r--  1 root root  84975 Aug 23 13:21 smoke_2000_predictions.json
-rw-r--r--  1 root root    947 Aug  9 01:01 smoke_meta_metrics.json
-rw-r--r--  1 root root   5127 Aug  9 00:53 smoke_meta_predictions.json
-rw-r--r--  1 root root  12467 Aug 22
…（已截断，原文 1786 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Let me now start part 1: examining the log files. I'll do head/tail checks in parallel batches. Let me be efficient by grouping multiple checks per bash call but that could be long. Let me use head + tail on multiple files at once.

Batch 1: stage2_20260807_200248.log and stage2_final.log in training_logs/ (root-level).

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/training_logs && ls -la && for f in *.log; do echo "=== $f ==="; head -c 600 "$f"; echo; echo "--- tail:"; tail -c 900 "$f"; echo; done 2>/dev/null | head -150
description: Show root training_logs files head and tail
```

</details>

<details><summary>输出摘要</summary>

```
total 880
drwxr-xr-x  2 root root    113 Aug  8 01:40 .
drwxr-xr-x 19 root root   4096 Sep  8 12:49 ..
-rw-r--r--  1 root root 390624 Aug  8 01:40 stage2_20260807_200248.log
-rw-r--r--  1 root root 313154 Aug  8 01:40 stage2_final.log
-rw-r--r--  1 root root 186377 Aug  8 01:40 stage2_trainer_state.json
=== stage2_20260807_200248.log ===
[2026-08-07 20:02:50,989] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2026-08-07 20:02:53,265] [WARNING] [runner.py:215:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
[2026-08-07 20:02:53,266] [INFO] [runner.py:607:main] cmd = /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python3.10 -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMF19 --master_
…（已截断，原文 3324 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

So root training_logs has:
- stage2_20260807_200248.log: started 2026-08-07 20:02, tail at 2026-08-08 01:40 "Process 149724 exits successfully." — 390KB. Ends successfully? It says "Process exits successfully" from deepspeed launcher. But is it a full run? Let me check if it has training loop progress. The tail shows only warnings and exit. So it's the whole log; apparently it ran ~5.5h. Wait, but exit successfully. Hmm, but did training complete epochs? Need to check if there are train_runtime entries.
- stage2_final.log: started same time 2026-08-07 20:02:53 — same run duplicate (second nohup capture). Tail shows training progress bar 1072/1072 and train_runtime 20204s with epoch 2.0. So both same run of Aug 7 20:02. One ends with "Process exits successfully". Both same-run duplicates of the same launch. Different sizes because one captured deepspeed launcher output (including stderr?), other captured stdout of training. Actually both captured same process differently. Both are from same run 20:02:53 on Aug 7. stage2_final.log shows the trainer summary epoch 2.0 with train_runtime ~20204s (5.6h) ending around Aug 8 01:40 (matches exit time). So this is the successful stage2 full? Wait this training dir at root (vtimellm) might be an earlier model. The tail of stage2_20260807 shows 01:40:18 exit. So the run completed (epoch 2.0) successfully. Duplicate = yes, same run (both started 20:02:53, same timestamp).

Wait — need to confirm stage2_final.log tail: "train_runtime: 20204" epoch 2.0 — completed 2 epochs. So the run was successful. Both files are copies of the same run, one with more launcher detail. Let me confirm whether stage2_final.log is a different capture of same launch — both started at 20:02:53 same second. Yes, same launch, two captures (probably nohup.out style vs tee). Both completed.

Hmm — but "是否同一 run 的重复?" Yes. Both end at the same point. Whether run is "成功" — yes, finished epochs (train_runtime logged). But maybe this was a stage2 training run that was later superseded? That's beyond file status; but the user asks mainly duplication.

Let me now examine the mllm/training_logs files in batches. First the early stage2_full ones (Aug 9-10).

</details>

Root training_logs files checked. Now let me examine the mllm/training_logs batch files.

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && for f in stage2_full_20260809_170938.log stage2_full_20260809_171216.log stage2_full_20260809_171708.log stage2_full_20260809_184921.log; do echo "
…（已截断，原文 301 字符）
description: Show head/tail of stage2_full Aug 9 logs
```

</details>

<details><summary>输出摘要</summary>

```
########## stage2_full_20260809_170938.log ##########
--- HEAD:
============================================================
B4DL Stage2 全量训练
启动时间: Sun Aug  9 17:09:38 CST 2026
数据: b4dl_dataset/stage2_full_train.json (118,722 条)
日志文件: ./training_logs/stage2_full_20260809_170938.log
--- TAIL:
    from peft import PeftMixedModel, PeftModel
ImportError: cannot import name 'PeftMixedModel' from 'peft' (/root/miniconda3/lib/python3.13/site-packages/peft/__init__.py)
[2026-08-09 17:09:54,256] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 234851
[2026-08-09 17:09:54,256] [ERROR] [launch.py:325:sigkill_handler] ['/root/miniconda3/bin/python3.13', '-u', 'vtimellm/train/train_mem.py', '--local_rank=0', '--deepspeed', './scripts/zero3.json', '--lora_enable', 'True', '--model_name_or_path'
…（已截断，原文 5810 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && for f in stage2_full_resume_20260810_092622.log stage2_full_resume_20260810_093210.log stage2_full_resume_20260810_093455.log stage2_full_seq_20260
…（已截断，原文 362 字符）
description: Show head/tail of resume and seq Aug 10/12 logs
```

</details>

<details><summary>输出摘要</summary>

```
########## stage2_full_resume_20260810_092622.log ##########
--- HEAD:
启动: Mon Aug 10 09:26:22 CST 2026
从 checkpoint-2000 恢复训练
--- TAIL:
启动: Mon Aug 10 09:26:22 CST 2026
从 checkpoint-2000 恢复训练

########## stage2_full_resume_20260810_093210.log ##########
--- HEAD:
============================================================
恢复训练 from checkpoint-2000
启动: Mon Aug 10 09:32:10 CST 2026
============================================================
[2026-08-10 09:32:11,914] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
--- TAIL:
wandb: Run data is saved locally in /root/autodl-tmp/wql/mmb4dl/mllm/wandb/offline-run-20260810_093319-vzxwut2k
  0%|          | 0/2781 [00:00<?, ?it/s] 36%|███▌      | 1001/2781 [00:25<00:45, 39.33it/s]                  
…（已截断，原文 7365 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Good progress. Findings so far:

1. `stage2_full_20260809_170938.log`: started 17:09:38, crashed at 17:09:54 with ImportError `PeftMixedModel` — environment problem. Near-empty crash log.
2. `stage2_full_20260809_171216.log`: 148 bytes, only header text — instant start then empty (probably a nohup wrapper that got killed before training started, or a script that printed header then exited silently). Actually 148 bytes with just start message. Barely started. Wait — possibly it launched in background and the log only shows the wrapper. Ends abruptly with no error. Likely a failed quick launch (or it was interrupted). Could be nohup double-write... Actually the script writes the header then runs training command that also writes to the same log. 148 bytes means it never even got to trainer. Just-exit.
3. `stage2_full_20260809_171708.log`: started 17:17:08, killed by SIGTERM at 18:48 (1.5h in, epoch 0.34). Manually killed or OOM? "Main process received SIGTERM, exiting" — killed (likely to restart with resume). Actually the next run 184921 started 18:49:21, right after kill. So 171708 was terminated to restart with new config? Possibly interrupted. Not complete.
4. `stage2_full_20260809_184921.log`: started 18:49:21 Aug 9, ran 11.5+ hours to 82% (epoch 2.45), tail at 06:18 Aug 10 — wait, the tail file mtime is Aug 10 06:18. Tail shows epoch 2.45, 2274/2781 at 11:26 elapsed — no crash message visible. Hmm, the log ends mid-training at 82%? Let me check whether it ended abruptly (no exit message) — but the launcher exit message may come from a different part. Let me check the last lines more carefully and look for CUDA OOM or exit. File size 671KB and time 06:18. Let me search for "OOM", "Error", "exits with return code", "out of memory", "saving checkpoint", "Process.*exits".

Actually the wrapper prints "训练完成" or "训练完成时间" only if script ends. No such line visible in tail. Let me grep.

5. `stage2_full_resume_20260810_092622.log`: 70 bytes — header only. Instant exit. Probably killed instantly or a failed resume attempt.
6. `stage2_full_resume_20260810_093210.log`: started 09:32:10, SIGTERM at 09:34:36 after ~2.5 min (1003/2781). Manually killed — probably because it was slow or wrong (actually resumed at 1001+ steps with high it/s 39 for first 1001 steps in 25 s — that's odd; likely fast-skipping data loading... whatever) then killed. It was killed to restart again at 093455.
7. `stage2_full_resume_20260810_093455.log`: started 09:34:55, "Process 324982 exits successfully" at 13:31:13 — completed (need to check epoch/steps). File 322KB. Presumably the successful resume run that finished the remainder of stage2-full run (from checkpoint-2000). Let me verify tail more. Also compare: does the earlier 184921 run + this resume cover the full training? The final successful run 093455 shows what steps/epochs — check for train_runtime summary in that log.

8. `stage2_full_seq_20260812_171413.log`: CUDA unknown error — CPU only detected; then ValueError --tf32 requires Ampere... exit code 1. Crash (GPU env issue).
9. `stage2_full_seq_20260812_171640.log`: started 17:16:40 with same CUDA unknown error (still on CPU), got further — dataloader warnings, started 0/2781 progress. Ends mid start... tail shows dataloader warning. What happened next? Probably crashed too. Need to see more of the tail. Ends at 17:19? mtime 17:19. File 10860 bytes. Let me grep for exit/error.

So the "real" seq run (with GPU, from checkpoints) is stage2_full_seq_20260822_094534.log and resume stage2_full_seq_resume_20260822_230654.log (with nohup twins). 08-12 files were failed GPU-env attempts (first-run seq try), unrelated to the 08-22 run (different data prep?). The user asks: "08-12 与 08-22/23 的 seq 日志是否同一 run?" — 08-12 was aborted before any real training (CPU-only + tf32 error / then started on CPU...). Not the same run as 08-22. Need to verify: did 171640 actually train? It showed progress bar 0/2781 then dataloader warnings — but it's on CPU (pin_memory warning, no accelerator found). Actually earlier it detected no CUDA (CUDA unknown error at start). So likely never trained meaningfully. Check the end for crash. Possibly CUDA error caused trainer crash or it hung and was killed.

Let me look at more detail: grep for "Error|error|OOM|out of memory|Killing|SIGTERM|exits with return code" in those logs. And check the 184921 tail after the visible region for abrupt ending — the tail already showed ending with a step line then nothing (no exit message, no wrapper "完成" line). So it likely got killed/OOM silently? Let me grep.

Then continue with other batches.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && for f in stage2_full_20260809_184921.log stage2_full_resume_20260810_093455.log stage2_full_seq_20260812_171640.log; do echo "########## $f #######
…（已截断，原文 458 字符）
description: Grep errors and completion markers in three logs
```

</details>

<details><summary>输出摘要</summary>

```
########## stage2_full_20260809_184921.log ##########
--- last progress line / summary:
2333: 82%|████████▏ | 2268/2781 [11:26:12<2:32:29, 17.84s/it] 82%|████████▏ | 2269/2781 [11:26:30<2:32:45, 17.90s/it]                                                        {'loss': 1.1507, 'grad_norm': 0.6215547919273376, 'learning_rate': 8.631899538660476e-06, 'epoch': 2.45}
2334: 82%|████████▏ | 2269/2781 [11:26:30<2:32:45, 17.90s/it] 82%|████████▏ | 2270/2781 [11:26:48<2:32:01, 17.85s/it]                                                        {'loss': 1.1484, 'grad_norm': 0.6363041996955872, 'learning_rate': 8.599214683967172e-06, 'epoch': 2.45}
2335: 82%|████████▏ | 2270/2781 [11:26:48<2:32:01, 17.85s/it] 82%|████████▏ | 2271/2781 [11:27:06<2:32:08, 17.90s/it]                            
…（已截断，原文 6248 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && for f in stage1_95k_20260825_161616.log stage1_95k_20260825_162055.log; do echo "########## $f ##########"; echo "--- HEAD:"; head -8 "$f"; echo "-
…（已截断，原文 236 字符）
description: Show head/tail of stage1 95k logs
```

</details>

<details><summary>输出摘要</summary>

```
########## stage1_95k_20260825_161616.log ##########
--- HEAD:
[2026-08-25 16:16:18,209] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2026-08-25 16:16:20,824] [WARNING] [runner.py:215:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
[2026-08-25 16:16:20,824] [INFO] [runner.py:607:main] cmd = /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python3.10 -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMF19 --master_addr=127.0.0.1 --master_port=29571 --enable_each_rank_log=None vtimellm/train/train_mem.py --deepspeed ./scripts/zero3.json --model_name_or_path ./base_model/vicuna-v1-5-7b --version plain --data_path ./b4dl_dataset/stage1_train.json --feat_folder ../encoders/lidarclip/b4dl/st
…（已截断，原文 8867 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

More findings:

- `stage2_full_20260809_184921.log`: no error trace found in grep. Ends abruptly mid-step at epoch 2.45, 2274/2781, without exit messages. File mtime Aug 10 06:18. That means it died/was killed silently (e.g., machine/process killed without SIGTERM handling or log truncation). Actually grep found no error text at all and no wrapper "完成" line. So this is an incomplete run (82%) that abruptly stopped. The resume run 093455 continued from checkpoint-2000 and completed 3.0 epochs with train_runtime 14114s (train_loss 0.317). So the successful full run = 184921 (partial) + resume_093455 (completed epoch 3.0). Both partial and resume are "the run". Resume 093455 is the record of a successful completion.

- `stage2_full_seq_20260812_171640.log`: continued past model loading, started at 0/2781, then last lines are pin_memory warnings... 10860 bytes total, mtime 17:19. Let me check actual tail lines (already saw 8 lines - ended with dataloader warnings). So died right after data loading started. Probably killed or crashed. Grep found no error. It's an aborted early attempt (CPU mode). Need the very last line — from tail output above: ends with pin_memory warning. No error, no progress beyond step 0. Possibly hung and killed (e.g. when the person realized GPU wasn't visible). 08-12 files = environment failure attempts; the real seq run was 08-22/23 (094534 + resume_230654 + nohup twins). Different run entirely.

- stage1_95k pair: 161616 started 16:16:24, first run was killed at ~8 steps (0.01 epoch) — presumably killed to restart at 16:20:59 with... wait, both have identical cmd (same data 95k? Both stage1_train.json with 742 steps). Actually 162055 started 16:21 and completed: 742/742, train_runtime 4975s (~1h23m), exits successfully at 17:46. So 161616 = interrupted early attempt (8 steps), 162055 = successful completion. Both same run config (master_port 29571 both) — retry of same run. The user said 95k. dataset stage1_train.json (95k). Good.

Interesting: 161616.log head shows start 16:16; tail shows up to step 8/742 at 01:16 elapsed. Ends without error message? Grep... no error shown but I didn't grep this file. The tail ends at step 8. Likely killed by user to fix something (e.g. too slow? 14.4s/it initially). Actually the 162055 log tail shows 6.3-6.9 s/it. Both are same "run" (same config), the first was aborted early and the second completed. Wait — but why would they retry within 4 minutes at 16:20:59... the second started at 16:21. First ran from 16:16 to ~16:17+ (8 steps * ~10s = 80s+ plus startup). Hmm but first log's mtime is 16:19. So killed around 16:19 (was killed). Let me quickly check if 161616 has an error at the end or was just killed — need to check for the wrapper line. Let me grep both for exit status text. Also for the first file, maybe there's an "exits with return code" line.

Also check whether 161616's tail has "SIGTERM"/"return code" — earlier grep didn't include these files. Quick check.

Now next: seqv2 batch (Aug 24, 6 files + nohup), seqv3_merge, seqv3_mixed_b1, b2 batches. Also need to double check stage2_full_seq_20260822/23 statuses for comparison context, though user mostly focuses on listed files. Let me continue grepping in parallel.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && grep -nE "SIGTERM|return code|exits successfully|Killing|训练完成|训练时间" stage1_95k_20260825_161616.log | tail -3; echo "=== 161616 tail 3 lines ==="; t
…（已截断，原文 599 字符）
description: Check stage1 95k first log exit, seqv2 early logs head/tail
```

</details>

<details><summary>输出摘要</summary>

```
=== 161616 tail 3 lines ===
  1%|          | 6/742 [01:02<2:00:00,  9.78s/it]  1%|          | 7/742 [01:09<1:50:14,  9.00s/it]                                                 {'loss': 1.4055, 'grad_norm': 0.712291419506073, 'learning_rate': 0.00030434782608695655, 'epoch': 0.01}
  1%|          | 7/742 [01:09<1:50:14,  9.00s/it]  1%|          | 8/742 [01:16<1:40:07,  8.18s/it]                                                 {'loss': 1.4317, 'grad_norm': 0.8031586408615112, 'learning_rate': 0.00034782608695652176, 'epoch': 0.01}
  1%|          | 8/742 [01:16<1:40:07,  8.18s/it]=== seq 08-12 tail 3 ===
wandb: Run data is saved locally in /root/autodl-tmp/wql/mmb4dl/mllm/wandb/offline-run-20260812_171930-hej4731z
  0%|          | 0/2781 [00:00<?, ?it/s]/root/autodl-tmp/.conda-stuff/e
…（已截断，原文 13121 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && echo "===== seqv2_131353 ====="; head -3 stage2_full_seqv2_20260824_131353.log; echo "--TAIL:"; tail -4 stage2_full_seqv2_20260824_131353.log; echo
…（已截断，原文 611 字符）
description: Show seqv2 131353/131601/nohup and seqv3 merge logs
```

</details>

<details><summary>输出摘要</summary>

```
===== seqv2_131353 =====
[2026-08-24 13:13:56,413] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2026-08-24 13:14:02,393] [WARNING] [runner.py:215:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
[2026-08-24 13:14:02,393] [INFO] [runner.py:607:main] cmd = /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python3.10 -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMF19 --master_addr=127.0.0.1 --master_port=29577 --enable_each_rank_log=None vtimellm/train/train_mem.py --deepspeed ./scripts/zero3.json --lora_enable True --model_name_or_path ./base_model/vicuna-v1-5-7b --version v1 --data_path ./b4dl_dataset/stage2_full_train_seqv2_148k.json --feat_folder ../encoders/lidarclip/b4dl/stage2_f
…（已截断，原文 9598 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Great progress. Findings so far:

seqv2 series Aug 24:
- 114030: started 11:40:35, killed with SIGTERM return code -9 at 12:14:45 (~34 min, 74/2781 steps, epoch 0.08). Interrupted.
- 121607: started 12:16:11, killed -9 at 12:41:59 (54/2781). Interrupted. Note save_steps changed from 500 to 200 in this run config.
- 124310: started 12:43:13, killed -9 at 12:47:27 — just started (0 steps). Interrupted.
- 125007: started 12:50:10, killed SIGTERM at 13:13:23 (38/2781). Interrupted. 
- 131353: started 13:13:56, killed -9 at 13:15:59 — just after model load (0 steps, model loading at 13:15). Interrupted. Wait, but this uses data seqv2_148k. This looks like a test? It got killed ~2min in after loading model. Actually this is the same attempt as the nohup log (seqv2_nohup shows same PID 583487 killed at 13:15:59).
- seqv2_nohup: shows header of PER-SEQUENCE V2 TRAINING wrapper started 13:13:53 and ends with the same kill of PID 583487 at 13:15:59, "End: Mon Aug 24 13:16:01" — duplicate of the 131353 run (nohup wrapper log). So seqv2_nohup == the wrapper log around 131353 attempt (killed before training began).
- 131601 (1.1MB, successful big run): started 13:16:02 (2 min after 131353 was killed — retry). Exited successfully at Aug 25 09:46:08 "Process 601035 exits successfully", wrapper "End: Tue Aug 25 12:25:36 CST 2026 (manual)". Hmm — exit successfully at 09:46 but wrapper end at 12:25 "(manual)". Weird. Let me check progress: how many epochs/steps completed? 2781 steps, 3 epochs? Data is 148k now (was 118,722 → 2781 steps for 3 epochs; 148k would be ~...). Let me check the last progress in that file: search for epoch count. Actually let me grep the summary line train_runtime in 131601. Also whether full 3 epochs done. Check train_runtime in the seqv2_131601 log.

Note the training data config changed: 131601 still says 2781 total? Actually the cmd shows same; total steps not visible in cmd. But progress bar from the earlier logs: 74/2781 with data 118,722条 (that was with original seqv2 data). For 131601 using 148k data, steps = ? Let me grep for max step/epochs.

- seqv3_merge 002121: completed normal merge of seqv3 model — "Merged model saved... Use ... for stage3 training". 1.6KB, successful small merge step, followed by stage3_seqv3 log the same day. Normal completion. Keep as record? It's a small but complete record of the merge step. Value: low but it is a log of pipeline step. Hmm.

Next: b1 (Aug30-31) and b2 (Sep 1-3) batches and b3/b4a etc. Also the seq 08-22 runs and their resume+nohup twins, and stage2_full_20260822_094534 (the actual full seq run?) Let me look at those too, and grep checkpoints dirs for what run checkpoints exist.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && echo "== seqv2_131601 summary =="; grep -nE "train_runtime|epoch': 3|exits successfully|End:|save_steps" stage2_full_seqv2_20260824_131601.log | he
…（已截断，原文 531 字符）
description: Check seqv2 big run completion and seq Aug22 logs
```

</details>

<details><summary>输出摘要</summary>

```
== seqv2_131601 summary ==
3:[2026-08-24 13:16:05,238] [INFO] [runner.py:607:main] cmd = /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python3.10 -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMF19 --master_addr=127.0.0.1 --master_port=29577 --enable_each_rank_log=None vtimellm/train/train_mem.py --deepspeed ./scripts/zero3.json --lora_enable True --model_name_or_path ./base_model/vicuna-v1-5-7b --version v1 --data_path ./b4dl_dataset/stage2_full_train_seqv2_148k.json --feat_folder ../encoders/lidarclip/b4dl/stage2_features --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin --output_dir ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv2 --bf16 True --num_train_epochs 3 --per_device_train_batch_size 8 --gradient_accumulation_steps 16
…（已截断，原文 9631 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && for f in stage2_full_seqv3_mixed_b1_20260830_213612.log stage2_full_seqv3_mixed_b1_20260831_135828.log stage2_full_seqv3_mixed_b1_20260831_164030.l
…（已截断，原文 342 字符）
description: Head/tail of b1 seqv3 mixed logs
```

</details>

<details><summary>输出摘要</summary>

```
########## stage2_full_seqv3_mixed_b1_20260830_213612.log ##########
--- HEAD:
[2026-08-30 21:36:14,345] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2026-08-30 21:36:17,161] [WARNING] [runner.py:215:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
[2026-08-30 21:36:17,161] [INFO] [runner.py:607:main] cmd = /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python3.10 -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMF19 --master_addr=127.0.0.1 --master_po
--- TAIL:
 91%|█████████ | 3157/3474 [16:08:11<1:42:25, 19.39s/it] 91%|█████████ | 3158/3474 [16:08:31<1:42:46, 19.51s/it]                                                        {'loss': 1.0745, 'grad_norm': 0.612700879573822, 'le
…（已截断，原文 4026 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Key findings:

seqv2 (Aug 24):
- 131601 was the big successful run: 3474/3474 steps, 3.0 epochs completed, train_runtime 73656s (~20.5h), process exits successfully Aug 25 09:46. Note: seqv2 data 148k = 3474 steps for 3 epochs (1158 steps/epoch). Wait — but earlier logs showed 2781 total steps for 118,722 data (927/epoch). Here with 148k, total is 3474 for 3 epochs (1158/epoch). Yes, consistent. So seqv2_131601 is the successful big seqv2 run.
- All earlier ones (114030, 121607, 124310, 125007, 131353) are aborted attempts — return code -9 (SIGKILL). Likely each was the user stopping and adjusting config (e.g., save_steps 500→200; data to 148k version). Also note seqv2_nohup log duplicates the 131353 attempt window (killed before real training). Wait actually seqv2_nohup start 13:13:53, killed process 583487 at 13:15:59 = same kill as 131353 (583487). So yes: seqv2_nohup.log covers 131353 attempt, killed before training loop start. Actually careful: 131353 started 13:13:56 and the log's own subprocess... Both mention process 583487 killed at 13:15:59.505. And the nohup log's tail also shows 583487 kill. So both = the same attempt. But which is the "real" log? 131353 is deepspeed launcher output; nohup is wrapper (with header). Then run 131601 started at 13:16:02 (~3 s after kill) and ran to completion.

- seq Aug 22 (094534): started 09:45, killed -9 at 18:59 (~9.3h, 1488/2781, epoch 1.6). Killed (probably disk/then resumed?). Actually killed to be resumed at 23:06 (resume_230654 from checkpoint-1000). Wait, resume from checkpoint-1000 but the killed run reached step 1488? Odd — maybe checkpoint saves every 500 steps but they had checkpoint-1000 as the resume point... whatever. The resume run started 23:06 Aug 22. Its tail? Need to check whether it completed. The file is 625KB, ended Aug 23 12:37. Let me grep its summary. Also nohup twins: stage2_full_seq_nohup.log == stage2_full_seq_20260822_094534.log? Sizes: 435483 vs 435643 — nearly identical sizes, same mtime Aug 22 18:59. And stage2_full_seq_resume_nohup.log == resume (625842 vs 625702, same mtime Aug 23 12:37). Both are duplicate captures (nohup double-write). Need to verify they're same content: diff -q may differ by a few bytes due to escape. Compare first lines & last lines — likely same run, written via both nohup and direct tee. The sizes differ slightly (~160 bytes) — could be identical content with one having a wrapper header. Let me check head lines of nohup versions and compare tail. Let me diff them quickly (read-only diff is fine).

- b1 mixed batch (Aug 30-31):
  - 213612: started Aug 30 21:36, ran 16h to 91% (3161/3474), then SIGKILL at Aug 31 13:48 (return code -9 presumably; cut off). Aborted at 91%.
  - 135828: started Aug 31 13:58 (right after the kill — resume?), exited successfully at 16:30 (2.5h). This is the completion of the remainder (resume from 91%?) or a rerun? 2.5 hours for remaining ~9%? plausible with resume. Let me check head for --resume_from_checkpoint flag. Also whether it reports train summary.
  - 164030: started Aug 31 16:40, exited successfully at 16:43 — 3 min only! That can't be a full run. Hmm. 81KB in 3 minutes. Probably a quick merge or eval? It says "Process 475565 exits successfully" at 16:43:05, started 16:40:31. So this one is short. What was it? Perhaps it was a run that immediately errored out but reported "successfully"? No — process exits successfully. Maybe it was a sanity run / final model save. Actually wait — perhaps the wrapper is training via train_mem.py... 3 minutes. Maybe it crashed internally after a couple steps without the launcher catching an error? Let me grep for progress bar lines and any error. Also check the actual cmd (data path, resume flag) in each of these three.

b2 mixed batch (Sep 1-3): six files. Let me check those heads/tails next, plus examine the b1 135828 tail more fully (summary lines).

Let me now run the next batch of checks in parallel:
1. b2 files (six logs)
2. b1 logs detail: grep cmd data files and resume flags, summaries, progress maxima
3. seq resume 08-22/23 tail summary
4. nohup twins diff
5. b1_pipeline logs content (4 files + b4a_pipeline)
6. smoke_test.log tail

Then du measurements and eval_results check and docs grep.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && for f in stage2_full_seqv3_mixed_b2_20260901_112252.log stage2_full_seqv3_mixed_b2_20260901_150156.log stage2_full_seqv3_mixed_b2_20260901_205231.l
…（已截断，原文 807 字符）
description: Summarize b2 mixed log endings and configs
```

</details>

<details><summary>输出摘要</summary>

```
########## stage2_full_seqv3_mixed_b2_20260901_112252.log ##########
-- data/resume:
stage2_full_train_seqv3_148k.json
--num_train_epochs 3
save_steps 200
-- progress max:
600/3474 [3:24:09
-- errors/exit:
650: 17%|█▋        | 600/3474 [3:24:09<15:07:10, 18.94s/it][2026-09-01 14:51:52,575] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 18
651:[2026-09-01 14:51:52,578] [ERROR] [launch.py:325:sigkill_handler] ['/root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python3.10', '-u', 'vtimellm/train
-- TAIL:
 17%|█▋        | 600/3474 [3:24:09<15:07:10, 18.94s/it][2026-09-01 14:51:52,575] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 188367
[2026-09-01 14:51:52,578] [ERROR] [launch.py:325:sigkill_handler] ['/root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python3.10', '-u', 'vtimellm
…（已截断，原文 5108 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && echo "===== b1 213612 data/resume ====="; grep -oE "stage2_full_train[^ ]*\.json|resume_from_checkpoint [^ ]*|save_steps [0-9]+" stage2_full_seqv3_
…（已截断，原文 1455 字符）
description: Detail b1 logs configs, seq resume summary, nohup twin diff
```

</details>

<details><summary>输出摘要</summary>

```
===== b1 213612 data/resume =====
stage2_full_train_seqv3_148k.json
save_steps 200
stage2_full_train_seqv3_148k.json
===== b1 135828 data/resume =====
stage2_full_train_seqv3_148k.json
resume_from_checkpoint ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b1/checkpoint-3000
stage2_full_train_seqv3_148k.json
100%|█████████▉| 3473/3474 [2:28:20<00:18, 18.03s/it]100%|██████████| 3474/3474 [2:28:39<00:00, 18.23s/it]                                                   
                                                     {'train_runtime': 8936.0727, 'train_samples_per_second': 49.777, 'train_steps_per_second': 0.389, 'train_loss': 0.14848897458152244, 'epoch': 3.0
===== b1 164030 data/resume =====
stage2_full_train_seqv3_148k.json
resume_from_checkpoint ./checkpoints/vtimellm-v
…（已截断，原文 1419 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Great findings on b2:

- b2 112252: killed -9 (return code) at 14:51 after 600/3474 (17%). Interrupted.
- b2 150156: killed -9 at 20:42 after 1292/3474 (37%). Interrupted.
- b2 205231: COMPLETED: 3474/3474, train_runtime 48008s (~13.3h), Process 954109 exits successfully at Sep 2 10:15. Full successful b2 run.
- b2 142959: killed at 17:04 after 400/3474 — interrupted attempt with new data (meta2).
- b2 171433: killed 17:17 — no progress line (0 steps? just launched, killed ~4 min in). Wait — 171433 started 17:14:33? and killed at 17:17:12 — before any progress. Aborted immediately.
- b2 172716: same: started 17:27? killed 17:29 — before progress. Aborted immediately.

Hmm wait, but where is the b2 run that succeeded on meta2 data? Only these logs Aug/Sep 3 end with kills. Sizes 8291 each for 171433/172716. They have no progress bars - killed right at startup. Suspicious: maybe OOM right at startup? They were killed ~2 min after start. Return code -9 (killed). Possibly the user tried to launch on busy GPU then killed. Whatever, they're failed/no-progress runs.

Wait: did a successful "b2" with meta2 data exist later? There might be more logs after Sep 3? The listing shows b3 (Sep 4 194430) completed Sep 5 13:59, and b4a (Sep 7 060023) completed Sep 8 07:40. But no completed b2-meta2 run in training_logs. Hmm. Actually maybe eval_results/stage2_full_seqv3_mixed_b2 has metrics though. Wait — user's background says b2 has "meta2" data meaning 2-meta runs? Let's not overthink; the question was "哪个是完整成功 run，其余是什么" for b2 files listed: answer = 205231 is complete success; 112252, 150156 are interrupted attempts (killed -9 at 17%/37%); 142959 killed at 400 steps; 171433/172716 killed before any training step. Note: 171433/172716 logs might be identical (both 8291 bytes, same size). Interesting — killed 12 minutes apart. They are repeated attempts both killed within ~3-4 min (probably by a watcher script? mixed_eval_watcher?). Check if identical content.

- b1 213612: full start (no resume) killed -9 at 91% (3161/3474) Aug 31 13:48 — interrupted (probably to restart with adjusted config?). Then:
- b1 135828: resume from checkpoint-3000, completed 3474/3474, train_runtime 8936s. epoch 3.0 — success (resume completion). Note train_loss 0.148. This is the completion of run b1.
- b1 164030: resume from checkpoint-3474 (already complete!) — 0 steps, train_runtime 0.8s — effectively an immediate no-op run (run that resumes at final checkpoint → finishes instantly, saves final model?). It "exits successfully" — it's a trivial run to produce final merged/saved adapter state. So it's a "dummy/finalize" run. Not training. Value: minimal.

Interesting: so the "b1" training spans 213612 (interrupted at 91%) + 135828 (resume to completion). Both needed for full picture? The b1 model final = after 135828.

- seq resume Aug 23: no train_runtime found by grep, and no "exits successfully"? Odd. Let me look at the actual tail lines of resume_230654. The mtime Aug 23 12:37. Earlier grep pattern had epoch': 3 with trailing... my grep printed nothing for both patterns. Let me check tail of that file directly and count progress steps max. Also diff nohup twins: "differ" — but the nohup files have a wrapper header line ("PER-SEQUENCE TRAINING (no tf32)"), plus maybe an "End:" trailer. So they're twin captures with extra header/trailer. The sizes: 435483 vs 435643 (+160 bytes = header+End lines probably). And resume: 625702 vs 625842 (+140). So nohup files = same training stdout + wrapper header lines. Duplicates of same run (nohup双写).

Let me check:
1. tail of stage2_full_seq_resume_20260822_230654.log — real tail text (the earlier tail cut off?). Also grep max progress and any kill message.
2. b2 171433 vs 172716 identical?
3. b1_pipeline.log, b1_eval_pipeline.log, b2_pipeline.log, b3_pipeline.log, b4a_pipeline.log contents.
4. smoke_test.log (root) head/tail.
5. stage1_162k_b1 logs (Aug 30) — not in the user's list but they're in training_logs; the user mentioned files to judge are those listed. But we should note other log files like eval_stage2_full_seqv3_mixed.log, mixed_eval_watcher.log, stage1_162k_b1 logs. They're also candidates perhaps, but the user's list is specific. I'll add brief notes.

Also need to check the earlier stage2 (root training_logs) files: stage2_20260807_200248 vs stage2_final: both same run? Confirm stage2_final.log first line timestamp identical 20:02:53, yes both start Aug 7 20:02. Sizes differ (390624 vs 313154). One file starts with launcher logs (deepseek... from runner 607) while final starts directly with training output. So both same run: one was launched via nohup (captured the launcher + training) the other via... Actually likely the same command run twice? No — both start at exactly same second 20:02:53. So they're two tee'd outputs of the same process (e.g., `nohup ... > a.log 2>&1 &` and also `script`/another redirect), OR the user launched with both `2>&1 | tee`. Most likely duplicate captures of the same run — same run: the launcher in 200248 log shows deepspeed launch; final shows trainer output only, no launcher spam. Different capture modes of same run.

Hmm wait, actually looking again at 200248 head: it has real_accelerator + runner.py fetch_hostfile + cmd = ... deepspeed.launcher.launch. The stage2_final head: starts at fetch_hostfile too but then shows transformers FutureWarning, then "You are using a model of type llama..." Direct train script output. The deeper question: 200248.log tail ends with "Process 149724 exits successfully" at 01:40:18 and has param.ds_status warnings. stage2_final tail ends with progress bar 1072/1072 + train_runtime. So both ends at the same completion point. Yes, same run duplicated (one full stdout of launcher, one of the wrapper/trainer without the launcher's final line? or with). Anyway both = same run, completed 2 epochs at 01:40 Aug 8.

Let me now check the two are the same run even more carefully: 200248 has the trainer progress too? size 390KB similar to 313KB — both contain trainer progress. Actually stage2_final contains progressbar lines w/ tqdm 1072/1072. Whether 200248 contains train_runtime? It ends with exit at 01:40. I'd verify: grep 'train_runtime' in 200248. Also grep loss lines epoch 2.0.

Actually, note stage2_final log loss text shows 'loss': 0.2963 etc. Fine.

Now let me also figure out max progress & outcome for the 08-07 run — both exit around 01:40 with epoch 2.0 summary. Wait: train_runtime 20204s = 5.6h from 20:02 → ends 01:40. Yes! matches. epoch 2.0 = 1072 steps? If data 118,722 → 3 epochs was 2781 steps. 1072 steps ≈ 3.86 steps/epoch * ... hmm 1072 steps with epoch 2.0. Different: maybe data ~113k*2/...? num_train_epochs=... unknown. Whatever: run completed 2 epochs and exited successfully (epoch 2.0 means it finished... hmm 2.0 not 3.0). It exited successfully so completed whatever epochs were configured. Let me check 200248 for 'epoch 3' or num_train_epochs in the cmd. Not critical: both duplicate & completed successfully.

Next big tasks:
1. pipeline logs content
2. seq resume tail
3. smoke_test.log
4. du measurements
5. eval_results content listing
6. docs grep

Let me parallelize.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && echo "== b2 171433 vs 172716 =="; cmp -s stage2_full_seqv3_mixed_b2_20260903_171433.log stage2_full_seqv3_mixed_b2_20260903_172716.log && echo IDEN
…（已截断，原文 902 字符）
description: Compare b2 twin logs, seq resume tail, stage2 Aug7 markers
```

</details>

<details><summary>输出摘要</summary>

```
== b2 171433 vs 172716 ==
differ
1,10c1,10
< [2026-09-03 17:14:35,583] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
< [2026-09-03 17:14:38,415] [WARNING] [runner.py:215:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
< [2026-09-03 17:14:38,415] [INFO] [runner.py:607:main] cmd = /root/autodl-tmp/.conda-stuff/envs/wqlc/bin/python3.10 -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMF19 --master_addr=127.0.0.1 --master_port=29583 --enable_each_rank_log=None vtimellm/train/train_mem.py --deepspeed ./scripts/zero3.json --lora_enable True --model_name_or_path ./base_model/vicuna-v1-5-7b --version v1 --data_path ./b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json --feat_folder ../en
…（已截断，原文 6937 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && echo "########## b1_pipeline.log ##########"; cat training_logs/b1_pipeline.log; echo; echo "########## b1_eval_pipeline.log ##########"; cat training_logs/b1_eval_pi
…（已截断，原文 451 字符）
description: Show b1/b2/b3 pipeline log contents
```

</details>

<details><summary>输出摘要</summary>

```
Exit code 1
########## b1_pipeline.log ##########
cat: training_logs/b1_pipeline.log: No such file or directory

########## b1_eval_pipeline.log ##########
cat: training_logs/b1_eval_pipeline.log: No such file or directory

########## b2_pipeline.log ##########
cat: training_logs/b2_pipeline.log: No such file or directory

########## b3_pipeline.log (head 40) ##########
head: cannot open 'training_logs/b3_pipeline.log' for reading: No such file or directory
...
tail: cannot open 'training_logs/b3_pipeline.log' for reading: No such file or directory
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Ah, b1_pipeline.log files are in mllm/training_logs (from earlier listing). Also very important discovery:

- b2 171433 & 172716 are NOT identical but both are `--resume_from_checkpoint .../mixed-b3/checkpoint-400` — wait! They resume a b3 run (output_dir mixed-b3). These two logs are actually b3-phase resume attempts (run on Sep 3), that were killed after ~2.5 min before progress (17:14→17:17 and 17:27→17:29). Hmm, but they're named stage2_full_seqv3_mixed_b2. Odd naming: they resume `...-mixed-b3/checkpoint-400`, whole_scene True. The b3 final run then was stage2_full_seqv3_mixed_b3_20260904_194430.log (Sep 4). So these two "b2_20260903" logs are actually resume attempts of the b3 model that got killed. Interesting — they're mislabeled/misplaced, killed before training, no progress.

Wait, no. Let me reconsider: 142959 (Sep 3 14:29) used data seqv3_meta2_148k with no resume? Its output_dir — need to check. 142959 was killed after 400 steps. Hmm, 400 steps and no resume flag → output_dir probably mixed-b3? Let me check 142959 cmd output_dir. Actually 400 steps with save_steps 200 → checkpoint-400 exists. And 171433 resumes from mixed-b3/checkpoint-400. So the Sep 3 142959 run was training the mixed-b3 model (whole_scene? check), killed at step 400; the follow-up resumes (171433, 172716) from checkpoint-400 were also killed almost immediately (maybe GPU taken/conflicts). Then the real b3 run began Sep 4 194430 and completed Sep 5. Wait but earlier user says b3... The naming is confusing but evidence shows output_dir mixed-b3.

Actually wait — is that right? Let me confirm the output_dir of the Sep 3 runs: 171433's cmd: output_dir ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3 with whole_scene True. So yes. And what about 142959? need check. Also whole_scene flag True indicates "whole scene" mode - the b3 config feature.

Interesting: also the naming "b2_20260903" suggests the file names were assigned by a wrapper that mislabels, or the user reused wrapper script with wrong label. These are attempts for b3 line. But the user asked: "stage2_full_seqv3_mixed_b2_20260901_112252/_150156/_205231/_20260903_142959/_171433/_172716——同上(哪个是完整成功 run，其余是什么)". Answer: 205231 completed (Sep 1 20:52 → Sep 2 10:15, 13.3h, success, seqv3_148k data). The two Sep 1 runs were killed attempts (17% / 37%). The Sep 3 files: 142959 killed at 400 steps (that run actually targeted mixed-b3 per naming? verify), 171433/172716 both resumes of mixed-b3 checkpoint-400 killed before step 1 (their filenames say b2 but content targets b3 output — likely misnamed attempts preceding the real b3 run).

Let me verify 142959's output_dir & whole_scene.

- seq resume Aug 23 (230654) tail: ends mid-file with peft warnings and wandb sync message, no progress lines in tail? That means the last part of file contains warnings and... Actually wait, tail -c 1200 shows warnings + wandb message — no progress bar and no "exits" message. Odd for a file that's 625KB. Progress bars dominate size. Maybe the tail contains final checkpoint saving warnings. Let me look at actual last progress line and check for exit status: earlier grep of "exits successfully|End:|完成" printed nothing? In the earlier run, my grep output for resume file printed nothing visible... Actually in the second earlier output ("seq resume 08-23 tail summary" section) there was nothing printed after "===== seq resume 08-23 tail summary =====" header — meaning no train_runtime, no 'epoch 3', no exits/End lines. Hmm, that suggests the resume run was killed/crashed without the wrapper End line and without trainer summary. Let me confirm: max progress reached? Let me grep the file for the last "xx/2781" and for "Killing subprocess|return code". The mtime was Aug 23 12:37. Maybe run crashed with OOM (CUDA OOM) at the end — grep OOM.

Then the successful seq training (the run that ended fine) might be missing, and eval_results/stage2_full_seq presumably used the final checkpoint from checkpoint-1000-resumed run... but wait there may be later seq fine-tune "seq" eval results dated Aug 22/23. eval_log.txt (Aug 22 21:07) contains "seq_predictions.json.ckpt" — hmm that's in eval_results root. eval_results/stage2_full_seq dir mtime Aug 23 16:39. Let me check content there later.

So summary seq line: 08-12 两个文件 = GPU-env 故障尝试 (CPU-only, 未开始训练即失败/被终止); 08-22 094534 = 主 seq run 进行到 54% (epoch 1.6) 被 SIGKILL(-9) 于 18:59; 08-22 230654 resume = 从 checkpoint-1000 续训 — 需要确认结局。若 OOM/被杀则 seq 也没有完成版本？但 eval_results/stage2_full_seq 有产物 (Aug 23 16:39)。Let me check the resume run's ending carefully.

Also check whether the seq resume run saved checkpoints up to the end (checkpoints dir listing will show final checkpoint). Let me grep checkpoints dirs: mllm/checkpoints. Also we should check resume run max step: grep -oE "[0-9]+/2781 \[[0-9:]+" | tail.

- stage2 Aug 7 (root training_logs): num_train_epochs 2 — the 1072/1072 = 2 epochs × 536 steps? or so; run completed 2 epochs successfully at 01:40 Aug 8. Both stage2_20260807_200248.log and stage2_final.log same run (train_runtime present in final log). Wait grep -c train_runtime in 200248 = 1. Yes it also has summary. Both files complete. Duplicates of same run: yes.

Pipeline logs are at mllm/training_logs/. Rerun the cat with right paths.

</details>

The b2 "Sep 3" logs are actually mislabeled b3-phase resumes. Let me dig into seq resume ending, verify the 142959 target dir, and read the pipeline logs at the right path.

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && echo "== 142959 output_dir/whole_scene =="; grep -oE "output_dir ./checkpoints/[^ ]*|whole_scene True|stage2_full_train[^ ]*" stage2_full_seqv3_mix
…（已截断，原文 810 字符）
description: Check seq resume ending and 142959 target output dir
```

</details>

<details><summary>输出摘要</summary>

```
== 142959 output_dir/whole_scene ==
output_dir ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3
stage2_full_train_seqv3_meta2_148k.json
stage2_full_train_seqv3_meta2_148k.json',
whole_scene True
== seq resume 230654: last progress + errors ==
2780/2781 [13:27:19
2781/2781 [13:27:41
== seq 094534 had OOM? ==
0
== stage1_162k b1 logs endings ==
stage1_162k_b1_20260830_165411.log:[2026-08-30 17:15:47,431] [ERROR] [launch.py:325:sigkill_handler] ['/root/autodl-tmp/
stage1_162k_b1_20260830_185010.log:[2026-08-30 21:36:08,355] [INFO] [launch.py:351:main] Process 195126 exits successful
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && echo "########## b1_pipeline.log ##########"; cat b1_pipeline.log; echo; echo "########## b1_eval_pipeline.log ##########"; cat b1_eval_pipeline.lo
…（已截断，原文 363 字符）
description: Print contents of b1/b2/b3 pipeline logs
```

</details>

<details><summary>输出摘要</summary>

```
########## b1_pipeline.log ##########
===== 阶段1a: stage1 sample_token 特征重提 (2026-08-30 12:53:03) =====
阶段1a rc=0
stage1 特征数: 28130（预期 28,130）
===== 阶段1b: stage2 per-scene 特征重提 (2026-08-30 14:16:47) =====
阶段1b rc=0
stage2 特征数: 850（预期 850）
===== 阶段2: stage1 162K projector 重训 (2026-08-30 16:54:11) =====
阶段2 rc=247
阶段2 失败，链停止
===== 阶段2: stage1 162K projector 重训 (2026-08-30 18:50:10) =====
阶段2 rc=0
===== 阶段3: mixed-b1 混合重训 (2026-08-30 21:36:10) =====
阶段3 尝试 #1 失败，10 分钟后断点续训重试...
阶段3 尝试 #2 失败，10 分钟后断点续训重试...
阶段3 尝试 #3 失败，10 分钟后断点续训重试...
阶段3 三次尝试均失败，链停止

########## b1_eval_pipeline.log ##########
===== 阶段4: 同口径评测 b1 (2026-08-31 23:35:28) =====
阶段4 尝试 #1 rc=0
===== B1 链完成 (2026-09-01 04:10:30) =====
B1 评测结果（对照 B0：acc 0.7629 / mIoU 0.2696，ΔmIoU > +0.013 才显著）：
accuracy 0.7787 (B0 0.7629)
mIoU      0
…（已截断，原文 4367 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Very informative!

The pipeline logs reveal structure:
- b1_pipeline.log: 阶段1a/1b features extraction OK; 阶段2 stage1 162K projector retrain — first attempt rc=247 FAILED (this corresponds to stage1_162k_b1_20260830_165411.log killed 17:15); retry at 18:50 rc=0 OK (that's stage1_162k_b1_20260830_185010.log which exits successfully 21:36); 阶段3 mixed-b1 training at 21:36:10 (this is 213612 log) — three attempts failed → "链停止". Wait interesting: 阶段3 尝试 #1/#2/#3 失败 then stop at ~?; but we know 213612 ran from 21:36 Aug 30 to 13:48 Aug 31 (killed -9 at 91%!). Hmm — but pipeline says attempt #1 failed... Wait timeline confusion. Let me look at rest of b1_pipeline.log: "尝试 #1 失败，10 分钟后断点续训重试..." — those retries maybe correspond to early quick failures... but actual log 213612 shows one long run 16 hours that got SIGKILLed at 13:48 Aug 31. Then the resume at 13:58 (135828) succeeded to completion 16:30. But pipeline log claims all 3 attempts failed and chain stopped?! Hmm — b1_pipeline.log ends "阶段3 三次尝试均失败，链停止" and b1_eval_pipeline.log says 阶段4 started at 23:35 Aug 31 rc=0 → B1 chain completed Sep 1 04:10. So actually the eval happened 23:35 Aug 31 — after 135828 completed at 16:30. And pipeline b1_pipeline.log was probably written by a different (older/failed) driver, while actual driver logs b1_eval_pipeline.log for phase 4. Wait but b1_pipeline.log mtime Aug 31 16:53. It recorded 3 failed attempts in phase 3 then chain stopped, but b1_eval_pipeline.log (Sep 1 04:10) recorded phase 4 rc=0 and completion. So the phases may have been run manually/by separate scripts. The b1_pipeline.log is the record of "auto chain attempt" that failed phase 3 repeatedly (probably those failures = attempts at 13:58-16:30? no...).

Hmm — actually there were 3 stages in phase 3 attempts per pipeline. But we saw only 213612 (long run, killed at 91% = interrupted at 13:48 Aug 31) and 135828 resume completed at 16:30 and 164030 finalize at 16:43. Hmm... wait maybe the pipeline driver considered each "kill" = failure and retried via resume = attempt #2 (135828) — no it completed. Confusion: attempt #1 might = 213612 itself? The 16-hr run killed at 91% at 13:48 — hmm but who killed it? Possibly the pipeline watcher kills and resumes when GPU contention? Let me look at b3_pipeline.log tail and mixed_eval_watcher.log to understand. Actually, look: 213612 started 21:36:10 (matches 阶段3 start 21:36:10) and was killed at 13:48 Aug 31 (SIGKILL -9, probably by pipeline driver that misjudged?), then pipeline retried... hmm but pipeline log says attempts failed. Possibly the pipeline driver itself killed the long run (e.g., its own liveness watchdog), and retries 2,3 also killed quickly, and the pipeline then reported failure, and the user manually ran the resume 135828 to completion at 16:30, then finalize run 164030, then eval at 23:35 via a different script.

The b1_pipeline.log content says: 阶段2 rc=247 (first), then 阶段2 rerun rc=0 at 18:50, then 阶段3 at 21:36 → 尝试#1 失败 (probably = 213612 killed at 13:48? but wait there's no 10-min gap...). Actually maybe: the watcher script kills long-run train at 13:48 and restarts resume — that's 尝试 #2 — killed quickly at some point; 尝试 #3 (164030?) — hmm no, that one exits successfully with 0 steps.

Actually simpler interpretation: 尝试 #1 = 213612 (失败 = killed at 13:48 after 91%), 尝试 #2 = 135828 resume? No wait — that succeeded at 16:30... unless pipeline's definition of failure differed. Hold on: the mtime of b1_pipeline.log is Aug 31 16:53 — after 135828 (16:30) and 164030 (16:43). And b1_eval_pipeline.log mtime Sep 1 04:10. Since eval ran fine (阶段4 rc=0, results recorded), phase 3 must have effectively succeeded (via manual resumes). The b1_pipeline.log recording "三次尝试均失败，链停止" was likely the auto-driver's verdict while a parallel manual process did the actual successful resume; the driver's log is stale/contradicted by events.

This is quite deep. The user's ask for the pipeline logs: "内容分别覆盖哪些阶段，是否被后续 b4a_pipeline.log 或 eval_results 的正式产物取代". So:

- b1_pipeline.log (749B): covers stage1a/stage1b feature re-extraction + stage2 (stage1 162K retrain) + stage3 mixed-b1 training attempts — records 3 failed auto attempts, chain stopped. Its stage-3 content superseded by actual successful resume + later b4a pipeline? Partially — it documents the b1 training auto-chain failure. Historical value limited; final metrics live in b1_eval_pipeline.log and eval_results/.
- b1_eval_pipeline.log (357B): phase 4 eval results of B1 with B0 comparison → the actual metrics record. Small file with the headline numbers. This is later summarized in eval_results and possibly CLAUDE.md. This one might be the "正式产物" precursor. Need to see eval_results subdirs metrics_recomputed... Let's check what's in stage2_full_seqv3_mixed_b1/ dir.
- b2_pipeline.log: 阶段1 mixed-b2 retrain (尝试#1 #2 失败, #3 完成) + 阶段2 eval rc=0 + B2 metrics summary (Sep 2 13:45). This IS the authoritative b2 chain record (attempts 1,2 = 112252 & 150156 killed; #3 = 205231 complete; eval after). Keep as it's the run narrative.
- b3_pipeline.log (13.8KB): 阶段1 mixed-b3 retrain — waits for GPU (显存不足 loops from Sep 3 18:54 for 44+ times → likely till Sep 4 19:44 when b3 run 194430 started) then training etc. Need the tail to see its completion and stages.
- b4a_pipeline.log: b4a phases. Check content later.

Wait, but b2_pipeline.log says attempts #1, #2 failed then #3 完成. Meanwhile logs 112252 was killed at 14:51, 150156 at 20:42, 205231 completed. That matches pipeline 阶段1 at 11:22:50 (start of 112252), retries 10 min later... Actually 150156 started at 15:01:56 (~10 min after 14:51 kill? kill at 14:51:52 → 15:01 would be exactly 10 min). Yes! So the pipeline driver killed each attempt after some point?? Hmm — why would the driver kill? Look at b3_pipeline head: it waits for 显存 (VRAM) — when GPU memory not enough, it kills the run and retries 10 min later. The driver kills training when VRAM is low (i.e., GPU contention) — for b2: attempt #1 killed at 17% after 3.4h at 14:51... Actually wait, VRAM check every 10 min? The kill happened after 3h24m — perhaps the driver killed it because VRAM dropped below threshold at some moment (another process on GPU?). Anyway b2_pipeline.log documents: attempts failed, #3 completed. Good.

Now seq resume 230654: progress reached 2781/2781 [13:27:41] — it did complete all steps! But no train_runtime summary and no "exits successfully"? Let me check further: perhaps summary came after progress; grep for train_runtime failed earlier. Wait — earlier grep on that file for train_runtime printed nothing but the max progress shows 2781/2781 completed at 13:27:41 elapsed. Then the tail of the file shows peft warnings + wandb sync — looks like the final save & shutdown. But no summary line found?? The train summary would appear right after progress completes. But maybe the run hit an error at final save. File ends with wandb message. Let me print the actual last 20 lines raw (with -c 2000 maybe) to see if there's a summary or error after progress. Since tail -c 1200 showed only warnings + wandb, meaning after progress 2781/2781, there's ~<1200 bytes of warnings and then file end. Actually tail -c 1200 shows warnings + wandb; the progress line could be just before that. The summary {'train_runtime': ...} would be there. It printed not? My grep for train_runtime returned nothing — hmm that's weird. Let me double-check: maybe grep pattern "train_runtime" — resume file might not have it because it crashed during final checkpointing? Let me view raw last ~2500 bytes.

Also verify how the file ends exactly (is there "exits successfully" or crash?). And check the checkpoints for stage2-full-seq to see final state — do checkpoints exist up to 2781? Let me ls checkpoints dir.

Also do the eval_results/stage2_full_seq contents (metrics) to determine if the seq model is final.

Then smoke_test.log root file content; also eval_log.txt head, smoke_2000, smoke_seq logs; _meteor_sweep content; official_ckpt_nometa, official_model_stage2, stage23_seqv3 subdir contents; then du; then grep docs.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && echo "== seq resume raw last 2000 bytes =="; tail -c 2000 stage2_full_seq_resume_20260822_230654.log | tr '\r' '\n' | grep -vE "^\s*$" | tail -15; 
…（已截断，原文 341 字符）
description: Inspect seq resume file ending, b3 pipeline tail, b4a pipeline
```

</details>

<details><summary>输出摘要</summary>

```
== seq resume raw last 2000 bytes ==
s.NOT_AVAILABLE
WARNING:root:base_model.model.model.layers.30.mlp.gate_proj.lora_B.default.weight: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: ZeroParamStatus.NOT_AVAILABLE
WARNING:root:base_model.model.model.layers.30.mlp.up_proj.lora_B.default.weight: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: ZeroParamStatus.NOT_AVAILABLE
WARNING:root:base_model.model.model.layers.30.mlp.down_proj.lora_B.default.weight: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: ZeroParamStatus.NOT_AVAILABLE
WARNING:root:base_model.model.model.layers.31.self_attn.q_proj.lora_B.default.weight: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: ZeroParamStatus.NOT_AVAILABLE
WARNING:root:base_model.model.model.layers.31.self_attn.k_proj.lora_B.default.weight: param.ds
…（已截断，原文 7556 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: ls /root/autodl-tmp/wql/mmb4dl/mllm/checkpoints/ 2>/dev/null; echo ---; ls /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seq/ /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv
…（已截断，原文 669 字符）
description: List checkpoints and eval_results run directories
```

</details>

<details><summary>输出摘要</summary>

```
vtimellm-vicuna-v1-5-7b-stage1
vtimellm-vicuna-v1-5-7b-stage2
vtimellm-vicuna-v1-5-7b-stage2-full
vtimellm-vicuna-v1-5-7b-stage2-full-seq
vtimellm-vicuna-v1-5-7b-stage2-full-seq-merged
vtimellm-vicuna-v1-5-7b-stage2-full-seqv2
vtimellm-vicuna-v1-5-7b-stage2-full-seqv2-merged
vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed
vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b1
vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b2
vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b3
vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b4a
vtimellm-vicuna-v1-5-7b-stage2-seqv3
vtimellm-vicuna-v1-5-7b-stage2-seqv3-merged
vtimellm-vicuna-v1-5-7b-stage3-seqv3
---
/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seq/:
metrics.json
predictions.json

/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_f
…（已截断，原文 1728 字符）
```

</details>


### USER

The TodoWrite tool hasn't been used recently. If you're working on tasks that would benefit from tracking progress, consider using the TodoWrite tool to track progress. Also consider cleaning up the todo list if has become stale and no longer matches what you are working on. Only use it if it's relevant to the current work. This is just a gentle reminder - ignore if not applicable.


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Let me track progress with TodoWrite, then continue: smoke_test.log, seq resume tail conclusion, du measurements, eval_results root scattered files check, docs grep, then the final report.

Wait — the seq resume 230654 tail: file ends right at wandb sync message without train_runtime summary? Actually earlier grep for train_runtime found nothing. Hmm, wait. Actually I earlier grep'd "train_runtime|epoch': 3" and got nothing printed. Let me confirm whether the resume log contains a summary... The progress shows 2781/2781 completed. Then warnings during final save; then file ends abruptly at the wandb message. So run likely killed during final checkpoint save at 12:37 Aug 23 (mtime) — after training completed but before final summary/save? Weird. Let me check more carefully: search for last occurrence of "epoch 3" or check whether summary exists at all, e.g. "train_loss". If not, the run probably got SIGKILLed right after the final step (by pipeline watcher at 12:37). But note checkpoint-2781 probably saved at steps; save every 500 steps. Actually the tail shows warnings from PEFT save_and_load during a checkpoint save — that's a checkpoint-2500/2750-ish save mid-way? No—at the end. mtime 12:37. Summary absent.

Anyway for judging the file: the resume run executed all 2781 steps (13h27) but log ends w/o trainer summary/exit marker — killed during finalization (likely GPU contention watcher) — but training itself was complete. Outcome: training completed 3 epochs; final summary missing. 

Given the checkpoints dir has seq-merged: vtimellm-vicuna-v1-5-7b-stage2-full-seq-merged exists, meaning a merge step followed (maybe in another log or via script). Actually checkpoints listing shows "vtimellm-vicuna-v1-5-7b-stage2-full-seq-merged" — good. But no explicit "merged" log in training_logs listing. Hmm; whatever: seq lineage completed with merge.

Now, before finalizing the per-file judgment table, I still need:
1. smoke_test.log (root) head/tail.
2. du measurements.
3. eval_results root scattered files analysis + _meteor_sweep contents + official dirs content, eval_log.txt head, and whether stage2_full_seq etc. subdirs have newer same-type artifacts (yes: metrics.json/predictions.json Aug 23-ish; check mtimes & compare timestamps in headers). Also stage2_full_seqv2/eval_log.txt.
4. docs grep referencing file names.
5. Also check .claude or wiki? User said docs/, README*.md, CLAUDE.md, *.md root files. Also maybe docs subdirs.

Also check remaining uncertain judgments: stage1_95k first file end (killed or error?) — earlier grep found no error lines... Actually I did grep for SIGTERM|return code|exits successfully|Killing|训练完成|训练时间 in 161616 and got nothing. So the log just ends at step 8/742 — but mtime 16:19 — the run was killed (no error lines; SIGKILL leaves no trace). Then 162055 = rerun with master_port 29571 too but started 16:21? Hmm wait — 162055 started at 16:21, four minutes after 161616 was killed at 16:19? Wait 161616 log file name timestamp 16:16:16 = launch time 16:16. mtime 16:19. Steps 1-8 took 01:16 → 16:17:32 + startup... The kill must have occurred ~16:19 (mtime). Then second run started 16:20:59 per log head (name says 162055). Yes.

Interesting: 161616 (95k data) reached only step 8 and stopped. Was it killed due to some failure? or maybe it actually OOM'd? No error text in file. Whatever: interrupted early; the retry 162055 succeeded (742/742, summary, exits successfully at 17:46). So answer: 162055 is the successful one.

One more check on seqv2 131601's wrapper "End ... (manual)" — it says process exited successfully at 09:46 then "End: Tue Aug 25 12:25:36 (manual)" — the wrapper's End line at 12:25 (manual) probably means wrapper script finished 12:25 by manual Ctrl... or script ended 12:25 and the log mtime is 12:25. Wait mtime says Aug 25 12:25. And exit at 09:46?? train_runtime 73656s = 20.46h from 13:16 Aug 24 → ends 09:46 Aug 25. Yes consistent. Then why "End ... (manual)" at 12:25? Script waited 2h40 after trainer exit? Probably the wrapper ran some final steps (e.g., merge or just echo End when user Ctrl+C'd at 12:25). Not important: the training run completed fully and successfully at 09:46. The "(manual)" End is wrapper nuance. Hmm, wait actually maybe the wrapper slept... Whatever.

Also b1 batch: 213612 has no resume flag (fresh start), killed -9 13:48 (91%). But wait — b1_pipeline.log says "阶段3 尝试 #1 失败，10 分钟后断点续训重试... 尝试 #2 失败... 尝试 #3 失败，三次尝试均失败，链停止". If 213612 = 尝试 #1 (killed at 13:48 Aug 31 after 16h)... but hold on, mtime of b1_pipeline.log is Aug 31 16:53. And attempts #2/#3? Between 13:48 and 16:53 there are two more runs: 135828 (13:58→16:30, succeeded, exits successfully) and 164030 (16:40→16:43, success trivial resume from 3474). So attempts #2 and #3 (which would be at 13:58+10min=14:08? and 16:43+?)... doesn't line up. Alternatively attempt numbering: #1 = 213612 fresh run killed at 13:48 (killed BY the pipeline driver? e.g., driver kills long-running then resumes with --resume_from_checkpoint). Actually maybe: driver runs train; every 10 min checks VRAM/gpu; when conflict detected (e.g., VRAM dropped because someone else grabbed GPU?) it SIGKILLs and after 10 min retries with resume flag. Hmm but 213612 ran 16 h uninterrupted... 

Let me not over-model the driver. What matters: b1_pipeline.log claims three failed attempts at phase 3 (matches: 213612 killed at 13:48 [attempt #1], 135828 = resume attempt — wait, but that SUCCEEDED at 16:30. Unless the pipeline driver deemed attempt #2 failed too at some point... 135828 has no kill (exits successfully). Confusing.

Alternatively the three "attempts" in b1_pipeline.log could map to something else entirely (tries at 21:36, then +10 min, +10 min?). But the only logs are the three above. And b1_eval_pipeline.log (mtime Sep 1 04:10) says 阶段4 rc=0 and prints B1 results — meaning the model existed (from resume runs) and eval was done. The b1_pipeline.log's "chain stopped" statement conflicts with b1_eval_pipeline.log's success record — the phases were probably run by different driver invocations, and b1_pipeline.log is the record of the failed auto-chain (superseded by manual resume + separate eval pipeline). So b1_pipeline.log is mostly superseded narrative; b1_eval_pipeline.log holds B1 metrics — later duplicated into eval_results/stage2_full_seqv3_mixed_b1/metrics.json and CLAUDE.md possibly.

OK. Also mixed_eval_watcher.log and eval_stage2_full_seqv3_mixed.log (2.7MB) — eval of the seqv3 mixed model (B0). Not in the requested list but part of training_logs. And stage1_162k_b1 logs (2 files): first (165411) killed rc? (SIGKILL at 17:15 — grep shows sigkill_handler ERROR line), second (185010) exits successfully 21:36. The successful one corresponds to 阶段2 rc=0 in b1_pipeline. Also stage3_seqv3 log (Aug 26, 628KB) complete? Not in list; skip details but note. eval watcher log too. I'll keep focus on requested items, adding brief notes for the others as optional.

Also the root-level training_logs has stage2_trainer_state.json — not a log per se. Leave out; mention.

Also seqv3 mixed big log: stage2_full_seqv3_mixed_20260826_131938.log (1.1MB, Aug 27 12:14 mtime) — 1300+ lines? It's the B0 mixed run (Aug 26-27), not requested but related. And stage2_seqv3_20260825_174612.log, stage3_seqv3_20260826_002520.log.

Given time constraints, focus on requested items; du next; then docs grep; then eval_results analysis; then smoke test log; then final answer.

Let me do du now in one command (read-only), plus smoke_test tail, plus _meteor_sweep listing and eval_log head.

</details>

**🔧 TodoWrite**

<details><summary>输入</summary>

```
{"todos": [{"content": "Judge per-file outcome for all listed training logs", "status": "completed", "priority": "high"}, {"content": "Disk usage measurements (du) for eval_results/training_logs/wandb/checkpoints etc.", "status": "in_progress", "priority": "high"}, {"content": "Analyze eval_results root scattered files vs run subdirs", "status": "pending", "priority": "high"}, {"content": "Grep docs/README/CLAUDE.md for references to cleanup candidates", "status": "pending", "priority": "high"},
…（已截断，原文 598 字符）
```

</details>

<details><summary>输出摘要</summary>

```
{"oldTodos":[],"todos":[{"content":"Judge per-file outcome for all listed training logs","status":"completed","priority":"high"},{"content":"Disk usage measurements (du) for eval_results/training_logs/wandb/checkpoints etc.","status":"in_progress","priority":"high"},{"content":"Analyze eval_results root scattered files vs run subdirs","status":"pending","priority":"high"},{"content":"Grep docs/README/CLAUDE.md for references to cleanup candidates","status":"pending","priority":"high"},{"content":"Compile final safe-delete vs keep lists","status":"pending","priority":"high"}],"summary":{"total":5,"pending":3,"inProgress":1,"completed":1}}
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && echo "== eval_results total =="; du -sh mllm/eval_results/ 2>/dev/null; echo "== per run subdir =="; du -sh mllm/eval_results/*/ 2>/dev/null; echo "== eval_results ro
…（已截断，原文 447 字符）
description: Disk usage of eval_results tree
```

</details>

<details><summary>输出摘要</summary>

```
== eval_results total ==
94M	mllm/eval_results/
== per run subdir ==
4.8M	mllm/eval_results/_meteor_sweep/
2.5M	mllm/eval_results/official_ckpt_nometa/
4.0K	mllm/eval_results/official_model_stage2/
12M	mllm/eval_results/stage23_seqv3/
8.0M	mllm/eval_results/stage2_full_seq/
11M	mllm/eval_results/stage2_full_seqv2/
8.0M	mllm/eval_results/stage2_full_seqv3_mixed/
14M	mllm/eval_results/stage2_full_seqv3_mixed_b1/
11M	mllm/eval_results/stage2_full_seqv3_mixed_b2/
11M	mllm/eval_results/stage2_full_seqv3_mixed_b3/
12M	mllm/eval_results/stage2_full_seqv3_mixed_b4a/
== eval_results root files ==
884K	total
4.8M	mllm/eval_results/_meteor_sweep
2.5M	mllm/eval_results/official_ckpt_nometa
4.0K	mllm/eval_results/official_model_stage2
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && echo "== training_logs =="; du -sh mllm/training_logs training_logs 2>/dev/null; echo "== wandb =="; du -sh mllm/wandb wandb 2>/dev/null; ls mllm/wandb | grep -c offl
…（已截断，原文 565 字符）
description: Disk usage of training_logs, wandb, checkpoints, backups, dataset, top dirs
```

</details>

<details><summary>输出摘要</summary>

```
== training_logs ==
15M	mllm/training_logs
876K	training_logs
== wandb ==
137M	mllm/wandb
76K	wandb
26
== checkpoints ==
145G	mllm/checkpoints
61M	mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage1
2.4G	mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2
4.1G	mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full
13G	mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seq
13G	mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seq-merged
13G	mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv2
13G	mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv2-merged
556M	mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed
13G	mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b1
13G	mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage2-full-seqv3-mixed-b2
13G	mllm/checkpoin
…（已截断，原文 1513 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Disk usage measured. Note root top-level listing didn't show eval mllm subdirs etc. mllm/ 159G total but checkpoints 145G + b4dl_dataset 1.6G + wandb 137M + training_logs 15M + eval_results 94M + evaluation 25M. There's still ~12G unaccounted in mllm (features? encoders data? maybe mllm has other dirs). Fine — user asked only listed ones; report the top-level sums.

Check root-level wandb: 76K, root training_logs: 876K (root has 3 files). backups 898M. Top 10 of one-level dirs of project root: mllm 159G, models 15G, encoders 11G, .git 1.5G, backups 898M, .claude 601M, dataset 384M, datageneration 76M, assets 17M, docs 8.8M. Wait but these are only root-level children; also wandb 76K, training_logs 876K, Claude_record 112K. Also smoke_test.log 20K etc. "项目总占用 du -sh 一级子目录排序前10" — yes list above (need absolute ordering; there may be encoders' 11G etc. already listed). models 15G? Wait top-level "models/" 15G: base model vicuna? Yes model weights.

Now: offline-run count in mllm/wandb = 26.

Next: eval_results root scattered files:
- eval_log.txt (479KB Aug 22 21:07): head to see what it covers. Also compare: run dirs have own eval_log.txt. Which run did the root eval_log.txt correspond to? Probably stage2_full_seq eval (Aug 22 21:07 same time as seq_predictions.json.ckpt). And stage2_full_seq/metrics.json mtime? Need stat mtimes.
- seq_predictions.json.ckpt + smoke_predictions.json.ckpt + smoke_2000_* + smoke_meta_* + smoke_seq_*: all under eval_results root. Compare with per-run dirs products: predictions.json in stage2_full_seq dir with same timestamp (Aug 22 21:07?) If stage2_full_seq/predictions.json exists with same content/size as seq_predictions.json.ckpt → duplicate. Let me compare sizes and timestamps and content hash quickly (md5 of the two pairs is fine read-only).
- _meteor_sweep: contains hyp.txt/ref.txt etc. Check files and whether the recomputed metrics (metrics_recomputed_20260907_dual.json) superseded it: check recomputed file names inside run dirs and mtime Sep 7 21:26-21:29 (dir mtimes). Compare content briefly.
- official_ckpt_nometa and official_model_stage2 dirs: eval results referencing official models? not in the requested file list, but sizes matter (2.5M, tiny) — for completeness of eval_results structure. Not requested to judge; note their presence.

Also check smoke_test.log in root. And docs grep. Let me proceed.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/eval_results && echo "== root file mtimes & sizes =="; ls -la --time-style=long-iso *.json *.ckpt *.txt 2>/dev/null; echo "== _meteor_sweep =="; ls -la --time-style
…（已截断，原文 539 字符）
description: List eval_results root files with timestamps and subdir contents
```

</details>

<details><summary>输出摘要</summary>

```
== root file mtimes & sizes ==
-rw-r--r-- 1 root root 479487 2026-08-22 21:07 eval_log.txt
-rw-r--r-- 1 root root 692905 2026-08-22 21:07 seq_predictions.json.ckpt
-rw-r--r-- 1 root root  30254 2026-08-23 13:21 smoke_2000_log.txt
-rw-r--r-- 1 root root    991 2026-08-23 13:21 smoke_2000_metrics.json
-rw-r--r-- 1 root root  84975 2026-08-23 13:21 smoke_2000_predictions.json
-rw-r--r-- 1 root root    947 2026-08-09 01:01 smoke_meta_metrics.json
-rw-r--r-- 1 root root   5127 2026-08-09 00:53 smoke_meta_predictions.json
-rw-r--r-- 1 root root  12467 2026-08-22 21:09 smoke_predictions.json.ckpt
-rw-r--r-- 1 root root  24866 2026-08-22 21:23 smoke_seq_log.txt
-rw-r--r-- 1 root root   1056 2026-08-22 21:23 smoke_seq_metrics.json
-rw-r--r-- 1 root root  84274 2026-08-22 21:23 smoke_seq_predictions
…（已截断，原文 3229 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/eval_results && echo "== dup check: root ckpt vs run-dir json =="; for pair in "seq_predictions.json.ckpt stage2_full_seq/predictions.json" "smoke_seq_predictions.j
…（已截断，原文 1009 字符）
description: Compare root eval artifacts with run-dir products, preview logs
```

</details>

<details><summary>输出摘要</summary>

```
<persisted-output>
Output too large (470.5KB). Full output saved to: /root/.zcode/cli/exec/sess_subagent_agent_47e1ce38-7b3e-4a3d-a2f2-ebc1fbed0d41/call_01_YkTgCbDNBIJJaLX2Nma35942-stdout.log

Preview (first 2KB):
== dup check: root ckpt vs run-dir json ==
-- seq_predictions.json.ckpt vs stage2_full_seq/predictions.json:
692905 seq_predictions.json.ckpt
8318518 stage2_full_seq/predictions.json
 json-keys: ['results'] | ['existence', 'binary_qa', 'time_grounding', 'description', 'temporal_understanding']
-- smoke_seq_predictions.json vs stage2_full_seq/predictions.json:
84274 smoke_seq_predictions.json
8318518 stage2_full_seq/predictions.json
 json-keys: ['existence', 'binary_qa', 'time_grounding', 'description', 'temporal_understanding'] | ['existence', 'binary_qa', 'time_grounding', 'desc
…（已截断，原文 2186 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

The output got too big due to my python printing whole json. Need to avoid that. Key info so far:

- eval_log.txt (root, 479KB, Aug 22 21:07): eval of 30145 test QA items — big full eval (probably on stage2-full-seq model since seq_predictions.json.ckpt at same time and size 692KB vs 8.3MB full predictions). Actually seq_predictions.json.ckpt is 692KB with key 'results'; that was the ckpt intermediate of the seq full eval (savepoint while eval running?). stage2_full_seq/predictions.json is 8.3MB with task-level keys (existence, binary_qa, ...) mtime Aug 23 16:38/16:39 — later, the formal per-run results + metrics.json. And smoke_seq_predictions.json (84KB) also has those keys but smaller (smoke = subset). Root eval_log.txt at 21:07 same time as seq_predictions ckpt.

Wait actually more precisely: root eval_log.txt + seq_predictions.json.ckpt were generated by an eval run that started earlier Aug 22 and saved ckpt 21:07-21:09. smoke_seq (21:23) is a quick smoke on the same model. Then a *new* formal eval ran Aug 23 (16:38) writing into stage2_full_seq/{predictions.json,metrics.json}. So root-scattered files were pre-dir-era outputs; the run dirs are the formal successors. Both "same nature". Compare: which model was stage2_full_seq eval on? metrics.json contains details — I can read it.

Also there's no eval_log.txt inside stage2_full_seq/ (only metrics+predictions) — while other run dirs (stage2_full_seqv2, b1/b2/b3/b4a, stage23_seqv3, official_ckpt_nometa) include eval_log.txt. So root eval_log.txt might be the only detailed log of that seq eval... but wait the seq eval (Aug 23 16:38) was a rerun of the full eval; the root eval_log from Aug 22 covers an earlier interrupted/different eval? Both full-size. Actually root eval_log.txt probably corresponds to the same Aug-22 eval whose ckpt is seq_predictions.json.ckpt; this eval might have been killed at 21:07 (log ends? ckpt saved). Then smoke_seq at 21:23 was a 2k-sample smoke of the same seq model. Then on Aug 23 the formal eval rerun to completion into stage2_full_seq/.

Wait, seq_predictions.json.ckpt 692KB with 'results' structure might be a partial checkpoint of predictions (30k items → 8MB json; 692KB = partial). So yes, the Aug 22 eval was interrupted at 21:07 (saving ckpt), superseded by the complete Aug 23 formal eval in run dir.

Check root eval_log.txt tail to see how it ended (crash or completion). Also read the last lines of the persisted output file? No, I should rerun a smaller analysis. Let me:

1. tail of eval_log.txt.
2. cat stage2_full_seq/metrics.json and stage2_full_seqv2/metrics.json heads (small).
3. Determine whether stage2_full_seq dir lacks eval_log (yes), so the seq full-eval log isn't archived elsewhere... but the formal metrics exist. Root eval_log.txt might be the record of a failed eval. Even so, deleting is risky? It's a log of an interrupted run superseded by metrics.json. I'll mark as candidate but note it documents the full-eval pipeline output; there is another eval_log in stage2_full_seqv2 with similar structure.
4. _meteor_sweep: hyp.txt/ref.txt Sep 7 00:19 + sweep scripts + metrics_recomputed_20260907_dual.json produced Sep 7 21:30 in run dirs (after sweep). So the sweep's outcome is incorporated into metrics_recomputed_20260907_dual.json files in b1/b2/b3/mixed dirs. hyp.txt 2.3MB + ref.txt 2.6MB are intermediate. Confirm ref/hyp content relate to dual-meteor recompute. Quick peek first lines of sweep.py/sweep2.py and head of hyp/ref, and check metrics_recomputed_20260907_dual.json content keys to establish supersession.
5. smoke_test.log root: quick head/tail.
6. docs grep.

Also quickly verify seqv2 run dir metrics & stage23_seqv3 and official_ckpt_nometa exist as formal products (already listed). Also which runs dirs DO have eval_log: stage2_full_seqv2, stage23_seqv3, b1..b4a, official_ckpt_nometa. So root eval_log.txt (Aug 22, seq full eval interrupted) — the seq full eval log final version? The Aug 23 16:38 eval log would have been at root? no—only metrics+predictions in dir. Hmm, maybe eval on Aug 23 wrote log to root as eval_log2? no.

Let me just inspect tail of root eval_log.txt to see if interrupted, and its content model (which checkpoint it loads: probably stage2-full-seq merged). Enough to say: superseded-by formal seq products but is the only detailed log of that specific full-eval run... Actually if interrupted, not valuable.

One nuance: b1..b4a dirs eval_log.txt exist (6MB each) — likely those evals' detailed logs. root eval_log.txt 479KB is Aug 22 (seq model eval), formal = stage2_full_seq/{predictions,metrics}. It's likely the run that produced seq_predictions.json.ckpt. Was the whole 30k eval interrupted then? Aug 22 21:07 (ckpt), and resumed/redone Aug 23 16:38 complete (predictions.json 8.3MB). Check root eval_log tail for interruption marker (e.g., "Killed", "Error", or just ends). Let me check.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/eval_results && echo "== root eval_log.txt tail =="; tail -c 1500 eval_log.txt | tr '\r' '\n' | grep -vE "^\s*$" | tail -12; echo; echo "== stage2_full_seq/metrics.
…（已截断，原文 488 字符）
description: Inspect eval_log tail and formal metrics files
```

</details>

<details><summary>输出摘要</summary>

```
== root eval_log.txt tail ==
binary_qa:  28%|██▊       | 2081/7525 [04:34<25:46,  3.52it/s, ckpt=✓, rate=7.7/s, eta=12m]
binary_qa:  28%|██▊       | 2082/7525 [04:34<26:00,  3.49it/s, ckpt=✓, rate=7.7/s, eta=12m]
binary_qa:  28%|██▊       | 2083/7525 [04:34<28:48,  3.15it/s, ckpt=✓, rate=7.7/s, eta=12m]
binary_qa:  28%|██▊       | 2084/7525 [04:35<25:44,  3.52it/s, ckpt=✓, rate=7.7/s, eta=12m]
binary_qa:  28%|██▊       | 2085/7525 [04:35<21:30,  4.22it/s, ckpt=✓, rate=7.7/s, eta=12m]
binary_qa:  28%|██▊       | 2086/7525 [04:35<22:17,  4.07it/s, ckpt=✓, rate=7.7/s, eta=12m]
binary_qa:  28%|██▊       | 2087/7525 [04:35<23:45,  3.81it/s, ckpt=✓, rate=7.7/s, eta=12m]
binary_qa:  28%|██▊       | 2088/7525 [04:35<22:46,  3.98it/s, ckpt=✓, rate=7.7/s, eta=12m]
binary_qa:  28%|██▊       | 2089/75
…（已截断，原文 4259 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/eval_results/_meteor_sweep && head -25 sweep.py; echo "== sweep2 diff-ish =="; diff sweep.py sweep2.py | head -10; echo "== hyp/ref first lines =="; head -2 hyp.txt
…（已截断，原文 245 字符）
description: Inspect meteor sweep scripts and data
```

</details>

<details><summary>输出摘要</summary>

```
#!/usr/bin/env python
"""B0 (stage2_full_seqv3_mixed) 预测上的 Meteor-1.5 变体扫描。
目的：论文 METEOR=0.275 vs 当前 jar 口径 0.1729，考证论文所用变体。
用法: python sweep.py <variant...>  可选: default nonorm adq hter nopara exact
"""
import json
import subprocess
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
JAR = "/root/autodl-tmp/.conda-stuff/envs/wqlc/lib/python3.10/site-packages/pycocoevalcap/meteor/meteor-1.5.jar"
PRED = "/root/autodl-tmp/wql/mmb4dl/mllm/eval_results/stage2_full_seqv3_mixed/predictions.json"
TEXT_TASKS = ["description", "temporal_understanding", "comprehensive_reasoning"]

VARIANTS = {
    # 论文当前口径：语料级 EVAL 聚合 ≈ 批模式 Final score
    "default": ["-l", "en", "-norm"],
    "nonorm":  ["-l", "en"],
    "adq":     ["-l", "en", "-norm", "-t", "adq"],
    "hter":    ["-
…（已截断，原文 1904 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Now I have a very clear picture:

- _meteor_sweep: contains hyp.txt (predictions side), ref.txt (references) — text dumps from the B0 predictions (stage2_full_seqv3_mixed/predictions.json) for Meteor variant scanning, plus sweep.py/sweep2.py scripts. This scan was done Sep 7 00:19; results: metrics_recomputed_20260907_dual.json files in run dirs (Sep 7 21:26-21:31) — which include "meteor_pycocoevalcap" numbers. So the sweep outcome is固化 in the recomputed files. hyp.txt 2.3MB + ref.txt 2.6MB + scripts → the scripts may still be wanted for reproducibility but the hyp/ref data can be regenerated from predictions.json. Check: does the dual recompute exactly correspond to sweep result? The recomputed files are in b1/b2/b3 + mixed dirs (Sep 7 21:26+). mixed dir recomputed 20260907_dual too. That means the final canonical numbers now use dual meteor (meteor + meteor_pycocoevalcap). The sweep likely settled on "meteor_pycocoevalcap" as the reference for paper comparison. I'd call _meteor_sweep = disposable intermediate except scripts could be worth keeping. I'll mark whole dir as safe-to-delete with note (regenerable, results固化 in metrics_recomputed_20260907_dual.json), but it's only 4.8MB.

Hmm — the sweep compares B0. But dual recompute files exist for b1/b2/b3 and mixed. Fine.

- eval_log.txt root: seq full-eval run log interrupted mid binary_qa at 2092/7525 (28%) on Aug 22 21:07 — interrupted eval run (that's why ckpt seq_predictions.json.ckpt is partial with 'results' key). Superseded by stage2_full_seq/metrics.json + predictions.json (complete, Aug 23 16:38-39). So root eval_log.txt + seq_predictions.json.ckpt = leftovers of interrupted eval. smoke_predictions.json.ckpt (Aug 22 21:09, 12KB) = another ckpt leftover of the same interrupted eval (maybe per-task ckpt). smoke_seq_* (21:23) = smoke subset eval outputs of same seq model (metrics + predictions + log) — quick check that worked. But note stage2_full_seq/ dir has NO eval_log.txt and no smoke; so the smoke_seq files are the only quick-eval artifacts — but they're superseded by full metrics.

- smoke_meta_* (Aug 9, 947B + 5KB): smoke eval of the very early stage2 (Aug 9) model; probably superseded by everything later. smoke_2000_* (Aug 23 13:21): 2000-sample eval of seqv2? or stage2-full-seq model? "smoke_2000" Aug 23 13:21 — between seq resume end (12:37) and seq full eval (16:38)? Might be on seq model. Small.

- Root smoke_test.log (in project root): 20KB Aug 9 16:17 — 推理 smoke test from Aug 9. Will check head/tail quickly and grep docs referencing smoke files.

Now the smoke_test.log, plus what do run dirs' predictions.json contain for smoke? Doesn't matter.

Then docs grep. Files to grep: docs/ tree, README*.md, CLAUDE.md, *.md at root, wiki/ maybe (user said docs/ README CLAUDE md root files). I'll include wiki? The user listed: docs/ (含 docs/learn docs/ 子目录), README*.md, CLAUDE.md, *.md 根文件. I'll stick to those, but also .claude? No—that's chat history, not docs. Also mllm/README? README*.md pattern — root-level has README.md only; but find any README in repo.

Search patterns: the exact candidate filenames + tokens: "smoke_", "stage1_95k", "seqv2", "stage2_20260807", "b1_pipeline", "predictions_full", "evaluation_results.json", "eval_log", "metrics_recomputed", "stage2_full_resume", "stage2_full_seq", "stage2_full_seqv3_mixed_b1"... Report docs that reference paths that are candidates for deletion.

Approach: use grep -rn with -l across the doc files, then examine contexts. Let me list docs files first and then grep.

</details>

Now the docs/reference check and remaining small verifications.

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && find docs -type f | head -40; echo "---md files---"; find . -maxdepth 2 -name "*.md" -not -path "./.git/*" -not -path "./.claude/*" -not -path "./mllm/*" 2>/dev/null 
…（已截断，原文 277 字符）
description: List docs and markdown files for reference check
```

</details>

<details><summary>输出摘要</summary>

```
docs/B4DL_复现方案.md
docs/mmb4dl.pdf
docs/learn docs/B4DL_vs_LiDAR-LLM_comparison.md
docs/learn docs/VTimeLLM_analysis.md
docs/learn docs/B4DL_论文对齐审计_20260824.md
docs/learn docs/B4DL_seqv2评测对比与坍缩分析_20260825.md
docs/learn docs/B4DL_两阶段训练评测与混合法决策_20260826.md
docs/learn docs/B4DL_seqv3混合训练评测分析_20260827.md
docs/learn docs/B4DL_训练方法对比_论文vs复现_20260828.md
docs/learn docs/B4DL_评测数值对比与问题清单_20260828.md
docs/learn docs/B4DL_LiDARCLIP权重核对_20260828.md
docs/learn docs/B4DL_基线锁定与对比规则_20260829.md
docs/learn docs/B4DL_待修改清单_20260829.md
docs/learn docs/B4DL_全仓库论文对齐审查_20260829.md
docs/learn docs/B4DL_基线版本训练方法对比_20260830.md
docs/learn docs/B4DL_B3整场景meta修复mIoU超论文_20260906.md
docs/learn docs/B4DL_工作进度总结_20260906.md
docs/learn docs/B4DL_METEOR双口径溯源与评测规则_20260907.md
docs/learn docs/B4DL_训练代码全景解析_20260907.md
docs/le
…（已截断，原文 2364 字符）
```

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && echo "== smoke_test.log head =="; head -25 smoke_test.log; echo "== tail =="; tail -c 800 smoke_test.log | tr '\r' '\n' | grep -vE "^\s*$" | tail -8; echo "== evaluat
…（已截断，原文 256 字符）
description: Inspect smoke_test.log and root evaluation_results.json
```

</details>

<details><summary>输出摘要</summary>

```
== smoke_test.log head ==
You are using a model of type llama to instantiate a model of type VTimeLLM. This is not supported for all configurations of models and can yield errors.

Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
Loading checkpoint shards:  50%|█████     | 1/2 [00:00<00:00,  4.94it/s]
Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  5.82it/s]
Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  5.66it/s]
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new lm_head weights will be initialized from a multivariate normal distributi
…（已截断，原文 3280 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

smoke_test.log (root): inference smoke test (50 items per task? shows existence: 50) comparing paper vs ours — final table printed with paper vs ours on smoke subset. It's an Aug 9 early smoke (50 items). Value: historical debug record; the numbers are garbage-level (from early model), no docs reference likely. Small 20KB.

Root evaluation_results.json: trivial all-1.0 values (test harness output, likely dummy) Aug 8. Not in requested list, but I can mention it's a toy result. Careful: user didn't list it. Skip or 1-line mention.

Now the docs grep. Patterns to search in docs + root md files + wiki? The user specified: docs/ (含 "docs/learn docs/"), README*.md, CLAUDE.md, *.md root files. I'll also check mllm/README.md since it might matter (it's an md). But user explicitly limited scope to those sets + root *.md. I'll include mllm/README.md as a bonus but that's inside mllm (not candidates for deletion there). Fine.

Files: docs/*.md (B4DL_复现方案.md), docs/learn docs/*.md (the B4DL_* series), CLAUDE.md, README.md, Claude_record dir? not requested. wiki not requested. Focus:

grep -nE "smoke_|stage1_95k|seqv2|seqv3_merge|stage2_20260807|stage2_final|b[0-9]_pipeline|predictions_full|evaluation_results.json|eval_log|metrics_recomputed|stage2_full_resume|_meteor_sweep|hyp.txt|stage2_full_seq|stage2_full_seqv2|stage2_full_seqv3_mixed|training_logs/|eval_results" over those md files, and report references to files that are candidates for deletion.

Better: for each md file, list matches where they refer to specific files under training_logs or eval_results root artifacts. Many will refer to run dirs like eval_results/stage2_full_seqv3_mixed_b1/metrics.json — those dirs are NOT candidates (they're formal). I need doc refs that point at:
- training_logs/*.log files we're judging for deletion
- eval_results root scattered files
- _meteor_sweep
- root training_logs logs
- smoke_test.log
- root evaluation_results.json

Let me grep precisely.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && grep -rnE "smoke_|smoke_test|_meteor_sweep|stage1_95k|seqv2_nohup|stage2_full_resume|stage2_full_20260809|stage2_20260807|stage2_final|b1_pipeline|b2_pipeline|b3_pipe
…（已截断，原文 418 字符）
description: Grep docs for references to candidate cleanup filenames
```

</details>

<details><summary>输出摘要</summary>

```
docs/learn docs/B4DL_基线锁定与对比规则_20260829.md:124:- 评测驱动：`mllm/scripts/run_mixed_eval_after_train.sh`；评测日志：`mllm/training_logs/eval_stage2_full_seqv3_mixed.log`
docs/learn docs/B4DL_seqv3混合训练评测分析_20260827.md:144:- 评测日志：`mllm/training_logs/eval_stage2_full_seqv3_mixed.log`
docs/B4DL_复现方案.md:608:    --metrics_output ./evaluation/evaluation_results.json \
docs/learn docs/B4DL_B3整场景meta修复mIoU超论文_20260906.md:89:- 训练日志：`mllm/training_logs/stage2_full_seqv3_mixed_b3_*.log`、`b3_pipeline.log`
docs/learn docs/B4DL_评测数值对比与问题清单_20260828.md:113:- 评测日志：`mllm/training_logs/eval_stage2_full_seqv3_mixed.log`；watcher：`mllm/training_logs/mixed_eval_watcher.log`
docs/learn docs/B4DL_训练方法对比_论文vs复现_20260828.md:116:- 评测日志：`mllm/training_logs/eval_stage2_full_seqv3_mixed.log`
docs/learn docs/B4DL_LiDARCLIP权重核对_20260
…（已截断，原文 7075 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Very important references found:
- docs explicitly cite `mllm/training_logs/b3_pipeline.log` (B3 doc) and `mllm/training_logs/b4a_pipeline.log` (工作进度总结 doc) — so b3_pipeline.log and b4a_pipeline.log are referenced in docs. b1/b2 pipeline logs were not found referenced (only run_b1_pipeline.sh script). Let me double check b1_pipeline/b2_pipeline references more carefully with simpler pattern to be sure — my grep included b1_pipeline etc. patterns and matched only b4a & b3. So b1/b2 pipeline log files aren't referenced.
- `_meteor_sweep/{sweep,sweep2}.py` referenced in METEOR doc (Sep 7). So sweep scripts referenced; hyp.txt/ref.txt not referenced but scripts are.
- eval_stage2_full_seqv3_mixed.log referenced 4x in docs (Aug 27/28 docs) — file exists (2.7MB). mixed_eval_watcher.log referenced once.
- stage2_full_seqv3_mixed_b3_*.log + b3_pipeline.log referenced (B3 doc). So b3 pipeline log + the b3 run log (Sep 4 194430, 1MB) are doc-referenced. Actually "stage2_full_seqv3_mixed_b3_*.log" glob — refers to log stage2_full_seqv3_mixed_b3_20260904_194430.log.
- "training_logs/b4a_pipeline.log" referenced in 进度总结 doc.
- root CLAUDE.md mentions ./evaluation/evaluation_results.json (a generic eval script path — that's mllm/evaluation dir), and docs/B4DL_复现方案.md also references ./evaluation/evaluation_results.json — mllm/evaluation dir. That's the eval script's output path inside mllm/evaluation, not the root eval_results.json toy file.
- stage1_95k, seqv2, stage2_20260807, stage2_final: no matches → not doc-referenced. Wait, my grep combined patterns with -E and multiple alternations. stage1_95k found nothing. But earlier docs list references "stage2_full_seqv3_mixed_b3_*.log". Let me also search specifically for "smoke_" in docs: only smoke_train.py (a script, not our smoke files) matched. So smoke artifacts not referenced.
- Check references to "seq_predictions", "eval_log.txt", "stage2_full_seq", "metrics.json", "predictions.json" (formal run dir products) — need to know which docs cite run dirs (not candidates). Those matter less, but let me grep for candidate-file-specific names: "seq_predictions.json.ckpt", "smoke_2000", "smoke_meta", "smoke_seq", "metrics_recomputed", "official_ckpt_nometa", "stage2_full_seq/predictions" etc. metrics_recomputed IS referenced likely (dual metric rules doc Sep 7). Let me grep. Also CLAUDE.md reference check of anything else (e.g., does CLAUDE.md mention specific training logs?). Let me grep CLAUDE.md content for the candidate names, and grep remaining doc set for smoke_/metrics_recomputed etc. And check docs mentioning stage1_162k logs, root training_logs (vtimellm logs), stage3_seqv3_20260826, stage2_seqv3_20260825_174612, seqv3_mixed_20260826_131938 (B0 mixed big log) - referenced in docs? Those are candidate files maybe to delete. B0 doc (seqv3混合训练评测分析 20260827) may reference the stage2_full_seqv3_mixed_20260826 log. Let me grep these names.

</details>

The doc grep found key references. Let me complete the reference matrix with a second pass for remaining candidate names.

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && grep -rnE "metrics_recomputed|seq_predictions|smoke_seq|smoke_2000|smoke_meta|official_ckpt_nometa|official_model_stage2|stage2_full_seq/|stage2_full_seqv2/|stage23_s
…（已截断，原文 506 字符）
description: Second pass grep for run-dir and eval artifact references
```

</details>

<details><summary>输出摘要</summary>

```
docs/learn docs/B4DL_B3整场景meta修复mIoU超论文_20260906.md:90:- 评测：`mllm/eval_results/stage2_full_seqv3_mixed_b3/`（predictions/metrics/eval_log）
docs/learn docs/B4DL_B3整场景meta修复mIoU超论文_20260906.md:92:- 官方模型评测对照：`mllm/eval_results/official_ckpt_nometa/`（纯视觉 0.1737）
docs/learn docs/B4DL_METEOR双口径溯源与评测规则_20260907.md:38:## 三、B0-B3 双口径补算表（2026-09-07 实测，产物 metrics_recomputed_20260907_dual.json）
docs/learn docs/B4DL_METEOR双口径溯源与评测规则_20260907.md:65:- 补算产物：`mllm/eval_results/stage2_full_seqv3_mixed{,_b1,_b2,_b3}/metrics_recomputed_20260907_dual.json`
docs/learn docs/B4DL_seqv2评测对比与坍缩分析_20260825.md:83:- 评测：`mllm/eval_results/stage2_full_seq/`（seq-118k）、
docs/learn docs/B4DL_seqv2评测对比与坍缩分析_20260825.md:84:  `mllm/eval_results/stage2_full_seqv2/`（seqv2-148k）、
docs/learn docs/B4DL_seqv3混合训练评测分析_20260827.md:141
…（已截断，原文 1958 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Reference matrix is nearly complete. Additional important ones:
- `stage2_full_seqv3_mixed_20260826_131938.log` is referenced by 4 docs — this is a keep file (doc-referenced). Good, it wasn't in the user's list anyway.
- Run dirs are referenced (not deletion candidates).
- `metrics_recomputed_20260907_dual.json` referenced (in run dirs, not candidates).
- `official_ckpt_nometa/` referenced (in doc) — not a candidate but noted as useful.
- No doc references found for: stage1_95k logs, stage2_full_20260809*, stage2_full_resume*, stage2_full_seq_2026* logs, stage2_full_seqv2 logs, seqv3_merge log, stage2_seqv3_20260825_174612.log, stage3_seqv3_20260826_002520.log, stage1_162k logs, root training_logs stage2 files, smoke_* root eval artifacts, seq_predictions ckpt, eval_log.txt (root), _meteor_sweep hyp/ref (only sweep scripts referenced), pipeline logs b1/b2 (not referenced; b3/b4a referenced), stage2_20260807/stage2_final (not referenced).

Also check CLAUDE.md more comprehensively for the key run names (stage2_full_seq, stage2_full_seqv2, seqv3-mixed-b1...) to be sure CLAUDE.md doesn't reference training logs. Also check root README.md & wiki? not needed.

Let me also do a sanity grep for the filename list inside CLAUDE.md and root README only (they're part of the doc set), for "training_logs" occurrence at all to make sure nothing else.

Also: the smoke_test.log grep found nothing. Good.

One quick verify: b4a_pipeline.log tail (mtime Sep 8 13:14) to describe its phase coverage & b4a run log ending (stage2_full_seqv3_mixed_b4a_20260907_060023.log mtime Sep 8 07:40): exits successfully? tail check quickly. Also check b3 run log (194430) completion summary & stage2_seqv3/stage3 logs for completeness? The stage3_seqv3_20260826 log and stage2_seqv3_20260825_174612 log are other candidate files in mllm/training_logs (not explicitly in the user's list but they exist; user asked judgment for the listed files; these two were in the dir listing: stage2_seqv3_20260825_174612.log (396KB) and stage3_seqv3_20260826_002520.log (628KB)). For a complete "what can be cleaned" picture, do a quick tail check on those two plus stage2_full_seq 08-12 statuses? Those are in-list (08-12 done). Let me quickly tail stage3 & stage2_seqv3 logs and b3 run and b4a run & mixed_eval_watcher & eval_stage2_full_seqv3_mixed.

Also verify whether smoke-related root files (smoke_seq_metrics.json etc.) correspond to smoke on the seq merged model and whether predictions_full is referenced anywhere — grep found no predictions_full matches, meaning no file/doc named that? OK.

Now do these final log checks then compile results.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl/mllm/training_logs && for f in stage2_full_seqv3_mixed_b3_20260904_194430.log stage2_full_seqv3_mixed_b4a_20260907_060023.log stage2_seqv3_20260825_174612.log stage3_seq
…（已截断，原文 567 字符）
description: Check endings of remaining training logs and pipeline logs
```

</details>

<details><summary>输出摘要</summary>

```
########## stage2_full_seqv3_mixed_b3_20260904_194430.log ##########
[2026-09-04 19:44:32,419] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2026-09-04 19:44:35,605] [WARNING] [runner.py:215:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
-- tail:
[1;34mwandb[0m: Find logs at: [1;35mwandb/offline-run-20260904_194700-5yjskd6s/logs[0m
[2026-09-05 13:59:09,059] [INFO] [launch.py:351:main] Process 72617 exits successfully.
End: Sat Sep  5 13:59:11 CST 2026
########## stage2_full_seqv3_mixed_b4a_20260907_060023.log ##########
[2026-09-07 06:00:25,727] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2026-09-07 06:00:28,409] [WARNING] [runner.py:215:fe
…（已截断，原文 4401 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Complete picture now. Interesting details:

- b4a_pipeline.log: phase 1 completed at Sep 8 ~05:50+ (attempt #1) — training completed (07:40 exit) — but phase 2 eval rc=124 (timeout)?? b4a_pipeline.log is 7.7KB with tail at Sep 8 13:14 (mtime). Phase 2 attempt #1 rc=124 (timeout) — maybe retrying or subsequent manual eval? Since eval_results/stage2_full_seqv3_mixed_b4a has eval_log/metrics/predictions (Sep 8 18:18 dir mtime) — eval succeeded later (maybe rc=124 timeout on the pipeline then manual rerun of eval). The b4a_pipeline.log remains "latest chain record" - referenced by doc (工作进度总结 Sep 6)... doc references b4a_pipeline.log as ongoing run. Still, it's the current project phase record: keep.

- mixed_eval_watcher.log: shows OOM and "Terminated" — B0-era eval watcher, referenced by a doc (评测数值对比 doc: watcher). Small.

- stage2_seqv3_20260825_174612.log (B0 stage2 seqv3 train Aug 25-26, 396KB): completed successfully 00:21 Aug 26 (6.6h). And stage2_full_seqv3_mixed_20260826_131938.log (the "mixed B0" 1.1MB log) is doc-referenced 4 times (the 3,474-step full run, completed Aug 27 12:14). Both complete; the former not doc-referenced but it's the stage2 of the seqv3 lineage... wait — 20260825_174612 is "stage2_seqv3" run, and 131938 "stage2_full_seqv3_mixed" (B0). stage3_seqv3_20260826_002520.log (Aug 26 10:05, completed).

Careful about the seqv3 lineage story: 20260825_174612 (stage2-seqv3, output stage2-seqv3), merge at 002121 (into stage2-seqv3-merged), then stage3_seqv3 (002520) — then B0 "mixed" (131938) = stage2-full-seqv3-mixed. OK.

The user asked specifically about files in the list; the extras I've examined will inform the safe-delete summary. For safety I'll classify:

Not in user list but present in mllm/training_logs:
- eval_stage2_full_seqv3_mixed.log (2.7MB) — doc-referenced (4 docs). Keep.
- mixed_eval_watcher.log — doc-referenced. Keep (small).
- stage1_162k_b1_20260830_165411.log (49KB, killed 17:15) — superseded by the 185010 run (success). Actually referenced indirectly by b1_pipeline 阶段2 rc=247. Not doc-referenced by name. Failed attempt → deletable. 185010.log (371KB, success) = the actual stage1-162k retrain — the checkpoints of stage1 used by all stage2 (b-series reference ./checkpoints/stage1/mm_projector.bin?) — hmm... stage1 162k was used by b1+ (b-series stage2 used stage1 mm_projector). Actually the stage2 b-runs pretrain adapter comes from checkpoints/vtimellm-vicuna-v1-5-7b-stage1 (mm_projector.bin) — and 185010 run is the record of the final stage1 162k retrain (its checkpoints overwrote stage1). The successful log 185010: keep? It's the key stage1 retrain record (18:50→21:36). Since final checkpoints exist, logs optional. I'd say keep 185010 as the successful stage1 retrain record (unique), delete 165411 (failed attempt). But neither is doc-referenced by filename. To be conservative: put 185010 in "keep (unique成功留档)" and 165411 in safe-delete.
- stage2_full_seqv3_mixed_b3_20260904_194430.log — b3 run; docs reference glob stage2_full_seqv3_mixed_b3_*.log → keep.
- stage2_full_seqv3_mixed_b4a_20260907_060023.log — b4a run; keep (probably doc-glob referenced? docs mention b4a run via pipeline log & tmux; filename not explicit but stage2_full_seqv3_mixed_b3_*.log glob pattern was for b3; for b4a the docs reference b4a_pipeline.log and eval_results dir only). The b4a training log is the current run's log — keep.
- stage2_full_seq_20260822_094534.log + nohup twin + resume 230654 + nohup twin: the seq lineage. Docs: B4DL_seqv2评测对比与坍缩分析_20260825 references eval_results/stage2_full_seq/ only; nothing references the seq training logs by name. These four are duplicates (two runs × two captures). Deleting nohup twins is safe; keeping one of each pair gives the run record. seq = historical baseline (superseded by seqv2/seqv3). Given the checkpoint dirs remain, safe-delete both nohup duplicates; the primary ones (094534 partial + 230654 resume completed) could be kept as the run narrative. Resume 230654's training completed all steps (2781/2781) though the log lacks summary lines (was likely killed right at end by GPU watcher at 12:37? Hmm... mtime 12:37 Aug 23. Progress hit 2781/2781 at 13:27 elapsed → 23:06+13:27 = 12:33 Aug 23. Then killed ~12:37 during final checkpoint save before summary printed. So run = training complete but finalize interrupted... but wait: eval_results/stage2_full_seq products exist from Aug 23 16:38. And merged checkpoint exists (stage2-full-seq-merged dir). So the lineage completed.)

Let me reconsider whether to keep the two primary seq logs: The evals + checkpoint dirs hold the real products; logs of an interrupted-at-final-save resume + a killed-at-54% run are mostly noise except documenting hyperparams (which are in scripts). Docs (learn docs) describe all the decisions already. I'll classify them as "建议保留" only if unique record of run config. Actually user's instruction: 建议保留 = 文档引用、pipeline 关键记录、唯一留档. The seq logs could be deleted since superseded by seqv2 family, but I'll be more careful: user wants to decide usefulness; I'll present: nohup duplicates definitely deletable; keep one of the pair as唯一留档 optionally. For b2 files the pipeline log documents them.

Since the user asked a medium-scope check, I should give decisive but defensible recommendations. Strategy for the final lists:

【可安全删除】criteria: duplicates/empty/interrupted remnants superseded by later successful runs and no doc references:
1. mllm/training_logs/stage2_full_20260809_170938.log (crashed env ImportError)
2. stage2_full_20260809_171216.log (148B empty)
3. stage2_full_20260809_171708.log (killed at 11% — superseded by 184921+resume... wait careful: 184921 (82% then died) + resume 093455 (completed). So the pair 184921 + 093455 is the "successful stage2-full" lineage; both ended OK-ish (184921 died at 82% but its continuation 093455 completed 3 epochs). 171708 killed at 11% by user → superseded. Delete.
4. stage2_full_resume_20260810_092622.log (70B empty)
5. stage2_full_resume_20260810_093210.log (killed after 2.5min, superseded by 093455)
6. stage2_full_seq_20260812_171413.log (GPU/tf32 env crash)
7. stage2_full_seq_20260812_171640.log (GPU CPU-mode, died at step 0)
8. stage2_full_seq_nohup.log (duplicate of 094534)
9. stage2_full_seq_resume_nohup.log (duplicate of resume)
10. stage1_95k_20260825_161616.log (interrupted at 8 steps; retry 162055 succeeded) — delete
11. seqv2: 114030, 121607, 124310, 125007, 131353 (killed attempts), nohup (dup of 131353 attempt) — delete. 131601 = the successful one → keep as唯一成功留档 (1.1MB) — check doc ref: docs/learn docs/B4DL_seqv2评测对比与坍缩分析_20260825 (Aug 25) — but run finished Aug 25 12:25; doc references only eval_results dirs, not the training log. Keep anyway (big successful run, unique).
12. stage2_seqv3_merge_20260826_002121.log — merge step completed; small record; merge products exist (checkpoints merged dir). Doc refs? no. Deletable; but it's small & documents the merge used by stage3 — the same info is in the docs (stage3 log starts right after). I'll call it "可删（正常完成但无保留必要；merge 产物在 checkpoints）" — Actually user asked "merge 步骤？正常否" → answer 正常完成. I'd put in safe-delete as well but flag small size; hmm, safer to keep? Its content is a one-shot merge that can't be re-run identically... but merged checkpoint persists (13G). Deleting the log loses nothing about the model. Docs don't cite it. Put in safe-delete.
13. b1: 213612 (killed 91% superseded by 135828) — but this is a *fresh full run* record, killed at 91%, and resume 135828 (needed for history). Hmm 213612 is large (949KB) — the failed part; 135828 succeeded. I'll delete 213612? Its content (16h training) is superseded by 135828 which includes summary. And b1_pipeline.log says attempts failed. To be safe, keep 135828 as record (resume success = final b1 training stage) and mark 213612 + 164030 as deletable? 164030 is 3-min trivial run (resume from final ckpt, 0 steps — likely the "final adapter save" step). Deleting 213612 removes the loss-curve record of steps 0-3000... but wandb offline runs still hold those (wandb 137MB!). Hmm, user keeps wandb? wandb offline runs also keep logs. Not our call to delete; we just note.

Actually hold on. Let me reconsider deletion guidance for 213612: b1 eval exists (stage2_full_seqv3_mixed_b1/), checkpoints exist. The log 213612 covers training 0→91% and 135828 covers 91%→100%. b1_pipeline.log claims chain failed 3 times (contradicted by events; likely its "attempt" bookkeeping misfired). Given ambiguity, I'd mark 213612 as delete-safe remnant (killed run whose continuation is logged in 135828) but with caveat that it holds the unique loss curve for steps 0–3161 of b1... The 13G b1 checkpoint dir has intermediate ckpts with trainer_state. Docs describe the b1 numbers. OK: safe-delete candidate. And 164030 (no-op run) safe-delete.

14. b2: 112252, 150156, 142959, 171433, 172716 delete; 205231 keep (成功). b2_pipeline.log documents the chain incl. attempts — pipeline log keep.
   Wait: 142959/171433/172716 were actually b3-phase training attempts (output_dir mixed-b3!) — mislabeled with b2 names. Since b3_pipeline.log documents 阶段1 for b3 and b3 run succeeded (194430), these are superseded. Delete.
15. Root eval_results scattered files: 
   - seq_predictions.json.ckpt (partial ckpt of interrupted eval) → delete
   - smoke_predictions.json.ckpt → delete (tiny, ckpt of interrupted eval)
   - eval_log.txt (interrupted full eval at 28% binary_qa) → delete (superseded by stage2_full_seq/metrics+predictions) — note: the *formal* seq eval (Aug 23 16:38) has no saved eval_log.txt in stage2_full_seq dir... so root eval_log is the only "log" of the seq full eval, but it's of the *interrupted* run. Actually is root eval_log.txt possibly the log of the SAME Aug-23 formal eval? No: mtime Aug 22 21:07; seq_predictions ckpt 21:07; but stage2_full_seq eval products Aug 23 16:38/39. And smoke_seq* 21:23. So Aug-23 run has no log. Where did its log go? Possibly eval wrote log elsewhere (nohup overwritten?). Whatever: eval_log.txt root = interrupted-run log. Superseded. Delete. I'll note the caveat.
   - smoke_2000_log.txt/metrics/predictions (Aug 23 13:21) — smoke-scale (2000) eval of the seq/seq-merged model: superseded by formal full eval 16:38. Delete.
   - smoke_meta_* (Aug 9) — early smoke of stage2 (meta) — superseded long ago. Delete.
   - smoke_seq_* (Aug 22 21:23) — smoke of seq model — superseded. Delete. But note: same-model smoke outputs; predictions 84KB; the "正式" predictions exist in run dir. Delete.
   - _meteor_sweep: hyp.txt + ref.txt regenerable from predictions.json (scripts stay referenced by doc). The doc references the sweep scripts path (mllm/eval_results/_meteor_sweep/{sweep,sweep2}.py). If we delete whole dir, doc path breaks → per user criterion 4, docs DO reference _meteor_sweep scripts → 保留脚本,可删 hyp.txt/ref.txt (2.3+2.6MB). Careful: "只报告文档在引用将被清理对象". The doc references sweep scripts inside _meteor_sweep. So don't delete the .py files. hyp.txt/ref.txt are not referenced → deletable.
16. smoke_test.log root — 50-sample smoke on the stage2 model from Aug 9 — no doc ref, historical curiosity → safe-delete (no formal value; numbers garbage early model).

17. Root training_logs/: stage2_20260807_200248.log & stage2_final.log are duplicate captures of the same completed run (Aug 7-8, 2 epochs). docs don't reference them. Question asked: 是否同一 run 的重复 → Yes. Delete both? They are the old (pre-rewrite) vtimellm stage2 (vtimellm-vicuna-v1-5-7b-stage2 output = the "官方风格 stage2"? wait, checkpoints dir has stage2 2.4GB...). This older lineage (root training_logs = vtimellm/ base dir) was replaced by mllm work. Given duplicates + superseded by everything in mllm — safe-delete both (keep one if wary; recommend deleting stage2_20260807_200248.log which contains full launcher detail duplicate, keep stage2_final? no—both same run; delete either. My recommendation: both deletable; if keep one, keep stage2_final (cleaner trainer summary). Also stage2_trainer_state.json 186KB there (not asked, it's a trainer state json — historical state of that run; deletable if deleting logs).
18. Root evaluation_results.json — toy/placeholder (acc 1.0) — not referenced? CLAUDE.md line 193 uses ./evaluation/evaluation_results.json (mllm/evaluation path in scripts). Root evaluation_results.json not referenced. Not in the user's list though — mention as bonus? It's a small root file; mention it's a placeholder. Actually be careful about recommending deletion of things not asked; I'll mention in passing with a note (user asked for reference checking of it too actually — "evaluation_results.json" was in grep list). Root-level file: matches not in md docs. Deletable in my view (values 1.0 = dummy self-test).

【建议保留】:
- mllm/training_logs/b3_pipeline.log (doc-cited), b4a_pipeline.log (doc-cited)
- b1_eval_pipeline.log & b2_pipeline.log — pipeline 关键记录 with headline B1/B2 metrics and chain narrative (b2_pipeline documents attempts 1-2 fail then #3 success = directly corresponds to the three b2 logs, aiding interpretation). b1_eval_pipeline holds B1 numbers (mirrored in eval_results metrics.json? b1 metrics.json Sep 1 04:10 same time as eval_log — likely same numbers). Keep as pipeline records. Hmm b1_eval_pipeline.log 357B. b1_pipeline.log: record of failed auto-chain (superseded by manual; metrics exist in eval_results/). It is 749B. Not doc-referenced. Two options... it says phase1a/b and phase2 result (特征重提记录: 28130/850 counts) which is unique info (feature extraction stage logs). I'll recommend keep b1_pipeline.log too (small, unique 特征重提记录), or at least flag "可删但小". Actually the user's framing: "是否被后续 b4a_pipeline.log 或 eval_results 的正式产物取代" — for b1/b2: yes largely superseded by eval_results metrics and by newer pipeline logs; but b1_pipeline.log contains stage1a/1b feature-extraction records not in b4a_pipeline. I'll recommend: 保留 b1_eval_pipeline/b2_pipeline/b3_pipeline/b4a_pipeline（记录链+指标，b3/b4a 有文档引用）；b1_pipeline.log 可删或留——含特征重提记录，但小。 Put b1_pipeline in borderline → recommend keep? The user wants decisive lists: safe-delete needs "无文档引用" AND superseded. b1_pipeline.log is not doc-referenced; its stage2 (stage1-162k) outcome is in stage1_162k logs & checkpoints; stage3 outcome in 135828 + eval_results; phase1a/1b in eval logs of datageneration? I'll put b1_pipeline.log in "可安全删除"(低风险,与b1_eval_pipeline同代) BUT given its tiny size and unique phase1a/1b record, put in 建议保留 with reasoning "链记录：B1 链的失败重试叙述+特征重提记录，757B 仅" hmm. Let me decide: KEEP — it is the record of B1 chain attempts (why 3 attempts failed), explaining the naming of the 3 logs. It's the "pipeline 关键记录" category the user wants preserved. Same for b1_eval_pipeline.log.

- mllm/training_logs/eval_stage2_full_seqv3_mixed.log (doc-cited 4x) — keep
- mixed_eval_watcher.log (doc-cited) — keep
- stage2_full_seqv3_mixed_20260826_131938.log (doc-cited 4x, B0) — keep
- stage2_full_seqv2_20260824_131601.log — 成功大 run 唯一记录 keep
- stage2_full_seqv3_mixed_b2_20260901_205231.log — b2 成功 keep
- stage2_full_seqv3_mixed_b1_20260831_135828.log — b1 完成段 keep
- stage2_full_seqv3_mixed_b3_20260904_194430.log — doc glob 引用 keep
- stage2_full_seqv3_mixed_b4a_20260907_060023.log — b4a 当前 run keep
- stage1_95k_20260825_162055.log — 成功 stage1(95k) run keep? Note: stage1 95k is an older stage1; newer stage1-162k (Aug 30) supersedes it. The stage1 dir used by stage2 b-line is "stage1" (mm_projector.bin from 162k after Aug 30 overwrite). So stage1-95k 162055 was superseded by 185010 (162k retrain). Hmm, but 95k stage1 was used for seq/seqv2/seqv3-mixed-B0 lineage (through Aug 26-27). eval products exist; checkpoints stage1 (61MB) currently = 162k final. Decision: 162055 keep? It's 214KB. The user specifically asked which of the pair is successful — answer 162055; whether to delete depends on 留档 policy. Since B0 lineage (docs) depended on it, and docs describe the runs, but no direct filename reference → safe-delete possible. I'd put 161616 in delete; keep 162055 as 唯一成功留档 (like seqv2 131601). Wait actually, do I keep every successful log? Then safe-delete list = failed/dup/empty only. But user asked to judge "无用" logs; successful ones likely already archived/decided. Let me follow the two-bucket summary with reasons, and mark "可安全删除" to include superseded duplicates + failed attempts + interrupted evals, and "建议保留" for successful-run logs of versions that are still referenced in docs lineage + pipeline logs. But that leaves lots of logs (successful ones of superseded lineages: stage1_95k 162055, seqv2 131601, b1 135828, b2 205231...) in "keep" — checkpoints/lineages were kept (13GB each), so the project retains history; keeping their logs is consistent. Hmm — but seqv2 131601's model dir is 13GB + merged 13GB. Actually wait — maybe the user plans to delete checkpoints too later (they asked only for logs/eval now). Keep recommendation consistent: for each version family (seq, seqv2, B0-mixed, b1..b4a), keep the success logs because they are 唯一留档 of training narrative, delete only crashed attempts/duplicates/empties (each superseded because same-version success log exists).

Simplify final buckets:

【可安全删除】(失败尝试/空跑/重复/被成功版本取代, 无 doc 引用):
Root training_logs: both stage2_20260807_200248.log & stage2_final.log (mutual duplicates; superseded by mllm lineage; no doc refs) + note stage2_trainer_state.json (companion, same run).
mllm/training_logs:
- stage2_full_20260809_170938.log (ImportError PeftMixedModel crash, 17s)
- stage2_full_20260809_171216.log (148B, 无训练即停)
- stage2_full_20260809_171708.log (SIGTERM 11%, 后被 184921+resume 取代)
- stage2_full_resume_20260810_092622.log (70B 空)
- stage2_full_resume_20260810_093210.log (SIGTERM 2.5min, 被 093455 取代)
- stage2_full_seq_20260812_171413.log (CPU-only tf32 报错)
- stage2_full_seq_20260812_171640.log (CPU-only 0 step 中断)
- stage2_full_seq_nohup.log (与 094534 同 run 双写)
- stage2_full_seq_resume_nohup.log (与 resume 同 run 双写)
- stage1_95k_20260825_161616.log (8 步中断, 162055 重试成功)
- stage2_full_seqv2_20260824_114030.log / _121607 / _124310 / _125007 / _131353 / seqv2_nohup.log (全部被 131601 取代)
- stage2_seqv3_merge_20260826_002121.log — hmm. I said deletable. It's normal-completion merge log. Merge产物在 checkpoints 13G + stage3 日志 002520 紧随其后。无 doc 引用。Safe-delete. (1.6KB) But "merge 步骤唯一记录"? stage3_seqv3 log + merged ckpt exist; script exists presumably. OK delete... Actually the merge was of stage2-seqv3; stage2-seqv3 (13GB) & merged (13GB) checkpoints remain. Log deletable. Fine — but risk: none. Delete.
- b1: 213612? I said keep 135828 delete 213612/164030. Hmm wait, is that right? 213612 covers 0→91% of the b1 training. It is the LARGEST part. Deleting it while keeping 135828 means losing loss curve of 91% of b1 run. But keep policy: same-version success log = 135828 which contains steps 3000→3474 resume + final summary only. The doc for B3/b1 (基线版本训练方法对比 0830, B3 doc) may quote numbers, not curve. wandb offline-run-20260830_213741 has full curve (offline runs kept). Actually wandb offline-run dirs exist per launch: e.g. offline-run-20260830_2138xx for b1 attempt. Since mllm/wandb (137MB, 26 runs) isn't being cleaned now, curve preserved. So deleting 213612 acceptable but... it's the *primary* b1 training attempt record: after the kill at 91%, resume succeeded. Both parts form b1's 3 epochs. I'll recommend: 213612 可删（尾部 SIGKILL 91%, 续训成功段在 135828; 但若想保留 b1 全程曲线,保留 213612/135828 任一即可）. For decisiveness, I'll put 213612 in 可删 list with note; keep 135828. Hmm — actually, wait: the user asked "哪个是完整成功 run,其余是什么" — so they need the analysis; whether to delete 213612 is judgment. Given the b1 doc exists and wandb has curves, deletion OK.

Actually let me reconsider: maybe safer to keep 213612+135828 pair as the b1 training narrative (both halves), delete only 164030 (0-step finalize no-op). That is more defensible for a "medium" audit; files are moderate (949KB+229KB). Hmm. The overall goal: 清理无用日志. 213612 isn't useless — it's the main b1 training log (interrupted at end but majority of training). I'll classify as 建议保留(可选) with a note? The instruction demands two buckets: safe-delete vs keep. Judgment call: the loss curve and step history is in wandb; docs summarize; the log is one of a kind for its 0→91% segment but the same model result persisted. 

Final: I'll place 213612 into 可安全删除 bucket? Let me look at how they judged similar earlier: 184921 (82% kill) + resume 093455 = complete. If I delete 171708 (11%) but keep 184921... wait 184921 died at 82% silently (no wrapper end). If we apply symmetric logic (interrupted at 82% → resume completes), then 184921 would be deletable too keeping resume log. Consistency issue! For Aug 9-10 lineage: files 170938/171216/171708 = env crash/empty/early kill → delete. 184921 (82%, 671KB) + 093455 (resume, complete) — same run split across two files. Recommendation: keep BOTH? or delete 184921? Since the two are halves of one successful 3-epoch run, and resume 093455 is complete (with summary), the pair forms the record. To keep bucket count manageable and safe: 
- I'll mark 184921 as 可删 but with "(若需留档 82% 前半段曲线可选保留)"? That's not decisive.

Alternative cleaner rule that yields decisive lists and matches user intent:
- Logs that are "normal completion of a run whose products remain" → KEEP (唯一留档 of that run) — these are few per lineage: seqv2 131601; b2 205231; seqv3-mixed B0 131938 (doc-cited); b3 194430; b4a 060023; seq resume+094534? no...

Hmm, but then "safe delete" covers: crash/empty/duplicate/early-kill-attempt where a SIBLING success log in the same family exists. For stage2-full Aug9-10 family: the "family success" = 184921+093455. Both stay? The user can decide later.

I need to stop going back and forth. Decide with a documented rationale: Any log that ends interrupted (killed/error/empty) AND whose continuation/successor log exists that completed (with final summary) → 可删(残余). For the Aug9-10 stage2-full: family completed via resume 093455 (has final summary & 'exits successfully'). 184921 is the aborted 82% part. Following the rule → delete 184921 as well. But wait — actually check resume target: 093455 resumed from checkpoint-2000. So the first run (184921) reached step 2274 (82%)... hmm resume from 2000 means they went back to checkpoint-2000, losing steps 2000-2274?? The resumed run completed 2781 total with runtime 14114s. If resumed at step 2000 with ~17.7s/it → 781 steps ≈ 3h51m = 14114s ✓ consistent. So yes: 184921 progress after step 2000 (2274 reached) was discarded. Resume is authoritative full-run (2781/2781, epoch 3.0). 184921 = wasted partial (its steps 2001-2274 lost anyway) → cleanly deletable. 

Similarly seq family: 094534 killed at step 1488; resume started from checkpoint-1000 (log says resume_from checkpoint-1000) and went to 2781. So 094534 steps 1000-1488 were also discarded — resume is authoritative. 094534 deletable as残余; its nohup twin too. That's consistent: keep only resume 230654 (the completed run record; 625KB).

But hmm, resume 230654 lacks final summary (killed at save). Still it's the record of steps 1000→2781. And the eval products confirm completion. OK.

For b1: 213612 killed at 3161; resume 135828 resumed from checkpoint-3000 → so 213612's steps 3000-3161 were discarded → 213612 = 残余 → delete, keep 135828 (completed w/ summary). Consistent! And 164030 = 0-step finalize (no-op, probably for final save) → delete (its purpose fulfilled by product).

b2 family: 112252 (killed 600), 150156 (killed 1292), 205231 (fresh full completion, NOT a resume — pipeline retried attempt #3 fresh? or resumed? pipeline says 尝试#3 完成; the log 205231 start 20:52 Sep 1 — 10 min after 150156 kill at 20:42 → attempt #3, likely started fresh from checkpoint-1200 or fresh? It ran 13.3h to completion (48008s ≈ 13h20m from 20:52 → 10:15 Sep 2 ✓) — full 3474 steps in 13.3h... if resumed from 1200, 2274 steps × 18.5s = 11.7h ≈ close; can't confirm; whatever, 205231 contains the complete epoch-3 summary and exits successfully → keep as b2 success record; delete the two earlier kills + the three Sep-3 b3-phase mislabeled kills (142959/171433/172716) which were superseded by real b3 run (194430, Sep 4-5) — pipeline b3 log documents the wait then 阶段1 完成.

seqv3 family (Aug 25-26): stage2_seqv3_20260825_174612 (completed successfully 00:21) → this is B0 stage2-seqv3 train; merged & stage3 followed. Success run → keep? and stage3_seqv3_20260826_002520 (completed 10:05) → keep. And merge log → delete. Hmm — but do docs reference stage2_seqv3_20260825 log? No; docs reference 131938 (mixed B0). These two Aug-25/26 logs = seqv3 baseline lineage (superseded by mixed-B0 on Aug 26 13:19). But stage23_seqv3 eval dir exists (Aug 26 13:14) = products of stage2-seqv3 + stage3-seqv3 evals (referenced in docs 两阶段训练评测与混合法决策 118). The docs use the results, not logs. Delete or keep? Success logs, part of lineage with eval dir referenced by docs → suggest KEEP (小成本). These weren't in user's asked list; but they're in the folder for cleanup. I'll add to keep-list quietly (or note optional).

Given user question focuses listed files, final answer structure:
1. per-file verdict table (as asked)
2. du numbers
3. eval_results root files verdict
4. docs references summary
5. two buckets.

For the b1 pair question "哪个是完整成功 run": answer 135828 (resume完成 3474/3474 + summary) — together with 213612 (0-91%,被中断) covering the same run; 164030 = no-op. "完整成功" strictly = 135828 has final summary/exits-successfully; 213612 wasn't complete.

For b2 pair: 205231 完整成功 (3474/3474 exit success). Others killed attempts; Sep-3 trio were actually b3-phase training attempts mislabeled b2 (output_dir mixed-b3, killed at step 400/0/0), superseded by the b3 run.

seqv2: 131601 完整成功 (3474/3474, summary, exit成功; wrapper End "(manual)" 12:25 = 手动收尾). Others killed -9; nohup duplicates the 131353 attempt window (same PID 583487) — wait, actually let me double check: seqv2_nohup.log head says "Start: Mon Aug 24 13:13:53" and its tail is identical to 131353's tail (PID 583487, killed 13:15:59). So seqv2_nohup.log = wrapper duplicate of the 131353 launch (both captured same attempt). Slight difference: nohup log might cover additional wrapper text only. Since the 131353 launch died at 13:15:59 and 131601 started 13:16:02 — yes.

stage1_95k: 161616 = 启动即被中断 (8 steps, no exit msg; likely SIGKILL ~16:19), 162055 = 完整成功 (742/742, summary 17:46 exit成功). Both same config → 同 run 重试，成功的是 162055.

stage2_full Aug9 family answers as above.
stage2_full_resume family: 092622 empty (70B); 093210 killed after ~2.5 min (1003 steps?? wait — 1003/2781 in 2.5 minutes?! That's because resume did fast-forward? 1001 steps in 00:25 at 39 it/s — it seems it "fast-forwarded" through steps already done? No wait, epoch 1.08 at step 1001 — hmm from checkpoint-2000 resume at epoch... 2000/2781*3 = 2.16 epoch. But shows epoch 1.08?? Hmm epoch 1.08 at step 1001 means resume restarted at epoch ~1.08? checkpoint-2000 would be epoch 2.16 of 2781-step/3-epoch schedule... Actually the resume config may differ (resume from checkpoint-2000 of the *previous* run whose total was... whatever). Not critical. It ran 1003 steps in 2.5min?? 39 it/s then 11 it/s. Probably dataloader fast start then killed at 09:34:36 because user restarted with correct settings (093455). Yes: 093210 was started 09:32 and killed 09:34:40 (2.5 min). Then 093455 started 09:34:55 → the successful resume. So 093210 = premature kill (maybe wrong data or wrong node) — superseded by 093455. Delete.

Also confirm 093455 was "resume" config: head says 恢复训练 from checkpoint-2000. completed at 13:31 with train_runtime summary? earlier grep shows train_runtime 14114.9 & epoch 3.0 in 093455 ✓ and exit successfully 13:31:13.

Aug9-10 lineage final recommendation: keep 093455; delete 170938/171216/171708/184921? Hmm wait — earlier I reasoned delete 184921 as well. But is there any possibility the 184921 (non-resume, fresh run at 82%) recorded steps NOT in the resume log (0→2000 in resume log? resume log starts at 1001/2781? In the resume log 093455, the first progress shown would be ~step 2000 onward given resumed from checkpoint-2000, ~781 steps. But the tail shows epoch 3.0 and 2781 total. So resume log covers only ~2000→2781. The 0→2000 part is in 184921 only. Two halves again. Ugh: delete = lose 0→2000 loss curve; wandb offline-run-20260809_1851xx has it. wandb is retained. Docs don't need curves. OK: put 184921 in "可安全删除(残余半程, 全程曲线在 wandb offline-run)" — Hmm, but is mllm/wandb offline-run for 184921 retained? The run would have offline-run-20260809_1852xx? mllm/wandb contains 26 offline runs incl. those. Yes.

Hmm, actually simpler & safer: bucket 184921+171708 under "与成功run同一次训练的前半程/中断段，完整 3 epoch 记录在 093455（+wandb 曲线）→ 可删；若想留全程文本曲线，各留 1 个即可". I'll keep them classified 可删 with that caveat in one line. Wait, 171708 at 11% is definitely junk (user restarted). 184921: I'll mark as "可删(可选留)" → final bucket 可安全删除.

Actually — careful decision on deleting logs whose half-record forms a whole: 093455 IS the whole (2781/2781 w/ summary) since resumed at 2000 → covers 2000-2781 + summary; 184921 covers 0-2274. So both half-cover. Combined narrative = both. To preserve one-file per run policy it's fine to delete both?? No, keep 093455 at minimum.

Decision for Aug9-10: delete 170938/171216/171708/092622/093210 (5 junk), keep 184921+093455 (the run record pair) — safer and defensible. Actually 184921 = killed 82%, no wrapper end; the resume log starts at step 1001?? Let me not overclaim; keep both halves as 建议保留(该代成功 run 的两段). Fine — but hmm 184921 mtime Aug 10 06:18 and died silently at 06:18 (82%, step 2274 at 11:26 elapsed from 18:49 → 06:15 ✓). Wait: it died at step 2274 at ~11h28m → 18:49+11:28 = Aug 10 06:17. mtime 06:18 ✓. And resume from checkpoint-2000 started 09:26 (092622 empty), 09:32 (093210 killed), 09:34 (093455 OK → finished 13:31). Between 06:18 and 09:26, no log — user probably killed it or it OOMed at 06:18 (no message; possibly SIGKILL from driver/GPU loss). It's the "pre-resume" segment. Keep both → so seq family similarly: keep 094534 (0→1488+部分) AND 230654 (resume complete)? For seq: 094534 was SIGKILL -9 (return code -9 = explicit kill by watcher/user), resume started 23:06 from checkpoint-1000. Keep both halves too? sizes 435KB+625KB — no big deal. Hmm OK — pragmatic: KEEP half-pairs of successful runs where resume followed (184921+093455, 094534+230654) — but DELETE the pure duplicates (nohup twins, identical content + wrapper lines).

Wait, then for b1 (213612 + 135828) same pattern → KEEP both, delete 164030 only? The user explicitly asks "哪个是完整成功 run,其余是什么" — implies they may delete 其余. For b1, "其余" = 213612 (interrupted) and 164030 (no-op finalize). Following the half-pair logic, keep 213612? To be consistent I keep 213612 too...

Hmm, wait. But there's a subtlety with b1: 213612 ran 16h with data seqv3_148k (not meta2/whole_scene?) while b1... b1 = "mixed-b1 混合重训" per pipeline (start 21:36 = 213612). And 135828 resumed 13:58. b1's resume → complete. So yes half-pair.

OK so my bucket rule, finally:
【可安全删除】= 
(a) duplicates (same-run twin captures, keep one copy already present): nohup twins (seq ×2), root stage2 pair (stage2_20260807_200248 vs stage2_final — both same run; delete one... wait they are BOTH duplicate captures of same run but are they "keep one"? They're both the entire run (each complete w/ summary). Delete ONE of them (either). Since neither referenced, deleting both is possible but keep-one policy → delete stage2_20260807_200248.log (contains duplicated deepspeed launch spam) & keep stage2_final.log; or delete both. I'll recommend deleting stage2_20260807_200248.log and keeping stage2_final.log... but actually 200248 contains the launcher's final "exits successfully" line; final.log contains trainer summary. Both fine. I'll suggest: 同 run 双捕获 → 保留其一即可; 建议删 stage2_20260807_200248.log（与 stage2_final.log 同 run；final 版更完整含 train_runtime）. Then optionally stage2_final also deletable as superseded historical lineage — I'll place stage2_final.log in 保留(同run保留一份留档) — no wait. Decision: put BOTH in delete list with note "同一 run 双写拷贝；若保守可留 stage2_final.log 一份". Ugh.

Deep breath. The user's cleanup: root training_logs belongs to the OLD repo top level (vtimellm), superseded by mllm/ sub-repo. Everything in root training_logs is old-lineage (Aug 7-8 stage2 2-epoch run). The mllm/checkpoints still contains vtimellm-vicuna-v1-5-7b-stage2 (2.4GB, official_ckpt_nometa eval references ljq path for official; and official_model_stage2 symlinks point to /root/autodl-tmp/ljq/mmb4dl-main/... — the official model lives elsewhere). Root stage2 run (Aug 7) = the model that produced smoke_meta eval (Aug 9) and earlier comparisons. It's fully historical.

FINAL DECISION (keep it simple and defensible):
- 可安全删除: duplicates & crashed/empty/aborted runs where the same-version successful log (or products) remain. That includes root stage2 pair BOTH files (互相重复 + 属旧代 vtimellm 顶层目录训练, 已由 mllm/ 全部新代版本取代; 无 doc 引用; 如需留档二选一).
- I'll keep all "successful/completed" logs and half-pairs in keep list; user can then delete at will from delete list.

Hmm, but that makes the delete list miss things the user might want gone (successful but superseded run logs, e.g. stage1 95k 162055). The user's Q for stage1_95k: which is successful → they'll likely want to delete the failed one and keep the successful (as their "产物"留档). OK my rule covers: delete 161616, keep 162055.

Alright — finalize. One remaining check for part 4: do CLAUDE.md / README reference any candidate? Already grepped — no candidate refs in CLAUDE.md/README.md (only ./evaluation/evaluation_results.json generic path inside mllm? wait CLAUDE.md:193 `--metrics_output ./evaluation/evaluation_results.json` — that's the mllm/evaluation script example, fine).
Also check the .gitignore status of training_logs / eval_results (user said eval_results gitignored). For understanding: check .gitignore quickly. And confirm mllm/training_logs is tracked or ignored — affects whether "delete" needs git care. Quick check. Also confirm none of eval_results scattered files are tracked by git (they're gitignored, so delete is not a git change). Fine to quickly verify.

Also confirm stage2_full_seq_20260822_094534.log reach 1488 & return code -9 with "Killing subprocess 147747" and that resume used checkpoint-1000 (yes head shows --resume_from_checkpoint ./.../checkpoint-1000).

Wait — one more: 094534 killed at step 1488 (epoch 1.6) — after the kill, resume from checkpoint-1000 (they went back 488 steps). OK.

Let me quickly do the .gitignore check + confirm whether docs contain any "training_logs/stage2_full_seq_resume" etc. reference (already none) + confirm root file references to pipeline logs only in training_logs? Did grep for docs referencing b3_pipeline only in one doc and b4a_pipeline in two docs? Let me check remaining matches in docs for "training_logs" generally to catch any referenced log I might misclassify (like references to files I marked delete).

</details>

Let me run a final consistency check on all "training_logs/" references across docs, and confirm gitignore status.

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && grep -rnoE "training_logs/[A-Za-z0-9_./\-]*" --include="*.md" docs "docs/learn docs" CLAUDE.md README.md 2>/dev/null | sort -u; echo "=== gitignore ==="; cat .gitigno
…（已截断，原文 443 字符）
description: List all training_logs references in docs and gitignore status
```

</details>

<details><summary>输出摘要</summary>

```
docs/learn docs/B4DL_B3整场景meta修复mIoU超论文_20260906.md:89:training_logs/stage2_full_seqv3_mixed_b3_
docs/learn docs/B4DL_seqv3混合训练评测分析_20260827.md:141:training_logs/stage2_full_seqv3_mixed_20260826_131938.log
docs/learn docs/B4DL_seqv3混合训练评测分析_20260827.md:144:training_logs/eval_stage2_full_seqv3_mixed.log
docs/learn docs/B4DL_基线锁定与对比规则_20260829.md:124:training_logs/eval_stage2_full_seqv3_mixed.log
docs/learn docs/B4DL_基线锁定与对比规则_20260829.md:125:training_logs/stage2_full_seqv3_mixed_20260826_131938.log
docs/learn docs/B4DL_工作进度总结_20260906.md:101:training_logs/b4a_pipeline.log
docs/learn docs/B4DL_工作进度总结_20260906.md:68:training_logs/
docs/learn docs/B4DL_训练方法对比_论文vs复现_20260828.md:112:training_logs/stage2_full_seqv3_mixed_20260826_131938.log
docs/learn docs/B4DL_训练方法对比_论文vs复现_20260828.md:116:trai
…（已截断，原文 2092 字符）
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Doc references to training_logs paths (complete list):
- stage2_full_seqv3_mixed_b3_*.log (glob) — B3 doc — the actual file stage2_full_seqv3_mixed_b3_20260904_194430.log
- stage2_full_seqv3_mixed_20260826_131938.log — 4 docs (B0 mixed log)
- eval_stage2_full_seqv3_mixed.log — 4 docs
- b4a_pipeline.log — 1 doc (进度总结 101)
- mixed_eval_watcher.log — 1 doc
- B4DL_工作进度总结 line 68 mentions "training_logs/" directory as an item generally.

Note: mllm/training_logs itself is NOT in .gitignore (git check-ignore returned nothing for the training log — only eval_results and wandb matches). So training logs ARE git-tracked (mllm/training_logs isn't ignored: no output line for it means not ignored → tracked). Deleting them affects git history (large-ish). Important note for the parent: deletion of tracked files = git changes; files could be recovered from git history. eval_results + wandb gitignored → deleting is file-level only, not tracked.

Also mllm/checkpoints, mllm/b4dl_dataset, backups/ are gitignored (safe from git perspective).
smoke_test.log not ignored → tracked? check-ignore printed nothing for smoke_test.log → tracked by git (root). deletion would be a tracked change. Root training_logs: also not ignored → tracked. training_logs/stage2_final.log tracked.

Let me verify: git ls-files on a few candidate paths to be precise. Quick run.

</details>

**🔧 Bash**

<details><summary>输入</summary>

```
command: cd /root/autodl-tmp/wql/mmb4dl && echo "== tracked? =="; git ls-files --error-unmatch smoke_test.log training_logs/stage2_final.log mllm/training_logs/stage2_full_seqv3_mixed_b2_20260901_205231.log ml
…（已截断，原文 405 字符）
description: Check which candidate files are git-tracked
```

</details>

<details><summary>输出摘要</summary>

```
== tracked? ==
error: pathspec 'mllm/eval_results/eval_log.txt' did not match any file(s) known to git
Did you forget to 'git add'?
mllm/training_logs/b3_pipeline.log
mllm/training_logs/stage2_full_seqv3_mixed_b2_20260901_205231.log
smoke_test.log
training_logs/stage2_final.log
== how many training log files tracked ==
48
709
```

</details>


### ASSISTANT  ·  `deepseek-v4-flash`

<details><summary>思考过程</summary>

Interesting: 
- smoke_test.log, training_logs/* (root), mllm/training_logs files ARE git-tracked (48 files in the two training_logs dirs), while mllm/eval_results/* is NOT tracked (gitignored). 709 tracked files total. So deleting training logs affects the git repo (recoverable in history; also possibly committed already). Note that deletion of tracked files is a normal cleanup, but the user should be aware.

All checks are complete. Now compile the final comprehensive report in Chinese, in the requested format.

Let me organize all the data:

=== Part 1: per-file verdicts ===

Root training_logs/ (旧 vtimellm 顶层目录, git-tracked):
1. stage2_20260807_200248.log (390KB, Aug7 20:02→Aug8 01:40): 与 stage2_final.log 同一 run 的双捕获（同秒启动 20:02:53），深色 deepspeed launch 日志 + trainer 输出；run 正常完成（num_train_epochs 2，尾部 "Process 149724 exits successfully"，含 train_runtime×1）。结论: 正常完成 + 同 run 重复（与 stage2_final 同启动时间）。
2. stage2_final.log (313KB): 同上同一 run 的另一份捕获（无 launch 行，开头即 trainer 输出），尾部 1072/1072 + train_runtime 20204s epoch 2.0 → 正常完成。两份为同一 run 双写，保留一份即可。

mllm/training_logs/ (git-tracked):
3. stage2_full_20260809_170938.log (8.7KB): 17:09 启动 17:09:54 即崩 — ImportError: cannot import name 'PeftMixedModel' (python3.13/miniconda env 与 peft 版本不符) — 启动即失败。
4. stage2_full_20260809_171216.log (148B): 只有 wrapper 头，训练未起即停（近乎空）。
5. stage2_full_20260809_171708.log (91KB): 17:17 启动，18:48 SIGTERM 被杀（epoch 0.34，11%）；随后 184921 重新全量启动 → 失败重试前奏/被弃前半程。
6. stage2_full_20260809_184921.log (671KB): 18:49→Aug10 06:18 跑 82% (2274/2781, epoch 2.45) 后无任何错误信息猝停（无 exit 行，无 wrapper 完成行）；同日 09:26-09:34 三个 resume 尝试后由 093455 续到完成。→ 中断 run 前半段（被 checkpoint-2000 续训取代,其 2000 步后进度被弃）。
7. stage2_full_resume_20260810_092622.log (70B): 仅启动头，秒退（近空）。
8. stage2_full_resume_20260810_093210.log (7.9KB): 09:32 启动 2.5 分钟即 SIGTERM(1003/2781)——被杀重试前奏。
9. stage2_full_resume_20260810_093455.log (322KB): 09:34:55→13:31 续训 完成 2781/2781 epoch 3.0 train_runtime 14114s + "exits successfully" → 该代 stage2-full 唯一完整成功 run。
10. stage2_full_seq_20260812_171413.log (8.5KB): 17:14 CUDA unknown error（GPU 不可见→CPU）+ "--tf32 requires Ampere" ValueError exit 1 → 环境故障启动失败。
11. stage2_full_seq_20260812_171640.log (10.8KB): 17:16 重试，同样 CPU-only，模型加载后 0/2781 即停（无 error 无 exit，被杀/中断）。→ 8-12 两文件 = seq 首次尝试的环境故障/中断；与 8-22/23 的正式 seq run（094534+230654）不是同一 run（不同日期、8-22 日志 GPU 正常）。
12. stage2_full_seq_20260822_094534.log (435KB) + nohup 双胞胎 stage2_full_seq_nohup.log (435KB, 头多 "PER-SEQUENCE TRAINING (no tf32)" wrapper 行): 09:45→18:59 被杀(-9) at 54% (1488/2781 epoch 1.6)。nohup 版为同 run 双写（除 wrapper 头/尾几行外同内容）。
13. stage2_full_seq_resume_20260822_230654.log (625KB) + stage2_full_seq_resume_nohup.log (625KB): 23:06 从 checkpoint-1000 续训, 跑到 2781/2781 (13h27m)，但日志末尾停在 final save 的 peft 警告+wandb 行、无 train_runtime 总结行、无 exit 行（文件 mtime 12:37 = 完成训练后 ~4 分钟被杀）；no(hup 版同 run 双写。→ 训练已跑满但收尾被打断；seq 模型完成后有 merged ckpt + stage2_full_seq/ 评测产物（Aug 23 16:38）。8-22 09:45 run 之 checkpoint-1000 以后被此文件记录。
14. stage1_95k_20260825_161616.log (7.6KB): 16:16 启动只跑 8/742 步 (epoch 0.01) 即中断（无错误无 exit，mtime 16:19 → SIGKILL）。
15. stage1_95k_20260825_162055.log (214KB): 16:21 重试同配置, 742/742 完成 + train_runtime 4975s + 17:46 exits successfully → 同 run 重试中的成功者。
16. stage2_full_seqv2_20260824_114030.log (27KB): 11:40→12:14 被 SIGKILL(-9) at 74/2781 (3%)。
17. _121607.log (22KB): 12:16→12:41 -9 at 54/2781 (2%)。
18. _124310.log (7.8KB): 12:43→12:47 -9 at 0 步。
19. _125007.log (16KB): 12:50→13:13 SIGTERM at 38/2781。
20. _131353.log (6KB): 13:13:56→13:15:59 -9 at 0 步（模型刚加载完）。
21. _131601.log (1.12MB): 13:16:02→Aug25 09:46 完整成功 3474/3474 epoch 3.0 train_runtime 73656s, exits successfully → seqv2 唯一完整大 run（wrapper End "(manual)" 12:25 为手动收尾）。
22. seqv2_nohup.log (6.4KB): wrapper 日志,与 131353 同一 attempt（同 PID 583487 13:15:59 被杀）→ 131353 的双写。
23. stage2_seqv3_merge_20260826_002121.log (1.6KB): merge 步骤正常完成（00:21-00:24，"Merged model saved to ...stage2-seqv3-merged / Use ... for stage3"）→ 一次性的 merge 记录，merge 产物在 checkpoints（13G merged 目录），随后 stage3 日志(002520)接力。正常但可再生成（merge 产物仍存）。
24. stage2_full_seqv3_mixed_b1_20260830_213612.log (949KB): 21:36→Aug31 13:48 SIGKILL(-9) at 91% (3161/3474)；b1_pipeline.log 记为 阶段3 尝试#1 失败 → 中断大半程。
25. _20260831_135828.log (229KB): 13:58 从 checkpoint-3000 续训, 16:30 完成 3474/3474 + train_runtime 8936s + exits successfully → b1 完成段（完整成功 run 即 213612+135828 接力；单文件论 135828 是含完成总结的那半）。
26. _20260831_164030.log (81KB): 16:40 从 checkpoint-3474 续训 = 0 步、train_runtime 0.8s、exits successfully → no-op 收尾 run（保存最终 adapter 用），无训练内容。
27. stage2_full_seqv3_mixed_b2_20260901_112252.log (175KB): 11:22→14:51 -9 at 600/3474 (17%)。
28. _150156.log (265KB): 15:01→20:42 -9 at 1292/3474 (37%)。
29. _205231.log (778KB): 20:52→Sep2 10:15 完成 3474/3474 train_runtime 48008s + exits successfully → b2 完整成功 run（b2_pipeline: 尝试#3 完成）。
30. _20260903_142959.log (118KB): 14:29→17:04 -9 at 400/3474；注意其 output_dir=...-mixed-b3 (whole_scene+meta2) → 实为 b3 阶段训练尝试（文件名误标 b2）。
31. _20260903_171433.log (8.3KB) & _172716.log (8.3KB): 分别 17:14/17:27 启动、~2.5 分钟后被 -9、0 步；同为 resume mixed-b3/checkpoint-400 的尝试（文件名误标 b2）→ b3 正式 run(194430) 前的失败尝试；两者内容仅时间戳不同。
32. b1_pipeline.log (749B): 覆盖 B1 链 阶段1a/1b 特征重提(OK, 28130/850) + 阶段2 stage1-162k 重训（首试 rc=247 失败→18:50 重试 rc=0）+ 阶段3 mixed-b1 训练三试均败后链停止的记录。与后续实际成功（手动 resume+独立 eval）矛盾，叙述陈旧；关键数字已固化到 b1_eval_pipeline.log + eval_results/stage2_full_seqv3_mixed_b1/metrics.json。不被 md 引用（文档只引 run_b1_pipeline.sh）。
33. b1_eval_pipeline.log (357B): B1 链 阶段4 同口径评测 rc=0 + B1 vs B0 结果表（acc 0.7787/mIoU 0.2653...）→ B1 正式结果记录（与 eval_results b1/metrics.json 同值）。不被 md 引用。
34. b2_pipeline.log (638B): B2 链 阶段1（#1/#2 失败→#3 完成）+ 阶段2 评测 rc=0 + B2 结果表（0.7649/0.1992）→ b2 链正式记录,解释 3 个 b2 run 日志的相互关系。不被 md 引用。
35. b3_pipeline.log (13.8KB): B3 链：阶段1 显存门控等待 149 次(~24.8h) 后 阶段1 完成 + 阶段2 评测 rc=0 + B3 结果表（mIoU 0.3467 超论文）→ 被 B3 文档明确引用 → 保留。
36. b4a_pipeline.log (7.7KB): B4a 链：阶段1 门控等待 85 次后完成（Sep7 06:00→Sep8 07:40 训练）；阶段2 评测 尝试#1 rc=124（超时）—— 被进度总结文档引用 → 保留（当前链）。
37. smoke_test.log (根目录, 20KB, Aug 9): 推理 smoke（50 items × tasks）打印 paper vs ours 对照表，早期 stage2 模型冒烟。历史早期冒烟记录，数值已被正式评测取代；无 md 引用。

Part 2: du numbers (collected):
- mllm/eval_results/: 94M total. run dirs: _meteor_sweep 4.8M; official_ckpt_nometa 2.5M; official_model_stage2 4.0K; stage23_seqv3 12M; stage2_full_seq 8.0M; stage2_full_seqv2 11M; stage2_full_seqv3_mixed 8.0M; b1 14M; b2 11M; b3 11M; b4a 12M; root 散落文件 ~884K+root files? eval_log 479KB + seq_predictions 692KB + smoke files ~240KB ≈ 1.4M? my du -ch of *.log/*.json/*.ckpt gave 884K total... hmm 692+479+85+30+25+12+5+1...=~1.4M? du said 884K (blocks). fine: ~0.9MB.
- mllm/training_logs/: 15M; root training_logs/: 876K.
- mllm/evaluation/: 25M.
- mllm/wandb/: 137M, 26 offline-run dirs (mllm/wandb). Root wandb/: 76K.
- mllm/checkpoints/: 145G; per-model sizes listed above.
- backups/: 898M (B3_stage2_final tar 467MB + manifest + b4dl_seqv3_mixed_backup_20260827).
- mllm/b4dl_dataset/: 1.6G.
- Top-10 一级目录: mllm 159G, models 15G, encoders 11G, .git 1.5G, backups 898M, .claude 601M, dataset 384M, datageneration 76M, assets 17M, docs 8.8M. (plus wandb 76K, training_logs 876K, Claude_record 112K below).

Wait — note "du -sh */ .[^.]*" included wandb? It shows list; wandb/76K wasn't in output because sorted head-12 cut... it shows 10 entries; wandb likely small and omitted from head -12? List shows 12 lines incl training_logs and Claude_record; wandb and smoke etc. dropped. Fine, mention root wandb 76K separately (already measured).

Part 3: eval_results root scattered files:
- eval_log.txt (479KB, Aug22 21:07): seq 全量评测日志,尾部停在 binary_qa 2092/7525 (28%) → 中断的评测 run（与 seq_predictions.json.ckpt 配套）；正式完整结果 = stage2_full_seq/{predictions.json 8.3MB, metrics.json}(Aug 23 16:38-39)。注: stage2_full_seq/ 内没有 eval_log.txt,根 eval_log 是该次(中断)评测唯一日志,但评测未完成价值低。→ 被正式产物取代。
- seq_predictions.json.ckpt (692KB, Aug22 21:07): 中断评测的部分预测 ckpt（顶层 key 'results'）,被 stage2_full_seq/predictions.json (task keys, 8.3MB) 取代。
- smoke_predictions.json.ckpt (12KB, 21:09): 同上中断评测的 task ckpt 片段。
- smoke_seq_log.txt / smoke_seq_metrics.json / smoke_seq_predictions.json (21:23): seq 模型 2k-ish 冒烟评测全套;被 8-23 正式 stage2_full_seq 评测取代（metrics.json 同性质且更新）。
- smoke_2000_log.txt/metrics/predictions (Aug 23 13:21): 2000 样本冒烟（介于 resume 结束 12:37 与正式评测 16:38 之间,疑为 seq run 完后的快速冒烟）→ 被同日 16:38 正式评测取代。
- smoke_meta_* (Aug 9): 早期 stage2(meta token 时代) 冒烟 → 被后续一切取代。
- _meteor_sweep/: hyp.txt 2.3MB/ref.txt 2.6MB (Sep7 00:19) = B0 predictions 拆出的 text 中间文件 + sweep.py/sweep2.py（METEOR 变体扫描脚本, 被 20260907 METEOR 文档引用）。扫描成果已固化在各 run 目录 metrics_recomputed_20260907_dual.json (Sep7 21:26-31)。hyp/ref 可由 predictions.json 重新生成;脚本应保留。
- official_ckpt_nometa/ + official_model_stage2/ (Sep2): 官方模型对照评测 + 符号链接（指向 ljq 目录）→ 文档引用(official_ckpt_nometa 0.1737) → 保留（非清理对象）。
- root evaluation_results.json（根目录, 全 1.0 占位数值, Aug 8）: 无引用价值（dummy 自检输出）；gitignore 有 "evaluation_results.json" 模式。

Part 4: docs references (complete list above) — key ones:
- b3_pipeline.log ← B4DL_B3整场景meta修复mIoU超论文_20260906.md:89
- b4a_pipeline.log ← B4DL_工作进度总结_20260906.md:101
- stage2_full_seqv3_mixed_20260826_131938.log ← 4 docs
- eval_stage2_full_seqv3_mixed.log ← 4 docs
- mixed_eval_watcher.log ← 1 doc
- _meteor_sweep/{sweep,sweep2}.py ← METEOR doc
- stage2_full_seqv3_mixed_b3_*.log glob ← B3 doc
- No doc references any of the delete candidates (all stage1_95k/stage2_full_*/stage2_full_resume/seq*/seqv2/stage2_full_seq*/seqv3_merge/b1/b2 logs, smoke_*, eval_log root etc.)
- CLAUDE.md: only generic `./evaluation/evaluation_results.json` in a command example (not the root file).
- Note: 工作进度总结 068 references "training_logs/" generically (dir listing), 且 B4DL docs section headers mention "训练日志：..." but those are the kept ones.

Deletion buckets:

【可安全删除】— 失败/空跑/双写重复/被成功版本取代 且无任何 md 引用:
root training_logs/:
- /root/autodl-tmp/wql/mmb4dl/training_logs/stage2_20260807_200248.log (同 run 双捕获,保留 stage2_final.log 即可)
- (建议同删) stage2_final.log — same-run twin, older lineage... Hmm. Two files same run. Bucket: if deleting duplicates, both are duplicates of each other. Since no doc refs and lineage superseded → both deletable; conservative = keep one. I'll list both as 可删 with note "二选一保留亦可".

mllm/training_logs/:
- stage2_full_20260809_170938.log / _171216 / _171708 / stage2_full_20260809_184921.log / stage2_full_resume_092622 / _093210 — all from Aug 9-10 lineage, superseded by 093455 (成功) — wait, I earlier decided to keep 184921+093455 halves... Final call: DELETE 170938(崩),171216(空),171708(11%弃),092622(空),093210(2.5min弃). For 184921: interrupted 82% segment whose post-2000 progress was discarded by resume-from-2000; whole-run 唯一完整记录 = 093455. I'll put 184921 in 可删组 too but flag "若想留 0-2000 步曲线可留"(wandb 有曲线)。Cleaner: include in delete with note. Hmm, the instructions: "可安全删除（被正式产物取代/空跑/崩溃残留/重复拷贝，且无文档引用）"。184921 = 崩溃残留/被 093455+checkpoint 取代 → qualifies.
- stage2_full_seq_20260812_171413 / _171640 (GPU 环境故障)
- stage2_full_seq_nohup.log, stage2_full_seq_resume_nohup.log (双写重复)
- stage2_full_seq_20260822_094534.log: killed at 54%, superseded by resume 230654 (which covers 1000→2781 + products) → 可删（与 resume 同 run; resume 为完整记录）— note: resume log starts at 1001. The eval products & merged checkpoint are definitive. Include in delete with note. Actually hold on — is 230654 truly covering from step 1001? yes. Steps 0→1488 of 094534 partially redone. Since user asked "08-12 两个文件结局如何" and "08-12 与 08-22/23 是否同一 run" — they will likely delete 08-12 files. For 08-22 run: keep resume (main run log 有完整步数), delete 094534 + nohup twins. OK: delete 094534 too? It's "前半段"被弃. I'd classify: 可删(同 run 前段, checkpoint-1000 前进度仍被 resume 部分重跑; 完整记录=resume). To hedge, note "保守可留其一". I'll place 094534 in delete; 230654 in keep.
- stage1_95k_20260825_161616.log (8步中断; 162055 成功)
- seqv2: _114030, _121607, _124310, _125007, _131353, seqv2_nohup.log
- stage2_seqv3_merge_20260826_002121.log — normal merge but one-off; keep?? decide: 可删（merge 产物在 checkpoints 13G merged 目录仍存; stage3 日志接力; 无 doc 引用）。但用户问题 6 单列了此文件问"merge 步骤? 正常否" — answer: 正常完成。Bucket: it IS normal completed. Whether delete... it documents that stage2-seqv3 (not mixed) merge — the merged dir persists → log 无用。→ 可删。
- b1: _20260830_213612.log (91% -9, 续跑成功段在 135828) + _20260831_164030.log (0-step no-op) → 可删; keep 135828.
- b2: _20260901_112252, _150156, _20260903_142959, _20260903_171433, _20260903_172716 → 可删（前两个被 205231 取代; 后三个实为 b3 阶段失败尝试, 被 b3 run 194430 取代）
- stage1_162k_b1_20260830_165411.log → 可删 (rc=247 失败尝试, 被 185010 取代) [不在清单但同目录]
- mllm/eval_results/ root: eval_log.txt, seq_predictions.json.ckpt, smoke_predictions.json.ckpt, smoke_2000_*, smoke_meta_*, smoke_seq_*, _meteor_sweep/hyp.txt, _meteor_sweep/ref.txt
- root: smoke_test.log
- root evaluation_results.json — hmm careful; it's gitignored ("evaluation_results.json" pattern in .gitignore) and CLAUDE.md's example output path is ./evaluation/evaluation_results.json ≠ root file. All-1.0 placeholder. I'll mention as可删 candidate but not push (not requested). Include in list with note.

Estimated recoverable: root files ~20K+ logs ~0.8MB+..., mllm eval root ~2.3MB+_meteor_sweep hyp/ref 4.9MB, mllm training_logs 删除项 ~= 170938 8.7K + 171216 148 + 171708 91K + 184921 671K + 092622 70 + 093210 7.9K + seq1213 8.5K + 121640 10.9K + nohup×2 1.1MB + seq094534 435K + stage1 95k 7.6K + seqv2 5×~80K + nohup 6.4K + merge 1.6K + b1 213612 949K + 164030 81K + b2 112252 175K + 150156 265K + 142959 118K + 171433 8.3K + 172716 8.3K ≈ ~4.3MB + eval_results ~1.6MB (excl. meteor 4.9MB) → total ~9-11MB. Mention: 回收空间有限 (主要空间在 checkpoints 145G/wandb 137M/eval_results 94M)。

【建议保留】:
mllm/training_logs/:
- stage2_full_20260809_184921... no wait 184921 → delete. Keep 093455 (该代成功 run).
Hmm — 093455 resume 是成功完整 3 epoch run 记录 → keep. Wait — actually also stage2_full_20260809_184921... delete. So the whole lineage keeps ONE file 093455. Good.
Hmm, wait: 184921 阶段有 0→2000 而 093455 只有 2000→2781+summary。不完整但 summary 完整… 可以了,按计划。
- stage2_full_seq_resume_20260822_230654.log (seq 正式 run 主日志; nohup 版删除) — but hold on, is resume log really complete enough? yes keep.
- stage1_95k_20260825_162055.log (成功 stage1 95k, epoch 1.0 完成 summary) keep — 该代 stage1 唯一成功 run。
- stage2_full_seqv2_20260824_131601.log (seqv2 成功大 run 唯一日志)
- stage2_full_seqv3_mixed_20260826_131938.log (B0 mixed, doc 引用 ×4)
- stage2_full_seqv3_mixed_b1_20260831_135828.log (b1 完成段)
- stage2_full_seqv3_mixed_b2_20260901_205231.log (b2 成功 run)
- stage2_full_seqv3_mixed_b3_20260904_194430.log (b3 成功 run; doc glob 引用)
- stage2_full_seqv3_mixed_b4a_20260907_060023.log (b4a 当前成功 run)
- stage1_162k_b1_20260830_185010.log (stage1-162k 重训成功 run; b1 链阶段2 产物)
- stage2_seqv3_20260825_174612.log + stage3_seqv3_20260826_002520.log (se 阶段2/3 完成 run, stage23_seqv3 评测对应; 虽无文件名引用但为 8-26 决策依据日志;可选) — keep optional.
- b1_pipeline.log / b1_eval_pipeline.log / b2_pipeline.log / b3_pipeline.log / b4a_pipeline.log (pipeline 链记录; b3/b4a doc 引用; b1/b2 记录失败-重试关系并含结果表)
- eval_stage2_full_seqv3_mixed.log (doc ×4), mixed_eval_watcher.log (doc ×1)
- stage2_trainer_state.json? (root, 186KB trainer state 历史) — with stage2_final keep optional. skip.

eval_results: run dirs all keep (not candidates); _meteor_sweep/{sweep.py, sweep2.py} keep (doc ref); official_ckpt_nometa/ official_model_stage2/ keep.

Note git: training logs tracked (48 files tracked in the two dirs); eval_results & wandb & backups & checkpoints gitignored. Deleting tracked logs = real git change (recoverable via history). Should note.

Also note evaluation_results.json root is gitignored by pattern.

Sizes per top-10 listing again.

Length: It's a big deliverable; keep organized, use absolute paths, no emojis, no colon before tool calls (done with tools). Chinese output as user speaks Chinese.

Double-check per-file one-line format requested: "输出每个文件一行：结论短语 + 一两句证据". The user asked that under item 1 as a table-ish listing — do per-file lines grouped.

Final answer compose now. Given constraints, comprehensive but structured.

</details>

All checks complete. Here is the full read-only audit report.

# 日志与产物清理审计报告（mmb4dl）

**重要前置信息**：`mllm/training_logs/`（48 个文件）与根 `training_logs/`、`smoke_test.log` 均**已被 git 跟踪**（`git ls-files` 确认），删除会形成 git 变更但可从历史恢复；`mllm/eval_results/`、`mllm/wandb/`、`mllm/checkpoints/`、`backups/`、根 `evaluation_results.json` 均被 .gitignore 忽略，删除不影响 git。

## 1. 逐文件结局判定

### 根目录 training_logs/（旧 vtimellm 顶层，Aug 7-8 时代）
- `stage2_20260807_200248.log`（390KB）— **正常完成 + 同 run 重复**：20:02:53 启动、Aug 8 01:40 尾部 "Process 149724 exits successfully"，含 num_train_epochs=2 与 train_runtime；与 stage2_final.log **同一秒启动（20:02:53）**，是同一 run 的 deepspeed-launcher 版捕获。
- `stage2_final.log`（313KB）— **正常完成 + 同 run 重复**：同一 run 的另一捕获（无 launcher 行），尾部 1072/1072 + `train_runtime 20204s, epoch 2.0`。两者互备，留一份即可。

### mllm/training_logs/ — Aug 9-10 stage2-full 家系
- `stage2_full_20260809_170938.log`（8.7KB）— **启动即崩溃**：17:09:54 `ImportError: cannot import name 'PeftMixedModel'`（miniconda py3.13 环境问题），return code 1。
- `stage2_full_20260809_171216.log`（148B）— **近乎空**：仅 wrapper 头两行，训练未起即停。
- `stage2_full_20260809_171708.log`（91KB）— **中途被杀**：17:17→18:48 SIGTERM，仅 11%（epoch 0.34），随即被 184921 全量重开，属被弃前奏。
- `stage2_full_20260809_184921.log`（671KB）— **猝停的半程**：18:49→Aug 10 06:18 跑到 82%（2274/2781, epoch 2.45）后无任何错误/exit 行即断（疑 SIGKILL）；后续 resume 从 checkpoint-2000 续跑，其 2000 步后进度被弃。
- `stage2_full_resume_20260810_092622.log`（70B）— **近乎空**：仅 "启动/从 checkpoint-2000 恢复" 两行即止。
- `stage2_full_resume_20260810_093210.log`（7.9KB）— **被杀重试前奏**：09:32 启动 2.5 分钟即 SIGTERM（1003/2781），2 分钟后被 093455 取代。
- `stage2_full_resume_20260810_093455.log`（322KB）— **该代唯一完整成功 run**：09:34:55→13:31，2781/2781、epoch 3.0、train_runtime 14114s、exits successfully。

### mllm/training_logs/ — seq（Aug 12 与 Aug 22-23）
- `stage2_full_seq_20260812_171413.log`（8.5KB）— **环境故障启动失败**：CUDA unknown error（GPU 不可见降 CPU）+ `--tf32 requires Ampere` ValueError，exit 1。
- `stage2_full_seq_20260812_171640.log`（10.9KB）— **中断于 0 步**：同样 CPU-only 重试，模型加载完、0/2781 即停，无错误无 exit。
- 08-12 两文件与 08-22/23 的 seq 日志**不是同一 run**：前者是 GPU 环境故障的首次尝试（当天即废弃），正式 seq run 是 08-22 09:45 那次（GPU 正常、跑到 54% 才被杀）。
- `stage2_full_seq_20260822_094534.log`（435KB）— **被杀半程（主 run 前段）**：09:45→18:59 return code -9 at 54%（1488/2781, epoch 1.6）。
- `stage2_full_seq_nohup.log`（435KB）— **重复双写**：与 094534 同 run（多 wrapper 头 "PER-SEQUENCE TRAINING (no tf32)"，除头尾行外内容一致）。
- `stage2_full_seq_resume_20260822_230654.log`（625KB）— **训练跑满但收尾被打断**：23:06 从 checkpoint-1000 续训至 2781/2781（13h27m），但日志止于 final-save 的 peft 警告+wandb 行（mtime 12:37），无 train_runtime/exit 行；模型产物其后由 merged checkpoint 与 stage2_full_seq/ 评测（Aug 23 16:38）证实完成。
- `stage2_full_seq_resume_nohup.log`（625KB）— **重复双写**：与上一文件同 run（多 wrapper 头）。

### stage1 95k（Aug 25）
- `stage1_95k_20260825_161616.log`（7.6KB）— **同 run 重试的中断者**：16:16 启动仅 8/742 步（epoch 0.01）即无痕中断（mtime 16:19，疑 SIGKILL）。
- `stage1_95k_20260825_162055.log`（214KB）— **同 run 重试的成功者**：16:21 同配置重开，742/742 完成、train_runtime 4975s、17:46 exits successfully。

### seqv2（Aug 24）
- `stage2_full_seqv2_20260824_114030.log`（27KB）— **被杀尝试**：11:40→12:14 return code -9 at 74/2781（3%）。
- `_121607.log`（22KB）— **被杀尝试**：12:16→12:41 -9 at 54/2781（save_steps 已改 200）。
- `_124310.log`（7.8KB）— **被杀尝试**：12:43→12:47 -9 at 0 步。
- `_125007.log`（16KB）— **被杀尝试**：12:50→13:13 SIGTERM at 38/2781。
- `_131353.log`（6KB）— **被杀尝试**：13:13:56→13:15:59 -9 at 0 步（模型刚加载完，data 已换 148k 版）。
- `_131601.log`（1.12MB）— **唯一完整成功大 run**：13:16:02→Aug 25 09:46，3474/3474、epoch 3.0、train_runtime 73656s、exits successfully（wrapper "End (manual) 12:25" 只是手动收尾标记）。
- `stage2_full_seqv2_nohup.log`（6.4KB）— **重复双写**：与 131353 同一 attempt（同 PID 583487 于 13:15:59 被杀）。

### seqv3 家系
- `stage2_seqv3_merge_20260826_002121.log`（1.6KB）— **merge 步骤正常完成**：00:21→00:24 "Merged model saved to ...stage2-seqv3-merged / Use ... for stage3"；一次性步骤，产物（13G merged 目录）仍在，随后 stage3 日志 002520 接力。
- `stage2_full_seqv3_mixed_b1_20260830_213612.log`（949KB）— **被杀大半程**：21:36→Aug 31 13:48 -9 at 91%（3161/3474）；b1_pipeline.log 记为阶段3 尝试#1 失败。
- `_20260831_135828.log`（229KB）— **b1 完成段（完整成功 run 的后半）**：13:58 从 checkpoint-3000 续训，16:30 完成 3474/3474、train_runtime 8936s、exits successfully。
- `_20260831_164030.log`（81KB）— **no-op 收尾 run**：16:40 从 checkpoint-3474 续训=0 步，train_runtime 0.8s 即 exits successfully（用于落最终 adapter，无训练内容）。
- `stage2_full_seqv3_mixed_b2_20260901_112252.log`（175KB）— **被杀尝试**：11:22→14:51 -9 at 600/3474（17%）。
- `_150156.log`（265KB）— **被杀尝试**：15:01→20:42 -9 at 1292/3474（37%）。
- `_205231.log`（778KB）— **b2 完整成功 run**：20:52→Sep 2 10:15，3474/3474、train_runtime 48008s、exits successfully（b2_pipeline 记为尝试#3 完成）。
- `_20260903_142959.log`（118KB）— **被杀尝试（实为 b3 阶段）**：14:29→17:04 -9 at 400/3474；cmd 的 output_dir 是 `...-mixed-b3`（whole_scene+meta2），文件名误标 b2。
- `_20260903_171433.log`（8.3KB）与 `_172716.log`（8.3KB）— **0 步被杀（实为 b3 阶段）**：分别 17:14/17:27 启动约 2.5 分钟即 -9，均为 resume `mixed-b3/checkpoint-400` 的失败尝试（内容仅时间戳不同），随后真正的 b3 run 于 Sep 4 19:44 启动并成功。

### 其他被列文件
- `b1_pipeline.log`（749B）— **B1 自动链记录**：阶段1a/1b 特征重提 OK（28130/850）+ 阶段2 stage1-162k 重训（首试 rc=247 失败→18:50 重试 rc=0）+ 阶段3 mixed-b1 三试均败链停止；该"链停止"叙述与实际（手动 resume 成功 + 23:35 评测 rc=0）相矛盾，结论过时，关键数值已固化到 eval_results/。不被 md 引用。
- `b1_eval_pipeline.log`（357B）— **B1 正式结果记录**：阶段4 评测 rc=0 + B1 vs B0 结果表（acc 0.7787/mIoU 0.2653），与 eval_results/stage2_full_seqv3_mixed_b1/metrics.json 同值。不被 md 引用。
- `b2_pipeline.log`（638B）— **B2 链正式记录**：阶段1 尝试#1/#2 失败→#3 完成 + 阶段2 评测 rc=0 + 结果表（0.7649/0.1992），是解释 3 个 b2 run 日志关系的钥匙。不被 md 引用。
- `b3_pipeline.log`（13.8KB）— **B3 链正式记录**：阶段1 显存门控等待 149 次后完成 + 阶段2 评测 rc=0 + mIoU 0.3467 超论文结果。**被 md 引用**。
- `b4a_pipeline.log`（7.7KB）— **B4a 链（进行中/最新）**：阶段1 门控等待 85 次后训练完成（Sep 7 06:00→Sep 8 07:40），阶段2 评测尝试#1 rc=124（超时，后由 eval_results b4a 产物补上）。**被 md 引用**。
- `smoke_test.log`（根目录，20KB）— **早期冒烟记录**：Aug 9 的 50 条推理 smoke 并打印 paper-vs-ours 表；数值是早期模型的粗测，无 md 引用，参考价值低。

## 2. 磁盘占用实测（只报数）

| 路径 | 占用 |
|---|---|
| mllm/eval_results/ | **94M**（root 散落文件约 0.9M）|
| ├ _meteor_sweep/ | 4.8M |
| ├ official_ckpt_nometa/ | 2.5M |
| ├ official_model_stage2/ | 4K（含指向 /root/autodl-tmp/ljq/… 的软链）|
| ├ stage23_seqv3/ | 12M |
| ├ stage2_full_seq/ | 8.0M |
| ├ stage2_full_seqv2/ | 11M |
| ├ stage2_full_seqv3_mixed/ | 8.0M |
| ├ stage2_full_seqv3_mixed_b1/ | 14M |
| ├ stage2_full_seqv3_mixed_b2/ | 11M |
| ├ stage2_full_seqv3_mixed_b3/ | 11M |
| └ stage2_full_seqv3_mixed_b4a/ | 12M |
| mllm/training_logs/ | **15M** |
| training_logs/（根）| 876K |
| mllm/evaluation/ | **25M** |
| mllm/wandb/ | **137M**，offline-run 26 个 |
| wandb/（根）| 76K |
| mllm/checkpoints/ | **145G**（stage1 61M / stage2 2.4G / stage2-full 4.1G / stage2-full-seq 13G / seq-merged 13G / seqv2 13G / seqv2-merged 13G / seqv3-mixed 556M / mixed-b1 13G / mixed-b2 13G / mixed-b3 13G / mixed-b4a 13G / seqv3 13G / seqv3-merged 13G / stage3-seqv3 12G）|
| backups/ | **898M**（B3_stage2_final_20260907.tar.gz + manifest + b4dl_seqv3_mixed_backup_20260827.tar.gz）|
| mllm/b4dl_dataset/ | **1.6G** |
| 项目一级目录前 10 | mllm 159G / models 15G / encoders 11G / .git 1.5G / backups 898M / .claude 601M / dataset 384M / datageneration 76M / assets 17M / docs 8.8M |

## 3. mllm/eval_results/ 根目录散落文件判定

- `eval_log.txt`（479KB，Aug 22 21:07）— **中断评测日志**：尾部停在 binary_qa 2092/7525（28%）。配套 ckpt 是 seq_predictions.json.ckpt；正式完整产物是 `stage2_full_seq/{predictions.json 8.3MB, metrics.json}`（Aug 23 16:38-39，同性质且更新）。注意 stage2_full_seq/ 内没有别的 eval_log，这是该次（未完成）评测唯一日志，但评测本身未完成。
- `seq_predictions.json.ckpt`（692KB，21:07）— **中断评测的部分预测 ckpt**（顶层 key `results`，30k 全量应是 8.3MB 的 task 结构），被 stage2_full_seq/predictions.json 取代。
- `smoke_predictions.json.ckpt`（12KB，21:09）— 同上中断评测的另一个任务级 ckpt 残片，被取代。
- `smoke_seq_log.txt / smoke_seq_metrics.json / smoke_seq_predictions.json`（21:23，84KB）— seq 模型的冒烟评测全套，被次日 16:38 的正式 stage2_full_seq 评测取代。
- `smoke_2000_log.txt / _metrics.json / _predictions.json`（Aug 23 13:21）— 2000 条冒烟（在 resume 结束与正式评测之间），同样被同日正式评测取代。
- `smoke_meta_metrics.json / smoke_meta_predictions.json`（Aug 9）— 早期 stage2 meta 时代冒烟，被后续所有版本取代。
- `_meteor_sweep/`（Sep 7）— `hyp.txt`(2.3MB)+`ref.txt`(2.6MB) 是从 B0 predictions.json 拆出的文本中间文件，可由 `stage2_full_seqv3_mixed/predictions.json` 重新生成；扫描**成果已固化**为各 run 目录的 `metrics_recomputed_20260907_dual.json`（Sep 7 21:26-21:31，含 meteor_pycocoevalcap 双口径）。`sweep.py/sweep2.py` 脚本被 METEOR 文档引用，应保留。
- `official_ckpt_nometa/`、`official_model_stage2/` — 官方对照评测产物与软链，被文档引用（0.1737），非清理对象。

## 4. 文档引用关系（docs/ + docs/learn docs/ + CLAUDE.md + README.md + mllm/README.md）

**文档确实引用、删除会破坏路径指引的**：
- `mllm/training_logs/b3_pipeline.log` ← `docs/learn docs/B4DL_B3整场景meta修复mIoU超论文_20260906.md:89`（与 `stage2_full_seqv3_mixed_b3_*.log` glob 同处）
- `mllm/training_logs/b4a_pipeline.log` ← `docs/learn docs/B4DL_工作进度总结_20260906.md:101`
- `mllm/training_logs/stage2_full_seqv3_mixed_20260826_131938.log`（B0 mixed）← 4 个文档（B4DL_训练方法对比:3,112/116、seqv3混合训练评测分析:141、基线锁定:125）
- `mllm/training_logs/eval_stage2_full_seqv3_mixed.log` ← 4 个文档
- `mllm/training_logs/mixed_eval_watcher.log` ← 评测数值对比文档:113
- `mllm/eval_results/_meteor_sweep/{sweep,sweep2}.py` ← B4DL_METEOR双口径溯源与评测规则_20260907.md:36
- 各 `eval_results/<run>/` 目录、`official_ckpt_nometa/`、`metrics_recomputed_20260907_dual.json` 被多个文档引用（但这些不是清理对象）

**将被清理对象均无任何 md 引用**：stage1_95k_*、stage2_full_20260809_*、stage2_full_resume_*、stage2_full_seq_*（含 nohup 双胞胎）、stage2_full_seq_resume_*、stage2_full_seqv2_*、stage2_seqv3_merge_*、stage1_162k_b1_*、b1/b2 的 *_pipeline.log、seqv3_mixed_b1/b2 各 run 日志、根 eval_results 散落文件（smoke_*/eval_log.txt/*.ckpt）、根 smoke_test.log、根 training_logs 两文件。CLAUDE.md:193 的 `./evaluation/evaluation_results.json` 是命令示例（指向 mllm/evaluation 脚本路径），与根目录占位文件无关。

## 汇总

### 建议可安全删除（失败尝试/空跑/双写重复/被同版本成功 run 取代，且无文档引用；合计约 8-10MB）

**根目录 training_logs/**
- `/root/autodl-tmp/wql/mmb4dl/training_logs/stage2_20260807_200248.log`（与 stage2_final.log 同 run 双捕获，保留其一即可；保守可留 stage2_final.log）

**mllm/training_logs/**
- `/root/autodl-tmp/wql/mmb4dl/mllm/training_logs/stage2_full_20260809_170938.log`（ImportError 秒崩）
- `.../stage2_full_20260809_171216.log`（148B 空跑）
- `.../stage2_full_20260809_171708.log`（11% SIGTERM，被 184921 重开取代）
- `.../stage2_full_20260809_184921.log`（82% 猝停半程，其 2000 步后进度被 resume 弃用；完整记录=093455，曲线在 wandb offline-run）
- `.../stage2_full_resume_20260810_092622.log`（70B 空跑）
- `.../stage2_full_resume_20260810_093210.log`（2.5 分钟被杀，被 093455 取代）
- `.../stage2_full_seq_20260812_171413.log`、`.../stage2_full_seq_20260812_171640.log`（GPU 环境故障尝试）
- `.../stage2_full_seq_nohup.log`、`.../stage2_full_seq_resume_nohup.log`（与主日志双写重复）
- `.../stage2_full_seq_20260822_094534.log`（54% -9；完整记录=230654 resume）
- `.../stage1_95k_20260825_161616.log`（8 步中断；成功版=162055）
- `.../stage2_full_seqv2_20260824_114030.log`、`_121607.log`、`_124310.log`、`_125007.log`、`_131353.log`、`stage2_full_seqv2_nohup.log`（均被 131601 取代或为其双写）
- `.../stage2_seqv3_merge_20260826_002121.log`（merge 正常完成但产物在 13G merged checkpoint 中，无文档引用）
- `.../stage2_full_seqv3_mixed_b1_20260830_213612.log`（91% -9 半程，续跑成功段在 135828）
- `.../stage2_full_seqv3_mixed_b1_20260831_164030.log`（0 步 no-op 收尾）
- `.../stage2_full_seqv3_mixed_b2_20260901_112252.log`、`_150156.log`（被 205231 取代）
- `.../stage2_full_seqv3_mixed_b2_20260903_142959.log`、`_171433.log`、`_172716.log`（文件名误标 b2，实为 b3 阶段失败尝试，被 b3 run 194430 取代）
- `.../stage1_162k_b1_20260830_165411.log`（rc=247 失败尝试，成功版=185010；虽不在清单，同目录同类）

**mllm/eval_results/ 根目录散落**
- `.../eval_log.txt`、`seq_predictions.json.ckpt`、`smoke_predictions.json.ckpt`、`smoke_2000_log.txt`、`smoke_2000_metrics.json`、`smoke_2000_predictions.json`、`smoke_meta_metrics.json`、`smoke_meta_predictions.json`、`smoke_seq_log.txt`、`smoke_seq_metrics.json`、`smoke_seq_predictions.json`（均为中断评测残片或已被 run 目录正式产物取代的冒烟）
- `.../_meteor_sweep/hyp.txt`、`.../_meteor_sweep/ref.txt`（可由 predictions.json 再生成；**保留同目录 sweep.py/sweep2.py**）

**项目根目录**
- `/root/autodl-tmp/wql/mmb4dl/smoke_test.log`（Aug 9 早期冒烟，数值过时）
- `/root/autodl-tmp/wql/mmb4dl/evaluation_results.json`（全 1.0 占位自检输出，非真实评测；gitignore 已含该名）— 顺带发现，非清单内

### 建议保留（成功 run 唯一留档 / 文档引用 / pipeline 链记录）

**mllm/training_logs/**
- `stage2_full_resume_20260810_093455.log`（Aug 9-10 代 stage2-full 唯一完整 3-epoch 成功记录）
- `stage2_full_seq_resume_20260822_230654.log`（seq 正式 run 主日志，覆盖 checkpoint-1000→2781）
- `stage1_95k_20260825_162055.log`（该代 stage1 唯一成功 run）
- `stage2_full_seqv2_20260824_131601.log`（seqv2 唯一完整成功大 run）
- `stage2_full_seqv3_mixed_20260826_131938.log`（B0 mixed，**4 个文档引用**）
- `stage2_full_seqv3_mixed_b1_20260831_135828.log`、`stage2_full_seqv3_mixed_b2_20260901_205231.log`、`stage2_full_seqv3_mixed_b3_20260904_194430.log`（**文档 glob 引用**）、`stage2_full_seqv3_mixed_b4a_20260907_060023.log`（各代完整成功 run）
- `stage1_162k_b1_20260830_185010.log`（stage1-162k 重训成功 run，后续 b 系列 stage2 均依赖其 mm_projector）
- `b1_pipeline.log`、`b1_eval_pipeline.log`、`b2_pipeline.log`（B1/B2 链的失败-重试叙述与结果表，是解读各 run 日志相互关系的唯一钥匙；b1_pipeline 含阶段1a/1b 特征重提记录）、`b3_pipeline.log`、`b4a_pipeline.log`（**后两者文档引用**）
- `eval_stage2_full_seqv3_mixed.log`（**4 个文档引用**）、`mixed_eval_watcher.log`（**1 个文档引用**）
- 可选保留（该代虽被取代但属成功留档，成本低）：`stage2_seqv3_20260825_174612.log`、`stage3_seqv3_20260826_002520.log`（stage23_seqv3 评测的对应训练记录，决策文档依据）

**根目录**
- `training_logs/stage2_final.log`（若坚持留一份 Aug 7 旧代记录；与 200248 二选一）

**mllm/eval_results/**：全部 run 子目录及 `_meteor_sweep/sweep.py`、`sweep2.py`（文档引用）、`official_ckpt_nometa/`、`official_model_stage2/` 均保留。

**提醒**：真正的大头在 `mllm/checkpoints/`（145G，13 个 13G 级版本目录 + merged 双份）与 `mllm/wandb/`（137M/26 runs），本次日志清理可回收空间约 10MB 量级；checkpoints 中 `*-merged` 与源 `-seq/-seqv2/-seqv3` 双份 13G 结构、以及 seq/seqv2 等被取代版本的保留决策，建议作为下一步单独评估。

