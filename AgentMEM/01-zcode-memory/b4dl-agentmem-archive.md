---
name: b4dl-agentmem-archive
description: 本项目 Agent 记忆统一归档在仓库内 AgentMEM/（ZCode 记忆+会话导出、Claude Code
  记忆+原始会话、计划、Claude_record、历史快照），含刷新脚本与默认 gitignore 策略（2026-09-10 建立）
metadata:
  node_type: memory
  type: project
  originSessionId: sess_03e0f32b-5fc1-4f0d-9896-2f38649d5be6
---

2026-09-10 应需求建立 **`AgentMEM/`（仓库根内）**，把本项目的 agent 记忆集中归档，便于随迁移整体带走：

- 目录：`01-zcode-memory/`（ZCode 长期记忆 + `MEMORY.md` 索引）、`02-zcode-sessions/`（ZCode sqlite 只读导出的 65 个会话 markdown + `INDEX.md`，2026-08-23~09-10）、`03-zcode-plans/`、`04-claude-code-memory/`、`05-claude-code-sessions/`（Claude Code 52 个原始 `.jsonl` + `INDEX.md`，2026-05-17~08-13）、`07-agent-instructions/`（CLAUDE.md 快照 + `settings.local.json`）、`08-claude-records/`、`09-history-snapshots/`（09-09 记忆快照）。
- **刷新方法**：`python3 AgentMEM/tools/export_zcode_sessions.py`（只读 `/root/.zcode/cli/db/db.sqlite`）+ `python3 AgentMEM/tools/build_indexes.py`（重建 Claude 索引与密钥扫描）。记忆更新后重跑这两条即可。
- **安全**：`POTENTIAL_SECRETS.txt` 记录 8 个文件含 `sk-` 样式 key（7 个 Claude 原始会话 + `settings.local.json`）；`AgentMEM/.gitignore` 默认屏蔽 `05-claude-code-sessions/` 与 `settings.local.json`，让记忆本体可安全提交。
- 其它 agent 排查结论：Cursor/Codex 无本项目记忆（codex 记忆库为空表）；`/root/.claude-lhwt`、`.claude-gy`、`.claude-miller` 属他人目录；`/root/.claude/plans/` 里唯一的 plan 属 hyn/MAP_SAM 项目。
- 归档与《迁移清单与 SSH 搬运手册》（`docs/learn docs/B4DL_迁移清单与SSH搬运手册_20260910.md`）互补：前者管代码与数据，AgentMEM 管记忆与对话历史。

关联：[[b4dl-project-overview]]、[[b4dl-server-access-workflow]]、[[b4dl-disk-requirements]]、[[b4dl-git-commit-conventions]]
