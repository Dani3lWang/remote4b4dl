---
name: b4dl-git-commit-conventions
description: 用户的 git/文档规范——中文提交信息、feat/fix/docs 前缀、改动后立即 commit、同步更新 CLAUDE.md
metadata:
  node_type: memory
  type: feedback
  originSessionId: sess_57bc4f46-cf94-4f89-9779-568442660f70
---

用户在 mmb4dl/remote4b4dl 项目的协作规范（2026-08-23 明确要求写入）：

- 每次代码修改后立即 git commit（不必等用户催），使用**中文提交信息**，带 `feat:` / `fix:` / `docs:` / `chore:` 前缀，正文说明动机（如对齐论文哪一节）+ 改了什么。
- 提交后**及时 push 到 origin 远程同步**，不要长期积压本地领先提交（2026-08-28 用户明确要求）。
- 结构性改动要同步更新仓库根的 `CLAUDE.md`（有 AGENTS.md 也一并），把新机制/使用流程写进去（如 per-sequence 修改就写入了 CLAUDE.md，commit cd6d975）。
- 评测前先跑一次旧格式 baseline 评测留档，再上新格式，便于对比。

**Why:** 用户靠 git log + CLAUDE.md + 导出的对话 md 串联跨天、跨机器（Windows 本地 ↔ 远端服务器）的工作，提交信息是主要导航手段；远端同步保证 Windows 本地仓库能拉取到最新进展。
**How to apply:** 在该项目改完文件后主动执行规范 commit 并立即 push origin；commit message 中文、前缀式、正文含动机；涉及训练/评测流程变化时提醒更新 CLAUDE.md。

相关：[[b4dl-server-access-workflow]]
