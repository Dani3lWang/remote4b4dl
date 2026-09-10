---
name: git-auto-commit
description: 每次代码修改后自动进行 git commit
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7d4a1a02-9ca2-4b56-b958-17771d2780b0
---

每次完成代码更改后，自动创建对应的 git commit。使用中文提交信息，commit 前先 `git add` 具体修改的文件（不使用 `git add -A`）。

**Why:** 用户希望每次代码修改都有版本记录，便于追溯和回滚。

**How to apply:** 任何通过 Edit/Write 工具修改代码文件后，执行 `git add <具体文件>` + `git commit`。如果 claude.md 也有变更，同样需要提交。
