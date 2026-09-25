---
name: b4dl-agentmem-archive
description: AgentMEM/ 已于 2026-09-25 解散——记忆本体与 Qoder 会话归档迁入 git 跟踪的 docs/memory/，ZCode 65 会话全文（12M）移到仓库外保存，独有的 RTX5090_DEBUG_LOG.md 救回 docs/learn docs/；此条记明保留了什么、删了什么、各自在哪
metadata:
  node_type: memory
  type: project
  originSessionId: f554c11b-93d7-4687-a2ad-c0a5d4dec88a
---

2026-09-10 曾在仓库根建立 **`AgentMEM/`**，把本项目各 agent 的记忆集中归档以便随迁移带走。**该目录已于 2026-09-25 按用户指示解散删除**，且它整目录被 `.gitignore` 屏蔽、从未进过 git 历史，所以未迁移的部分**不可恢复**。

## 迁到了哪里

`docs/memory/`（git 跟踪）：

- `MEMORY.md` + 18 条记忆本体（原 `01-zcode-memory/`，含本次新增的 [[b4dl-reasonseg-seg-line-2026-09]]）
- `qoder-sessions/`：4 份 Qoder 会话正文归档 + `INDEX.md`（原 `06-qoder-sessions/`，2026-09-20~25 的 seg 训练线）
- `README.md`：本次迁移的说明与来源

## 丢了什么（未迁移即删除）

删除前实测本机 `AgentMEM/` 共 **12M**（不是旧 README 记的 59M —— **`05-claude-code-sessions/` 在当初迁到 autodl3 时就没带过来，本机从来不存在**，所以这次删除并没有丢 Claude 原始会话）：

| 原目录 | 内容 | 规模 | 处置 |
|---|---|---|---|
| `01-zcode-memory/` | 19 个文件：`MEMORY.md` + 18 条记忆 | 132K | **已迁** → `docs/memory/` |
| `06-qoder-sessions/` | 5 个文件：4 份 Qoder 会话 + INDEX | 92K | **已迁** → `docs/memory/qoder-sessions/` |
| `08-claude-records/` | 10 个文件 | 112K | 9 个与仓库根 `Claude_record/` 重复；**唯一独有的 `RTX5090_DEBUG_LOG.md` 已救回** → `docs/learn docs/RTX5090_DEBUG_LOG.md`（8K，旧机 sm_120 环境调试记录）。⚠ 没放进 `Claude_record/`：那个目录整目录被 `.gitignore:83` 屏蔽，放那儿等于没保住 |
| `02-zcode-sessions/` | **ZCode 65 个会话的 markdown 全文导出**（2026-08-23~09-10）+ INDEX，66 个文件 | **12M** | **移出仓库保存** → `/root/autodl-tmp/AgentMEM_zcode_sessions_20260925/`（不在 git 里，但本机可查原话） |
| `09-history-snapshots/` | 15 个文件：09-09 记忆快照 + `B4DL_记忆导出_合并版_20260909.md` | 140K | 删除；11 个是 `01` 的副本，合并版与 4 个旧版记忆均被现行版本取代 |
| `03-zcode-plans/` | 5 份 ZCode 计划（09-01~09-08） | 32K | 删除；均已被实际执行结果取代 |
| `07-agent-instructions/` | `CLAUDE.md` 的 2026-09-10 快照（18847B） | 20K | 删除；仓库根有现行 `CLAUDE.md`（7775B，内容已重写） |
| `04-claude-code-memory/` | `MEMORY.md` + `feedback_git_commit.md`（2026-05-24） | 8K | 删除；git 规范已并入现行 `CLAUDE.md` 与 [[b4dl-git-commit-conventions]] |
| `tools/`、`POTENTIAL_SECRETS.txt`、`README.md`、`.gitignore` | 归档生成脚本与密钥扫描报告 | 32K | 删除；脚本只对已不存在的 sqlite/jsonl 有意义 |

⇒ **没有不可恢复的损失**：记忆本体与 Qoder 会话归档进了 git（`docs/memory/`），ZCode 65 个会话全文移出到仓库外仍在盘上，独有的调试日志救回了 `docs/learn docs/`。真正删掉的只有已被取代的旧快照、旧计划与只对已不存在的 sqlite/jsonl 有意义的脚本（合计约 230K）。代价是 ZCode 会话全文**不再随仓库走**——换机器或克隆仓库时拿不到，需要时得回本机 `/root/autodl-tmp/` 取。

**顺带查清的一件事**：仓库根 `Claude_record/` 也被 `.gitignore:83` 整目录屏蔽，里面的 9 份 2026-05~06 记录（`PAPER_PLAN.md`、`TODO.md`、`project_analysis.md` 等）**同样从未进过 git**，只存在于本机磁盘。若哪天要清理磁盘或换机，这批和 ZCode 会话是同一类风险。

## 若要查当时原话

- **ZCode 65 个会话**：本机 `/root/autodl-tmp/AgentMEM_zcode_sessions_20260925/`（文件名形如 `01_<标题>__sess_<id>.md`，含 `INDEX.md`）。导出源在旧机 RTX 5090 的 `/root/.zcode/cli/db/db.sqlite`，若那台机器还在可重跑导出脚本再生成。
- **Qoder 会话**（2026-09-20~25 的 seg 训练线）：`docs/memory/qoder-sessions/`，每份文件头带会话 ID，可用 `read_chat_session` 取比归档更完整的版本（含工具输出）。
- **Claude Code 52 个原始会话**：本机从来没有（当初迁到 autodl3 时未带过来），原件在旧机项目内 `.claude/`。
- `08-claude-records/` 的内容看仓库根 `Claude_record/`（9 份，**注意该目录被 gitignore、未进 git**）；其中独有的 `RTX5090_DEBUG_LOG.md` 已单独救到 git 跟踪的 `docs/learn docs/RTX5090_DEBUG_LOG.md`。

关联：[[b4dl-project-overview]]、[[b4dl-server-access-workflow]]、[[b4dl-disk-requirements]]、[[b4dl-git-commit-conventions]]
