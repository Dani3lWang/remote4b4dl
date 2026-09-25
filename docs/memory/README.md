# docs/memory —— B4DL 项目的 Agent 长期记忆

**迁入时间**：2026-09-25 ｜ **来源**：仓库根 `AgentMEM/`（已删除）｜ **机器**：autodl3（`/root/autodl-tmp/mmb4dl`）

这里放的是**跨会话要记住的结论**：项目事实、已判死的方向、踩过的坑与运行口径。与 `docs/learn docs/` 的分工是——那边是**带日期的实验记录与分析报告**（一次实验一篇），这边是**不带日期、持续更新的记忆条目**（一个主题一篇，1–8 KB）。

## 怎么读

| 你想做什么 | 去哪 |
|---|---|
| 快速回顾项目结论与踩过的坑 | `MEMORY.md`（索引，每条一行摘要）→ 对应记忆文件 |
| 查 ReasonSeg seg 分支现在到哪一步了 | `b4dl-reasonseg-seg-line-2026-09.md` + `../learn docs/B4DL_ReasonSeg训练现状_20260925.md`（后者是带时刻的实测快照） |
| 核对某次 Qoder 对话的原话 | `qoder-sessions/INDEX.md` → 对应 md（含会话 ID，可用 `read_chat_session` 取更完整版本） |
| 查运行约定（环境、评测口径、git 规范） | **仓库根 `CLAUDE.md` 是权威**；这里的记忆只补充"为什么"与踩坑经过 |

## 记忆条目格式约定

```markdown
---
name: b4dl-<主题>
description: 一行摘要，写清结论而不是范围
metadata:
  node_type: memory
  type: project
  originSessionId: <产生这条记忆的会话 ID>
---

正文：稠密的事实 + 判死的结论 + 机制解释。
交叉引用用 [[b4dl-其它条目]]，末尾一行「关联：」。
```

写记忆的三条规矩（都是踩过坑才有的）：

1. **记结论要带机制**，不要只记"没涨"。例如损失杠杆判死，要记的是"正例占 3e-4 → BCE 把正例梯度稀释 3000 倍 → 沉默与对冲的损失面持平（下坡仅 4.9%）"，否则下次还会再试一遍。
2. **负结果与正结果同等入档**，并写明是哪个口径下的负结果（manifest 名 + 记录数 + 分母口径）。
3. **自己的方法学错误也要记**。`b4dl-reasonseg-seg-line-2026-09.md` 里专门有一节记"判据挑错轴复发三次"——这类错误的复发成本远高于一次实验失败。

## 2026-09-25 的迁移：保留了什么、丢了什么

原 `AgentMEM/` 整目录被 `.gitignore` 屏蔽、**从未进过 git**，删除即不可恢复。本次按用户指示只迁移有用的记忆本体（删除前实测本机共 **12M**；旧 README 记的 59M 含 `05-claude-code-sessions/`，而那个目录当初迁到 autodl3 时就没带过来，本机从来不存在）：

| 迁入本目录 | 原位置 |
|---|---|
| `MEMORY.md` + 18 条记忆 | `AgentMEM/01-zcode-memory/`（132K） |
| `qoder-sessions/`（4 份会话 + INDEX） | `AgentMEM/06-qoder-sessions/`（92K） |
| — | `AgentMEM/08-claude-records/RTX5090_DEBUG_LOG.md`（唯一独有的一份，8K）救回 **`docs/learn docs/RTX5090_DEBUG_LOG.md`**。⚠ 没放仓库根 `Claude_record/`——那个目录整目录被 `.gitignore:83` 屏蔽，放那儿等于没保住；该目录另 9 个文件与 `Claude_record/` 内容重复 |
| — | `AgentMEM/02-zcode-sessions/`（ZCode **65 个会话全文**，12M / 66 文件）**移出仓库保存** → `/root/autodl-tmp/AgentMEM_zcode_sessions_20260925/`。不在 git 里，但本机可查原话；换机器即拿不到 |

| **删除** | 规模 | 说明 |
|---|---|---|
| `09-history-snapshots/`：09-09 快照 + 合并版导出 | 140K | 11 个是 `01` 的副本，其余被现行记忆取代 |
| `03-zcode-plans/`：5 份旧计划 | 32K | 已被实际执行结果取代 |
| `07-agent-instructions/CLAUDE.md`：09-10 快照（18847B） | 20K | 仓库根有现行版（7775B，已重写） |
| `04-claude-code-memory/`：2026-05-24 的 2 个文件 | 8K | git 规范已并入现行 `CLAUDE.md` |
| `tools/`、`POTENTIAL_SECRETS.txt`、`README.md`、`.gitignore` | 32K | 脚本只对已不存在的 sqlite/jsonl 有意义 |

⇒ **没有不可恢复的损失**，删掉的约 230K 全是已被取代的旧快照与失效脚本。逐条清单见 `b4dl-agentmem-archive.md`（含各原件位置）。

## 安全

本目录**已进 git 跟踪**，因此写入前必须确认无密钥。迁移时对全部迁入文件按 `sk-[A-Za-z0-9]{10,}`、`api_key=`、`BEGIN PRIVATE KEY`、`ghp_`、`password=` 扫过，仅 `b4dl-disk-requirements` 的文件名撞上 `sk-` 模式（两处误报），无真实密钥。原先靠 `AgentMEM/tools/build_indexes.py` 自动生成 `POTENTIAL_SECRETS.txt`，该脚本已随目录删除 —— **现在只能手工 grep，往这里加内容前记得扫。**
