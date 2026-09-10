# AgentMEM —— B4DL 项目的 Agent 记忆归档

**生成时间**：2026-09-10 11:24 ｜ **源机**：AutoDL RTX 5090（`/root/autodl-tmp/wql/mmb4dl`）
**内容**：本项目相关的 ZCode 与 Claude Code 记忆、计划、会话记录与项目记录
**规模**：177 个文件，59 MB（其中两个会话归档 `02-zcode-sessions/` 与 `05-claude-code-sessions/` 合计约 60 MB）

---

## 0. 先看这里

| 你想做什么 | 去哪个目录 |
|---|---|
| 快速回顾项目结论与踩过的坑 | `01-zcode-memory/`（先看 `MEMORY.md` 索引，每条记忆 1–4 KB） |
| 查某次 ZCode 对话的细节 | `02-zcode-sessions/INDEX.md` → 对应 md（65 个会话，含每会话待办） |
| 查某次 Claude Code 会话聊了什么 | `05-claude-code-sessions/INDEX.md`（含每条会话的首句用户指令摘要） |
| 看当初的训练/评测计划 | `03-zcode-plans/`（ZCode 计划）、`08-claude-records/`（含 `PAPER_PLAN.md`、`TODO.md`） |
| 迁移到新机器 | 见 §3「恢复到另一台机器」 |
| 提交到 git 前 | 见 §4「安全与 git」，先读 `POTENTIAL_SECRETS.txt` |

---

## 1. 归档内容与来源

| 目录 | 内容 | 原始位置（本机） | 规模 | 时间范围 |
|---|---|---|---|---|
| `01-zcode-memory/` | ZCode 长期记忆 16 条 + `MEMORY.md` 索引 | `/root/.zcode/cli/memories/projects/project-64d8f08ddafc8e25/memory/` | 17 文件 / 96K | 2026-08-24 ~ 09-10 |
| `02-zcode-sessions/` | ZCode 会话全文导出（markdown）+ `INDEX.md` | `/root/.zcode/cli/db/db.sqlite`（只读导出，见 `tools/export_zcode_sessions.py`） | 65 会话 + 索引 / 12M | 2026-08-23 ~ 09-10 |
| `03-zcode-plans/` | ZCode 计划文档（`plan-sess_*.md`） | 项目内 `.zcode/plans/` | 5 文件 / 32K | 2026-09-01 ~ 09-08 |
| `04-claude-code-memory/` | Claude Code 记忆（`MEMORY.md` + `feedback_git_commit.md`） | 项目内 `.claude/memory/` | 2 文件 / 8K | 2026-05-24 |
| `05-claude-code-sessions/` | Claude Code 原始会话记录（`.jsonl`）+ 子代理记录 + `INDEX.md` | 项目内 `.claude/*.jsonl`、`.claude/sessions-{parent,mllm,datageneration}/` | 52 会话 / 48M | 2026-05-17 ~ 08-13 |
| `07-agent-instructions/` | 项目 `CLAUDE.md` 快照、Claude Code 权限配置 | 仓库根 `CLAUDE.md`、`.claude/settings.local.json` | 2 文件 / 24K | 2026-09-10 快照 |
| `08-claude-records/` | Claude 记录的项目文档（含 `MANIFEST.md`、`PAPER_PLAN.md`、`TODO.md`、`RTX5090_DEBUG_LOG.md`、`project_analysis.md`） | 项目内 `Claude_record/` | 10 文件 / 112K | 2026-05 ~ 06 |
| `09-history-snapshots/` | 2026-09-09 的 ZCode 记忆快照（单条 + 合并版 `B4DL_记忆导出_合并版_20260909.md`） | `backups/zcode-memory-export-20260909/` | 15 文件 / 140K | 2026-09-09 |
| `tools/` | 归档生成脚本（可重跑） | — | 2 文件 | 2026-09-10 |
| `POTENTIAL_SECRETS.txt` | 密钥扫描报告（只列路径与命中次数，不含密钥） | 由 `tools/build_indexes.py` 生成 | 1 文件 | 2026-09-10 |
| `README.md`、`.gitignore` | 本说明与 git 策略 | — | 2 文件 | 2026-09-10 |

**重跑方式**（记忆更新后想刷新归档）：

```bash
cd /root/autodl-tmp/wql/mmb4dl
python3 AgentMEM/tools/export_zcode_sessions.py   # 重新导出 ZCode 会话（只读 sqlite，不写入原始库）
python3 AgentMEM/tools/build_indexes.py           # 重建 Claude 会话索引 + 密钥扫描
```

`02-zcode-sessions/` 的导出约定：对话正文全文保留；`思考过程` 折叠在 `<details>` 里；工具调用只记「工具名 + 标题 + 输入/输出摘要（截断 500/800 字符）」，因此**归档无法替代原始记录**——需要完整工具输出时仍应查 `.claude/*.jsonl` 或 ZCode 会话库。另外，导出时正在进行的那个会话（本次归档对应的 `sess_03e0f32b`，编号 64「云服务器训练项目磁盘容量估算」）只是**当时的快照**，内容不完整；需要完整版可随时重跑上面的命令刷新。

---

## 2. 记忆里都记了什么（速览）

`01-zcode-memory/MEMORY.md` 是权威索引，摘要如下：

- **项目总览 / 服务器工作流**：mmb4dl 复现架构、6 任务 benchmark、Windows 本地仓库 + SSH 到 AutoDL 的工作方式。
- **基线演进（B0→B4a）**：B3（整场景 + meta2）mIoU **0.3467** 超论文 0.311，为当前最优；B4a 的 TG 高帧段过采样是负结果（0.3271），病灶是输入缺帧号信号。
- **per-sequence 改造**：三重错配的确认、seqv2/seqv3/两阶段/mixed 演进、评测需 `--per_sequence` 且与训练代次对齐。
- **评测方法学陷阱**：NLTK 与 pycocoevalcap 的 METEOR 虚高差异、BERTScore OOM、skip 缩水、ckpt 断点续跑。
- **数据/权重代次**：Stage1 官方 162K 对齐（161,629 条 / sample_token 键控 / 特征 28,130）；LiDAR-CLIP 本地 ckpt 是 ONCE 而非官方 nuScenes 版。
- **工程规范与边界**：中文 commit + 改后即 commit + 及时 push；写操作仅限项目文件夹（服务器多项目共用）。
- **清理记录**：checkpoints 145G→1.7G（09-09）、`b4dl_dataset` 1.5G→501M（09-10）。
- **待办方向**：RL/RLVR 引进计划（奖励函数可复用 `evaluate_model` 的规则函数）。
- **迁移与容量**：磁盘需求实测（sweeps 338G 全代码零引用）与《迁移清单与 SSH 搬运手册》。

---

## 3. 恢复到另一台机器

| 类别 | 放回位置 | 注意 |
|---|---|---|
| ZCode 记忆 | `<新机>/.zcode/cli/memories/projects/<project-id>/memory/` | project-id 由工作目录派生（本机为 `project-64d8f08ddafc8e25`），换路径后 id 会变——最稳妥是让 ZCode 在新机上先跑一次以生成目录，再把 `01-zcode-memory/` 内容拷进去（`MEMORY.md` 是索引，必须一并拷贝） |
| Claude Code 记忆 | `<repo>/.claude/memory/` | Claude Code 通过 `~/.claude/projects/<slug>` 指向项目内 `.claude/`（本机该软链即指向本项目） |
| Claude 原始会话 | `<repo>/.claude/`（含 `sessions-*` 子目录） | 体积大、含密钥样式内容，按 §4 处理 |
| 项目指令 | 仓库根 `CLAUDE.md` | 覆盖前先比对，新机上可能已有新版 |
| 计划文档 | `<repo>/.zcode/plans/`、`<repo>/.claude/plans/` | — |
| 交接阅读材料 | 直接读本目录即可，无需恢复 | — |

> 迁移服务器时请与 `docs/learn docs/B4DL_迁移清单与SSH搬运手册_20260910.md` 配合使用：那边管**代码与数据**，本目录管**记忆与对话历史**。

---

## 4. 安全与 git（重要，提交前必读）

`POTENTIAL_SECRETS.txt` 是自动扫描结果：**8 个文件命中 `sk-` 样式的密钥**——7 个 Claude 原始会话记录 + `07-agent-instructions/settings.local.json`（合计 70 处命中）。归档里不复制密钥内容，只记路径与次数。

本目录自带 `.gitignore`，默认策略是「记忆本体可提交、原始会话与含密钥配置不提交」：

```
05-claude-code-sessions/
07-agent-instructions/settings.local.json
```

- 若确实要提交原始会话：先跑 `tools/build_indexes.py` 看扫描报告，把命中的 key 替换为 `REDACTED` 再提交。
- 若这些 key 仍在生产使用，建议在平台上**直接轮换**（`CLAUDE.md` 亦记录过 "git 历史中仍有旧 key，公开发布前需清理历史"）。
- 本归档**尚未 commit**，是否纳入 git 由你决定；不提交也完全可用（打包带走即可）。

---

## 5. 有意未纳入的内容

| 未纳入 | 原因 |
|---|---|
| `.claude/worktrees/`（452M） | git 工作树副本，不是记忆 |
| `~/.claude/` 下的 `file-history/`、`shell-snapshots/`、`todos/`、`tasks/`、`jobs/` | 运行时状态，非记忆 |
| `/root/.claude-lhwt`、`/root/.claude-miller`、`/root/.claude-gy` | 属于其他同学的账号目录 |
| `/root/.cursor/projects/`、`/root/.cursor-server/` | 检索后确认无本项目相关内容 |
| `/root/.codex/`（`memories_1.sqlite` 等） | 记忆库为空表（`stage1_outputs` / `thread_goals` 均 0 行），无本项目记忆 |
| ZCode 原始数据（`db.sqlite` 64M、`rollout/` 36M 模型 I/O、`agents/` 133M） | 二进制/原始 I/O；可读文本已导出到 `02-zcode-sessions/` |
| `/root/.claude/plans/lazy-chasing-torvalds.md` | 属于其它项目（`hyn/MAP_SAM`），非本项目 |
| `Claude_record/RTX5090_DEBUG_LOG.md` 之外的历史截图、`assets/` 图片 | 已在仓库中，且非记忆 |

---

## 6. 数字核对（生成时）

```
01-zcode-memory                96K   17 文件（16 条记忆 + MEMORY.md）
02-zcode-sessions              12M   66 文件（65 会话 + INDEX）
03-zcode-plans                 32K    5 文件
04-claude-code-memory         8.0K    2 文件
05-claude-code-sessions        48M   55 文件（52 会话 + 子代理记录 + INDEX）
07-agent-instructions          24K    2 文件
08-claude-records             112K   10 文件
09-history-snapshots          140K   15 文件
tools                          16K    2 文件
README.md / .gitignore /
POTENTIAL_SECRETS.txt          20K    3 文件
合计                           59M  177 文件
```

ZCode 会话时间跨度：2026-08-23 → 2026-09-10（65 个会话，全部工作目录为本项目）；Claude Code 会话时间跨度：2026-05-17 → 2026-08-13。
