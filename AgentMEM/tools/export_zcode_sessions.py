#!/usr/bin/env python3
"""把 ZCode 的会话库（sqlite）导出为可读 markdown 归档。

数据源：/root/.zcode/cli/db/db.sqlite （只读打开，不会写入）
输出：  AgentMEM/02-zcode-sessions/<序号>_<标题>__<会话ID>.md  以及 INDEX.md

设计取舍：
  - assistant/user 的 text 部分全文保留（这是对话主干）
  - reasoning 折叠进 <details>（保留可追溯性，但不干扰阅读）
  - tool 调用只记「工具名 + 标题 + 输入/输出摘要（截断）」，避免归档被日志淹没
  - step-start / step-finish 等无信息量的分片丢弃；timeline 记为模型切换行

用法：
    python3 AgentMEM/tools/export_zcode_sessions.py [--db /path/to/db.sqlite] [--out DIR]
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_DB = "/root/.zcode/cli/db/db.sqlite"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "AgentMEM" / "02-zcode-sessions"

TOOL_IN_LIMIT = 500
TOOL_OUT_LIMIT = 800
TITLE_LIMIT = 60


def ts(ms):
    if not ms:
        return "-"
    return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")


def slug(text, limit=TITLE_LIMIT):
    s = "".join(c if c.isalnum() or c in " -_" else "" for c in (text or "未命名"))
    s = "-".join(s.split())
    return (s[:limit] or "untitled").strip("-")


def clip(text, limit):
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…（已截断，原文 {len(text)} 字符）"


def render_tool(part):
    st = part.get("state") or {}
    name = part.get("tool") or "?"
    title = st.get("title") or ""
    if title.strip().lower() == name.strip().lower():
        title = ""  # 多数工具（如 Bash）的 title 就是工具名，重复显示无意义
    lines = [f"**🔧 {name}**{(' — ' + title) if title else ''}"]
    inp = st.get("input")
    if isinstance(inp, dict) and inp:
        kv = []
        for k in ("command", "file_path", "pattern", "path", "query", "prompt", "description"):
            if k in inp:
                kv.append(f"{k}: {clip(inp[k], 200)}")
        if not kv:
            kv.append(clip(json.dumps(inp, ensure_ascii=False), TOOL_IN_LIMIT))
        lines.append("\n<details><summary>输入</summary>\n\n```\n"
                     + "\n".join(kv) + "\n```\n\n</details>")
    out = st.get("output")
    if out:
        lines.append("\n<details><summary>输出摘要</summary>\n\n```\n"
                     + clip(out, TOOL_OUT_LIMIT) + "\n```\n\n</details>")
    return "\n".join(lines)


def render_message(msg, parts, out):
    role = (msg.get("role") or "?").upper()
    model = msg.get("modelID") or (msg.get("model") or {}).get("modelID")
    head = f"### {role}" + (f"  ·  `{model}`" if model and role == "ASSISTANT" else "")
    out.append(head)
    out.append("")
    for p in parts:
        t = p.get("type")
        if t == "text":
            txt = (p.get("text") or "").strip()
            if txt:
                out.append(txt)
                out.append("")
        elif t == "reasoning":
            txt = (p.get("text") or "").strip()
            if txt:
                out.append("<details><summary>思考过程</summary>\n")
                out.append(txt)
                out.append("\n</details>\n")
        elif t == "tool":
            out.append(render_tool(p))
            out.append("")
        elif t == "file":
            out.append(f"- 📎 附件：{p.get('filename', '?')}")
            out.append("")
        elif t == "timeline":
            tm = p.get("timelineType")
            if tm == "model_change":
                f = (p.get("fromModel") or {}).get("label")
                to = (p.get("toModel") or {}).get("label")
                out.append(f"> 🔀 模型切换：{f} → {to}\n")
    out.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    sessions = list(con.execute(
        "select id, title, directory, parent_id, time_created, time_updated "
        "from session order by time_created"))

    index_rows = []
    for i, (sid, title, directory, parent_id, t0, t1) in enumerate(sessions, 1):
        msgs = list(con.execute(
            "select id, data, sequence from message where session_id=? order by sequence", (sid,)))
        parts_by_msg = {}
        for pid, mid, pdata, pseq in con.execute(
                "select id, message_id, data, sequence from part where session_id=? order by sequence", (sid,)):
            parts_by_msg.setdefault(mid, []).append(json.loads(pdata))
        todos = list(con.execute(
            "select content, status, priority from todo where session_id=? order by position", (sid,)))

        body, n_tool, n_text, models, tokens = [], 0, 0, [], 0
        for mid, mdata, _seq in msgs:
            m = json.loads(mdata)
            parts = parts_by_msg.get(mid, [])
            n_tool += sum(1 for p in parts if p.get("type") == "tool")
            n_text += sum(1 for p in parts if p.get("type") == "text")
            if m.get("modelID") and m["modelID"] not in models:
                models.append(m["modelID"])
            tk = m.get("tokens") or {}
            tokens += int(tk.get("total") or 0)
            render_message(m, parts, body)

        fname = f"{i:02d}_{slug(title)}__{sid[:20]}.md"
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            f.write(f"# {title or '（无标题会话）'}\n\n")
            f.write("| 项 | 值 |\n|---|---|\n")
            f.write(f"| 会话 ID | `{sid}` |\n")
            f.write(f"| 工作目录 | `{directory}` |\n")
            if parent_id:
                f.write(f"| 父会话 | `{parent_id}` |\n")
            f.write(f"| 时间 | {ts(t0)} → {ts(t1)} |\n")
            f.write(f"| 模型 | {', '.join(models) if models else '-'} |\n")
            f.write(f"| 消息数 | {len(msgs)}（文本块 {n_text}，工具调用 {n_tool}） |\n")
            if todos:
                f.write("\n## 会话待办\n\n")
                for c, st, pr in todos:
                    f.write(f"- [{ 'x' if st == 'completed' else ' ' }] {c}  _({st}, {pr})_\n")
            f.write("\n---\n\n")
            f.write("\n".join(body))
        index_rows.append((i, title or "（无标题）", sid, ts(t0), len(msgs), n_tool,
                           len(models), fname, (out_dir / fname).stat().st_size))

    with open(out_dir / "INDEX.md", "w", encoding="utf-8") as f:
        f.write("# ZCode 会话索引\n\n")
        f.write(f"来源：`{args.db}`（只读导出） ｜ 共 {len(index_rows)} 个会话，"
                f"全部工作目录为 `/root/autodl-tmp/wql/mmb4dl`\n\n")
        f.write("| # | 标题 | 会话 ID | 起始时间 | 消息 | 工具调用 | 模型数 | 归档 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for i, title, sid, t0, nmsg, ntool, nmodel, fname, size in index_rows:
            f.write(f"| {i} | {title} | `{sid}` | {t0} | {nmsg} | {ntool} | {nmodel} | "
                    f"[{fname}](./{fname.replace(' ', '%20')}) |\n")

    print(f"导出完成：{len(index_rows)} 个会话 → {out_dir}")
    print(f"总大小：{sum(r[8] for r in index_rows) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
