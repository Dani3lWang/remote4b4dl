#!/usr/bin/env python3
"""为 AgentMEM 归档生成两份辅助文件：

1. AgentMEM/05-claude-code-sessions/INDEX.md
   —— Claude Code 会话记录（.jsonl）清单：文件、体积、时间、首条用户指令摘要
2. AgentMEM/POTENTIAL_SECRETS.txt
   —— 全归档的密钥模式扫描结果（只列路径与命中次数，绝不打印密钥内容）

用法：python3 AgentMEM/tools/build_indexes.py
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAUDE_DIR = ROOT / "05-claude-code-sessions"

SECRET_PATTERNS = [
    ("OpenAI 风格 key (sk-…)", re.compile(r"sk-[A-Za-z0-9_\-]{20,}")),
    ("Google API key (AIza…)", re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
    ("GitHub token (ghp_/gho_…)", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("HuggingFace token (hf_…)", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("私有块 (BEGIN … PRIVATE KEY)", re.compile(r"BEGIN [A-Z ]*PRIVATE KEY")),
    ("通用 Bearer/Token 赋值", re.compile(r"(?i)(api[_-]?key|access[_-]?token|secret)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{24,}")),
]


def first_user_text(path):
    """取 jsonl 里第一条真实用户指令，截 100 字。"""
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("type") != "user":
                    continue
                msg = d.get("message") or {}
                content = msg.get("content")
                text = ""
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text += c.get("text", "")
                text = " ".join(text.split())
                if text and not text.startswith("<") and "command-name" not in text:
                    return text[:100]
    except Exception:
        pass
    return ""


def build_claude_index():
    files = sorted(p for p in CLAUDE_DIR.rglob("*.jsonl") if p.is_file())
    rows = []
    for p in files:
        st = p.stat()
        rows.append((p.relative_to(CLAUDE_DIR), st.st_size,
                     datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                     first_user_text(p)))
    with open(CLAUDE_DIR / "INDEX.md", "w", encoding="utf-8") as f:
        f.write("# Claude Code 会话索引\n\n")
        f.write(f"共 {len(rows)} 个 `.jsonl` 会话记录（原始格式，逐行 JSON）。"
                "本索引便于定位「哪次会话聊了什么」。\n\n")
        f.write("| 文件 | 体积 | 最后修改 | 首条用户指令 |\n|---|---|---|---|\n")
        for rel, size, mt, text in rows:
            f.write(f"| `{rel}` | {size/1024:.0f} KB | {mt} | {text or '（未解析到）'} |\n")
    return len(rows), sum(r[1] for r in rows)


def build_secrets_scan():
    findings = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.name == "POTENTIAL_SECRETS.txt":
            continue
        if p.suffix.lower() in {".png", ".jpg", ".gif", ".pt", ".ckpt", ".tar", ".gz", ".bin", ".safetensors"}:
            continue
        try:
            data = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        hits = []
        for label, rx in SECRET_PATTERNS:
            n = len(rx.findall(data))
            if n:
                hits.append((label, n))
        if hits:
            findings.append((p.relative_to(ROOT), hits))

    with open(ROOT / "POTENTIAL_SECRETS.txt", "w", encoding="utf-8") as f:
        f.write("AgentMEM 敏感信息扫描报告（仅列路径与命中次数，不含密钥内容）\n")
        f.write("生成时间：" + datetime.now().strftime("%Y-%m-%d %H:%M") + "\n")
        f.write("=" * 72 + "\n\n")
        if not findings:
            f.write("未发现密钥样式的内容。\n")
        else:
            f.write(f"共 {len(findings)} 个文件命中密钥样式，**提交/推送前请先处理**：\n\n")
            for rel, hits in findings:
                f.write(f"- {rel}\n")
                for label, n in hits:
                    f.write(f"    - {label}：{n} 处\n")
            f.write("\n处置建议：\n")
            f.write("1. AgentMEM 默认不要纳入 git（见 README「安全与 git」一节）；\n")
            f.write("2. 若必须提交，先删除 05-claude-code-sessions/ 与 07-agent-instructions/settings.local.json，"
                    "或把命中的 key 替换为 REDACTED；\n")
            f.write("3. 会话记录里出现的 key 若仍在用，建议在平台上直接轮换。\n")
    return len(findings)


if __name__ == "__main__":
    n, size = build_claude_index()
    print(f"Claude 会话索引：{n} 个文件，{size/1e6:.1f} MB → 05-claude-code-sessions/INDEX.md")
    k = build_secrets_scan()
    print(f"密钥扫描：{k} 个文件命中 → POTENTIAL_SECRETS.txt")
