#!/usr/bin/env python3
"""Build a self-contained HTML dashboard from grounding diagnostic artifacts.

Reads the per-object JSONL produced by ``diagnose_reasonseg_grounding.py`` for
several runs and emits one HTML file with the data embedded, so it opens in any
browser with no server, no Plotly and no Gradio — the reasonseg conda env on the
4090 box has neither Plotly nor spare GPU while a training run is live.

Rows are joined across runs by ``index``. That is only valid because every run
used the same manifest, the same sample limit and the same ordering; the join is
asserted rather than assumed, and a mismatch aborts instead of silently
comparing different objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

# 逐物体表里保留的字段。loc_iou/loc_hit/point_count/class_id/hit_random_k 省略：
# 前两者与 loc_containment 讲同一件事且后者更直观，hit_random_k 恒为 0（地板校验），
# point_count 与 class_id 可由 target_size 与 class_name 代替。
ROW_FIELDS = (
    "query",
    "ordinal_bucket",
    "class_name",
    "same_class_instances",
    "target_size",
    "predicted_size",
    "auc",
    "mean_prob_positive",
    "mean_prob_negative",
    "loc_containment",
    "iou_thresholded",
    "hit_thresholded",
    "iou_top_k",
    "hit_top_k",
    "hit_top_k_in_loc",
)
FLOAT_FIELDS = frozenset({
    "auc", "mean_prob_positive", "mean_prob_negative", "loc_containment",
    "iou_thresholded", "iou_top_k",
})
METRICS = (
    ("auc", "AUC"),
    ("mean_prob_positive", "prob_pos"),
    ("recall_thresholded", "recall@0.5"),
    ("recall_top_k", "recall_topK"),
    ("loc_containment_mean", "LOC 包含率"),
)


def load_run(path: Path) -> tuple[Dict[str, object], List[dict]]:
    report = json.loads((path.parent / "report.json").read_text(encoding="utf-8"))
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            source = json.loads(line)
            row = {"index": source["index"], "sample_token": source["sample_token"][:8]}
            for field in ROW_FIELDS:
                value = source[field]
                if isinstance(value, float):
                    # 概率跨度达 6 个数量级，统一 6 位有效数字足够且能把体积压下来。
                    row[field] = float(f"{value:.6g}")
                else:
                    row[field] = value
            rows.append(row)
    return report, rows


def discover(eval_root: Path, names: Sequence[str]) -> Dict[str, tuple[dict, list]]:
    runs: Dict[str, tuple[dict, list]] = {}
    for name in names:
        path = eval_root / f"_grounding_{name}" / "report.per_object.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"run '{name}' 缺少 {path}")
        runs[name] = load_run(path)
    return runs


def assert_aligned(runs: Dict[str, tuple[dict, list]]) -> None:
    reference_name = next(iter(runs))
    reference = runs[reference_name][1]
    keys = [(row["index"], row["sample_token"], row["query"]) for row in reference]
    for name, (_, rows) in runs.items():
        if name == reference_name:
            continue
        if len(rows) != len(reference):
            raise RuntimeError(
                f"run '{name}' 有 {len(rows)} 个物体，'{reference_name}' 有 "
                f"{len(reference)} 个，无法配对比较"
            )
        other = [(row["index"], row["sample_token"], row["query"]) for row in rows]
        mismatch = next(
            (i for i, (a, b) in enumerate(zip(keys, other)) if a != b), None
        )
        if mismatch is not None:
            raise RuntimeError(
                f"run '{name}' 与 '{reference_name}' 第 {mismatch} 行不是同一个物体，"
                "配对比较无效（manifest、样本上限或顺序不一致）"
            )


def bucket_values(rows: List[dict], field: str) -> List[str]:
    return sorted({str(row[field]) for row in rows})


def svg_bars(labels: Sequence[str], values: Sequence[float], *, width: int = 620,
             height: int = 190, fmt: str = "{:.3f}") -> str:
    """Inline SVG bar chart; no plotting dependency required."""
    if not values:
        return ""
    peak = max(max(values), 1e-9)
    left, bottom, top = 132, 26, 12
    plot_width = width - left - 14
    slot = (height - bottom - top) / len(values)
    bar_height = max(6.0, slot * 0.62)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="img" class="chart">'
    ]
    for position in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + plot_width * position
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" '
            'class="grid"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - bottom + 14}" class="tick" '
            f'text-anchor="middle">{fmt.format(peak * position)}</text>'
        )
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + slot * index + (slot - bar_height) / 2
        bar_width = max(1.0, plot_width * value / peak)
        parts.append(
            f'<text x="{left - 8}" y="{y + bar_height * 0.78:.1f}" class="label" '
            f'text-anchor="end">{label}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" rx="2" class="bar"/>'
        )
        parts.append(
            f'<text x="{left + bar_width + 6:.1f}" y="{y + bar_height * 0.78:.1f}" '
            f'class="value">{fmt.format(value)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--fg:#dfe3ea;--mut:#8b93a3;
--good:#4ec9a5;--bad:#e06777;--warn:#e0b352;--acc:#5b9dd9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
h1{font-size:19px;margin:0 0 4px}h2{font-size:15px;margin:26px 0 8px;color:var(--acc)}
.wrap{max-width:1400px;margin:0 auto;padding:20px 22px 60px}
.sub{color:var(--mut);font-size:12px;margin-bottom:18px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:14px 16px;margin-bottom:14px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;
white-space:nowrap}
th{color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--panel);
cursor:pointer;user-select:none}
th:first-child,td:first-child{text-align:left}
td.txt{text-align:left;white-space:normal;max-width:430px}
tr:hover td{background:#1d222b}
.pos{color:var(--good)}.neg{color:var(--bad)}.flat{color:var(--mut)}
.controls{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:end}
.ctl{display:flex;flex-direction:column;gap:3px}
.ctl label{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
select,input[type=text],input[type=number]{background:#0c0e13;color:var(--fg);
border:1px solid var(--line);border-radius:5px;padding:5px 8px;font:inherit;font-size:13px}
input[type=text]{min-width:230px}
.chk{display:flex;gap:6px;align-items:center;font-size:12px;color:var(--fg)}
.count{color:var(--mut);font-size:12px;margin:8px 0 6px}
.scroll{max-height:620px;overflow:auto;border:1px solid var(--line);border-radius:6px}
.chart{display:block}.grid{stroke:var(--line);stroke-width:1}
.tick,.label{fill:var(--mut);font-size:10px}
.label{font-size:11px}.value{fill:var(--fg);font-size:11px}
.bar{fill:var(--acc)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:980px){.grid2{grid-template-columns:1fr}}
.note{color:var(--mut);font-size:12px;margin-top:6px}
code{background:#0c0e13;padding:1px 4px;border-radius:3px}
"""

JS = """
const DATA = __DATA__;
const RUNS = DATA.runs, ROWS = DATA.rows, OVER = DATA.overall;
const $ = (id) => document.getElementById(id);

function fillSelect(el, values, chosen) {
  el.innerHTML = values.map(v =>
    `<option value="${v}"${v === chosen ? ' selected' : ''}>${v}</option>`).join('');
}

function metricCell(v, digits) {
  if (v === null || v === undefined || Number.isNaN(v)) return '<td class="flat">—</td>';
  return `<td>${v.toFixed(digits)}</td>`;
}

function deltaCell(d, digits) {
  if (d === null || d === undefined || Number.isNaN(d)) return '<td class="flat">—</td>';
  const cls = d > 1e-9 ? 'pos' : (d < -1e-9 ? 'neg' : 'flat');
  const sign = d > 0 ? '+' : '';
  return `<td class="${cls}">${sign}${d.toFixed(digits)}</td>`;
}

function renderSummary() {
  const base = $('baseline').value;
  const keys = ['auc_mean', 'mean_prob_positive', 'recall_thresholded',
                'recall_top_k', 'loc_containment_mean'];
  const heads = ['AUC', 'prob_pos', 'recall@0.5', 'recall_topK', 'LOC 包含率'];
  let html = '<table><thead><tr><th>run</th>' + heads.map(h => `<th>${h}</th>`).join('')
           + '<th>尺寸中位 pred</th><th>尺寸中位 GT</th><th>比值</th></tr></thead><tbody>';
  for (const run of RUNS) {
    const o = OVER[run] || {};
    const ratio = (o.target_size_median > 0)
      ? (o.predicted_size_median / o.target_size_median) : null;
    const ratioCls = ratio === null ? 'flat'
      : (ratio > 3 ? 'neg' : (ratio < 0.5 ? 'neg' : 'pos'));
    html += `<tr${run === base ? ' style="background:#1b2230"' : ''}><td>${run}</td>`
         + keys.map(k => metricCell(o[k], 4)).join('')
         + metricCell(o.predicted_size_median, 0)
         + metricCell(o.target_size_median, 0)
         + `<td class="${ratioCls}">${ratio === null ? '—' : ratio.toFixed(2) + '×'}</td></tr>`;
  }
  html += '</tbody></table>';
  $('summary').innerHTML = html;
  $('chartRecall').innerHTML = DATA.charts.recall;
  $('chartAuc').innerHTML = DATA.charts.auc;
}

function renderBuckets() {
  const field = $('bucketField').value;
  const keys = ['auc_mean', 'mean_prob_positive', 'recall_thresholded', 'recall_top_k'];
  const heads = ['AUC', 'prob_pos', 'recall@0.5', 'recall_topK'];
  const buckets = DATA.buckets[field];
  let html = `<table><thead><tr><th>${field}</th><th>n</th>`;
  for (const run of RUNS) html += `<th colspan="${heads.length}">${run}</th>`;
  html += '</tr><tr><th></th><th></th>';
  for (const run of RUNS) html += heads.map(h => `<th>${h}</th>`).join('');
  html += '</tr></thead><tbody>';
  for (const b of buckets) {
    const n = (OVER[RUNS[0]].by[field] || {})[b];
    html += `<tr><td>${b}</td><td class="flat">${n === undefined ? '' : n.n}</td>`;
    for (const run of RUNS) {
      const g = ((OVER[run].by || {})[field] || {})[b] || {};
      html += keys.map(k => metricCell(g[k], 4)).join('');
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  $('buckets').innerHTML = html;
}

let sortKey = 'hit_thresholded', sortAsc = true;

function selectedRows() {
  const base = $('baseline').value, cmp = $('compare').value;
  const bucket = $('fBucket').value, cls = $('fClass').value;
  const text = $('fText').value.trim().toLowerCase();
  const modes = new Set(
    [...document.querySelectorAll('input[name=mode]:checked')].map(e => e.value));
  const baseRows = ROWS[base], cmpRows = ROWS[cmp];
  const out = [];
  for (let i = 0; i < baseRows.length; i++) {
    const b = baseRows[i], c = cmpRows[i];
    if (bucket !== '__all' && b.ordinal_bucket !== bucket) continue;
    if (cls !== '__all' && b.class_name !== cls) continue;
    if (text && !b.query.toLowerCase().includes(text)
             && !b.sample_token.includes(text)) continue;
    const empty = c.predicted_size === 0;
    const over = c.target_size > 0 && c.predicted_size > 3 * c.target_size;
    const hitNow = c.hit_thresholded > 0.5, hitBefore = b.hit_thresholded > 0.5;
    const fixed = hitNow && !hitBefore, broke = !hitNow && hitBefore;
    const both = (c.iou_thresholded > 0 && c.iou_thresholded < 0.5);
    if (modes.size) {
      const ok = (modes.has('empty') && empty) || (modes.has('over') && over)
              || (modes.has('fixed') && fixed) || (modes.has('broke') && broke)
              || (modes.has('partial') && both);
      if (!ok) continue;
    }
    out.push({ b, c, i, dIou: c.iou_thresholded - b.iou_thresholded,
               dAuc: c.auc - b.auc, empty, over, fixed, broke });
  }
  return out;
}

function renderRows() {
  const base = $('baseline').value, cmp = $('compare').value;
  const rows = selectedRows();
  const key = sortKey, asc = sortAsc;
  rows.sort((x, y) => {
    let a, b;
    if (key === 'dIou') { a = x.dIou; b = y.dIou; }
    else if (key === 'dAuc') { a = x.dAuc; b = y.dAuc; }
    else { a = x.c[key]; b = y.c[key]; }
    if (typeof a === 'string') return asc ? a.localeCompare(b) : b.localeCompare(a);
    return asc ? a - b : b - a;
  });
  const limit = Math.min(rows.length, Number($('limit').value) || 400);
  const cols = [
    ['#', r => r.i], ['query', r => r.c.query, 'txt'],
    ['桶', r => r.c.ordinal_bucket], ['类别', r => r.c.class_name],
    ['同类数', r => r.c.same_class_instances],
    ['GT点数', r => r.c.target_size],
    [`${cmp} 预测点数`, r => r.c.predicted_size],
    ['尺寸比', r => r.c.target_size ? (r.c.predicted_size / r.c.target_size).toFixed(2) : '—'],
    [`${cmp} IoU`, r => r.c.iou_thresholded.toFixed(3)],
    [`${base} IoU`, r => r.b.iou_thresholded.toFixed(3)],
    ['ΔIoU', r => r.dIou.toFixed(3), 'num dIou'],
    [`${cmp} AUC`, r => r.c.auc.toFixed(3)],
    ['ΔAUC', r => r.dAuc.toFixed(3), 'num dAuc'],
    ['prob_pos', r => r.c.mean_prob_positive.toExponential(2)],
    ['LOC包含', r => r.c.loc_containment.toFixed(2)],
    ['标记', r => [r.empty ? '空掩码' : '', r.over ? '过触发' : '',
                   r.fixed ? '修好' : '', r.broke ? '弄坏' : '']
                   .filter(Boolean).join(' ') || '—'],
  ];
  let html = '<table><thead><tr>' + cols.map((c, idx) => {
    const sk = c[2] && c[2].includes('dIou') ? 'dIou'
             : (c[2] && c[2].includes('dAuc') ? 'dAuc' : null);
    const sortable = sk || ['target_size', 'predicted_size', 'iou_thresholded',
                            'auc', 'mean_prob_positive', 'same_class_instances']
                           .includes(c[0].split(' ').pop());
    return `<th data-sort="${sk || ''}">${c[0]}${sortKey === sk ? (asc ? ' ▲' : ' ▼') : ''}</th>`;
  }).join('') + '</tr></thead><tbody>';
  for (const r of rows.slice(0, limit)) {
    html += '<tr>' + cols.map(c => {
      const v = c[1](r);
      let cls = c[2] || '';
      if (c[0] === 'ΔIoU' || c[0] === 'ΔAUC') {
        const n = parseFloat(v);
        cls = n > 1e-9 ? 'pos' : (n < -1e-9 ? 'neg' : 'flat');
      }
      if (c[0] === '标记' && v !== '—') cls = 'neg';
      return `<td class="${cls.includes('txt') ? 'txt' : ''} ${cls.includes('pos') ? 'pos'
               : cls.includes('neg') ? 'neg' : cls.includes('flat') ? 'flat' : ''}">${v}</td>`;
    }).join('') + '</tr>';
  }
  html += '</tbody></table>';
  $('rows').innerHTML = html;
  $('rowCount').textContent =
    `显示 ${limit} / 命中 ${rows.length} / 共 ${ROWS[base].length} 个物体`
    + (rows.length > limit ? '（调大上限或收紧筛选）' : '');
  document.querySelectorAll('#rows th').forEach(th => {
    th.onclick = () => {
      const k = th.dataset.sort;
      if (!k) return;
      if (sortKey === k) sortAsc = !sortAsc; else { sortKey = k; sortAsc = false; }
      renderRows();
    };
  });
}

function refresh() { renderSummary(); renderBuckets(); renderRows(); }

fillSelect($('baseline'), RUNS, DATA.default_baseline);
fillSelect($('compare'), RUNS, DATA.default_compare);
fillSelect($('bucketField'), Object.keys(DATA.buckets), 'ordinal_bucket');
for (const [id, values] of [['fBucket', DATA.ordinal_buckets], ['fClass', DATA.classes]]) {
  $(id).innerHTML = '<option value="__all">全部</option>'
    + values.map(v => `<option value="${v}">${v}</option>`).join('');
}
for (const id of ['baseline', 'compare', 'bucketField', 'fBucket', 'fClass', 'fText', 'limit']) {
  $(id).addEventListener('input', refresh);
}
document.querySelectorAll('input[name=mode]').forEach(e => e.addEventListener('change', refresh));
refresh();
"""


def build_payload(runs: Dict[str, tuple[dict, list]], baseline: str, compare: str,
                  charts: Dict[str, str]) -> dict:
    names = list(runs)
    overall = {}
    for name, (report, _) in runs.items():
        entry = dict(report.get("overall", {}))
        entry["by"] = {
            "ordinal_bucket": report.get("by_ordinal_bucket", {}),
            "same_class_instances": report.get("by_same_class_instances", {}),
            "class_name": report.get("by_class", {}),
        }
        overall[name] = entry
    reference = runs[names[0]][1]
    return {
        "runs": names,
        "overall": overall,
        "rows": {name: runs[name][1] for name in names},
        "buckets": {
            "ordinal_bucket": bucket_values(reference, "ordinal_bucket"),
            "same_class_instances": bucket_values(reference, "same_class_instances"),
            "class_name": bucket_values(reference, "class_name"),
        },
        "ordinal_buckets": bucket_values(reference, "ordinal_bucket"),
        "classes": bucket_values(reference, "class_name"),
        "default_baseline": baseline,
        "default_compare": compare,
        "charts": charts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("eval_results"))
    parser.add_argument(
        "--runs",
        default="internal679,p21_prioroff,reasonseg-lossA1-bal-dice,"
                "reasonseg-lossA2-plain-tversky,reasonseg-lossA3-bal-tversky",
        help="逗号分隔的 run 名，对应 eval_results/_grounding_<name>/",
    )
    parser.add_argument("--baseline", default=None, help="配对比较的基准 run，默认第一个")
    parser.add_argument("--compare", default=None, help="配对比较的对照 run，默认最后一个")
    parser.add_argument("--output", type=Path, default=Path("grounding_dashboard.html"))
    args = parser.parse_args()

    names = [name.strip() for name in args.runs.split(",") if name.strip()]
    if len(names) < 2:
        raise SystemExit("至少需要两个 run 才能做配对比较")
    runs = discover(args.eval_root, names)
    assert_aligned(runs)

    baseline = args.baseline or names[0]
    compare = args.compare or names[-1]
    for chosen in (baseline, compare):
        if chosen not in runs:
            raise SystemExit(f"--baseline/--compare 指定的 '{chosen}' 不在 --runs 里")

    charts = {
        "recall": svg_bars(names, [runs[n][0]["overall"]["recall_thresholded"] for n in names]),
        "auc": svg_bars(names, [runs[n][0]["overall"]["auc_mean"] for n in names]),
    }
    payload = build_payload(runs, baseline, compare, charts)
    html = (
        "<!doctype html><html lang=zh><meta charset=utf-8>"
        "<title>ReasonSeg 接地诊断仪表板</title>"
        f"<style>{CSS}</style><body><div class=wrap>"
        "<h1>ReasonSeg 接地诊断仪表板</h1>"
        "<div class=sub>逐物体配对比较 · 数据来自 diagnose_reasonseg_grounding.py "
        "的 report.per_object.jsonl · 本文件自包含，可离线打开</div>"
        "<div class=panel><div class=controls>"
        "<div class=ctl><label>基准 run</label><select id=baseline></select></div>"
        "<div class=ctl><label>对比 run</label><select id=compare></select></div>"
        "</div><div class=note>“尺寸比”= 预测点数 / GT 点数。&gt;3× 记过触发、"
        "&lt;0.5× 记欠触发，两者都标红——Phase 2.2 里 balanced 臂正是 136× 过触发。</div></div>"
        "<h2>总体指标</h2><div class=panel id=summary></div>"
        "<div class=grid2><div class=panel><h2 style='margin-top:0'>recall@0.5（阈值 0.5）</h2>"
        "<div id=chartRecall></div></div>"
        "<div class=panel><h2 style='margin-top:0'>掩码 logit 秩 AUC</h2>"
        "<div id=chartAuc></div></div></div>"
        "<h2>分桶下钻</h2><div class=panel><div class=controls>"
        "<div class=ctl><label>分桶维度</label><select id=bucketField></select></div>"
        "</div><div class=scroll style='max-height:340px'><div id=buckets></div></div>"
        "<div class=note>序数桶看“第 n 近”的指代难度，同类实例桶看拥挤度，类别看尺寸偏置。</div></div>"
        "<h2>逐物体浏览器</h2><div class=panel><div class=controls>"
        "<div class=ctl><label>序数桶</label><select id=fBucket></select></div>"
        "<div class=ctl><label>类别</label><select id=fClass></select></div>"
        "<div class=ctl><label>query / token 搜索</label>"
        "<input type=text id=fText placeholder='如 17th-nearest 或 sample_token 前缀'></div>"
        "<div class=ctl><label>显示上限</label>"
        "<input type=number id=limit value=400 min=20 max=2000 step=20></div>"
        "<div class=ctl><label>失败模式</label><div style='display:flex;gap:12px;flex-wrap:wrap'>"
        "<label class=chk><input type=checkbox name=mode value=empty>空掩码</label>"
        "<label class=chk><input type=checkbox name=mode value=over>过触发 &gt;3×</label>"
        "<label class=chk><input type=checkbox name=mode value=fixed>相对基准修好</label>"
        "<label class=chk><input type=checkbox name=mode value=broke>相对基准弄坏</label>"
        "<label class=chk><input type=checkbox name=mode value=partial>部分重叠 IoU∈(0,0.5)</label>"
        "</div></div></div><div class=count id=rowCount></div>"
        "<div class=scroll id=rows></div>"
        "<div class=note>点 ΔIoU / ΔAUC 表头可排序（默认降序）。“修好/弄坏”是相对基准 run "
        "在 IoU≥0.5 上的翻转，用来快速定位一次损失改动到底动了哪些样本。</div></div>"
        "</div><script>"
        + JS.replace("__DATA__", json.dumps(payload, ensure_ascii=False,
                                            separators=(",", ":")))
        + "</script></body></html>"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    size_mb = args.output.stat().st_size / 1e6
    print(f"runs: {', '.join(names)}")
    print(f"baseline={baseline}  compare={compare}")
    print(f"objects per run: {len(runs[names[0]][1])}")
    print(f"written: {args.output}  ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
