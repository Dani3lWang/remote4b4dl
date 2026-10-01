#!/usr/bin/env python3
"""对 oracle 探针各臂的 test 逐物体指标做 bootstrap，给出 95% CI 与臂间配对差。

为什么必须补这一步：test 只在选定轮评**一次**，而空对照 a2 已经证明"同样信息换一种
排列"就能把 test AUC 推 0.0028 —— 点估计单独不足以支撑任何"改善/恶化"的说法。
逐物体数据来自各臂 `report.json` 的 `test.object_metrics`（evaluate() 落盘）。

配对键 = (记录序号, sample_token, 该记录内的第几个可评物体)。不可达被跳过的物体在
各臂数量不同（21–23），所以配对只在两臂键集合的交集上做。
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np

ARMS = {
    "single_20ep": "_multitask_encoder",
    "single_30ep": "_multitask_encoder_ext",
    "a2_repeat": "_temporal_a2_repeat_f3",
    "a3_naive": "_temporal_a3_naive_f3",
    "a1_compensated": "_temporal_a1_compensated_f3",
}
METRICS = ("auc", "iou", "hit", "iou_top_k", "hit_top_k")


def load_rows(root: pathlib.Path, arm: str):
    report = root / ARMS[arm] / "report.json"
    if not report.is_file():
        return None, f"缺 {report}"
    data = json.loads(report.read_text(encoding="utf-8"))
    rows = (data.get("test") or {}).get("object_metrics")
    if not rows:
        return None, f"{report.name} 里没有 test.object_metrics"
    # 记录内序号：同一 sample_token 可能有多个 query
    seen = {}
    out = {}
    for row in rows:
        key = (row["index"], row["sample_token"], seen.get(row["index"], 0))
        seen[row["index"]] = seen.get(row["index"], 0) + 1
        out[key] = row
    return out, str(report)


def boot(values: np.ndarray, reps: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(reps, values.size))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/root/autodl-tmp/mmb4dl/mllm/eval_results")
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = pathlib.Path(args.root)

    table, payload, available = {}, {}, []
    for arm in ARMS:
        rows, note = load_rows(root, arm)
        if rows is None:
            print(f"[跳过] {arm}: {note}")
            continue
        available.append(arm)
        table[arm] = rows
        print(f"[载入] {arm}: {len(rows)} 个可评物体  <- {note}")

    print("\n=== 各臂均值与 95% CI（bootstrap over objects）===")
    summary = {}
    for arm in available:
        rows = table[arm]
        per = {}
        for metric in METRICS:
            values = np.array([r[metric] for r in rows.values()], dtype=np.float64)
            mean, lo, hi = boot(values, args.reps, args.seed)
            per[metric] = {"mean": mean, "ci95": [lo, hi], "n": int(values.size)}
        sizes = np.array([r["target_size"] for r in rows.values()], dtype=np.float64)
        pred = np.array([r["predicted_size"] for r in rows.values()], dtype=np.float64)
        per["target_size_median"] = {"median": float(np.median(sizes))}
        per["predicted_size_median"] = {"median": float(np.median(pred))}
        summary[arm] = per
        print(f"{arm:<17} AUC {per['auc']['mean']:.5f} [{per['auc']['ci95'][0]:.5f}, "
              f"{per['auc']['ci95'][1]:.5f}]  IoU {per['iou']['mean']:.4f} "
              f"[{per['iou']['ci95'][0]:.4f}, {per['iou']['ci95'][1]:.4f}]  "
              f"R@0.5 {per['hit']['mean']:.4f} [{per['hit']['ci95'][0]:.4f}, "
              f"{per['hit']['ci95'][1]:.4f}]  topK_R {per['hit_top_k']['mean']:.4f}")

    print("\n=== 臂间配对差（同一批物体，键交集）===")
    pairs = [("a1_compensated", "a2_repeat"), ("a1_compensated", "a3_naive"),
             ("a1_compensated", "single_30ep"), ("a3_naive", "a2_repeat"),
             ("a2_repeat", "single_20ep"), ("a2_repeat", "single_30ep")]
    diff = {}
    for left, right in pairs:
        if left not in table or right not in table:
            continue
        keys = sorted(set(table[left]) & set(table[right]))
        a = np.array([table[left][k]["auc"] for k in keys], dtype=np.float64)
        b = np.array([table[right][k]["auc"] for k in keys], dtype=np.float64)
        d = a - b
        mean, lo, hi = boot(d, args.reps, args.seed)
        hits = int((d > 0).sum())
        verdict = "可分辨" if lo > 0 or hi < 0 else "与噪声不可分"
        key = f"{left} - {right}"
        diff[key] = {"n_paired": len(keys), "delta_auc": mean, "ci95": [lo, hi],
                     "share_left_better": hits / max(len(keys), 1), "verdict": verdict}
        print(f"{key:<34} ΔAUC {mean:+.5f} [{lo:+.5f}, {hi:+.5f}]  n={len(keys)}  "
              f"逐物体同向 {hits/len(keys):.1%}  => {verdict}")

    payload = {"reps": args.reps, "seed": args.seed, "arms": ARMS,
               "summary": summary, "paired": diff}
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
        print(f"\n落盘 -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
