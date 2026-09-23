#!/usr/bin/env python3
"""Apply the pre-registered Phase 2.2 gates to the three loss-rebalance arms.

Reads the control curve from the existing internal679 run (same initialization,
data and seed, so epochs 0..N-1 are a valid no-cost control) and each arm's
training log plus grounding report, then prints the PASS / PARTIAL / FAIL
verdict together with the guardrail checks.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

ARMS = (
    ("A1", "reasonseg-lossA1-bal-dice", "balanced + dice"),
    ("A2", "reasonseg-lossA2-plain-tversky", "plain + tversky(0.3,0.7)"),
    ("A3", "reasonseg-lossA3-bal-tversky", "balanced + tversky(0.3,0.7)"),
)

PASS_AUC = 0.93
PARTIAL_AUC = 0.88
PASS_PROB = 0.35
PARTIAL_PROB = 0.25
GUARD_AUC = 0.80
SIZE_RATIO_GUARD = 3.0
SIGNIFICANCE = 0.013


def read_epochs(path: Path) -> list[dict]:
    """Extract the per-epoch teacher-forcing rows from a training log."""
    values = []
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if "teacher_forcing_mean_iou" in record:
            values.append(record)
    return values


def read_step_stats(path: Path) -> dict:
    """Fraction of training steps whose mask_seg sits at the near-perfect level."""
    segments = []
    if not path.is_file():
        return {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if "mask_seg" in record:
            segments.append(record["mask_seg"])
    if not segments:
        return {}
    segments.sort()
    count = len(segments)
    return {
        "steps": count,
        "mask_seg_mean": statistics.fmean(segments),
        "mask_seg_p50": segments[count // 2],
        "frac_seg_below_0.05": sum(1 for v in segments if v < 0.05) / count,
    }


def read_grounding(path: Path) -> dict:
    if not path.is_file():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    overall = report.get("overall", {})
    return {
        "auc": overall.get("auc_mean"),
        "prob_positive": overall.get("mean_prob_positive"),
        "recall_thresholded": overall.get("recall_thresholded"),
        "recall_top_k": overall.get("recall_top_k"),
        "target_size_median": overall.get("target_size_median"),
        "predicted_size_median": overall.get("predicted_size_median"),
    }


def verdict(grounding: dict, mean_iou: float | None, control_iou: float) -> str:
    auc = grounding.get("auc")
    prob = grounding.get("prob_positive")
    target = grounding.get("target_size_median") or 0.0
    predicted = grounding.get("predicted_size_median") or 0.0
    if auc is None or prob is None:
        return "NO_DATA（缺接地诊断报告）"
    if auc < GUARD_AUC:
        return f"GUARDRAIL：AUC {auc:.3f} < {GUARD_AUC}，换来的是无差别触发，不采纳"
    if target > 0 and predicted > SIZE_RATIO_GUARD * target:
        return (
            f"GUARDRAIL：predicted_size_median {predicted:.0f} > "
            f"{SIZE_RATIO_GUARD:g}x target {target:.0f}，过度触发，不采纳"
        )
    if mean_iou is None:
        return "NO_DATA（缺 epoch TF 指标）"
    if auc >= PASS_AUC and prob >= PASS_PROB:
        return f"PASS：扩到 20 epoch（TF mIoU {mean_iou:.4f} vs 对照 {control_iou:.4f}）"
    partial_metric = (PARTIAL_AUC <= auc < PASS_AUC) or (PARTIAL_PROB <= prob < PASS_PROB)
    if partial_metric and mean_iou >= control_iou + SIGNIFICANCE / 2:
        return f"PARTIAL：TF mIoU {mean_iou:.4f}，仅扩最优一臂到 20 epoch"
    if auc <= PARTIAL_AUC and prob <= 0.20:
        return f"FAIL：AUC {auc:.3f} 且 prob {prob:.3f}，损失再平衡判死 → 转 Phase 2.3"
    return f"INCONCLUSIVE：auc={auc:.3f} prob={prob:.3f} mIoU={mean_iou:.4f}，需人工裁定"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--control-log",
        type=Path,
        default=Path("training_logs/chain_overnight/stage2_train_internal679.log"),
    )
    parser.add_argument("--arm-log-dir", type=Path, default=Path("training_logs/phase2"))
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    root = args.root

    control = read_epochs(root / args.control_log)
    control_curve = [row["teacher_forcing_mean_iou"] for row in control[: args.epochs]]
    control_iou = control_curve[-1] if control_curve else float("nan")
    print("=== 对照：internal679 已有曲线 epoch 0..%d ===" % (args.epochs - 1))
    print("  TF mIoU:", " ".join("%.4f" % v for v in control_curve))
    control_stats = read_step_stats(root / args.control_log)
    if control_stats:
        print("  mask_seg mean=%.4f p50=%.4f frac<0.05=%.3f (steps=%d)" % (
            control_stats["mask_seg_mean"], control_stats["mask_seg_p50"],
            control_stats["frac_seg_below_0.05"], control_stats["steps"]))

    summary = {}
    for label, name, description in ARMS:
        log = root / args.arm_log_dir / f"{name}.log"
        epochs = read_epochs(log)
        curve = [row["teacher_forcing_mean_iou"] for row in epochs[: args.epochs]]
        mean_iou = curve[-1] if curve else None
        grounding = read_grounding(root / f"eval_results/_grounding_{name}/report.json")
        stats = read_step_stats(log)
        summary[label] = {
            "name": name,
            "description": description,
            "tf_miou_curve": curve,
            "tf_miou_last": mean_iou,
            "step_stats": stats,
            "grounding": grounding,
            "verdict": verdict(grounding, mean_iou, control_iou),
        }

    print()
    print("=== 三臂结果（判据见 run_loss_rebalance_arms.sh 头部注释）===")
    for label, entry in summary.items():
        grounding = entry["grounding"]
        print("\n[%s] %s  (%s)" % (label, entry["name"], entry["description"]))
        if entry["tf_miou_curve"]:
            print("  TF mIoU :", " ".join("%.4f" % v for v in entry["tf_miou_curve"]),
                  " | 对照 %.4f" % control_iou)
        else:
            print("  TF mIoU : 无（训练未完成或日志缺失）")
        if grounding:
            print("  AUC=%.4f prob_pos=%.4f recall_thresh=%.4f recall_topK=%.4f" % (
                grounding["auc"], grounding["prob_positive"],
                grounding["recall_thresholded"], grounding["recall_top_k"]))
            print("  size median: predicted %.0f vs target %.0f" % (
                grounding["predicted_size_median"], grounding["target_size_median"]))
        else:
            print("  接地诊断：无报告")
        if entry["step_stats"]:
            print("  mask_seg mean=%.4f p50=%.4f frac<0.05=%.3f (steps=%d)" % (
                entry["step_stats"]["mask_seg_mean"], entry["step_stats"]["mask_seg_p50"],
                entry["step_stats"]["frac_seg_below_0.05"], entry["step_stats"]["steps"]))
        print("  >>> 裁定:", entry["verdict"])

    out = root / args.arm_log_dir / "phase22_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "control": {"curve": control_curve, "step_stats": control_stats},
        "arms": summary,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nsummary written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
