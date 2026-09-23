#!/usr/bin/env python3
"""Decide whether the Phase 2.2 winner may be auto-extended to 20 epochs.

Deliberately conservative: it emits GO only when the pre-registered PASS criteria
are met by exactly one arm and no guardrail trips. Anything ambiguous — a tie, a
guardrail trip alongside a mechanistic pass, or missing data — yields NOGO so a
human decides. This exists because the size guardrail can misfire at epoch 4:
"dares to fire" arrives before "has learned to stop firing", so an arm may overshoot
predicted_size_median while still being the right direction.

The "AUC < 0.80 means indiscriminate firing" guardrail is deliberately absent here:
it cannot coexist with PASS_AUC = 0.93, so it would be unreachable. It still lives
in summarize_phase22.py, where verdicts are descriptive rather than a gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PASS_AUC = 0.93
PASS_PROB = 0.35
SIZE_RATIO_GUARD = 3.0

ARM_FLAGS = {
    "reasonseg-lossA1-bal-dice": "--bce-mode balanced --region-loss dice",
    "reasonseg-lossA2-plain-tversky": (
        "--bce-mode plain --region-loss tversky --tversky-alpha 0.3 --tversky-beta 0.7"
    ),
    "reasonseg-lossA3-bal-tversky": (
        "--bce-mode balanced --region-loss tversky --tversky-alpha 0.3 --tversky-beta 0.7"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("training_logs/phase2/phase22_summary.json"),
    )
    args = parser.parse_args()

    if not args.summary.is_file():
        print(f"NOGO missing_summary {args.summary}")
        return 0
    try:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"NOGO unreadable_summary {exc}")
        return 0

    control_curve = summary.get("control", {}).get("curve") or []
    if not control_curve:
        print("NOGO no_control_curve")
        return 0
    control_iou = control_curve[-1]

    arms = summary.get("arms") or {}
    if len(arms) != 3:
        print(f"NOGO arm_count {len(arms)}")
        return 0

    passing = []
    ambiguous = []
    for label in sorted(arms):
        entry = arms[label]
        grounding = entry.get("grounding") or {}
        name = entry.get("name")
        auc = grounding.get("auc")
        prob = grounding.get("prob_positive")
        mean_iou = entry.get("tf_miou_last")
        target = grounding.get("target_size_median") or 0.0
        predicted = grounding.get("predicted_size_median") or 0.0
        if None in (auc, prob, mean_iou):
            ambiguous.append(f"{label}:incomplete_data")
            continue
        guard_size = target > 0 and predicted > SIZE_RATIO_GUARD * target
        mechanistic_pass = auc >= PASS_AUC and prob >= PASS_PROB
        beats_control = mean_iou > control_iou
        if mechanistic_pass and (guard_size or not beats_control):
            reasons = []
            if guard_size:
                reasons.append(f"size {predicted:.0f}>{SIZE_RATIO_GUARD:g}x{target:.0f}")
            if not beats_control:
                reasons.append(f"mIoU {mean_iou:.4f}<=control {control_iou:.4f}")
            ambiguous.append(f"{label}:guardrail({'|'.join(reasons)})")
        elif mechanistic_pass and beats_control:
            passing.append((label, name, auc, prob, mean_iou))

    if ambiguous:
        print("NOGO needs_human " + ";".join(ambiguous))
        return 0
    if not passing:
        print(f"NOGO no_arm_passed control_mIoU={control_iou:.4f}")
        return 0
    if len(passing) > 1:
        print("NOGO tie " + ",".join(label for label, *_ in passing))
        return 0

    label, name, auc, prob, mean_iou = passing[0]
    flags = ARM_FLAGS.get(name)
    if not flags:
        print(f"NOGO unknown_arm {name}")
        return 0
    print(
        f"GO {name} {label} auc={auc:.4f} prob={prob:.4f} "
        f"mIoU={mean_iou:.4f}>control={control_iou:.4f} flags={flags}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
