#!/usr/bin/env python3
"""Recompute ReasonSeg free-generation metrics from saved npz masks.

`metrics.json` inside an eval directory was written by whatever metrics.py was
checked out at eval time.  The accumulator changed on 2026-09-22 (below-IoU-0.5
pairs used to be counted twice as FP/FN), so stored recall/precision can be
stale.  This replays the saved masks through the current accumulator, offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MLLM_ROOT = Path(__file__).resolve().parents[2]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))

from vtimellm.segmentation.metrics import SegmentationMetricAccumulator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    predictions = json.loads((args.eval_dir / "predictions.json").read_text(encoding="utf-8"))
    accumulator = SegmentationMetricAccumulator(args.num_classes)
    for entry in predictions:
        with np.load(args.eval_dir / entry["mask_file"]) as npz:
            accumulator.update(
                np.asarray(npz["predicted_masks"], dtype=bool),
                np.asarray(npz["predicted_classes"], dtype=np.int64),
                np.asarray(npz["target_masks"], dtype=bool),
                np.asarray(npz["target_classes"], dtype=np.int64),
                token_status=entry.get("status", "ok"),
            )
    metrics = accumulator.compute()
    metrics["recomputed_from"] = str(args.eval_dir)
    metrics["samples"] = len(predictions)
    stored_path = args.eval_dir / "metrics.json"
    if stored_path.is_file():
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
        metrics["stored_values"] = {
            key: stored.get(key)
            for key in ("cIoU", "gIoU", "instance_precision@0.5", "instance_recall@0.5")
        }
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
