#!/usr/bin/env python3
"""Report why teacher-forcing counts so few objects.

For every manifest record it measures, per target object, how many of its
panoptic points survive the encoder's point_cloud_range gate -- that gate is
exactly what validate_teacher_forcing() calls `valid`, so an object with zero
surviving points contributes union == 0 and is silently dropped from the metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

MLLM_ROOT = Path(__file__).resolve().parents[1]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))

from vtimellm.segmentation.config import ReasonSegConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = ReasonSegConfig()
    lower = np.array(config.point_cloud_range[:3], dtype=np.float32)
    upper = np.array(config.point_cloud_range[3:], dtype=np.float32)

    per_class: dict[str, Counter] = {}
    points_kept: list[float] = []
    records = objects = empty_objects = 0
    records_without_object = 0
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if args.limit and records >= args.limit:
            break
        record = json.loads(line)
        records += 1
        points = np.fromfile(args.dataroot / record["lidar_path"], dtype=np.float32)
        points = points.reshape(-1, 5)[:, :3]
        in_range = ((points >= lower) & (points < upper)).all(axis=1)
        points_kept.append(float(in_range.mean()))
        with np.load(args.dataroot / record["panoptic_path"]) as values:
            panoptic = np.asarray(values["data"]).reshape(-1).astype(np.int64)
        targets = record.get("targets") or []
        if not targets:
            records_without_object += 1
            continue
        for target in targets:
            objects += 1
            name = target.get("class_name", "unknown")
            tally = per_class.setdefault(name, Counter())
            mask = panoptic == int(target["panoptic_id"])
            kept = int((mask & in_range).sum())
            tally["objects"] += 1
            tally["points_total"] += int(mask.sum())
            tally["points_kept"] += kept
            if kept == 0:
                empty_objects += 1
                tally["empty"] += 1

    print(json.dumps({
        "manifest": str(args.manifest),
        "records": records,
        "records_without_targets": records_without_object,
        "target_objects": objects,
        "objects_with_zero_valid_points": empty_objects,
        "dropped_rate": round(empty_objects / max(1, objects), 4),
        "mean_points_in_range": round(float(np.mean(points_kept)), 4),
        "per_class": {
            name: {
                "objects": tally["objects"],
                "empty": tally["empty"],
                "avg_points_total": round(tally["points_total"] / tally["objects"], 1),
                "avg_points_kept": round(tally["points_kept"] / tally["objects"], 1),
            }
            for name, tally in sorted(per_class.items())
        },
    }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
