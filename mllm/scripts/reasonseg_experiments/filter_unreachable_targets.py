#!/usr/bin/env python3
"""Drop ReasonSeg records containing a target that the encoder cannot see.

`SpatialEncoder._voxelize` marks a point valid only when it lies inside
`point_cloud_range`, and the teacher-forcing denominator intersects GT with that
mask. A target whose points are all outside the range can therefore never be
matched: it is either silently dropped from the metric or, worse, counted with a
guaranteed-zero IoU. On val_thin that is 143 of 2440 targets (5.86%), mostly
distant cars (71) and trucks (53) with a median of 5 points.

Filtering granularity is the RECORD, not the target: a query names one specific
object and the answer carries one `<SEG>` per target, so removing a single target
from a multi-target record would desynchronize query, answer and targets. Records
where every target is reachable are kept byte-for-byte.

Partially out-of-range targets are NOT dropped — they are still reachable, though
their GT mask is truncated. They are reported separately so the residual
distortion stays visible instead of being quietly folded in.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--point-cloud-range",
        type=float,
        nargs=6,
        default=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        help="必须与 ReasonSegConfig.point_cloud_range 一致，否则过滤判据与编码器不符",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    lower = np.array(args.point_cloud_range[:3], dtype=np.float32)
    upper = np.array(args.point_cloud_range[3:], dtype=np.float32)
    root = Path(args.dataroot)
    manifest = Path(args.manifest)

    kept_lines: list[str] = []
    dropped_records = 0
    dropped_targets = 0
    dropped_by_class: Counter = Counter()
    partial_targets = 0
    total_targets = 0
    # 同一帧会被多条 query 复用，但各记录的 targets 列表不同，所以缓存必须按
    # panoptic_id 存"框内点数 / 总点数"这种与记录无关的量，不能存列表下标。
    frame_cache: dict[str, tuple[dict, dict]] = {}

    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            token = record["sample_token"]
            targets = record.get("targets", [])
            total_targets += len(targets)

            if token not in frame_cache:
                points = np.fromfile(root / record["lidar_path"], dtype=np.float32)
                points = points.reshape(-1, 5)[:, :3]
                with np.load(root / record["panoptic_path"]) as values:
                    panoptic = np.asarray(values["data"]).reshape(-1)
                if panoptic.shape[0] != points.shape[0]:
                    raise RuntimeError(
                        f"point/panoptic length mismatch for {token}: "
                        f"{points.shape[0]} != {panoptic.shape[0]}"
                    )
                in_range = ((points >= lower) & (points < upper)).all(axis=1)
                all_ids, all_counts = np.unique(panoptic, return_counts=True)
                in_ids, in_counts = np.unique(
                    panoptic[in_range], return_counts=True
                )
                frame_cache[token] = (
                    dict(zip(all_ids.tolist(), all_counts.tolist())),
                    dict(zip(in_ids.tolist(), in_counts.tolist())),
                )
            totals_map, inside_map = frame_cache[token]

            unreachable, partial = [], 0
            for index, target in enumerate(targets):
                panoptic_id = target["panoptic_id"]
                size = totals_map.get(panoptic_id, 0)
                if size == 0:
                    raise RuntimeError(f"target {panoptic_id} absent in {token}")
                inside = inside_map.get(panoptic_id, 0)
                if inside == 0:
                    unreachable.append(index)
                elif inside < size:
                    partial += 1
            partial_targets += partial

            if unreachable:
                dropped_records += 1
                dropped_targets += len(unreachable)
                for index in unreachable:
                    dropped_by_class[targets[index]["class_name"]] += 1
                continue
            kept_lines.append(line.rstrip("\n"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")

    summary = {
        "manifest": str(manifest),
        "output": str(args.output),
        "point_cloud_range": args.point_cloud_range,
        "records_in": dropped_records + len(kept_lines),
        "records_kept": len(kept_lines),
        "records_dropped": dropped_records,
        "targets_total": total_targets,
        "targets_unreachable_dropped": dropped_targets,
        "targets_partially_out_of_range_kept": partial_targets,
        "dropped_by_class": dict(dropped_by_class.most_common()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
