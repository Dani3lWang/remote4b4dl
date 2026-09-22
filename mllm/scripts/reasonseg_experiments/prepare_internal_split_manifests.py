#!/usr/bin/env python3
"""Prepare the manifests for a leakage-free ReasonSeg retrain.

Emits (a) the scene-token list of the internal validation split, so
``build_reasonseg_scene_subset.py --exclude-scenes`` can keep those scenes out
of training, and (b) an early-stopping subset of the internal validation set
that is round-robin balanced across scenes.  The full internal val stays the
final reporting set: validating on all 20k records every epoch would add hours.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--early-stopping-samples", type=int, default=2248)
    args = parser.parse_args()

    data = args.data_dir
    val_path = data / "reasonseg_val_internal.jsonl"
    records = [
        json.loads(line)
        for line in val_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    scenes = sorted({record["scene_token"] for record in records})
    scenes_path = data / "scenes_reasonseg_internalval.json"
    scenes_path.write_text(json.dumps(scenes), encoding="utf-8")

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_scene[record["scene_token"]].append(record)

    # Round-robin over scenes so the subset keeps the full scene coverage of the
    # internal val instead of clustering in the first scenes of the sorted file.
    ordered: list[dict] = []
    cursors = {scene: 0 for scene in scenes}
    while len(ordered) < args.early_stopping_samples:
        progressed = False
        for scene in scenes:
            bucket = by_scene[scene]
            index = cursors[scene]
            if index < len(bucket):
                ordered.append(bucket[index])
                cursors[scene] = index + 1
                progressed = True
                if len(ordered) >= args.early_stopping_samples:
                    break
        if not progressed:
            break

    subset_path = data / "reasonseg_val_internal_es.jsonl"
    with subset_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({
        "internal_val_records": len(records),
        "internal_val_scenes": len(scenes),
        "scenes_file": str(scenes_path),
        "early_stopping_records": len(ordered),
        "early_stopping_scenes": len({record["scene_token"] for record in ordered}),
        "early_stopping_file": str(subset_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
