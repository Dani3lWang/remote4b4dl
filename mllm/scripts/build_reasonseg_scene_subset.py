#!/usr/bin/env python3
"""Build a ReasonSeg train manifest that spans many scenes at fixed epoch cost.

Taking every frame of every nuScenes scene multiplies epoch time linearly, so
this builder instead picks a fixed number of frames per scene.  Scene diversity
can therefore be raised (e.g. 680 scenes instead of 200) while the number of
training records stays comparable to the existing thin manifest.

Record construction is reused verbatim from ``build_reasonseg_nuscenes`` so the
query/answer text, targets and split bookkeeping stay byte-compatible.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

MLLM_ROOT = Path(__file__).resolve().parents[1]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))

from scripts.build_reasonseg_nuscenes import (  # noqa: E402
    THING_CLASSES,
    _category_mapping,
    _load_excluded_scenes,
    build_frame_records,
)


def class_of(record: dict) -> str:
    targets = record.get("targets") or []
    return targets[0].get("class_name", "unknown") if targets else "unknown"


def balanced_pick(records: list[dict], per_frame: int, counts: Counter, rng: random.Random) -> list[dict]:
    """One query per frame, spread over classes; negative frames only as fallback."""

    positive = [record for record in records if record.get("targets")]
    candidates = positive or list(records)
    chosen: list[dict] = []
    for _ in range(per_frame):
        if not candidates:
            break
        lowest = min(counts[class_of(record)] for record in candidates)
        pool = [record for record in candidates if counts[class_of(record)] == lowest]
        record = rng.choice(pool)
        candidates.remove(record)
        chosen.append(record)
        counts[class_of(record)] += 1
    if not chosen and records:
        chosen = [records[0]]
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-scene", type=int, default=11)
    parser.add_argument("--per-frame-queries", type=int, default=1)
    parser.add_argument("--exclude-scenes", type=Path)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    dataroot = args.dataroot.resolve()
    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=False)
    splits = create_splits_scenes()
    train_names = set(splits["train"])
    excluded = _load_excluded_scenes(args.exclude_scenes)
    idx_to_name = _category_mapping(nusc)
    class_to_id = {name: index for index, name in enumerate(THING_CLASSES)}
    panoptic_by_sample_data = {
        str(record.get("sample_data_token") or record.get("token")): record
        for record in nusc.panoptic
    }

    samples_by_scene: dict[str, list[dict]] = defaultdict(list)
    for sample in nusc.sample:
        samples_by_scene[sample["scene_token"]].append(sample)

    frames: list[tuple[dict, dict]] = []
    scenes_used = 0
    for scene in nusc.scene:
        name = scene["name"]
        if name not in train_names:
            continue
        if scene["token"] in excluded or name in excluded:
            continue
        samples = samples_by_scene.get(scene["token"], [])
        if not samples:
            continue
        rng = random.Random(f"{args.seed}:{scene['token']}")
        order = list(range(len(samples)))
        rng.shuffle(order)
        for index in sorted(order[: args.frames_per_scene]):
            frames.append((scene, samples[index]))
        scenes_used += 1

    counts: Counter = Counter()
    records: list[dict] = []
    frames_used = 0
    for scene, sample in frames:
        lidar_token = sample["data"]["LIDAR_TOP"]
        panoptic_record = panoptic_by_sample_data.get(lidar_token)
        if panoptic_record is None:
            raise RuntimeError(f"missing panoptic record for LIDAR_TOP {lidar_token}")
        frame_records = build_frame_records(
            sample_token=sample["token"],
            scene_token=scene["token"],
            split="train",
            lidar_path=dataroot / nusc.get("sample_data", lidar_token)["filename"],
            panoptic_path=dataroot / panoptic_record["filename"],
            dataroot=dataroot,
            idx_to_name=idx_to_name,
            class_to_id=class_to_id,
            seed=args.seed,
        )
        frame_rng = random.Random(f"{args.seed}:{sample['token']}")
        records.extend(balanced_pick(frame_records, args.per_frame_queries, counts, frame_rng))
        frames_used += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({
        "scenes": scenes_used,
        "frames": frames_used,
        "records": len(records),
        "negatives": sum(1 for record in records if not record.get("targets")),
        "classes": dict(sorted(counts.items())),
        "output": str(args.output),
    }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
