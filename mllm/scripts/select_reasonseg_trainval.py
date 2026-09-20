#!/usr/bin/env python3
"""Pick a trainval scene subset and emit the blobs it needs.

Outputs (all under --output-dir):
  scenes_keep.json     scene names to use
  scenes_exclude.json  every other train/val scene name, for build --exclude-scenes
  lidar_members.txt    tar member paths of the LIDAR_TOP files to extract
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--train-scenes", type=int, default=200)
    parser.add_argument("--validation-scenes", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    splits = create_splits_scenes()
    train_names = sorted(set(splits["train"]))
    val_names = sorted(set(splits["val"]))
    keep_train = train_names[: args.train_scenes]
    keep_val = val_names[: args.validation_scenes]
    keep = keep_train + keep_val
    exclude = train_names[len(keep_train):] + val_names[len(keep_val):]

    name_to_token = {scene["name"]: scene["token"] for scene in nusc.scene}
    keep_tokens = {name_to_token[name] for name in keep if name in name_to_token}

    members = []
    bytes_needed = 0
    for sample in nusc.sample:
        if sample["scene_token"] not in keep_tokens:
            continue
        filename = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["filename"]
        members.append(filename)
        path = args.dataroot / filename
        bytes_needed += path.stat().st_size if path.exists() else 665000

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "scenes_keep.json").write_text(json.dumps(keep, indent=1), encoding="utf-8")
    (output / "scenes_exclude.json").write_text(json.dumps(exclude, indent=1), encoding="utf-8")
    (output / "lidar_members.txt").write_text("\n".join(members) + "\n", encoding="utf-8")

    print(json.dumps({
        "train_scenes": len(keep_train),
        "val_scenes": len(keep_val),
        "excluded_scenes": len(exclude),
        "lidar_files": len(members),
        "lidar_gib": round(bytes_needed / 1024**3, 2),
        "outliers_cam_note": "only LIDAR_TOP is required by the ReasonSeg manifests",
    }, indent=1))
    print(f"members -> {output / 'lidar_members.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
