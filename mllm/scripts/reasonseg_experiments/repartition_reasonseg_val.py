#!/usr/bin/env python3
"""Repartition existing official-train ReasonSeg records into leakage-free splits.

The full manifest ``reasonseg_train.jsonl`` was built from official-train scenes
only, so its records are directly reusable: instead of rebuilding records from
nuScenes, this script re-splits them with ``partition_development_scenes`` and
additionally emits a strict-unseen validation set containing only scenes that
neither the thin (tvenc) nor the 680x11 (tvenc680) training runs ever saw.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MLLM_ROOT = Path(__file__).resolve().parents[2]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))

from reasonseg_splits import partition_development_scenes  # noqa: E402


def scene_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tokens.add(json.loads(line)["scene_token"])
    return tokens


def record_pairs(path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                record = json.loads(line)
                pairs.add((record["sample_token"], record["query"]))
    return pairs


def write_records(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--data-dir", type=Path,
                        default=MLLM_ROOT / "reasonseg_data_trainval")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()

    data = args.data_dir
    train_manifest = data / "reasonseg_train.jsonl"
    thin_manifest = data / "reasonseg_train_thin.jsonl"
    wide_manifest = data / "reasonseg_train_680x11.jsonl"
    for path in (train_manifest, thin_manifest, wide_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    splits = create_splits_scenes()
    scene_token_to_split = partition_development_scenes(
        nusc.scene,
        official_train_names=splits["train"],
        excluded_scenes=set(splits["val"]),
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    thin_scenes = scene_tokens(thin_manifest)
    wide_scenes = scene_tokens(wide_manifest)
    # tvenc (thin) trained every keyframe of the 200-scene world with one query
    # each, and tvenc680 covered all 700 official-train scenes, so no frame is
    # unseen by both. The strictest shared-fairness val is record-level:
    # internal-val (sample_token, query) pairs absent from both manifests.
    trained_pairs = record_pairs(thin_manifest) | record_pairs(wide_manifest)
    records = [
        json.loads(line)
        for line in train_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    train_records = [r for r in records if scene_token_to_split.get(r["scene_token"]) == "train"]
    val_records = [r for r in records if scene_token_to_split.get(r["scene_token"]) == "val"]
    val_fresh_records = [
        r for r in val_records
        if (r["sample_token"], r["query"]) not in trained_pairs
    ]
    val_scenes = {r["scene_token"] for r in val_records}

    write_records(data / "reasonseg_train_internal.jsonl", train_records)
    write_records(data / "reasonseg_val_internal.jsonl", val_records)
    write_records(data / "reasonseg_val_internal_fresh.jsonl", val_fresh_records)
    # Scenes of the 200-scene world: feed to build_reasonseg_scene_subset.py
    # --exclude-scenes to build an OOV val from official-train scenes tvenc
    # never saw (they were still seen by tvenc680's 11-frames-per-scene run).
    (data / "scenes_reasonseg_200world.json").write_text(
        json.dumps(sorted({r["scene_token"] for r in records})), encoding="utf-8"
    )

    print(json.dumps({
        "official_train_scenes_in_manifest": len({r["scene_token"] for r in records}),
        "internal_train": {"scenes": len(scene_token_to_split) - len(val_scenes),
                           "records": len(train_records)},
        "internal_val": {"scenes": len(val_scenes), "records": len(val_records)},
        "internal_val_fresh": {"scenes": len({r["scene_token"] for r in val_fresh_records}),
                               "records": len(val_fresh_records)},
        "wide_scene_count": len(wide_scenes),
        "thin_subset_of_wide": thin_scenes <= wide_scenes,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
