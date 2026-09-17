#!/usr/bin/env python3
"""Build deterministic, auditable nuScenes panoptic reasoning manifests.

This script never feeds labels, boxes or answer-derived frame ranges to the
model.  It only converts panoptic ground truth into targets and text supervision
for train/validation manifests.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


THING_CLASSES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-scenes", type=Path)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    dataroot = args.dataroot.resolve()
    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=False)
    if not hasattr(nusc, "panoptic") or not nusc.panoptic:
        raise RuntimeError(
            "nuScenes panoptic metadata is unavailable. Install/extract the panoptic "
            "expansion before building instance masks. lidarseg alone is insufficient."
        )
    idx_to_name = _category_mapping(nusc)
    class_to_id = {name: index for index, name in enumerate(THING_CLASSES)}
    splits = create_splits_scenes()
    scene_name_to_split = {
        name: split
        for split in ("train", "val")
        for name in splits[split]
    }
    excluded = _load_excluded_scenes(args.exclude_scenes)
    panoptic_by_sample_data = {
        str(record.get("sample_data_token") or record.get("token")): record
        for record in nusc.panoptic
    }
    scene_by_token = {scene["token"]: scene for scene in nusc.scene}
    records_by_split: Dict[str, List[dict]] = defaultdict(list)
    processed = 0
    for sample in nusc.sample:
        scene = scene_by_token[sample["scene_token"]]
        split = scene_name_to_split.get(scene["name"])
        if split not in ("train", "val") or scene["token"] in excluded or scene["name"] in excluded:
            continue
        lidar_token = sample["data"]["LIDAR_TOP"]
        panoptic_record = panoptic_by_sample_data.get(lidar_token)
        if panoptic_record is None:
            raise RuntimeError(f"missing panoptic record for LIDAR_TOP {lidar_token}")
        lidar_record = nusc.get("sample_data", lidar_token)
        lidar_path = dataroot / lidar_record["filename"]
        panoptic_path = dataroot / panoptic_record["filename"]
        frame_records = build_frame_records(
            sample_token=sample["token"],
            scene_token=scene["token"],
            split=split,
            lidar_path=lidar_path,
            panoptic_path=panoptic_path,
            dataroot=dataroot,
            idx_to_name=idx_to_name,
            class_to_id=class_to_id,
            seed=args.seed,
        )
        records_by_split[split].extend(frame_records)
        processed += 1
        if args.max_frames and processed >= args.max_frames:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        destination = args.output_dir / f"reasonseg_{split}.jsonl"
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records_by_split[split]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{split}: {len(records_by_split[split]):,} records -> {destination}")
    return 0


def build_frame_records(
    *,
    sample_token: str,
    scene_token: str,
    split: str,
    lidar_path: Path,
    panoptic_path: Path,
    dataroot: Path,
    idx_to_name: Dict[int, str],
    class_to_id: Dict[str, int],
    seed: int,
) -> List[dict]:
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :4]
    with np.load(panoptic_path) as values:
        panoptic = np.asarray(values["data"]).reshape(-1)
    if len(points) != len(panoptic):
        raise RuntimeError(f"point/panoptic mismatch for {sample_token}")
    instances = []
    for panoptic_id in np.unique(panoptic):
        semantic_id = int(panoptic_id) // 1000
        instance_id = int(panoptic_id) % 1000
        if instance_id == 0:
            continue
        raw_name = idx_to_name.get(semantic_id)
        class_name = normalize_category(raw_name or "")
        if class_name not in class_to_id:
            continue
        mask = panoptic == panoptic_id
        if mask.sum() < 3:
            continue
        center = points[mask, :3].mean(axis=0)
        instances.append(
            {
                "panoptic_id": int(panoptic_id),
                "class_id": class_to_id[class_name],
                "class_name": class_name,
                "center": center,
            }
        )
    if not instances:
        return []
    rng = random.Random(f"{seed}:{sample_token}")
    records: List[dict] = []
    by_class: Dict[str, List[dict]] = defaultdict(list)
    descriptor_by_panoptic_id: Dict[int, str] = {}
    for instance in instances:
        by_class[instance["class_name"]].append(instance)
    for class_instances in by_class.values():
        class_instances.sort(key=lambda value: float(np.linalg.norm(value["center"][:2])))
        descriptors = describe_same_class_instances(class_instances)
        for instance, descriptor in zip(class_instances, descriptors):
            descriptor_by_panoptic_id[instance["panoptic_id"]] = descriptor
            query = f"Segment the {descriptor} in the current LiDAR frame."
            records.append(_record(sample_token, scene_token, split, lidar_path, panoptic_path,
                                   dataroot, query, [instance], [descriptor]))
    if len(instances) >= 2:
        ordered = sorted(instances, key=lambda value: (value["class_name"], value["panoptic_id"]))
        first, second = rng.sample(ordered, 2)
        relation = relative_relation(first["center"], second["center"])
        first_description = descriptor_by_panoptic_id[first["panoptic_id"]]
        second_description = descriptor_by_panoptic_id[second["panoptic_id"]]
        query = (
            f"Segment the {first_description} and the {second_description}, which is "
            f"located {relation} the first object, in the current LiDAR frame."
        )
        records.append(_record(sample_token, scene_token, split, lidar_path, panoptic_path,
                               dataroot, query, [first, second],
                               [first_description, second_description]))
    absent = next((name for name in THING_CLASSES if name not in by_class), None)
    if absent is not None:
        records.append(
            _record(
                sample_token,
                scene_token,
                split,
                lidar_path,
                panoptic_path,
                dataroot,
                f"Segment every {absent} in the current LiDAR frame.",
                [],
                [],
            )
        )
    return records


def describe_same_class_instances(instances: Sequence[dict]) -> List[str]:
    if len(instances) == 1:
        return [instances[0]["class_name"]]
    descriptions = []
    for index, instance in enumerate(instances):
        center = instance["center"]
        distance = (
            "nearest"
            if index == 0
            else "farthest"
            if index == len(instances) - 1
            else f"{_ordinal(index + 1)}-nearest"
        )
        side = "left-side" if center[1] > 0 else "right-side"
        descriptions.append(f"{distance} {side} {instance['class_name']}")
    return descriptions


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def relative_relation(first: np.ndarray, second: np.ndarray) -> str:
    delta = second - first
    if abs(delta[1]) >= abs(delta[0]):
        return "to the left of" if delta[1] > 0 else "to the right of"
    return "in front of" if delta[0] > 0 else "behind"


def normalize_category(name: str) -> str:
    value = name.lower()
    if value.startswith("human.pedestrian"):
        return "pedestrian"
    if value.startswith("movable_object.barrier"):
        return "barrier"
    if value.startswith("movable_object.trafficcone"):
        return "traffic_cone"
    if value.startswith("vehicle.bus"):
        return "bus"
    vehicle_aliases = {
        "vehicle.bicycle": "bicycle",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }
    return vehicle_aliases.get(value, value)


def _record(
    sample_token: str,
    scene_token: str,
    split: str,
    lidar_path: Path,
    panoptic_path: Path,
    dataroot: Path,
    query: str,
    targets: Sequence[dict],
    descriptions: Sequence[str],
) -> dict:
    target_values = [
        {key: value for key, value in target.items() if key != "center"}
        for target in targets
    ]
    if targets:
        if len(descriptions) != len(targets):
            raise ValueError("each target must have one unambiguous description")
        answer = "I found " + ", and ".join(
            f"the {description} <LOC> <SEG>" for description in descriptions
        ) + "."
    else:
        answer = "No matching object is present <NOOBJ>."
    return {
        "sample_token": sample_token,
        "scene_token": scene_token,
        "split": split,
        "lidar_path": str(lidar_path.relative_to(dataroot)).replace("\\", "/"),
        "panoptic_path": str(panoptic_path.relative_to(dataroot)).replace("\\", "/"),
        "query": query,
        "answer": answer,
        "targets": target_values,
    }


def _category_mapping(nusc) -> Dict[int, str]:
    mapping = getattr(nusc, "lidarseg_idx2name_mapping", None)
    if mapping:
        return {int(key): str(value) for key, value in mapping.items()}
    mapping_path = Path(nusc.dataroot) / nusc.version / "lidarseg.json"
    if mapping_path.is_file():
        values = json.loads(mapping_path.read_text(encoding="utf-8"))
        if isinstance(values, dict) and "index" in values:
            return {int(key): str(value) for key, value in values["index"].items()}
    raise RuntimeError("cannot determine nuScenes lidarseg category-index mapping")


def _load_excluded_scenes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    values = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(values, dict):
        values = values.get("scene_tokens") or values.get("scenes") or []
    return {str(value) for value in values}


if __name__ == "__main__":
    raise SystemExit(main())
