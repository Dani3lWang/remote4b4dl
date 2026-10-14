"""Prepare complete single-frame supervision from original Panoptic nuScenes.

All configured thing masks are retained in the model's ROI. An ambiguous
panoptic-mask/annotation correspondence rejects the WHOLE frame, so missing
targets can never silently become background set-prediction supervision.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from .config import Grounded4DConfig, THING_CLASSES
from .data import SCHEMA, load_scene_tokens
from .taxonomy import IGNORE_IDS, LEAF_TO_CLASS, decode_panoptic, policy_info, validate_category_table


REQUIRED_TABLES = ("scene", "sample", "sample_data", "sample_annotation", "instance",
                   "category", "calibrated_sensor", "sensor", "ego_pose")


def file_info(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(Path(path).resolve()), sha256=digest.hexdigest(), bytes=Path(path).stat().st_size)


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("source path escapes declared root")
    return path


def table_rows(path):
    """Use streaming JSON when available, otherwise the standard-library parser."""
    try:
        import ijson
    except ImportError:
        with Path(path).open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        yield from rows
    else:
        with Path(path).open("rb") as handle:
            yield from ijson.items(handle, "item", use_float=True)


def quaternion_matrix(values):
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("invalid quaternion")
    norm = np.linalg.norm(quaternion)
    if norm == 0 or not np.isclose(norm, 1.0, atol=1e-3):
        raise ValueError("quaternion is not normalized")
    w, x, y, z = quaternion / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def vector(values):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("invalid translation/size vector")
    return result


def box_fraction(points_global, annotation):
    center, size = vector(annotation["translation"]), vector(annotation["size"])
    if (size <= 0).any():
        raise ValueError("nonpositive annotation box size")
    rotation = quaternion_matrix(annotation["rotation"])
    local = (points_global - center) @ rotation
    half_extents = size[[1, 0, 2]] / 2
    return float((np.abs(local) <= half_extents + 1e-6).all(1).mean())


def prepare_frame(sample, lidar, annotations, instances, categories, calibration,
                  pose, panoptic, semantic_names, args, config, output):
    source_point = None
    for source_root in args.point_source_roots:
        candidate = safe_path(source_root, lidar["filename"])
        if candidate.is_file():
            source_point = candidate
            break
    if source_point is None:
        raise ValueError("missing_original_lidar_file")
    point_root = args.point_root.resolve()
    if not source_point.is_relative_to(point_root):
        raise ValueError("point_file_outside_training_point_root")
    raw = source_point.read_bytes()
    if not lidar["filename"].startswith("samples/LIDAR_TOP/"):
        raise ValueError("expected_original_keyframe_lidar_filename")
    if not raw or len(raw) % 20:
        raise ValueError("invalid_original_lidar_layout")
    points = np.frombuffer(raw, dtype="<f4").reshape(-1, 5)
    if not np.isfinite(points).all():
        raise ValueError("nonfinite_original_lidar")
    ring = points[:, 4]
    if ((ring < 0) | (ring > 31) | (ring != np.floor(ring))).any():
        raise ValueError("invalid_original_lidar_ring")
    source_mask = safe_path(args.panoptic_root, panoptic["filename"])
    mask_raw = source_mask.read_bytes()
    encoded = decode_panoptic(mask_raw, len(points))
    semantic_ids = encoded // 1000
    if not set(np.unique(semantic_ids)).issubset(semantic_names):
        raise ValueError("unknown_panoptic_semantic_index")
    bounds = np.asarray(config.point_cloud_range)
    valid = ((points[:, :3] >= bounds[:3]) & (points[:, :3] < bounds[3:])).all(1)
    if not valid.any():
        raise ValueError("no_points_in_model_roi")
    global_xyz = (points[:, :3].astype(np.float64) @ quaternion_matrix(calibration["rotation"]).T
                  + vector(calibration["translation"]))
    global_xyz = global_xyz @ quaternion_matrix(pose["rotation"]).T + vector(pose["translation"])
    candidates = defaultdict(list)
    for annotation in annotations:
        if annotation.get("num_lidar_pts", 0) <= 0:
            continue
        leaf = categories[instances[annotation["instance_token"]]["category_token"]]["name"]
        if LEAF_TO_CLASS.get(leaf) in config.class_names:
            candidates[leaf].append(annotation)
    labels, class_ids, tokens, evidence = [], [], [], []
    for panoptic_id in np.unique(encoded):
        leaf = semantic_names[int(panoptic_id) // 1000]
        class_name = LEAF_TO_CLASS.get(leaf)
        if class_name not in config.class_names:
            continue
        mask = encoded == panoptic_id
        point_count = int((mask & valid).sum())
        if point_count == 0:
            continue
        if point_count < args.min_target_points:
            raise ValueError("instance_has_insufficient_in_range_foreground_points")
        if int(panoptic_id) % 1000 == 0:
            raise ValueError("configured_thing_has_no_panoptic_instance")
        scores = [(box_fraction(global_xyz[mask], annotation), annotation)
                  for annotation in candidates[leaf]]
        scores.sort(key=lambda item: item[0], reverse=True)
        best = scores[0][0] if scores else 0.0
        second = scores[1][0] if len(scores) > 1 else 0.0
        if not scores or best < args.min_box_overlap:
            raise ValueError("panoptic_instance_has_no_supported_annotation_match")
        if best - second < args.min_match_margin:
            raise ValueError("ambiguous_panoptic_annotation_match")
        annotation = scores[0][1]
        token = annotation["instance_token"]
        if token in tokens:
            raise ValueError("multiple_panoptic_instances_match_same_annotation_instance")
        labels.append(mask.astype(np.int8))
        class_ids.append(config.class_names.index(class_name))
        tokens.append(token)
        evidence.append(dict(panoptic_id=int(panoptic_id), instance_token=token,
                             sample_annotation_token=annotation["token"],
                             inside_annotation_box_fraction=best, runner_up_fraction=second,
                             point_count=int(mask.sum())))
    if not labels:
        raise ValueError("no_configured_instance_in_frame_roi")
    label_array = np.stack(labels)
    label_array[:, np.isin(semantic_ids, IGNORE_IDS)] = -1
    point_hash = hashlib.sha256(raw).hexdigest()
    relative_mask = "masks/" + lidar["token"] + ".npz"
    mask_path = safe_path(output, relative_mask)
    part = mask_path.with_suffix(".npz.part")
    with part.open("xb") as handle:
        np.savez_compressed(handle, labels=label_array, instance_tokens=np.asarray(tokens, dtype="U32"),
                            point_indices=np.arange(len(points), dtype=np.int64),
                            point_file_sha256=np.asarray(point_hash), label_kind=np.asarray("panoptic_gt"),
                            mask_verified=np.asarray(True))
    part.replace(mask_path)
    frame = dict(sample_data_token=lidar["token"], sample_token=sample["token"],
                 timestamp_us=lidar["timestamp"], class_ids=class_ids, mask_verified=True,
                 point_file=dict(path=str(source_point.relative_to(point_root)), sha256=point_hash),
                 mask_file=dict(path=relative_mask, sha256=file_info(mask_path)["sha256"]),
                 lidar_to_anchor=np.eye(4).tolist(),
                 source_panoptic_file=dict(path=str(source_mask), sha256=hashlib.sha256(mask_raw).hexdigest()),
                 original_sample_data_filename=lidar["filename"],
                 instance_mapping_evidence=evidence)
    return frame


def run(args):
    config = Grounded4DConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8-sig")))
    if any(name not in THING_CLASSES for name in config.class_names):
        raise ValueError("preparer supports the ten nuScenes thing classes only")
    if any(value is not None and value <= 0 for value in (args.train_scenes, args.dev_scenes, args.frames_per_scene)):
        raise ValueError("scene and frame budgets must be positive")
    if not 0 < args.min_box_overlap <= 1 or not 0 < args.min_match_margin <= 1:
        raise ValueError("annotation correspondence thresholds must lie in (0,1]")
    if args.min_target_points < 1:
        raise ValueError("min-target-points must be positive")
    output = args.output_dir.resolve()
    immutable_roots = [args.metadata_root, args.panoptic_root]
    immutable_roots.extend(root / sub for root in args.point_source_roots for sub in ("samples", "sweeps"))
    for source in immutable_roots:
        if output.is_relative_to(source.resolve()):
            raise ValueError("output must be outside original source directories")
    output.mkdir(parents=True, exist_ok=False)
    (output / "masks").mkdir()
    source_info = {}

    def read(name, predicate=None):
        path = args.metadata_root / (name + ".json")
        source_info[name] = file_info(path)
        values = {}
        for row in table_rows(path):
            if predicate is None or predicate(row):
                if row["token"] in values:
                    raise ValueError(f"duplicate token in {name}")
                values[row["token"]] = row
        print(json.dumps(dict(event="metadata_loaded", table=name, retained_rows=len(values))), flush=True)
        return values

    if args.version == "v1.0-mini":
        from nuscenes.utils.splits import mini_train, mini_val
        scenes = read("scene")
        by_name = {row["name"]: token for token, row in scenes.items()}
        if set(by_name) != set(mini_train + mini_val):
            raise ValueError("mini metadata must contain exactly the ten official mini scenes")
        train_budget, dev_budget = args.train_scenes or len(mini_train), args.dev_scenes or len(mini_val)
        if train_budget > len(mini_train) or dev_budget > len(mini_val):
            raise ValueError("mini scene budgets exceed official mini split sizes")
        train_scenes = {by_name[name] for name in mini_train[:train_budget]}
        dev_scenes = {by_name[name] for name in mini_val[:dev_budget]}
    else:
        whitelist = load_scene_tokens(args.train_scene_tokens)
        scenes = read("scene", lambda row: row["token"] in whitelist)
        order = sorted(scenes)
        random.Random(args.seed).shuffle(order)
        train_budget, dev_budget = args.train_scenes or 64, args.dev_scenes or 8
        count = train_budget + dev_budget
        if len(order) < count:
            raise ValueError("insufficient official training scenes for requested budgets")
        train_scenes, dev_scenes = set(order[:train_budget]), set(order[train_budget:count])
    for name, tokens in (("train_scene_tokens", train_scenes), ("dev_scene_tokens", dev_scenes)):
        (output / (name + ".json")).write_text(json.dumps(sorted(tokens)), encoding="utf-8")
    selected_scenes = train_scenes | dev_scenes
    samples = read("sample", lambda row: row["scene_token"] in selected_scenes)
    grouped = defaultdict(list)
    for sample in samples.values():
        grouped[sample["scene_token"]].append(sample)
    selected_samples = {}
    for scene in sorted(grouped):
        rows = sorted(grouped[scene], key=lambda row: row["timestamp"])
        budget = args.frames_per_scene or (len(rows) if args.version == "v1.0-mini" else 12)
        indices = np.linspace(0, len(rows) - 1, min(len(rows), budget)).astype(int)
        selected_samples.update({rows[index]["token"]: rows[index] for index in indices})
    sensor = read("sensor")
    calibration = read("calibrated_sensor")
    lidar_calibrations = {token for token, row in calibration.items()
                          if sensor[row["sensor_token"]]["channel"] == "LIDAR_TOP"}
    sample_data = read("sample_data", lambda row: row["sample_token"] in selected_samples
                       and row["is_key_frame"] and row["calibrated_sensor_token"] in lidar_calibrations)
    data_by_sample = {}
    for row in sample_data.values():
        if row["sample_token"] in data_by_sample:
            raise ValueError("multiple keyframe LiDAR entries for sample")
        data_by_sample[row["sample_token"]] = row
    annotations = read("sample_annotation", lambda row: row["sample_token"] in selected_samples)
    anns_by_sample = defaultdict(list)
    for row in annotations.values():
        anns_by_sample[row["sample_token"]].append(row)
    instances = read("instance")
    categories = read("category")
    pose_tokens = {row["ego_pose_token"] for row in sample_data.values()}
    poses = read("ego_pose", lambda row: row["token"] in pose_tokens)
    category_path = args.panoptic_root / args.version / "category.json"
    panoptic_path = args.panoptic_root / args.version / "panoptic.json"
    semantic_names = validate_category_table(table_rows(category_path))
    panoptic = {row["sample_data_token"]: row for row in table_rows(panoptic_path)}
    source_info["panoptic_category"], source_info["panoptic_table"] = file_info(category_path), file_info(panoptic_path)
    source_info["train_scene_whitelist"] = file_info(output / "train_scene_tokens.json")
    source_info["dev_scene_whitelist"] = file_info(output / "dev_scene_tokens.json")
    if args.version == "v1.0-trainval":
        source_info["official_train_scene_whitelist"] = file_info(args.train_scene_tokens)
    counts, reasons = Counter(), Counter()
    class_counts = Counter()
    handles = {name: (output / (name + ".jsonl.part")).open("x", encoding="utf-8")
               for name in ("train", "internal_dev", "rejected")}
    try:
        for index, sample in enumerate(sorted(selected_samples.values(), key=lambda row: (row["scene_token"], row["timestamp"]))):
            counts["selected_frames"] += 1
            try:
                lidar = data_by_sample[sample["token"]]
                frame = prepare_frame(sample, lidar, anns_by_sample[sample["token"]], instances,
                                      categories, calibration[lidar["calibrated_sensor_token"]],
                                      poses[lidar["ego_pose_token"]], panoptic[lidar["token"]],
                                      semantic_names, args, config, output)
            except (ValueError, OSError, KeyError) as error:
                reason = str(error)
                counts["rejected_frames"] += 1
                reasons[reason] += 1
                handles["rejected"].write(json.dumps(dict(sample_token=sample["token"],
                                                       scene_token=sample["scene_token"], reason=reason)) + "\n")
                continue
            split = "train" if sample["scene_token"] in train_scenes else "internal_dev"
            row = dict(schema_version=SCHEMA, clip_id="panoptic-" + lidar["token"],
                       dataset_version=args.version,
                       scene_token=sample["scene_token"], split=split, label_kind="panoptic_gt",
                       annotation_scope="all_configured_instances", coordinate_frame="anchor_lidar",
                       track_verified=False, grounding_verified=False, frames=[frame])
            handles[split].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            counts[split + "_frames"] += 1
            counts[split + "_instances"] += len(frame["class_ids"])
            class_counts.update(config.class_names[class_id] for class_id in frame["class_ids"])
            if (index + 1) % 50 == 0:
                print(json.dumps(dict(event="frames_prepared", counts=dict(counts))), flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    summary = dict(schema_version=SCHEMA, dataset_version=args.version, stage="spatial", label_kind="panoptic_gt",
                   counts=dict(counts), rejection_reasons=dict(reasons), class_counts=dict(class_counts),
                   train_scene_tokens=sorted(train_scenes), internal_dev_scene_tokens=sorted(dev_scenes),
                   source_files=source_info, point_root=str(args.point_root.resolve()),
                   point_source_roots=[str(path.resolve()) for path in args.point_source_roots],
                   config=config.to_dict(), seed=args.seed, frames_per_scene=args.frames_per_scene,
                   min_box_overlap=args.min_box_overlap, min_match_margin=args.min_match_margin,
                   class_admission=policy_info(config.class_names, args.min_target_points),
                   original_point_order_preserved=True, instance_identity="conservative_panoptic_to_annotation_geometry_match",
                   qa_grounding_verified=False, temporal_identity_verified=False,
                   model_performance_accepted=False)
    if not counts["train_frames"] or not counts["internal_dev_frames"]:
        (output / "failed_preparation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise ValueError("preparation produced an empty train or internal-dev set")
    for name in handles:
        (output / (name + ".jsonl.part")).replace(output / (name + ".jsonl"))
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v1.0-mini", "v1.0-trainval"], default="v1.0-mini")
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--panoptic-root", required=True, type=Path)
    parser.add_argument("--point-root", required=True, type=Path)
    parser.add_argument("--point-source-roots", required=True, nargs="+", type=Path)
    parser.add_argument("--train-scene-tokens", type=Path,
                        default=Path(__file__).resolve().parents[2] / "datageneration/tools/assets/train_scene_tokens.json")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-scenes", type=int, help="Defaults to all 8 mini train scenes, or 64 trainval scenes")
    parser.add_argument("--dev-scenes", type=int, help="Defaults to both mini val scenes, or 8 trainval internal-dev scenes")
    parser.add_argument("--frames-per-scene", type=int, help="Defaults to all mini frames, or 12 frames per trainval scene")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-box-overlap", type=float, default=0.8)
    parser.add_argument("--min-match-margin", type=float, default=0.15)
    parser.add_argument("--min-target-points", type=int, default=1,
                        help="Reject the whole frame if any configured ROI instance has fewer foreground points")
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
