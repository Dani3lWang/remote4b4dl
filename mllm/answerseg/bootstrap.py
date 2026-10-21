"""Explicit geometric supervision, separate from reviewed natural-language bindings.

No B3 answers, candidate queries or model predictions enter label construction.
Every label can be recomputed from the audited native panoptic instance masks.
"""

import math

from . import SCHEMA

BOOTSTRAP_SCHEMA = SCHEMA + "-geometry-bootstrap"
RULE_VERSION = "same-class-xy-centroid-extrema-v1"


def geometry_records(frame, clip, features, config, *, min_points=20, minimum_gap=0.5):
    from .data import digest

    if type(min_points) is not int or min_points < 2:
        raise ValueError("geometry bootstrap requires at least two foreground points")
    if not math.isfinite(minimum_gap) or minimum_gap <= 0:
        raise ValueError("minimum_gap must be positive and finite")
    source = clip["frames"][0]
    if (
        clip.get("label_kind") != "panoptic_gt"
        or source.get("mask_verified") is not True
    ):
        raise ValueError("geometry bootstrap requires verified panoptic GT")
    points, labels = frame["points"].cpu().double(), frame["labels"].cpu()
    bounds = points.new_tensor(config.point_cloud_range)
    valid = ((points[:, :3] >= bounds[:3]) & (points[:, :3] < bounds[3:])).all(1)
    groups = {}
    for index, category in enumerate(frame["class_ids"].tolist()):
        foreground = labels[index].eq(1) & valid
        count = int(foreground.sum())
        if count >= min_points:
            centroid = points[foreground, :2].mean(0)
            groups.setdefault(category, []).append(
                (float(centroid.norm()), index, count)
            )
    rows = []
    for category, objects in sorted(groups.items()):
        # A single category candidate is NEVER sufficient for this supervision.
        if len(objects) < 2:
            continue
        name = config.class_names[category].replace("_", " ")
        for relation in ("nearest", "farthest"):
            ordered = sorted(objects, reverse=relation == "farthest")
            distance, index, count = ordered[0]
            gap = abs(distance - ordered[1][0])
            if gap < minimum_gap:
                continue
            question = (
                f"Consider visible objects of class {name} with at least {min_points} "
                f"points inside the LiDAR XYZ range {list(config.point_cloud_range)}. "
                f"Segment the {relation} one, measuring distance from the sensor "
                "to the mean XY position of its points inside that range."
            )
            answer = f"The {relation} {name}."
            row = {
                "schema_version": BOOTSTRAP_SCHEMA,
                "request_id": f"geometry:{frame['sample_data_token']}:{category}:{relation}",
                "scene_token": clip["scene_token"],
                "sample_data_token": frame["sample_data_token"],
                "split": clip["split"],
                "question": question,
                "answer": answer,
                "features": features,
                "targets": [
                    {
                        "span": [4, len(answer) - 1],
                        "status": "matched",
                        "instance_token": frame["instance_tokens"][index],
                    }
                ],
                "answer_source": "panoptic_geometry_template",
                "supervision_kind": "geometry_bootstrap",
                "review_status": "not_human_reviewed",
                "grounding_verified": False,
                "spatial_reference": {
                    key: source[key]
                    for key in ("point_file", "mask_file", "source_panoptic_file")
                },
                "geometry": {
                    "rule": RULE_VERSION,
                    "relation": relation,
                    "class_id": category,
                    "min_points": min_points,
                    "minimum_gap_m": minimum_gap,
                    "point_cloud_range": list(config.point_cloud_range),
                    "eligible_same_class_instances": len(objects),
                    "target_points": count,
                    "target_centroid_distance_m": distance,
                    "runner_up_gap_m": gap,
                },
            }
            row["source_fingerprint"] = digest(row)
            rows.append(row)
    return rows


def verify_geometry_record(row, frame, clip, config):
    """Recompute the rule and the complete record, not just a user-written hash."""
    params = row.get("geometry", {})
    candidates = geometry_records(
        frame,
        clip,
        row["features"],
        config,
        min_points=params.get("min_points"),
        minimum_gap=params.get("minimum_gap_m", float("nan")),
    )
    if row not in candidates:
        raise ValueError("geometry supervision does not match audited panoptic rule")


def bootstrap(args, spec):
    import json
    from collections import Counter

    from .data import AnswerSegDataset, load_features, read_jsonl, verify_feature_frames
    from .runner import make_spatial_dataset

    spatial, policy = make_spatial_dataset(
        spec, development=args.split == "internal_dev"
    )
    index = {}
    for entry in read_jsonl(args.feature_index):
        key = (entry["scene_token"], entry["sample_data_token"])
        if key in index:
            raise ValueError("duplicate scene/frame feature mapping")
        index[key] = entry["features"]
    part = args.output_dir / "bootstrap.jsonl.part"
    counts = Counter()
    with part.open("x", encoding="utf-8") as handle:
        for i, clip in enumerate(spatial.records):
            key = (clip["scene_token"], clip["frames"][0]["sample_data_token"])
            if key not in index:
                continue
            frame = spatial[i]["frames"][0]
            records = geometry_records(
                frame,
                clip,
                index[key],
                spatial.config,
                min_points=args.min_points,
                minimum_gap=args.minimum_gap,
            )
            for row in records:
                # Check the language input before publishing any supervision.
                load_features(spec["feature_root"], row)
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
                counts[row["geometry"]["relation"]] += 1
    dataset = AnswerSegDataset(part, spatial, spec["feature_root"])
    report = dataset.preflight()
    inputs = verify_feature_frames(
        dataset.records, spec["metadata_root"], spatial.records[0]["dataset_version"]
    )
    part.replace(args.output_dir / "bootstrap.jsonl")
    return {
        "complete": True,
        "supervision_kind": "geometry_bootstrap",
        "report": report,
        "source_split": policy,
        "feature_inputs": inputs,
        "relations": dict(counts),
        "human_reviewed_bindings": 0,
        "reasoning_segmentation_accepted": False,
    }
