"""Recompute numerical supervision from raw annotations, without the QA builder."""
from collections import defaultdict
from pathlib import Path
import numpy as np

from .geometry import CATEGORY_MAP
from .io import iter_array, read_jsonl, sha256, unique_index


def raw_pose(row):
    w, x, y, z = np.asarray(row["rotation"], dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if not np.isfinite(norm) or abs(norm-1) > 1e-3:
        raise ValueError("invalid raw quaternion")
    w, x, y, z = np.asarray([w, x, y, z])/norm
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    result = np.eye(4)
    result[:3, :3] += 2*w*skew + 2*skew@skew
    result[:3, 3] = row["translation"]
    return result


def audit_supervision(dataset, tables):
    tables = Path(tables)
    for path, expected in dataset.manifest["sources"].items():
        if sha256(path) != expected:
            raise ValueError("source changed since preparation: " + path)
    frame_ids = {t for scene in dataset.manifest["scenes"].values() for t in scene["frame_tokens"]}
    lidar = unique_index(r for r in iter_array(tables/"sample_data.json") if r["token"] in frame_ids)
    sample_ids = {r["sample_token"] for r in lidar.values()}
    pose_ids = {r["ego_pose_token"] for r in lidar.values()}
    poses = unique_index(r for r in iter_array(tables/"ego_pose.json") if r["token"] in pose_ids)
    annotations = unique_index(r for r in iter_array(tables/"sample_annotation.json") if r["sample_token"] in sample_ids)
    instance_ids = {r["instance_token"] for r in annotations.values()}
    instances = unique_index(r for r in iter_array(tables/"instance.json") if r["token"] in instance_ids)
    categories = unique_index(iter_array(tables/"category.json"))
    support = None
    if "input_scope" in dataset.manifest:
        support = unique_index(read_jsonl(dataset.manifest["input_scope"]["support_records"]), "sample_token")
    calibrations = unique_index(iter_array(tables/"calibrated_sensor.json"))
    by_sample = defaultdict(list)
    for row in annotations.values():
        by_sample[row["sample_token"]].append(row)
    frames, numeric_checks = {}, 0
    for ident, record in dataset.manifest["scenes"].items():
        bundle = dataset.scene(ident)
        for index, token in enumerate(record["frame_tokens"]):
            sd = lidar[token]
            pose = raw_pose(poses[sd["ego_pose_token"]])
            calibration = raw_pose(calibrations[sd["calibrated_sensor_token"]])
            if not np.allclose(pose, bundle["frame_geometry"][index], atol=1e-9, rtol=0) or not np.allclose(calibration, bundle["ego_from_lidar"][index], atol=1e-9, rtol=0):
                raise ValueError("raw pose/bundle mismatch")
            if sd["timestamp"] != record["timestamps_us"][index]:
                raise ValueError("raw timestamp mismatch")
            inverse = np.linalg.inv(pose)
            objects = []
            for ann in by_sample[sd["sample_token"]]:
                category = CATEGORY_MAP.get(categories[instances[ann["instance_token"]]["category_token"]]["name"])
                if not category or ann["num_lidar_pts"] <= 0:
                    continue
                if support is not None and support[sd["sample_token"]]["annotation_counts"][ann["token"]] <= 0:
                    continue
                xyz = (inverse @ np.r_[ann["translation"], 1])[:3]
                objects.append({"category": category, "instance": ann["instance_token"], "annotation": ann["token"],
                                "xyz": xyz, "distance": float(np.linalg.norm(xyz[:2]))})
            frames[(ident, index)] = objects
    for request in dataset.requests:
        target = dataset.targets[request["request_id"]]
        frame = frames[(request["scene_id"], request["frames"][0])]
        provenance = target["provenance"]
        task = request["task"]
        if task == "existence":
            category = request["question"].split("Is there a ", 1)[1].split(" supported by", 1)[0].replace(" ", "_")
            selected = [o for o in frame if o["category"] == category]
            if set(provenance["annotations"]) != {o["annotation"] for o in selected}:
                raise ValueError("existence provenance mismatch")
            expected = {"count": len(selected)}
        else:
            selected = [o for o in frame if o["instance"] in provenance["instances"]]
            if len(selected) != len(provenance["instances"]):
                raise ValueError("missing start identity")
            for obj in selected:
                distances = sorted(o["distance"] for o in frame if o["category"] == obj["category"])
                if abs(obj["distance"] - distances[0]) > 1e-8 or len(distances) > 1 and distances[1]-distances[0] <= 1e-6:
                    raise ValueError("nonunique or nonnearest start object")
            if task == "distance":
                expected = {"distance_"+o["category"]: o["distance"] for o in selected}
            elif task == "position":
                expected = dict(zip(("x", "y"), selected[0]["xyz"][:2]))
            else:
                end_frame = frames[(request["scene_id"], request["frames"][1])]
                end = [o for o in end_frame if o["instance"] == selected[0]["instance"]]
                if len(end) != 1 or provenance["annotations"] != [selected[0]["annotation"], end[0]["annotation"]]:
                    raise ValueError("motion identity/provenance mismatch")
                d0, d1 = selected[0]["distance"], end[0]["distance"]
                expected = {"d0": d0, "d1": d1, "delta_d": d1-d0, "delta_y": end[0]["xyz"][1]-selected[0]["xyz"][1]}
        if set(expected) != set(target["evidence"]):
            raise ValueError("evidence keys mismatch")
        for key, value in expected.items():
            if not np.isclose(value, target["evidence"][key], atol=1e-8, rtol=0):
                raise ValueError("raw numeric supervision mismatch: " + request["request_id"])
            numeric_checks += 1
    return {"status": "raw_supervision_audit_passed", "dataset_id": dataset.manifest["dataset_id"],
            "questions": len(dataset.requests), "numeric_values_recomputed": numeric_checks,
            "frames": len(frames), "baseline_checkpoint": None,
            "note": "Inverse homogeneous transforms and annotation identity checks; no model quality measurement."}
