"""Build and load a separate CoT dataset bound to existing geometry inputs."""
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
import math
import re

import numpy as np

from ..audit import raw_pose
from ..data import TABLES
from ..geometry import CATEGORY_MAP, CLASSES
from ..inputs import GeometryDataset
from ..io import digest, iter_array, read_json, read_jsonl, safe_file, sha256, unique_index, write_json, write_jsonl
from . import SCHEMA
from .protocol import make_trace, public_prompt
from .rewards import validate_target


class UnscorableSample(ValueError):
    """A known annotation limitation makes this question unsuitable for reward."""


def public_request(request):
    result = {key: request[key] for key in ("request_id", "scene_id", "split", "task", "predicate",
              "question", "frames", "feature_length", "kind", "options")}
    for key in ("semantic_id", "option_rotation"):
        if key in request:
            result[key] = request[key]
    result.update(schema=SCHEMA, parent_input_fingerprint=request["input_fingerprint"],
                  prompt=public_prompt(request))
    result["input_fingerprint"] = digest(result)
    return result


def verify_sources(geometry, table_root, support_path):
    """Allow relocated source files only when their original bytes match."""
    table_root, support_path = Path(table_root), Path(support_path)
    sources = geometry.manifest["sources"]
    for name in TABLES:
        matches = [value for path, value in sources.items() if PurePosixPath(path.replace("\\", "/")).name == name + ".json"]
        if len(matches) != 1 or sha256(table_root / (name + ".json")) != matches[0]:
            raise ValueError("raw source hash mismatch: " + name)
    scope = geometry.manifest.get("input_scope")
    if not scope or scope.get("support_method") != "retained_input_points_inside_annotation_oriented_box":
        raise ValueError("reasoning supervision requires verified retained-input support")
    if sha256(support_path) != scope["support_sha256"]:
        raise ValueError("retained-input support hash mismatch")
    return {name: sha256(table_root / (name+".json")) for name in TABLES} | {"support": sha256(support_path)}


def reference_frames(geometry, table_root, support_path):
    """Independently invert raw poses and retain only input-supported objects."""
    root = Path(table_root)
    wanted = {token for scene in geometry.manifest["scenes"].values() for token in scene["frame_tokens"]}
    lidar = unique_index(r for r in iter_array(root/"sample_data.json") if r["token"] in wanted)
    if set(lidar) != wanted:
        raise ValueError("missing frame sample_data")
    sample_ids = {row["sample_token"] for row in lidar.values()}
    pose_ids = {row["ego_pose_token"] for row in lidar.values()}
    poses = unique_index(r for r in iter_array(root/"ego_pose.json") if r["token"] in pose_ids)
    annotations = [r for r in iter_array(root/"sample_annotation.json") if r["sample_token"] in sample_ids]
    annotation_index = unique_index(annotations)
    instance_ids = {r["instance_token"] for r in annotations}
    instances = unique_index(r for r in iter_array(root/"instance.json") if r["token"] in instance_ids)
    categories = unique_index(iter_array(root/"category.json"))
    support = unique_index((r for r in read_jsonl(support_path) if r["sample_token"] in sample_ids), "sample_token")
    if set(support) != sample_ids:
        raise ValueError("retained-input support coverage mismatch")
    by_sample = defaultdict(list)
    for ann in annotations:
        by_sample[ann["sample_token"]].append(ann)
    frames = {}
    for scene_id, record in geometry.manifest["scenes"].items():
        bundle = geometry.scene(scene_id)
        for index, token in enumerate(record["frame_tokens"]):
            sd = lidar[token]
            pose = poses[sd["ego_pose_token"]]
            retained = support[sd["sample_token"]]
            if (sd["timestamp"] != pose["timestamp"] or sd["timestamp"] != record["timestamps_us"][index]
                    or retained["scene_id"] != scene_id or retained["frame_index"] != index
                    or retained["lidar_sample_data_token"] != token):
                raise ValueError("frame/pose/support identity mismatch")
            raw = raw_pose(pose)
            if not np.allclose(raw, bundle["frame_geometry"][index], atol=1e-9, rtol=0):
                raise ValueError("raw pose differs from model input")
            anns = by_sample[sd["sample_token"]]
            if set(retained["annotation_counts"]) != {ann["token"] for ann in anns}:
                raise ValueError("support annotation coverage mismatch")
            inverse = np.linalg.inv(raw)
            objects, unsupported = [], set()
            for ann in anns:
                name = CATEGORY_MAP.get(categories[instances[ann["instance_token"]]["category_token"]]["name"])
                if not name:
                    continue
                count = retained["annotation_counts"][ann["token"]]
                raw_count = ann["num_lidar_pts"]
                if type(count) is not int or count < 0 or type(raw_count) is not int or raw_count < 0:
                    raise ValueError("invalid input support count")
                if raw_count == 0:
                    unsupported.add(name)
                    continue
                if count == 0:
                    continue
                xyz = (inverse @ np.r_[ann["translation"], 1.])[:3]
                objects.append({"class": name, "instance": ann["instance_token"], "annotation": ann["token"],
                                "xyz": xyz, "distance": float(np.linalg.norm(xyz[:2]))})
            frames[(scene_id, index)] = {"objects": objects, "unsupported": unsupported}
    return frames, annotation_index


def _question_class(request):
    patterns = {"existence": r"Is there a (.*?) supported by LiDAR",
                "position": r"nearest LiDAR-supported (.*?) in the",
                "motion": r"same LiDAR-supported (.*?) that is nearest"}
    match = re.search(patterns[request["task"]], request["question"])
    name = match.group(1).replace(" ", "_") if match else None
    if name not in CLASSES:
        raise ValueError("cannot verify question category")
    return name


def verify_truth(request, target, frames):
    """Verify selection, GT values and instance continuity against raw tables."""
    start = frames[(request["scene_id"], request["frames"][0])]
    objects, provenance, evidence = start["objects"], target["provenance"], target["evidence"]
    task = request["task"]
    endpoints = None
    if task == "existence":
        category = _question_class(request)
        selected = [obj for obj in objects if obj["class"] == category]
        if not selected and category in start["unsupported"]:
            raise UnscorableSample("unsupported_object_cannot_establish_absence")
        if {o["annotation"] for o in selected} != set(provenance["annotations"]):
            raise ValueError("existence annotation mismatch")
        actual = {"count": len(selected)}
    else:
        selected = [o for o in objects if o["instance"] in provenance["instances"]]
        if len(selected) != len(provenance["instances"]):
            raise ValueError("start instance mismatch")
        for obj in selected:
            ds = sorted(o["distance"] for o in objects if o["class"] == obj["class"])
            if (not ds or not math.isclose(obj["distance"], ds[0], abs_tol=1e-8, rel_tol=0)
                    or len(ds) > 1 and ds[1]-ds[0] <= 1e-6):
                raise ValueError("start instance is not uniquely nearest")
        if task == "distance":
            if {o["class"] for o in selected} != set(request["options"].values()):
                raise ValueError("distance options do not match annotation classes")
            actual = {"distance_"+o["class"]: o["distance"] for o in selected}
        else:
            if len(selected) != 1 or selected[0]["class"] != _question_class(request):
                raise ValueError("question/annotation category mismatch")
            obj = selected[0]
            if task == "position":
                actual = {"x": float(obj["xyz"][0]), "y": float(obj["xyz"][1])}
            else:
                end_frame = frames[(request["scene_id"], request["frames"][1])]
                found = [o for o in end_frame["objects"] if o["instance"] == obj["instance"]]
                if len(found) != 1 or provenance["annotations"] != [obj["annotation"], found[0]["annotation"]]:
                    raise ValueError("motion endpoint identity mismatch")
                end = found[0]
                endpoints = {"y0": float(obj["xyz"][1]), "y1": float(end["xyz"][1])}
                actual = {"d0": obj["distance"], "d1": end["distance"],
                          "delta_d": end["distance"]-obj["distance"],
                          "delta_y": endpoints["y1"]-endpoints["y0"]}
    if set(actual) != set(evidence) or any(not math.isclose(actual[k], evidence[k], abs_tol=1e-8, rel_tol=0) for k in actual):
        raise ValueError("raw geometry does not match private target: " + request["request_id"])
    return endpoints


def build_data(geometry_root, table_root, support_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new reasoning dataset directory")
    geometry = GeometryDataset(geometry_root)
    source_hashes = verify_sources(geometry, table_root, support_path)
    frames, _ = reference_frames(geometry, table_root, support_path)
    requests, targets, exclusions = [], [], []
    for request in geometry.requests:
        truth = geometry.targets[request["request_id"]]
        try:
            endpoints = verify_truth(request, truth, frames)
        except UnscorableSample as error:
            exclusions.append({"request_id": request["request_id"], "reason": str(error)})
            continue
        target = {"request_id": request["request_id"], "answer": truth["answer"],
                  "trace": make_trace(request, truth, endpoints), "provenance": truth["provenance"],
                  "quality": truth["quality"]}
        validate_target(request, target)
        requests.append(public_request(request))
        targets.append(target)
    if not requests:
        raise ValueError("empty reasoning dataset")
    output.mkdir(parents=True)
    write_jsonl(output/"requests.jsonl", requests)
    write_jsonl(output/"supervision.jsonl", targets)
    write_jsonl(output/"exclusions.jsonl", exclusions)
    manifest = {"schema": SCHEMA, "parent_dataset_id": geometry.manifest["dataset_id"],
                "parent_manifest_sha256": sha256(Path(geometry_root)/"manifest.json"),
                "source_hashes": source_hashes, "input_scope": geometry.manifest["input_scope"]["description"],
                "generator_sha256": {name: sha256(Path(__file__).parent/name) for name in ("data.py", "protocol.py")},
                "counts": dict(Counter(r["split"]+"/"+r["task"] for r in requests)),
                "selected_questions": len(requests), "excluded_questions": len(exclusions),
                "files": {name: sha256(output/name) for name in ("requests.jsonl", "supervision.jsonl", "exclusions.jsonl")}}
    manifest["dataset_id"] = digest(manifest)
    write_json(output/"manifest.json", manifest)
    return manifest


class ReasoningDataset:
    def __init__(self, root, geometry_root, split=None):
        self.root = Path(root)
        self.manifest = read_json(self.root/"manifest.json")
        if (self.manifest["schema"] != SCHEMA
                or self.manifest["dataset_id"] != digest({k: v for k, v in self.manifest.items() if k != "dataset_id"})):
            raise ValueError("reasoning manifest identity mismatch")
        if self.manifest["generator_sha256"] != {name: sha256(Path(__file__).parent/name) for name in ("data.py", "protocol.py")}:
            raise ValueError("reasoning data generator differs from recorded identity")
        for name, expected in self.manifest["files"].items():
            if sha256(safe_file(self.root, name)) != expected:
                raise ValueError("reasoning dataset file hash mismatch")
        self.geometry = GeometryDataset(geometry_root)
        if (self.geometry.manifest["dataset_id"] != self.manifest["parent_dataset_id"]
                or sha256(Path(geometry_root)/"manifest.json") != self.manifest["parent_manifest_sha256"]):
            raise ValueError("parent geometry dataset changed")
        parent = unique_index(self.geometry.requests, "request_id")
        requests = read_jsonl(self.root/"requests.jsonl")
        all_requests = unique_index(requests, "request_id")
        self.targets = unique_index(read_jsonl(self.root/"supervision.jsonl"), "request_id")
        self.exclusions = unique_index(read_jsonl(self.root/"exclusions.jsonl"), "request_id")
        if (set(all_requests) != set(self.targets) or set(all_requests) & set(self.exclusions)
                or set(all_requests) | set(self.exclusions) != set(parent)
                or len(self.exclusions) != self.manifest["excluded_questions"]
                or len(all_requests) != self.manifest["selected_questions"]):
            raise ValueError("request/target/parent coverage mismatch")
        if any(set(row) != {"request_id", "reason"} or row["reason"] != "unsupported_object_cannot_establish_absence"
               for row in self.exclusions.values()):
            raise ValueError("unknown exclusion reason")
        for ident, request in all_requests.items():
            if request != public_request(parent[ident]):
                raise ValueError("public request differs from parent or exposes private data")
            validate_target(request, self.targets[ident])
        if split not in (None, "train", "internal_dev", "test"):
            raise ValueError("unknown split")
        self.requests = [r for r in requests if split is None or r["split"] == split]
        self.parent = parent

    def model_input(self, request, control=None):
        parent = self.parent[request["request_id"]]
        item = self.geometry.model_input(parent, control)
        item["prompt"] = request["prompt"]
        return item


def audit_data(dataset, table_root, support_path):
    hashes = verify_sources(dataset.geometry, table_root, support_path)
    if hashes != dataset.manifest["source_hashes"]:
        raise ValueError("reasoning source identity mismatch")
    frames, _ = reference_frames(dataset.geometry, table_root, support_path)
    for request in dataset.requests:
        parent = dataset.parent[request["request_id"]]
        truth = dataset.geometry.targets[request["request_id"]]
        endpoints = verify_truth(parent, truth, frames)
        target = dataset.targets[request["request_id"]]
        if target["trace"] != make_trace(parent, truth, endpoints) or target["provenance"] != truth["provenance"]:
            raise ValueError("reasoning supervision differs from independently verified source")
        validate_target(request, target)
    for ident, exclusion in dataset.exclusions.items():
        parent = dataset.parent[ident]
        try:
            verify_truth(parent, dataset.geometry.targets[ident], frames)
        except UnscorableSample as error:
            if str(error) != exclusion["reason"]:
                raise ValueError("exclusion reason changed") from error
        else:
            raise ValueError("excluded sample is now scorable")
    return {"status": "reasoning_raw_audit_passed", "dataset_id": dataset.manifest["dataset_id"],
            "questions": len(dataset.requests), "excluded_questions": len(dataset.exclusions),
            "frames": len(frames), "sources": hashes,
            "performance_measured": False}
