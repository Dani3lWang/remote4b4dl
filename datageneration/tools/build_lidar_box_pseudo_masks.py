#!/usr/bin/env python3
"""Audit and generate box-geometry pseudo masks, never verified segmentation GT.

CPU-only, NumPy + stdlib; no training/model imports. A separately established
original-point manifest is required for generation (see the companion docs).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np


_spec = importlib.util.spec_from_file_location(
    "_box_pseudo_candidate_contract", Path(__file__).with_name("build_stage3_seg_candidates.py"))
candidate_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(candidate_builder)
safe_file = candidate_builder.safe_file
file_info = candidate_builder.file_info
SCHEMA_VERSION = "b4dl-lidar-box-pseudo-masks-v1"
POINT_SCHEMA = "nuscenes-original-lidar-manifest-v1"
LAYOUT = "float32_le_xyz_intensity_ring"
FLAGS = {"mask_verified": False, "grounding_verified": False, "supervision_ready": False}
REQUIRED = candidate_builder.NuScenesTables.REQUIRED + ("ego_pose",)
CONVENTIONS = {
    "point_layout": LAYOUT, "point_coordinate_frame": "lidar_sensor",
    "point_order": "original_file_rows", "box_coordinate_frame": "global",
    "box_size_order": "width_length_height", "quaternion_order": "wxyz",
    "transform": "p_global = R_ego @ (R_sensor @ p_lidar + t_sensor) + t_ego",
    "box_test": "abs(R_box.T @ (p_global - center)) <= [length,width,height]/2",
    "boundary": "inclusive; float64 with 64*eps*coordinate_scale roundoff allowance",
    "overlap_policy": "ignore target-box points inside any other annotated box in this sample",
    "label_values": {"1": "inside target box only (pseudo positive)",
                     "0": "outside target box (geometry only, not semantic background)",
                     "-1": "inside target and another annotation box (ignore)"},
    "official_sources": [
        {"url": "https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py",
         "blob_sha": "fcc757b839079895865a788a077fd7f733535544",
         "evidence": "LidarPointCloud.from_file; Box.corners"},
        {"url": "https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py",
         "blob_sha": "e64270870c402450f1fb5e93dec8e14b093976cf",
         "evidence": "get_box; get_sample_data"},
        {"url": "https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md",
         "blob_sha": "e69415ca97e49f50fd659aac5cafcd39bffa3cf0",
         "evidence": "sample_annotation; calibrated_sensor; ego_pose"},
    ],
}


class Refusal(ValueError):
    """An input does not establish the required geometry/point identity."""


def require(condition, reason):
    if not condition:
        raise Refusal(reason)


def reject_json_constant(value):
    raise Refusal("nonfinite_json_constant")


def vector(value, length, reason):
    require(isinstance(value, list) and len(value) == length
            and all(type(v) in (int, float) and math.isfinite(v) for v in value), reason)
    return np.asarray(value, dtype=np.float64)


def rotation(value):
    q = vector(value, 4, "invalid_quaternion")
    norm = float(q @ q)
    require(math.isclose(norm, 1.0, rel_tol=0, abs_tol=1e-3), "invalid_quaternion_norm")
    # Same unit-quaternion normalization as pyquaternion.rotation_matrix.
    w, x, y, z = q / np.sqrt(norm)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def geometry(ann):
    center = vector(ann.get("translation"), 3, "invalid_box_translation")
    size = vector(ann.get("size"), 3, "invalid_box_size")
    require(bool(np.all(size > 0)), "invalid_box_size")
    return center, size[[1, 0, 2]] / 2, rotation(ann.get("rotation"))


def points_in_box(points_global, ann):
    center, half_extent, orient = geometry(ann)
    # Row-vector form of R_box^-1; use all three rotation axes, not yaw only.
    local = (points_global - center) @ orient
    # Translation of large global coordinates and inverse rotations introduce
    # roundoff at exact faces. This is a machine-precision bound, not a tunable
    # geometric enlargement (and is recorded in the convention audit).
    scale = (np.max(np.abs(points_global), axis=1, keepdims=True)
             + np.max(np.abs(center)) + np.max(half_extent) + 1)
    tolerance = 64 * np.finfo(np.float64).eps * scale
    return np.all(np.abs(local) <= half_extent + tolerance, axis=1)


class SnapshotTables(candidate_builder.NuScenesTables):
    """Reuse scene-chain checks, hash exactly the metadata bytes we parse."""
    REQUIRED = REQUIRED

    def __init__(self, root, version):
        self.root, self.version = root.resolve(), version
        self.table_dir = safe_file(self.root, version)
        self.missing = [str(self.table_dir / (name + ".json")) for name in self.REQUIRED
                        if not (self.table_dir / (name + ".json")).is_file()]
        self.available = not self.missing
        self.tables, self.by_sample, self.lidar, self._scene_cache = {}, defaultdict(list), {}, {}
        self.sources = []
        if not self.available:
            return
        for name in self.REQUIRED:
            path = safe_file(self.root, version + "/" + name + ".json")
            raw = path.read_bytes()
            rows = json.loads(raw, parse_constant=reject_json_constant)
            require(isinstance(rows, list) and all(isinstance(r, dict) for r in rows), "invalid_nuscenes_table")
            self.tables[name] = candidate_builder.unique_index(rows, "token", name)
            self.sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        for sd in self.tables["sample_data"].values():
            if sd.get("is_key_frame") is not True:
                continue
            calibration = self.tables["calibrated_sensor"].get(sd.get("calibrated_sensor_token"), {})
            sensor = self.tables["sensor"].get(calibration.get("sensor_token"), {})
            if sensor.get("channel") == "LIDAR_TOP" and sensor.get("modality") == "lidar":
                require(sd.get("sample_token") not in self.lidar, "duplicate_lidar_top_keyframe")
                self.lidar[sd["sample_token"]] = sd
        for ann in self.tables["sample_annotation"].values():
            self.by_sample[ann["sample_token"]].append(ann)


class Evidence:
    def __init__(self, root, version, point_manifest):
        self.tables = SnapshotTables(root, version)
        self.root = self.tables.root
        self.missing = self.tables.missing
        self.table_hashes = {Path(s["path"]).stem: s["sha256"] for s in self.tables.sources}
        self.manifest_source = None
        self.manifest_error = None
        self.point_entries = {}
        self.provenance = None
        if point_manifest is None:
            self.manifest_error = "point_provenance_manifest_missing"
            return
        try:
            raw = point_manifest.read_bytes()
            self.manifest_source = {"path": str(point_manifest.resolve()),
                                    "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            manifest = json.loads(raw, parse_constant=reject_json_constant)
            require(isinstance(manifest, dict) and manifest.get("schema_version") == POINT_SCHEMA,
                    "invalid_point_manifest_schema")
            provenance = manifest.get("provenance")
            require(isinstance(provenance, dict) and provenance.get("original_files") is True
                    and isinstance(provenance.get("source"), str) and bool(provenance["source"].strip()),
                    "original_point_provenance_missing")
            self.provenance = provenance
            hashes = manifest.get("metadata_sha256")
            require(isinstance(hashes, dict) and all(
                hashes.get(name) == self.table_hashes.get(name) and name in self.table_hashes
                for name in REQUIRED), "point_manifest_metadata_mismatch")
            entries = manifest.get("points")
            require(isinstance(entries, list) and all(isinstance(e, dict) for e in entries),
                    "invalid_point_manifest_entries")
            self.point_entries = candidate_builder.unique_index(entries, "sample_data_token", "point manifest")
        except (ValueError, OSError, TypeError, KeyError) as error:
            self.manifest_error = str(error)

    def frame(self, candidate):
        require(not self.missing, "nuscenes_tables_unavailable")
        require(self.manifest_error is None, self.manifest_error)
        frame = candidate.get("frame")
        require(isinstance(frame, dict), "missing_candidate_frame")
        tab = self.tables.tables
        sample = tab["sample"].get(frame.get("sample_token"))
        sd = tab["sample_data"].get(frame.get("lidar_sample_data_token"))
        require(sample is not None and sd is not None, "unknown_sample_or_lidar_token")
        require(sample.get("scene_token") == candidate.get("scene_token"), "candidate_scene_mismatch")
        require(type(frame.get("frame_index")) is int and frame["frame_index"] >= 0,
                "invalid_scene_frame_index")
        try:
            samples = self.tables.scene_samples(candidate["scene_token"])
            require(frame["frame_index"] < len(samples)
                    and samples[frame["frame_index"]]["token"] == sample["token"], "scene_frame_sample_mismatch")
        except ValueError as error:
            raise Refusal(str(error)) from error
        require(sd.get("sample_token") == sample["token"] and sd.get("is_key_frame") is True,
                "not_corresponding_lidar_keyframe")
        require(self.tables.lidar.get(sample["token"], {}).get("token") == sd["token"],
                "not_lidar_top_keyframe")
        require(type(sd.get("timestamp")) is int and type(sample.get("timestamp")) is int,
                "missing_frame_timestamp")
        calibration = tab["calibrated_sensor"].get(sd.get("calibrated_sensor_token"))
        pose = tab["ego_pose"].get(sd.get("ego_pose_token"))
        require(calibration is not None and pose is not None, "missing_sensor_or_ego_transform")
        filename = sd.get("filename")
        path = safe_file(self.root, filename)
        require(frame.get("lidar_filename") == filename, "candidate_lidar_filename_mismatch")
        require(isinstance(filename, str) and filename.startswith("samples/LIDAR_TOP/")
                and filename.endswith(".bin") and sd.get("fileformat") == "bin", "unsupported_point_file_layout")
        # Absolute paths in candidates may belong to the original host. They are
        # retained as provenance, never used to redirect reads away from metadata.
        entry = self.point_entries.get(sd["token"])
        require(entry is not None, "point_manifest_entry_missing")
        require(entry.get("filename") == filename and entry.get("sample_token") == sample["token"],
                "point_manifest_identity_mismatch")
        require(entry.get("layout") == LAYOUT, "unproven_point_layout")
        require(entry.get("coordinate_frame") == "lidar_sensor", "unproven_point_coordinate_frame")
        require(entry.get("point_order") == "original_file_rows", "unproven_point_order")
        require(isinstance(entry.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]),
                "invalid_point_sha256")
        require(path.is_file(), "missing_lidar_file")
        # Parse and hash the very same immutable bytes to avoid a hash/read race.
        raw = path.read_bytes()
        point_hash = hashlib.sha256(raw).hexdigest()
        require(point_hash == entry["sha256"], "point_sha256_mismatch")
        require(len(raw) > 0 and len(raw) % 20 == 0, "invalid_lidar_file_size")
        points = np.frombuffer(raw, dtype="<f4").reshape(-1, 5)
        n = len(points)
        require(type(entry.get("point_count")) is int and entry["point_count"] == n
                and type(frame.get("lidar_point_count")) is int and frame["lidar_point_count"] == n,
                "point_count_mismatch")
        require(bool(np.isfinite(points).all()), "nonfinite_point_values")
        rings = points[:, 4]
        require(bool(np.all((rings >= 0) & (rings < 32) & (rings == np.floor(rings)))),
                "invalid_hdl32_ring_layout")
        sensor_rot = rotation(calibration.get("rotation"))
        ego_rot = rotation(pose.get("rotation"))
        sensor_t = vector(calibration.get("translation"), 3, "invalid_sensor_translation")
        ego_t = vector(pose.get("translation"), 3, "invalid_ego_translation")
        xyz_global = (points[:, :3].astype(np.float64) @ sensor_rot.T + sensor_t) @ ego_rot.T + ego_t
        require(bool(np.isfinite(xyz_global).all()), "nonfinite_transformed_points")
        evidence = {"sample_token": sample["token"], "lidar_sample_data_token": sd["token"],
                    "sample_timestamp": sample["timestamp"], "lidar_timestamp": sd["timestamp"],
                    "point_file": {"path": str(path), "filename": filename, "sha256": point_hash,
                                   "bytes": len(raw), "point_count": n},
                    "point_manifest_entry": entry, "calibrated_sensor": calibration, "ego_pose": pose,
                    "point_index_mapping": "label column i corresponds to raw file row i; no filtering/reordering",
                    "conventions": CONVENTIONS}
        return xyz_global, evidence

    def masks(self, candidate):
        require(candidate.get("schema_version") == candidate_builder.SCHEMA_VERSION,
                "unsupported_candidate_schema")
        require(candidate.get("status") == "candidate_pending_mask_and_grounding"
                and candidate.get("association_level") == "answer_class_present_in_frame"
                and candidate.get("mask_status") == "not_generated"
                and not candidate.get("reasons"), "not_an_accepted_unverified_candidate")
        require(all(candidate.get(key) is False for key in FLAGS), "candidate_verification_flags_not_false")
        require(isinstance(candidate.get("candidate_id"), str) and bool(candidate["candidate_id"]),
                "missing_candidate_id")
        require(isinstance(candidate.get("source"), dict), "missing_candidate_source")
        targets = candidate.get("annotation_candidates")
        require(isinstance(targets, list) and bool(targets)
                and all(isinstance(t, dict) for t in targets), "missing_annotation_candidates")
        require(len({t.get("sample_annotation_token") for t in targets}) == len(targets),
                "duplicate_annotation_candidates")
        require(candidate.get("multiple_instance_candidates") is (len(targets) > 1),
                "candidate_multiplicity_mismatch")
        xyz, frame_evidence = self.frame(candidate)
        tab = self.tables.tables
        sample_token = frame_evidence["sample_token"]
        all_annotations = self.tables.by_sample[sample_token]
        require(bool(all_annotations), "sample_annotations_missing")
        by_token = {a["token"]: a for a in all_annotations}
        for target in targets:
            ann = by_token.get(target.get("sample_annotation_token"))
            require(ann is not None, "annotation_sample_mismatch")
            instance = tab["instance"].get(ann.get("instance_token"))
            category = tab["category"].get(instance.get("category_token")) if instance else None
            require(instance is not None and category is not None, "broken_instance_category_link")
            require(target.get("instance_token") == instance["token"]
                    and target.get("category_token") == category["token"]
                    and target.get("category_name") == category.get("name"), "candidate_instance_category_mismatch")
            require(category["name"] in candidate_builder.CLASS_CATEGORIES.get(candidate.get("category"), ()),
                    "candidate_class_mismatch")
            require(target.get("annotation_kind") == "3d_box" and target.get("box_global") == {
                k: ann.get(k) for k in ("translation", "size", "rotation")}, "candidate_box_geometry_mismatch")
            require(type(ann.get("num_lidar_pts")) is int and ann["num_lidar_pts"] > 0
                    and target.get("num_lidar_pts") == ann["num_lidar_pts"], "annotation_lidar_count_mismatch")
        overlap_count = np.zeros(len(xyz), dtype=np.uint32)
        target_tokens = {t["sample_annotation_token"] for t in targets}
        membership_by_token = {}
        for ann in all_annotations:
            # Even boxes of other categories and with zero advertised LiDAR
            # points participate; malformed overlap geometry refuses the frame.
            inside = points_in_box(xyz, ann)
            overlap_count += inside
            if ann["token"] in target_tokens:
                membership_by_token[ann["token"]] = inside
        membership = np.stack([membership_by_token[t["sample_annotation_token"]] for t in targets])
        labels = membership.astype(np.int8)
        labels[membership & (overlap_count[None, :] > 1)] = -1
        instances = []
        for i, target in enumerate(targets):
            count = int(membership[i].sum())
            require(count > 0, "empty_box_pseudo_mask")
            ann = by_token[target["sample_annotation_token"]]
            instances.append({**target, "mask_row": i, "inside_box_point_count": count,
                              "pseudo_positive_point_count": int((labels[i] == 1).sum()),
                              "ignored_overlap_point_count": int((labels[i] == -1).sum()),
                              "outside_box_point_count": int((labels[i] == 0).sum()),
                              "advertised_num_lidar_pts": ann["num_lidar_pts"],
                              "count_matches_annotation": count == ann["num_lidar_pts"],
                              "visibility_token": ann.get("visibility_token"),
                              "attribute_tokens": ann.get("attribute_tokens", []), **FLAGS})
        frame_evidence["overlap_annotation_tokens"] = [a["token"] for a in all_annotations]
        return labels, membership, overlap_count, instances, frame_evidence


def write_json(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def run(args):
    input_path = args.candidates.resolve()
    output_dir = args.output_dir.resolve()
    root = args.nuscenes_root.resolve()
    require(not output_dir.is_relative_to(root), "output_directory_inside_nuscenes_root")
    require(not output_dir.exists(), "output_directory_already_exists")
    # Freeze input bytes; the audit source hash covers exactly the parsed rows.
    raw = input_path.read_bytes()
    candidate_source = {"path": str(input_path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    evidence = Evidence(root, args.nuscenes_version, args.point_manifest)
    output_dir.mkdir(parents=True)
    (output_dir / "masks").mkdir()
    counts = Counter(input_candidates=0, generated_candidates=0, generated_instance_pseudo_masks=0,
                     rejected_candidates=0, mask_verified=0, grounding_verified=0, supervision_ready=0)
    reasons = Counter()
    seen_ids = set()
    names = ("audit", "pseudo_masks", "rejected")
    handles = {name: (output_dir / (name + ".jsonl.part")).open("x", encoding="utf-8") for name in names}
    try:
        for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            counts["input_candidates"] += 1
            row = {"schema_version": SCHEMA_VERSION, "candidate_source": {**candidate_source, "line_number": line_number},
                   "label_kind": "box_geometry_pseudo_mask", "eligible_for_training": False, **FLAGS}
            try:
                candidate = json.loads(line, parse_constant=reject_json_constant)
                require(isinstance(candidate, dict), "candidate_record_not_object")
                row["candidate"] = candidate
                candidate_id = candidate.get("candidate_id")
                require(isinstance(candidate_id, str), "missing_candidate_id")
                require(candidate_id not in seen_ids, "duplicate_candidate_id")
                seen_ids.add(candidate_id)
                labels, membership, overlap, instances, frame = evidence.masks(candidate)
            except (Refusal, ValueError, KeyError, TypeError, OSError, IndexError) as error:
                reason = str(error)
                row.update({"status": "rejected", "mask_status": "not_generated", "reasons": [reason]})
                reasons[reason] += 1
                counts["rejected_candidates"] += 1
                write_json(handles["rejected"], row)
            else:
                # Output failures abort the run; they must not be mislabeled as
                # input refusals or publish a successful completion summary.
                name = hashlib.sha256(candidate_id.encode()).hexdigest() + ".npz"
                path = output_dir / "masks" / name
                part = path.with_suffix(".npz.part")
                with part.open("xb") as handle:
                    np.savez_compressed(
                        handle, labels=labels, box_membership=membership,
                        overlap_count=overlap, point_indices=np.arange(labels.shape[1], dtype=np.int64),
                        sample_annotation_tokens=np.asarray([i["sample_annotation_token"] for i in instances]),
                        instance_tokens=np.asarray([i["instance_token"] for i in instances]),
                        point_file_sha256=np.asarray(frame["point_file"]["sha256"]),
                        label_kind=np.asarray("box_geometry_pseudo_mask"),
                        **{key: np.asarray(False) for key in FLAGS})
                part.replace(path)
                row.update({"status": "pseudo_mask_generated_unverified", "mask_status": "box_pseudo_generated",
                            "candidate_id": candidate_id, "frame_evidence": frame,
                            "instances": instances, "multiple_instance_candidates": len(instances) > 1,
                            "mask_file": {**file_info(path), "relative_path": "masks/" + name,
                                          "shape": list(labels.shape), "dtype": "int8"}})
                counts["generated_candidates"] += 1
                counts["generated_instance_pseudo_masks"] += len(instances)
                write_json(handles["pseudo_masks"], row)
            write_json(handles["audit"], row)
    finally:
        for handle in handles.values():
            handle.close()
    summary = {"schema_version": SCHEMA_VERSION, "counts": dict(counts),
               "rejection_reason_candidates": dict(reasons), "candidate_source": candidate_source,
               "point_manifest_source": evidence.manifest_source, "point_provenance": evidence.provenance,
               "point_manifest_error": evidence.manifest_error, "metadata_sources": evidence.tables.sources,
               "missing_nuscenes_tables": evidence.missing, "builder": file_info(Path(__file__)),
               "candidate_contract_code": file_info(Path(candidate_builder.__file__)),
               "config": {"nuscenes_root": str(root), "nuscenes_version": args.nuscenes_version},
               "conventions": CONVENTIONS, "label_kind": "box_geometry_pseudo_mask",
               "mask_status": "box_pseudo_generated" if counts["generated_candidates"] else "not_generated",
               "eligible_for_training": False, **FLAGS,
               "scope": "Box geometry pseudo labels only. No panoptic GT, language-grounding verification or true segmentation supervision."}
    part = output_dir / "summary.json.part"
    part.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name in names:
        (output_dir / (name + ".jsonl.part")).replace(output_dir / (name + ".jsonl"))
    part.replace(output_dir / "summary.json")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path, help="Candidate builder candidates.jsonl")
    parser.add_argument("--nuscenes-root", required=True, type=Path)
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument("--point-manifest", type=Path,
                        help="Independently established original-point provenance JSON; omission permits rejection audit only")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory outside nuScenes root")
    return parser.parse_args()


def main():
    try:
        summary = run(parse_args())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Box pseudo-mask builder failed: {error}")
        return 2
    print(json.dumps({"counts": summary["counts"],
                      "rejection_reason_candidates": summary["rejection_reason_candidates"],
                      "missing_nuscenes_tables": summary["missing_nuscenes_tables"], **FLAGS}, ensure_ascii=False, indent=2))
    # Empty input is an audited no-op. Partial/total input refusal is observable
    # by callers as nonzero, while the completed audit is still published.
    return 1 if summary["counts"]["rejected_candidates"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
