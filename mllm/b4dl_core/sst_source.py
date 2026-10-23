"""Audit historical SST artifacts and measure support in their actual input crop."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .io import digest, iter_array, read_json, sha256, unique_index, write_json

SCOPE = ("Consider only annotated objects with LiDAR returns retained in the historical "
         "CAM_BACK_RIGHT-cropped SST input (camera-centred crop and voxel range). "
         "Measure positions in the ego frame, not the camera frame.")
SUPPORT_METHOD = "retained_input_points_inside_annotation_oriented_box"


def source_inventory(encoder_root, tables, metadata, features, output):
    import torch
    from nuscenes.utils.splits import create_splits_scenes
    encoder_root, tables, features = map(Path, (encoder_root, tables, features))
    scenes = unique_index(iter_array(tables / "scene.json"))
    metas = list(iter_array(metadata))
    official = create_splits_scenes()
    names = {scenes[m["scene_token"]]["name"] for m in metas}
    if names != set(official["train"] + official["val"]) or len(metas) != 850:
        raise ValueError("complete official trainval scene coverage required")
    if len({m["scene_id"] for m in metas}) != 850 or len({m["scene_token"] for m in metas}) != 850:
        raise ValueError("duplicate feature scene identities")
    records = {}
    for meta in metas:
        path = features / (meta["scene_id"] + ".npy")
        values = np.load(path, allow_pickle=False)
        if values.shape != (meta["num_frames"], 768) or not np.isfinite(values).all():
            raise ValueError("invalid feature array: " + str(path))
        scene = scenes[meta["scene_token"]]
        split = "train" if scene["name"] in official["train"] else "validation"
        if meta["split"] != ("train" if split == "train" else "test"):
            raise ValueError("metadata disagrees with official split")
        records[meta["scene_id"]] = {"sha256": sha256(path), "shape": list(values.shape),
            "dtype": str(values.dtype), "scene_token": meta["scene_token"], "scene_name": scene["name"], "split": split}
    if {p.stem for p in features.glob("*.npy")} != set(records):
        raise ValueError("unexpected or missing feature scenes")
    checkpoint = encoder_root / "ckpt_anneal/lidarclip_mm/last.ckpt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = {k: v for k, v in payload["state_dict"].items() if k.startswith("lidar_encoder.")}
    state_hash = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        state_hash.update(json.dumps([key, list(tensor.shape), str(tensor.dtype)]).encode())
        state_hash.update(tensor.detach().contiguous().numpy().tobytes())
    source_files = ["extract_pc_features.py", "extract_pc_features_sample_token.py", "lidarclip/loader.py",
        "lidarclip/model/sst.py", "lidarclip/model/sst_encoder_only_config.py", "lidarclip/model/_mmdet3d_compat.py",
        "lidarclip/model/attention_pool.py", "logs/extract_stage2_b1.log", "logs/anneal_chain.log"]
    report = {"schema": "b4dl-sst-inventory-v1", "status": "historical_artifacts_identified",
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "lidar_state_sha256": state_hash.hexdigest(), "lidar_state_tensors": len(state),
        "epoch": payload.get("epoch"), "global_step": payload.get("global_step"),
        "lightning_version": payload.get("pytorch-lightning_version"),
        "feature_root": str(features), "metadata": str(metadata), "metadata_sha256": sha256(metadata),
        "features": records, "feature_index_sha256": digest(records), "scene_count": len(records),
        "frame_count": sum(r["shape"][0] for r in records.values()),
        "official_scene_counts": {"train": len(official["train"]), "validation": len(official["val"])},
        "files_sha256": {str(encoder_root / name): sha256(encoder_root / name) for name in source_files},
        "feature_reproduction_verified": False,
        "note": "Historical scripts/logs identify a candidate; numerical reproduction is a separate receipt."}
    write_json(output, report)
    return report


def load_context(tables, sample_tokens=None):
    tables = Path(tables)
    sensors = unique_index(iter_array(tables / "sensor.json"))
    calibrations = unique_index(iter_array(tables / "calibrated_sensor.json"))
    samples = unique_index(r for r in iter_array(tables / "sample.json")
                           if sample_tokens is None or r["token"] in sample_tokens)
    channel = {key: sensors[c["sensor_token"]]["channel"] for key, c in calibrations.items()}
    data, by_sample = {}, defaultdict(dict)
    for row in iter_array(tables / "sample_data.json"):
        if row["sample_token"] not in samples or not row["is_key_frame"]:
            continue
        name = channel[row["calibrated_sensor_token"]]
        if name in ("LIDAR_TOP", "CAM_BACK_RIGHT"):
            if name in by_sample[row["sample_token"]]:
                raise ValueError("duplicate keyframe sensor")
            data[row["token"]] = row
            by_sample[row["sample_token"]][name] = row["token"]
    wanted_poses = {r["ego_pose_token"] for r in data.values()}
    poses = unique_index(r for r in iter_array(tables / "ego_pose.json") if r["token"] in wanted_poses)
    for token, sample in samples.items():
        if set(by_sample[token]) != {"LIDAR_TOP", "CAM_BACK_RIGHT"}:
            raise ValueError("missing input sensor for sample: " + token)
        sample["data"] = by_sample[token]
    return {"sample": samples, "sample_data": data, "ego_pose": poses, "calibrated_sensor": calibrations}


def retained_points(context, sample_token, data_root):
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import view_points
    from pyquaternion import Quaternion
    sample = context["sample"][sample_token]
    lidar = context["sample_data"][sample["data"]["LIDAR_TOP"]]
    camera = context["sample_data"][sample["data"]["CAM_BACK_RIGHT"]]
    path = Path(data_root) / lidar["filename"]
    raw = path.read_bytes()
    points = np.frombuffer(raw, dtype=np.float32).reshape(-1, 5)[:, :4].T.copy()
    pc = LidarPointCloud(points)
    for table, token in (("calibrated_sensor", lidar["calibrated_sensor_token"]), ("ego_pose", lidar["ego_pose_token"])):
        record = context[table][token]
        pc.rotate(Quaternion(record["rotation"]).rotation_matrix)
        pc.translate(np.asarray(record["translation"]))
    global_points = pc.points[:3].T.copy()
    for table, token in (("ego_pose", camera["ego_pose_token"]), ("calibrated_sensor", camera["calibrated_sensor_token"])):
        record = context[table][token]
        pc.translate(-np.asarray(record["translation"]))
        pc.rotate(Quaternion(record["rotation"]).rotation_matrix.T)
    calibration = context["calibrated_sensor"][camera["calibrated_sensor_token"]]
    projected = view_points(pc.points[:3], np.asarray(calibration["camera_intrinsic"]), normalize=True)
    width, height = camera["width"], camera["height"]
    keep = ((pc.points[2] > .5) & (projected[0] > width//2-height//2) &
            (projected[0] < width//2+height//2) & (projected[1] >= 0) & (projected[1] <= height))
    input_points = pc.points.T[:, [2, 0, 1, 3]].copy()
    input_points[:, 1:3] *= -1
    input_points[:, 3] /= 255.
    cropped = input_points[keep].copy()
    within = ((input_points[:, :3] >= [0., -20., -2.]) & (input_points[:, :3] < [40., 20., 4.])).all(-1)
    retained = keep & within
    return cropped, global_points[retained], {"raw_lidar_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_points": len(points.T), "camera_crop_points": int(keep.sum()), "retained_points": int(retained.sum()),
        "lidar_sample_data_token": lidar["token"], "camera_sample_data_token": camera["token"],
        "camera_image_size": [width, height]}


def box_support(points, annotations):
    from pyquaternion import Quaternion
    from scipy.spatial import cKDTree
    tree = cKDTree(points) if len(points) else None
    result = {}
    for row in annotations:
        centre = np.asarray(row["translation"], dtype=np.float64)
        half = np.asarray(row["size"], dtype=np.float64)[[1, 0, 2]] / 2
        if not np.isfinite(centre).all() or not np.isfinite(half).all() or (half <= 0).any():
            raise ValueError("invalid annotation box")
        candidates = tree.query_ball_point(centre, np.linalg.norm(half)+1e-5) if tree is not None else []
        local = (points[candidates] - centre) @ Quaternion(row["rotation"]).rotation_matrix
        result[row["token"]] = int((np.abs(local) <= half+1e-5).all(-1).sum())
    return result


def prepare_support(tables, metadata, data_root, output):
    tables, output = Path(tables), Path(output)
    context = load_context(tables)
    print("input camera/pose context loaded", flush=True)
    annotations = defaultdict(list)
    for row in iter_array(tables / "sample_annotation.json"):
        annotations[row["sample_token"]].append({k: row[k] for k in ("token", "translation", "rotation", "size", "num_lidar_pts")})
    by_scene = defaultdict(list)
    for sample in context["sample"].values():
        by_scene[sample["scene_token"]].append(sample)
    samples_by_lidar = {context["sample_data"][s["data"]["LIDAR_TOP"]]["filename"]: s for s in context["sample"].values()}
    totals, seen = Counter(), set()
    with output.open("x", encoding="utf-8") as stream:
        for n, meta in enumerate(sorted(iter_array(metadata), key=lambda r: r["scene_id"])):
            for index in range(meta["num_frames"]):
                sample = samples_by_lidar[meta["paths"]["PATH_LIDAR_TOP"][f"PATH_{index:03d}"]]
                if sample["scene_token"] != meta["scene_token"] or sample["token"] in seen:
                    raise ValueError("support metadata scene/order coverage mismatch")
                seen.add(sample["token"])
                cropped, global_points, receipt = retained_points(context, sample["token"], data_root)
                rows = annotations[sample["token"]]
                counts = box_support(global_points, rows)
                record = {"scene_id": meta["scene_id"], "frame_index": index, "sample_token": sample["token"],
                          "annotation_counts": counts, **receipt}
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                totals.update({"frames": 1, "annotations": len(rows), "supported_annotations": sum(v > 0 for v in counts.values()),
                    "raw_points": receipt["raw_points"], "retained_points": len(global_points),
                    "raw_positive_but_outside_input": sum(r["num_lidar_pts"] > 0 and counts[r["token"]] == 0 for r in rows)})
            if (n+1) % 25 == 0:
                print("support scenes", n+1, "frames", totals["frames"], flush=True)
    if seen != set(context["sample"]):
        raise ValueError("not all raw keyframes received support records")
    report = {"schema": "b4dl-input-support-v1", "status": "all_trainval_input_support_computed", "method": SUPPORT_METHOD,
        "input_scope": SCOPE, "camera": "CAM_BACK_RIGHT", "point_cloud_range": [0, -20, -2, 40, 20, 4],
        "box_tolerance_m": 1e-5, "totals": dict(totals), "records": str(output), "records_sha256": sha256(output),
        "source_tables_sha256": {name: sha256(tables/(name+".json")) for name in
            ("sample", "sample_data", "ego_pose", "calibrated_sensor", "sample_annotation")},
        "metadata_sha256": sha256(metadata),
        "limitation": "Oriented-box point containment is geometric visibility support, not panoptic point-instance ground truth."}
    write_json(output.with_suffix(".receipt.json"), report)
    return report


def reproduce(encoder_root, tables, metadata, data_root, features, output):
    import os
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    from .b3 import device_lease
    encoder_root = Path(encoder_root).resolve()
    sys.path.insert(0, str(encoder_root))
    os.chdir(encoder_root)
    torch.set_num_threads(4)
    from lidarclip.model.sst import LidarEncoderSST
    from lidarclip.loader import NuscenesImageLidarDataset_with_path
    samples = list(iter_array(Path(tables)/"sample.json"))
    starts = [0, len(samples)//2//8*8, (len(samples)-8)//8*8]
    selected = [s["token"] for start in starts for s in samples[start:start+8]]
    context = load_context(tables, set(selected))
    metas = list(iter_array(metadata))
    source_row = {path: (m["scene_id"], int(key.split("_")[1])) for m in metas
                  for key, path in m["paths"]["PATH_LIDAR_TOP"].items()}
    class SmallNuScenes:
        dataroot = str(data_root)
        def get(self, table, token):
            return context[table][token]
    dataset = object.__new__(NuscenesImageLidarDataset_with_path)
    dataset._nusc, dataset._frames = SmallNuScenes(), selected
    dataset.NUSCENES_CAM_NAMES, dataset.min_dist = ["CAM_BACK_RIGHT"], .5
    dataset._img_transform = lambda image: torch.zeros(3, 224, 224)
    image_sizes = {str(Path(data_root)/r["filename"]): (r["width"], r["height"])
                   for r in context["sample_data"].values() if "width" in r and r.get("width", 0) > 0}
    records = []
    with device_lease("cuda:0", "/root/autodl-tmp/.b4dl_gpu0_eval.lock"):
        checkpoint = encoder_root / "ckpt_anneal/lidarclip_mm/last.ckpt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        state = {k[len("lidar_encoder."):]: v for k, v in payload["state_dict"].items() if k.startswith("lidar_encoder.")}
        model = LidarEncoderSST("lidarclip/model/sst_encoder_only_config.py", 768)
        loaded = model.load_state_dict(state, strict=False)
        if loaded.missing_keys:
            raise ValueError("incomplete encoder parameters")
        model.cuda().eval()
        for start in range(0, len(selected), 8):
            point_clouds = []
            for index in range(start, start+8):
                # Image pixels never enter the LiDAR branch; official dimensions define the crop.
                with patch("lidarclip.loader.Image.open", side_effect=lambda name: SimpleNamespace(size=image_sizes[str(name)])):
                    _, points, path = dataset[index]
                independent, _, receipt = retained_points(context, selected[index], data_root)
                np.testing.assert_array_equal(points.numpy(), independent)
                point_clouds.append(points.cuda())
            torch.manual_seed(0)
            with torch.no_grad(): actual = model(point_clouds)[0].cpu().numpy()
            for index, values in enumerate(actual, start):
                sample = context["sample"][selected[index]]
                sd = context["sample_data"][sample["data"]["LIDAR_TOP"]]
                ident, row = source_row[sd["filename"]]
                expected = np.load(Path(features)/(ident+".npy"), allow_pickle=False)[row]
                cosine = float(np.dot(values, expected)/(np.linalg.norm(values)*np.linalg.norm(expected)))
                records.append({"sample_token": selected[index], "scene_id": ident, "frame_index": row,
                    "max_abs_error": float(np.max(np.abs(values-expected))), "cosine": cosine,
                    "matches": bool(np.allclose(values, expected, atol=5e-4, rtol=1e-4) and cosine > .999999),
                    "preprocessing_exact_against_historical_loader": True})
            print("reproduced historical batch", start//8+1, flush=True)
        source_files, synthetic_modules = {}, {}
        for module in list(sys.modules.values()):
            name = getattr(module, "__file__", None)
            if name:
                path = Path(name).resolve()
                if path.suffix == ".py" and path.is_relative_to(encoder_root):
                    if path.is_file():
                        source_files[str(path)] = sha256(path)
                    else:
                        synthetic_modules[getattr(module, "__name__", str(name))] = str(name)
        report = {"schema": "b4dl-sst-reproduction-v1", "status": "passed" if all(r["matches"] for r in records) else "failed",
            "checkpoint_sha256": sha256(checkpoint), "samples": records, "batch_size": 8,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "dtype": "float32", "model_mode": "eval",
            "tolerance": {"atol": 5e-4, "rtol": 1e-4, "minimum_cosine": .999999},
            "source_files_sha256": source_files, "non_file_module_metadata": synthetic_modules,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
            "scope": "24 deterministic raw-data samples across beginning/middle/end; remaining assets inventoried by hash, not re-extracted"}
        report["camera_image_dimensions_source"] = "official sample_data width/height; image pixels are not used by the LiDAR branch"
        write_json(output, report)
        if report["status"] != "passed":
            raise ValueError("historical feature reproduction mismatch; see receipt")
    return report


def seal_provenance(inventory_path, reproduction_path, output):
    inventory, reproduction = read_json(inventory_path), read_json(reproduction_path)
    if (reproduction["status"] != "passed" or not reproduction["samples"]
            or not all(r["matches"] and r["preprocessing_exact_against_historical_loader"] for r in reproduction["samples"])
            or inventory["checkpoint_sha256"] != reproduction["checkpoint_sha256"]):
        raise ValueError("source inventory and numerical reproduction do not agree")
    if sha256(inventory["checkpoint"]) != inventory["checkpoint_sha256"]:
        raise ValueError("encoder changed since verification")
    files = {**inventory["files_sha256"], **reproduction["source_files_sha256"]}
    for path, expected in files.items():
        if sha256(path) != expected:
            raise ValueError("extraction source changed: " + path)
    for ident, record in inventory["features"].items():
        if sha256(Path(inventory["feature_root"])/(ident+".npy")) != record["sha256"]:
            raise ValueError("feature changed since inventory")
    result = {"schema": "b4dl-sst-provenance-v1", "verified": True,
        "verification_scope": "all_feature_hashes_and_sampled_raw_data_numerical_reproduction",
        "checkpoint": {"path": inventory["checkpoint"], "sha256": inventory["checkpoint_sha256"],
                       "epoch": inventory["epoch"], "global_step": inventory["global_step"]},
        "metadata_sha256": inventory["metadata_sha256"], "features": inventory["features"],
        "feature_index_sha256": inventory["feature_index_sha256"], "source_files_sha256": files,
        "receipts": {str(inventory_path): sha256(inventory_path), str(reproduction_path): sha256(reproduction_path)},
        "input_scope": {"camera": "CAM_BACK_RIGHT", "min_camera_depth_m": .5,
            "crop": "width//2-height//2 < pixel_x < width//2+height//2; 0 <= pixel_y <= height",
            "coordinates": "camera-z, negative-camera-x, negative-camera-y", "intensity_divisor": 255.,
            "point_cloud_range": [0, -20, -2, 40, 20, 4], "model_mode": "eval", "dtype": "float32"},
        "limitation": "Historical extraction RNG was not archived; sampled numerical agreement is not a full bitwise re-extraction."}
    result["provenance_id"] = digest(result)
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "support", "reproduce"))
    for name in ("tables", "metadata", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--encoder-root")
    parser.add_argument("--features")
    parser.add_argument("--data-root")
    args = parser.parse_args()
    if Path(args.output).exists(): parser.error("use a new output path")
    if args.command == "inventory":
        report = source_inventory(args.encoder_root, args.tables, args.metadata, args.features, args.output)
    elif args.command == "support":
        report = prepare_support(args.tables, args.metadata, args.data_root, args.output)
    else:
        report = reproduce(args.encoder_root, args.tables, args.metadata, args.data_root, args.features, args.output)
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
