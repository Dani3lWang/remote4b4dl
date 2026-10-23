"""Build aligned frame bundles and geometry QA from raw nuScenes JSON tables."""
from collections import Counter, defaultdict
from pathlib import Path
import math
import numpy as np

from . import SCHEMA, UPSTREAM_COMMIT
from .geometry import (CATEGORY_MAP, CLASSES, MOTIONS, SECTORS, global_to_ego,
                       motion_margin, motion_truth, motion_values, sector, sector_margin, transform)
from .io import digest, iter_array, read_json, safe_file, sha256, unique_index, write_json, write_jsonl

TABLES = ("scene", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor",
          "sample_annotation", "instance", "category")
RULES = {
    "motion_offset_keyframes": 8, "closer_m": -2., "farther_m": 2.,
    "left_m": 1., "right_m": -1., "similar_distance_m_strict": 1.,
    "similar_lateral_m_strict": .5, "distance_tie_epsilon_m": 1e-6,
    "sector_width_deg": 45., "lidar_points_positive": True,
    "motion_frame": "each_endpoint_ego_frame", "distance": "box_centre_ground_plane",
}


def assign_splits(scene_rows, official, seed=42):
    """Official validation is test; internal dev comes only from official train."""
    if set(official) != {"version", "source", "train", "validation"}:
        raise ValueError("split manifest requires version/source/train/validation")
    train, test = set(official["train"]), set(official["validation"])
    if len(train) != len(official["train"]) or len(test) != len(official["validation"]) or train & test:
        raise ValueError("duplicate or overlapping official scenes")
    if not train or not test or not official["source"]:
        raise ValueError("empty official split or missing provenance")
    names = [s["name"] for s in scene_rows]
    if len(set(names)) != len(names) or set(names) - (train | test):
        raise ValueError("unknown or duplicate scene names")
    # Assign from the COMPLETE official train list so subsets retain their split.
    ordered = sorted(train, key=lambda name: digest([seed, name]))
    if len(ordered) < 2:
        raise ValueError("need at least two official training scenes")
    dev_count = max(1, round(len(ordered) * .2))
    dev = set(ordered[:dev_count])
    return {s["token"]: "test" if s["name"] in test else "internal_dev" if s["name"] in dev else "train" for s in scene_rows}


def load_aligned_scenes(table_root, scene_metadata, feature_root, official, scene_ids=None, seed=42,
                        input_support=None):
    table_root, feature_root = Path(table_root), Path(feature_root)
    raw_scenes = unique_index(iter_array(table_root / "scene.json"))
    metas = [r for r in iter_array(scene_metadata) if r["scene_token"] in raw_scenes
             and (scene_ids is None or r["scene_id"] in scene_ids)]
    meta_index = unique_index(metas, "scene_id")
    if not metas or (scene_ids is not None and set(meta_index) != set(scene_ids)):
        raise ValueError("requested scenes not fully represented in metadata")
    if len({r["scene_token"] for r in metas}) != len(metas):
        raise ValueError("multiple feature scene IDs for one physical scene")
    selected = {r["scene_token"] for r in metas}
    samples = unique_index(r for r in iter_array(table_root / "sample.json") if r["scene_token"] in selected)
    sensors = unique_index(iter_array(table_root / "sensor.json"))
    calibrations = unique_index(iter_array(table_root / "calibrated_sensor.json"))
    lidar, lidar_samples = {}, set()
    for r in iter_array(table_root / "sample_data.json"):
        if r["sample_token"] not in samples or not r["is_key_frame"]:
            continue
        calibration = calibrations[r["calibrated_sensor_token"]]
        if sensors[calibration["sensor_token"]]["channel"] != "LIDAR_TOP":
            continue
        if r["filename"] in lidar or r["sample_token"] in lidar_samples:
            raise ValueError("duplicate LIDAR_TOP keyframe")
        lidar_samples.add(r["sample_token"])
        lidar[r["filename"]] = r
    pose_ids = {r["ego_pose_token"] for r in lidar.values()}
    poses = unique_index(r for r in iter_array(table_root / "ego_pose.json") if r["token"] in pose_ids)
    annotations = defaultdict(list)
    annotation_ids, instance_ids = set(), set()
    for r in iter_array(table_root / "sample_annotation.json"):
        if r["sample_token"] in samples:
            if r["token"] in annotation_ids:
                raise ValueError("duplicate annotation token")
            annotation_ids.add(r["token"])
            annotations[r["sample_token"]].append(r)
            instance_ids.add(r["instance_token"])
    instances = unique_index(r for r in iter_array(table_root / "instance.json") if r["token"] in instance_ids)
    categories = unique_index(iter_array(table_root / "category.json"))
    splits = assign_splits([raw_scenes[t] for t in sorted(selected)], official, seed)
    support = None
    if input_support is not None:
        from .sst_source import SCOPE, SUPPORT_METHOD
        from .io import read_jsonl
        receipt = read_json(Path(input_support).with_suffix(".receipt.json"))
        if receipt["method"] != SUPPORT_METHOD or receipt["records_sha256"] != sha256(input_support):
            raise ValueError("input support identity mismatch")
        if receipt["metadata_sha256"] != sha256(scene_metadata):
            raise ValueError("input support metadata changed")
        for name, expected_hash in receipt["source_tables_sha256"].items():
            if sha256(table_root / (name+".json")) != expected_hash:
                raise ValueError("input support source table changed")
        support = unique_index(read_jsonl(input_support), "sample_token")
    result = []
    for meta in sorted(metas, key=lambda x: x["scene_id"]):
        ident, token = meta["scene_id"], meta["scene_token"]
        path = safe_file(feature_root, ident + ".npy")
        features = np.load(path, allow_pickle=False)
        paths = meta["paths"]["PATH_LIDAR_TOP"]
        n = int(meta["num_frames"])
        if features.shape != (n, 768) or not np.issubdtype(features.dtype, np.floating) or not np.isfinite(features).all():
            raise ValueError("invalid SST features: " + ident)
        expected = [f"PATH_{i:03d}" for i in range(n)]
        if set(paths) != set(expected) or n != raw_scenes[token]["nbr_samples"]:
            raise ValueError("feature row metadata is incomplete")
        frames, seen_samples = [], set()
        previous_sample = None
        for i, key in enumerate(expected):
            sd = lidar[paths[key]]
            sample = samples[sd["sample_token"]]
            if sample["scene_token"] != token or sample["token"] in seen_samples:
                raise ValueError("wrong scene or repeated sample")
            if i == 0 and sample["token"] != raw_scenes[token]["first_sample_token"]:
                raise ValueError("first SST row is not scene first keyframe")
            if previous_sample is not None and (previous_sample["next"] != sample["token"] or sample["prev"] != previous_sample["token"]):
                raise ValueError("SST rows disagree with sample chronology")
            seen_samples.add(sample["token"])
            previous_sample = sample
            ego = poses[sd["ego_pose_token"]]
            if type(sd["timestamp"]) is not int or ego["timestamp"] != sd["timestamp"]:
                raise ValueError("LiDAR/ego pose timestamp mismatch")
            visibility = support[sample["token"]] if support is not None else None
            if visibility is not None and (visibility["scene_id"] != ident or visibility["frame_index"] != i
                    or visibility["lidar_sample_data_token"] != sd["token"]
                    or set(visibility["annotation_counts"]) != {a["token"] for a in annotations[sample["token"]]}):
                raise ValueError("input support frame/annotation coverage mismatch")
            objects, unsupported, seen_instances = [], set(), set()
            for ann in annotations[sample["token"]]:
                ins = instances[ann["instance_token"]]
                category = CATEGORY_MAP.get(categories[ins["category_token"]]["name"])
                if category is None:
                    continue
                if ann["instance_token"] in seen_instances:
                    raise ValueError("same instance annotated twice in a keyframe")
                seen_instances.add(ann["instance_token"])
                count = ann["num_lidar_pts"]
                if type(count) is not int or count < 0:
                    raise ValueError("invalid lidar point count")
                if count == 0:
                    unsupported.add(category)
                    continue
                if visibility is not None:
                    count = visibility["annotation_counts"][ann["token"]]
                    if type(count) is not int or count < 0:
                        raise ValueError("invalid retained-input point support")
                    if count == 0:
                        continue
                xyz = global_to_ego(ann["translation"], ego)
                objects.append({"class": category, "instance_token": ann["instance_token"],
                                "annotation_token": ann["token"], "xyz": xyz.tolist(),
                                "distance": float(np.linalg.norm(xyz[:2])), "num_lidar_pts": count})
            frames.append({"index": i, "sample_token": sample["token"], "sample_data_token": sd["token"],
                           "timestamp_us": sd["timestamp"], "global_from_ego": transform(ego).tolist(),
                           "ego_from_lidar": transform(calibrations[sd["calibrated_sensor_token"]]).tolist(),
                           "objects": objects, "unsupported_classes": sorted(unsupported)})
        if previous_sample["next"] or previous_sample["token"] != raw_scenes[token]["last_sample_token"]:
            raise ValueError("last SST row is not scene last keyframe")
        stamps = np.asarray([f["timestamp_us"] for f in frames], dtype=np.int64)
        if (np.diff(stamps) <= 0).any():
            raise ValueError("nonincreasing LiDAR timestamps")
        result.append({"scene_id": ident, "scene_token": token, "scene_name": raw_scenes[token]["name"],
                       "split": splits[token], "feature_sha256": sha256(path),
                       "features": features, "frames": frames})
        if support is not None:
            result[-1]["input_scope"] = SCOPE
    return result


def nearest(objects):
    groups = defaultdict(list)
    for obj in objects:
        groups[obj["class"]].append(obj)
    chosen = {}
    for name, items in groups.items():
        items.sort(key=lambda x: (x["distance"], x["instance_token"]))
        if len(items) > 1 and items[1]["distance"] - items[0]["distance"] <= RULES["distance_tie_epsilon_m"]:
            continue
        chosen[name] = items[0]
    return chosen


def response_instruction(keys):
    return ('Return only JSON with keys "answer" and "evidence". Evidence must have exactly these numeric keys: '
            + ", ".join(keys) + ". Use metres for distances and displacements; use an integer for count.")


def make_question(scene, task, question, answer, evidence, frames, *, predicate="", options=None, provenance=None, quality=None, variant=0):
    if scene.get("input_scope"):
        question = scene["input_scope"] + "\n" + question
    base = {"scene_id": scene["scene_id"], "task": task, "question": question,
            "frames": frames, "predicate": predicate, "options": options or {}}
    request = {"schema": SCHEMA, "request_id": digest([base, variant])[:24],
               "scene_id": scene["scene_id"], "split": scene["split"], "task": task,
               "predicate": predicate, "question": question, "frames": frames,
               "feature_length": len(scene["frames"]), "kind": "mcq" if options else "binary",
               "options": options or {}, "evidence_keys": sorted(evidence),
               "response_format": "geometry_json_v1"}
    request["prompt"] = question + ("\n" + "\n".join(f"{k}. {v}" for k, v in options.items()) if options else "") + "\n" + response_instruction(request["evidence_keys"])
    request["input_fingerprint"] = digest({k: v for k, v in request.items() if k != "request_id"})
    target = {"request_id": request["request_id"], "answer": answer, "evidence": evidence,
              "provenance": provenance or {}, "quality": quality or {}}
    return request, target


def generate_scene_questions(scene, seed=42):
    """No teacher-selected identity or GT measurement is included in requests."""
    rows = []
    frames = scene["frames"]
    for frame in frames:
        i = frame["index"]
        chosen = nearest(frame["objects"])
        for category in CLASSES:
            objects = [o for o in frame["objects"] if o["class"] == category]
            if not objects and category in frame["unsupported_classes"]:
                continue  # annotation with zero returns is unknown, not an absent object
            q = f"Is there a {category.replace('_', ' ')} supported by LiDAR returns in frame {i:03d}?"
            rows.append(make_question(scene, "existence", q, "Yes" if objects else "No", {"count": len(objects)}, [i],
                                      provenance={"annotations": [o["annotation_token"] for o in objects]}))
        for category, obj in chosen.items():
            x, y, _ = obj["xyz"]
            if math.hypot(x, y) < 1e-8:
                continue
            label = sector(x, y)
            distractors = [s for s in SECTORS if s != label]
            false_sector = distractors[int(digest([seed, scene["scene_id"], i, category])[:8], 16) % 7]
            for candidate in (label, false_sector):
                q = (f"In frame {i:03d}, is the nearest LiDAR-supported {category.replace('_', ' ')} "
                     f"in the {candidate} sector relative to the ego vehicle? Use eight 45-degree sectors centred on forward/left/back/right and their diagonals.")
                rows.append(make_question(scene, "position", q, "Yes" if candidate == label else "No", {"x": x, "y": y}, [i],
                                          predicate=candidate, provenance={"instances": [obj["instance_token"]]},
                                          quality={"sector_margin_deg": sector_margin(x, y), "num_lidar_pts": obj["num_lidar_pts"]}))
            j = i + RULES["motion_offset_keyframes"]
            if j >= len(frames):
                continue
            end = next((o for o in frames[j]["objects"] if o["instance_token"] == obj["instance_token"]), None)
            if end is None:
                continue
            if end["class"] != category:
                raise ValueError("instance category changed")
            values = motion_values(obj["xyz"], end["xyz"])
            descriptions = {"closer": "decrease its ground-plane distance by at least 2 m",
                            "farther": "increase its ground-plane distance by at least 2 m",
                            "left": "increase its ego-relative y coordinate by at least 1 m",
                            "right": "decrease its ego-relative y coordinate by at least 1 m",
                            "similar": "change its distance by less than 1 m AND its y coordinate by less than 0.5 m in magnitude"}
            for predicate in MOTIONS:
                q = (f"Follow the same LiDAR-supported {category.replace('_', ' ')} that is nearest in frame {i:03d}. "
                     f"From frame {i:03d} to {j:03d}, does it {descriptions[predicate]}? "
                     "Measure each endpoint in its own ego frame, with x forward and y left.")
                rows.append(make_question(scene, "motion", q, "Yes" if motion_truth(values)[predicate] else "No", values, [i, j],
                                          predicate=predicate, provenance={"instances": [obj["instance_token"]],
                                          "annotations": [obj["annotation_token"], end["annotation_token"]]},
                                          quality={"threshold_margin_m": motion_margin(values, predicate),
                                                   "min_lidar_pts": min(obj["num_lidar_pts"], end["num_lidar_pts"]),
                                                   "elapsed_s": (frames[j]["timestamp_us"]-frame["timestamp_us"])/1e6}))
        if len(chosen) >= 4:
            categories = sorted(chosen, key=lambda c: digest([seed, scene["scene_id"], i, c]))[:4]
            distances = {c: chosen[c]["distance"] for c in categories}
            order = sorted(categories, key=lambda c: distances[c])
            margin = distances[order[1]] - distances[order[0]]
            if margin <= RULES["distance_tie_epsilon_m"]:
                continue
            semantic_id = digest([scene["scene_id"], i, sorted(categories), "distance"])[:24]
            q = f"In frame {i:03d}, which listed class has the nearest LiDAR-supported instance by ground-plane box-centre distance to the ego vehicle?"
            for v in range(4):
                rotated = categories[v:] + categories[:v]
                options = dict(zip("ABCD", rotated))
                answer = next(k for k, value in options.items() if value == order[0])
                pair = make_question(scene, "distance", q, answer, {"distance_"+c: distances[c] for c in sorted(categories)}, [i],
                                     options=options, provenance={"instances": [chosen[c]["instance_token"] for c in sorted(categories)]},
                                     quality={"distance_margin_m": margin}, variant=v)
                pair[0]["semantic_id"], pair[0]["option_rotation"] = semantic_id, v
                # Recompute after adding public fields.
                pair[0]["input_fingerprint"] = digest({k: val for k, val in pair[0].items() if k not in ("request_id", "input_fingerprint")})
                rows.append(pair)
    return rows


def balance_questions(rows, seed=42, max_per_label=256):
    if max_per_label < 1:
        raise ValueError("max_per_label must be positive")
    groups, distance = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for request, target in rows:
        if request["kind"] == "mcq":
            distance[(request["split"], request["semantic_id"])].append((request, target))
        else:
            groups[(request["split"], request["task"], request["predicate"])][target["answer"]].append((request, target))
    selected, shortages = [], []
    for key, labels in sorted(groups.items()):
        size = min(len(labels["Yes"]), len(labels["No"]), max_per_label)
        if size == 0:
            shortages.append({"bucket": list(key), "yes": len(labels["Yes"]), "no": len(labels["No"])})
        for answer in ("Yes", "No"):
            selected.extend(sorted(labels[answer], key=lambda pair: digest([seed, pair[0]["request_id"]]))[:size])
    for split in ("train", "internal_dev", "test"):
        keys = sorted([k for k in distance if k[0] == split], key=lambda k: digest([seed, list(k)]))[:max_per_label]
        for key in keys:
            if len(distance[key]) != 4:
                raise ValueError("incomplete option orbit")
            selected.extend(distance[key])
    selected.sort(key=lambda pair: pair[0]["request_id"])
    unique_index([r for r, _ in selected], "request_id")
    return selected, shortages


def build_dataset(table_root, scene_metadata, feature_root, split_source, output, *, seed=42, max_per_label=256,
                  scene_ids=None, input_support=None, encoder_provenance=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("use a new output directory")
    official = read_json(split_source)
    scenes = load_aligned_scenes(table_root, scene_metadata, feature_root, official, scene_ids, seed, input_support)
    encoder = read_json(encoder_provenance) if encoder_provenance is not None else None
    if encoder is not None:
        if (encoder["schema"] != "b4dl-sst-provenance-v1" or not encoder["verified"]
                or encoder["provenance_id"] != digest({k: v for k, v in encoder.items() if k != "provenance_id"})
                or encoder["metadata_sha256"] != sha256(scene_metadata)):
            raise ValueError("encoder provenance is not verified or metadata changed")
        if sha256(encoder["checkpoint"]["path"]) != encoder["checkpoint"]["sha256"]:
            raise ValueError("encoder checkpoint changed")
        for scene in scenes:
            if encoder["features"][scene["scene_id"]]["sha256"] != scene["feature_sha256"]:
                raise ValueError("features differ from encoder provenance")
    rows = [pair for scene in scenes for pair in generate_scene_questions(scene, seed)]
    selected, shortages = balance_questions(rows, seed, max_per_label)
    if not selected:
        raise ValueError("no eligible questions")
    output.mkdir(parents=True)
    (output / "scenes").mkdir()
    scene_manifest = {}
    for scene in scenes:
        frames = scene["frames"]
        stamps = np.asarray([f["timestamp_us"] for f in frames], dtype=np.int64)
        bundle_path = output / "scenes" / (scene["scene_id"] + ".npz")
        np.savez_compressed(bundle_path, features=scene["features"],
                            frame_times=(stamps-stamps[0]).astype(np.float64)/1e6,
                            frame_geometry=np.asarray([f["global_from_ego"] for f in frames], dtype=np.float64),
                            ego_from_lidar=np.asarray([f["ego_from_lidar"] for f in frames], dtype=np.float64))
        scene_manifest[scene["scene_id"]] = {"scene_token": scene["scene_token"], "scene_name": scene["scene_name"],
            "split": scene["split"], "bundle": bundle_path.relative_to(output).as_posix(), "sha256": sha256(bundle_path),
            "source_feature_sha256": scene["feature_sha256"], "frames": len(frames),
            "frame_tokens": [f["sample_data_token"] for f in frames], "timestamps_us": stamps.tolist()}
    write_jsonl(output / "requests.jsonl", [r for r, _ in selected])
    write_jsonl(output / "supervision.jsonl", [t for _, t in selected])
    for split in ("train", "internal_dev", "test"):
        write_jsonl(output / (split + ".jsonl"), [{"request": r, "target": t} for r, t in selected if r["split"] == split])
    manifest = {"schema": SCHEMA, "dataset_version": official["version"], "seed": seed,
        "max_per_label": max_per_label, "rules": RULES, "upstream_reference": UPSTREAM_COMMIT,
        "baseline_checkpoint": None, "encoder_checkpoint_verified": False,
        "usage": "mini_engineering_validation" if official["version"] == "v1.0-mini" else "geometry_supervision",
        "base_model_may_have_seen_internal_dev": True, "scenes": scene_manifest,
        "candidate_questions": len(rows), "selected_questions": len(selected), "shortages": shortages,
        "counts": dict(Counter(r["split"]+"/"+r["task"] for r, _ in selected)),
        "generator_sha256": {name: sha256(Path(__file__).parent / name) for name in ("__init__.py", "data.py", "geometry.py", "io.py")},
        "sources": {str(Path(table_root)/ (name+".json")): sha256(Path(table_root)/(name+".json")) for name in TABLES}}
    manifest["sources"].update({str(scene_metadata): sha256(scene_metadata), str(split_source): sha256(split_source)})
    if input_support is not None:
        from .sst_source import SCOPE, SUPPORT_METHOD
        manifest["input_scope"] = {"description": SCOPE, "support_method": SUPPORT_METHOD,
            "support_records": str(input_support), "support_sha256": sha256(input_support)}
        manifest["sources"][str(input_support)] = sha256(input_support)
        receipt = Path(input_support).with_suffix(".receipt.json")
        manifest["sources"][str(receipt)] = sha256(receipt)
        manifest["generator_sha256"]["sst_source.py"] = sha256(Path(__file__).parent/"sst_source.py")
    if encoder is not None:
        manifest["encoder_checkpoint_verified"] = True
        manifest["encoder_provenance"] = {"path": str(encoder_provenance), "sha256": sha256(encoder_provenance),
                                          "provenance_id": encoder["provenance_id"]}
        manifest["sources"][str(encoder_provenance)] = sha256(encoder_provenance)
    manifest["files"] = {p.name: sha256(p) for p in output.glob("*.jsonl")}
    manifest["dataset_id"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    return manifest
