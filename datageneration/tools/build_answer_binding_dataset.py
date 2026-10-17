#!/usr/bin/env python3
"""Package verified spatial masks and fail-closed answer-binding review records.

Uses official train QA and an existing scene-disjoint spatial dataset. Does not
infer grounding from a class match, and never converts missing targets to NOOBJ.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mllm"))
from grounded4d.taxonomy import FINE_NAMES, THING_CLASSES, audit_frame, decode_panoptic, validate_category_table
from build_stage3_seg_candidates import Metadata, category_mentions, qa_text, safe_file


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    """Stream a top-level JSON array with bounded memory and no extra dependency."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8-sig") as stream:
        buffer, eof = "", False

        def fill():
            nonlocal buffer, eof
            chunk = stream.read(1 << 16)
            buffer += chunk
            eof = not chunk

        def whitespace():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()

        whitespace()
        if not buffer.startswith("["):
            raise ValueError("expected JSON array: " + str(path))
        buffer = buffer[1:]
        whitespace()
        first = True
        while not buffer.startswith("]"):
            if not first:
                if not buffer.startswith(","):
                    raise ValueError("missing array separator")
                buffer = buffer[1:]
                whitespace()
                if buffer.startswith("]"):
                    raise ValueError("trailing comma")
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    if not isinstance(value, dict):
                        raise ValueError("expected array of objects")
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError("truncated/invalid JSON array: " + str(path))
                    fill()
            buffer = buffer[end:]
            yield value
            first = False
            whitespace()
        buffer = buffer[1:]
        whitespace()
        if buffer:
            raise ValueError("trailing JSON content")


def jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def emit(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def checked_bytes(root, record):
    raw = safe_file(root, record["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("source hash mismatch: " + record["path"])
    return raw


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-3):
        raise ValueError("invalid quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def verify_mask(frame, record, args):
    raw = checked_bytes(args.point_root, frame["point_file"])
    if not raw or len(raw) % 20:
        raise ValueError("invalid raw point layout")
    points = np.frombuffer(raw, dtype="<f4").reshape(-1, 5)
    if not np.isfinite(points).all() or not np.isin(points[:, 4], np.arange(32)).all():
        raise ValueError("invalid point values/ring")
    if not np.array_equal(np.asarray(frame["lidar_to_anchor"]), np.eye(4)):
        raise ValueError("only native single-frame point coordinates supported")
    mask_raw = checked_bytes(args.label_root, frame["mask_file"])
    with np.load(io.BytesIO(mask_raw), allow_pickle=False) as data:
        labels, tokens = data["labels"].copy(), data["instance_tokens"].tolist()
        if (data["label_kind"].item() != "panoptic_gt" or data["mask_verified"].item() is not True
                or data["point_file_sha256"].item() != frame["point_file"]["sha256"]
                or not np.array_equal(data["point_indices"], np.arange(len(points)))):
            raise ValueError("mask provenance or point order mismatch")
    classes = np.asarray(frame["class_ids"])
    if (labels.shape != (len(tokens), len(points)) or labels.dtype.kind not in "ib"
            or not np.isin(labels, [-1, 0, 1]).all() or len(set(tokens)) != len(tokens)
            or classes.shape != (len(tokens),) or classes.dtype.kind not in "iu"
            or ((classes < 0) | (classes >= len(THING_CLASSES))).any()):
        raise ValueError("invalid mask/class axes")
    encoded = decode_panoptic(checked_bytes(args.panoptic_root, frame["source_panoptic_file"]), len(points))
    bounds = np.array(args.point_cloud_range)
    valid = ((points[:, :3] >= bounds[:3]) & (points[:, :3] < bounds[3:])).all(1)
    audit_frame(frame, labels, classes, tokens, valid, encoded, THING_CLASSES, 1)
    return points, labels, tokens, valid, mask_raw


def phrase_context(answer, span):
    """Provisional sentence span; a reviewer must confirm the actual referent span."""
    start = max([0] + [m.end() for m in re.finditer(r"[.!?;]\s*", answer[:span[0]])])
    end_match = re.search(r"[.!?;]", answer[span[1]:])
    end = span[1] + end_match.end() if end_match else len(answer)
    while start < end and answer[start].isspace():
        start += 1
    return [start, end], answer[start:end]


def apply_review(candidate, review):
    if review.get("candidate_id") != candidate["candidate_id"]:
        raise ValueError("review identity mismatch")
    if review.get("decision") != "accept":
        raise ValueError("review import only accepts explicit accept decisions")
    for field in ("reviewer", "evidence", "reviewed_at"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError("missing review provenance: " + field)
    span = review.get("target_span")
    if (not isinstance(span, list) or len(span) != 2 or any(type(v) is not int for v in span)
            or not 0 <= span[0] < span[1] <= len(candidate["answer"])):
        raise ValueError("invalid reviewed answer span")
    if not (span[0] <= candidate["mention_span"][0] < candidate["mention_span"][1] <= span[1]):
        raise ValueError("reviewed phrase does not contain the candidate mention")
    selected = [obj for obj in candidate["objects"] if obj["instance_token"] == review.get("instance_token")]
    if len(selected) != 1:
        raise ValueError("reviewed instance is not a unique candidate in this frame")
    if review.get("candidate_fingerprint") != candidate["candidate_fingerprint"]:
        raise ValueError("review was made against different candidate content")
    return dict(candidate, target_span=span, target_phrase=candidate["answer"][span[0]:span[1]],
                selected_instance_token=selected[0]["instance_token"], target_mask_row=selected[0]["mask_row"],
                review=review, review_status="accepted", grounding_verified=True,
                eligible_for_binding_training=candidate["split"] == "train",
                eligible_for_binding_evaluation=candidate["split"] == "internal_dev")


def run(args):
    out = args.output_dir.resolve()
    protected = [args.label_root, args.metadata_root, args.panoptic_root, args.point_root]
    if any(out.is_relative_to(p.resolve()) for p in protected):
        raise ValueError("output must be separate from original data roots")
    out.mkdir(parents=True, exist_ok=False)
    (out / "spatial" / "masks").mkdir(parents=True)
    (out / "binding").mkdir()
    records, scenes_by_split, frame_tokens = {}, {}, set()
    sources = [*args.qa, args.scene_metadata, args.sequence_metadata, args.train_scene_tokens,
               args.train_manifest, args.dev_manifest, Path(__file__), Path(__file__).with_name("build_stage3_seg_candidates.py")]
    official_train = set(read(args.train_scene_tokens))
    for split, manifest in (("train", args.train_manifest), ("internal_dev", args.dev_manifest)):
        records[split] = list(jsonl(manifest))
        scenes_by_split[split] = {r["scene_token"] for r in records[split]}
        if not scenes_by_split[split] or not scenes_by_split[split] <= official_train:
            raise ValueError("non-training/empty scene partition")
        for row in records[split]:
            if (row.get("schema_version") != "b4dl-grounded4d-v1" or row.get("split") != split
                    or row.get("dataset_version") != "v1.0-trainval" or row.get("label_kind") != "panoptic_gt"
                    or row.get("annotation_scope") != "all_configured_instances" or len(row["frames"]) != 1):
                raise ValueError("not complete native single-frame spatial supervision")
            token = row["frames"][0]["sample_data_token"]
            if token in frame_tokens:
                raise ValueError("duplicate/cross-split frame")
            frame_tokens.add(token)
    if scenes_by_split["train"] & scenes_by_split["internal_dev"]:
        raise ValueError("scene leakage")
    scene_tokens = set.union(*scenes_by_split.values())
    metadata = Metadata([r for r in rows(args.scene_metadata) if r["scene_token"] in scene_tokens],
                        [r for r in rows(args.sequence_metadata) if r["scene_token"] in scene_tokens])
    tables = {}
    predicates = {
        "scene": lambda r: r["token"] in scene_tokens,
        "sample": lambda r: r["scene_token"] in scene_tokens,
        "sample_data": lambda r: r["token"] in frame_tokens,
    }
    # Build the sample-token set once: scanning sample_annotation must remain linear.
    selected_samples = {row["frames"][0]["sample_token"] for rr in records.values() for row in rr}
    predicates["sample_annotation"] = lambda r: r["sample_token"] in selected_samples
    for name in ("scene", "sample", "sample_data", "sample_annotation", "instance", "category", "calibrated_sensor", "ego_pose", "sensor"):
        path = args.metadata_root / (name + ".json")
        sources.append(path)
        predicate = predicates.get(name, lambda r: True)
        required_tokens = None
        if name == "instance":
            required_tokens = {r["instance_token"] for r in tables["sample_annotation"].values()}
        elif name == "calibrated_sensor":
            required_tokens = {r["calibrated_sensor_token"] for r in tables["sample_data"].values()}
        elif name == "ego_pose":
            required_tokens = {r["ego_pose_token"] for r in tables["sample_data"].values()}
        elif name == "sensor":
            required_tokens = {r["sensor_token"] for r in tables["calibrated_sensor"].values()}
        if required_tokens is not None:
            predicate = lambda r: r["token"] in required_tokens
        tables[name] = {r["token"]: r for r in rows(path) if predicate(r)}
        print(json.dumps(dict(stage="metadata", table=name, retained_rows=len(tables[name]))), flush=True)
    pan_category = args.panoptic_root / "v1.0-trainval/category.json"
    sources.append(pan_category)
    validate_category_table(read(pan_category))
    scene_frame_index = {}
    for scene in tables["scene"].values():
        token, seen = scene["first_sample_token"], set()
        while token:
            if token in seen:
                raise ValueError("cyclic sample chain")
            sample = tables["sample"][token]
            if sample["scene_token"] != scene["token"]:
                raise ValueError("cross-scene sample chain")
            scene_frame_index[token] = len(seen)
            seen.add(token)
            token = sample["next"]
        if len(seen) != scene["nbr_samples"] or len(seen) != metadata.scenes[scene["token"]]["num_frames"]:
            raise ValueError("scene length mismatch")
    frame_lookup, count = {}, Counter()
    with (out / "objects.jsonl").open("w", encoding="utf-8") as object_stream:
        for split, clips in records.items():
            write(out / "spatial" / (split + "_scene_tokens.json"), sorted(scenes_by_split[split]))
            with (out / "spatial" / (split + ".jsonl")).open("w", encoding="utf-8") as manifest_stream:
                for row in clips:
                    frame = row["frames"][0]
                    token, sample_token = frame["sample_data_token"], frame["sample_token"]
                    lidar = tables["sample_data"][token]
                    calibration = tables["calibrated_sensor"][lidar["calibrated_sensor_token"]]
                    if (lidar["sample_token"] != sample_token or not lidar["is_key_frame"]
                            or tables["sensor"][calibration["sensor_token"]]["channel"] != "LIDAR_TOP"
                            or tables["sample"][sample_token]["scene_token"] != row["scene_token"]
                            or lidar["timestamp"] != frame["timestamp_us"]):
                        raise ValueError("sample/LiDAR identity mismatch")
                    index = scene_frame_index[sample_token]
                    meta = metadata.frames[row["scene_token"]][index]
                    if meta["lidar_filename"] != lidar["filename"] or frame["point_file"]["path"] != lidar["filename"]:
                        raise ValueError("B4DL frame path mismatch")
                    if meta.get("sample_token", sample_token) != sample_token or meta.get("lidar_sample_data_token", token) != token:
                        raise ValueError("B4DL frame token mismatch")
                    points, labels, tokens, valid, mask_raw = verify_mask(frame, row, args)
                    pose = tables["ego_pose"][lidar["ego_pose_token"]]
                    xyz = points[:, :3].astype(float) @ rotation(calibration["rotation"]).T + calibration["translation"]
                    xyz = xyz @ rotation(pose["rotation"]).T + pose["translation"]
                    objects = []
                    for i, instance_token in enumerate(tokens):
                        evidence = frame["instance_mapping_evidence"][i]
                        annotation = tables["sample_annotation"][evidence["sample_annotation_token"]]
                        leaf = tables["category"][tables["instance"][instance_token]["category_token"]]["name"]
                        if (annotation["sample_token"] != sample_token or annotation["instance_token"] != instance_token
                                or leaf != FINE_NAMES[evidence["panoptic_id"] // 1000]):
                            raise ValueError("annotation identity mismatch")
                        mask = labels[i] == 1
                        local = (xyz[mask] - annotation["translation"]) @ rotation(annotation["rotation"])
                        size = np.array(annotation["size"])[[1, 0, 2]] / 2
                        if not np.isfinite(size).all() or (size <= 0).any():
                            raise ValueError("invalid box size")
                        overlap = float((np.abs(local) <= size + 1e-6).all(1).mean())
                        if not np.isclose(overlap, evidence["inside_annotation_box_fraction"], atol=1e-6):
                            raise ValueError("annotation overlap evidence mismatch")
                        obj = dict(instance_token=instance_token, mask_row=i, class_id=frame["class_ids"][i],
                                   class_name=THING_CLASSES[frame["class_ids"][i]], category_leaf=leaf,
                                   panoptic_id=evidence["panoptic_id"], sample_annotation_token=annotation["token"],
                                   point_count=int(mask.sum()), roi_point_count=int((mask & valid).sum()),
                                   center_lidar=points[mask, :3].mean(0).tolist())
                        objects.append(obj)
                        emit(object_stream, dict(obj, split=split, scene_token=row["scene_token"],
                                                 sample_data_token=token, mask_file=frame["mask_file"]))
                    relative = "masks/" + token + ".npz"
                    (out / "spatial" / relative).write_bytes(mask_raw)
                    frame["mask_file"] = dict(path=relative, sha256=hashlib.sha256(mask_raw).hexdigest())
                    # Keep spatial manifests consumable by the existing loader.
                    frame.pop("references", None)
                    row["grounding_verified"] = False
                    emit(manifest_stream, row)
                    frame_lookup[(row["scene_token"], index)] = dict(split=split, frame=frame, objects=objects)
                    count[split + "_frames"] += 1
                    count[split + "_instances"] += len(objects)
                    if sum(count[k] for k in ("train_frames", "internal_dev_frames")) % 100 == 0:
                        print(json.dumps(dict(stage="spatial", counts=dict(count))), flush=True)
    # Release metadata tables before streaming QA; do not keep point arrays in a cache.
    del tables
    reviews = {}
    if args.reviews:
        sources.append(args.reviews)
        for review in jsonl(args.reviews):
            if review["candidate_id"] in reviews:
                raise ValueError("duplicate review")
            reviews[review["candidate_id"]] = review
    used_reviews, reasons = set(), Counter()
    names = ("candidates", "rejected", "train", "internal_dev")
    streams = {name: (out / "binding" / (name + ".jsonl")).open("w", encoding="utf-8") for name in names}
    review_path = out / "binding/review.tsv"
    fields = ["candidate_id", "candidate_fingerprint", "split", "question", "answer", "target_phrase", "mention_span",
              "sample_data_token", "candidate_instance_tokens", "decision", "instance_token", "target_span", "reviewer", "reviewed_at", "evidence"]
    try:
        with review_path.open("w", encoding="utf-8-sig", newline="") as review_stream:
            writer = csv.DictWriter(review_stream, fields, delimiter="\t")
            writer.writeheader()
            for qa_path in args.qa:
                source_hash = sha(qa_path)
                for number, row in enumerate(rows(qa_path)):
                    count["qa_rows_scanned"] += 1
                    if row.get("scene_token") not in scene_tokens:
                        reasons["outside_current_spatial_scenes"] += 1
                        continue
                    count["qa_rows_in_scope"] += 1
                    source = dict(path=str(qa_path), sha256=source_hash, row_index=number)
                    def reject(reason, **extra):
                        reasons[reason] += 1
                        emit(streams["rejected"], dict(source=source, reason=reason, **extra))
                    question, answer = qa_text(row)
                    scene, errors = metadata.resolve_scene(row)
                    if errors or row.get("split") != "train" or scene.get("split") != "train":
                        reject("invalid_training_scene_identity")
                        continue
                    try:
                        indices, frame_source = metadata.resolve_frames(row, scene, question, answer)
                    except ValueError as exc:
                        reject(str(exc))
                        continue
                    mentions = [m for m in category_mentions(answer) if m["assertion"] == "positive_mention" and m["category"] in THING_CLASSES]
                    if not mentions:
                        reject("no_supported_positive_answer_mention")
                        continue
                    available = [i for i in indices if (scene["scene_token"], i) in frame_lookup]
                    if not available:
                        reject("referenced_frames_not_in_spatial_dataset", referenced_frame_indices=indices)
                        continue
                    for index in available:
                        entry = frame_lookup[(scene["scene_token"], index)]
                        frame = entry["frame"]
                        for mention in mentions:
                            objects = [o for o in entry["objects"] if o["category_leaf"] in mention["nuscenes_categories"]]
                            if not objects:
                                reject("no_matching_instance_in_frame_roi", frame_index=index, category=mention["category"])
                                continue
                            identity = [source_hash, number, frame["sample_data_token"], mention["span"]]
                            candidate_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
                            span, phrase = phrase_context(answer, mention["span"])
                            candidate = dict(schema_version="b4dl-answer-binding-v1", candidate_id=candidate_id,
                                source=source, answer_source="dataset_ground_truth", question=question, answer=answer,
                                target_phrase=phrase, target_span=span, phrase_status="provisional_sentence_context",
                                mention_text=mention["text"], mention_span=mention["span"], category=mention["category"],
                                scene_token=scene["scene_token"], scene_id=scene["scene_id"], split=entry["split"],
                                frame_index=index, frame_source=frame_source, referenced_frame_indices=indices,
                                available_frame_indices=available, reference_fully_covered=available == indices,
                                sample_data_token=frame["sample_data_token"], sample_token=frame["sample_token"],
                                timestamp_us=frame["timestamp_us"], point_file=frame["point_file"], mask_file=frame["mask_file"],
                                mask_root="spatial", objects=objects, selected_instance_token=None,
                                mask_verified=True, grounding_verified=False, review_status="pending",
                                temporal_grounding_verified=False, eligible_for_binding_training=False,
                                eligible_for_binding_evaluation=False)
                            candidate["candidate_fingerprint"] = hashlib.sha256(json.dumps(candidate, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                            if candidate_id in reviews:
                                candidate = apply_review(candidate, reviews[candidate_id])
                                used_reviews.add(candidate_id)
                                emit(streams[candidate["split"]], candidate)
                                count[candidate["split"] + "_reviewed_bindings"] += 1
                            emit(streams["candidates"], candidate)
                            count[entry["split"] + "_binding_candidates"] += 1
                            count["single_candidate_rows" if len(objects) == 1 else "multiple_candidate_rows"] += 1
                            writer.writerow(dict(candidate_id=candidate_id, candidate_fingerprint=candidate["candidate_fingerprint"],
                                split=entry["split"], question=question, answer=answer, target_phrase=phrase,
                                mention_span=json.dumps(mention["span"]), sample_data_token=frame["sample_data_token"],
                                candidate_instance_tokens=json.dumps([o["instance_token"] for o in objects]),
                                decision="pending"))
    finally:
        for stream in streams.values():
            stream.close()
    if set(reviews) != used_reviews:
        raise ValueError("review references an unknown/out-of-scope candidate")
    summary = dict(complete=True, scope="existing_64_train_8_dev_scene_spatial_dataset",
                   counts=dict(count), exclusions=dict(reasons), point_cloud_range=args.point_cloud_range,
                   source_roots=dict(points=str(args.point_root), panoptic=str(args.panoptic_root), metadata=str(args.metadata_root)),
                   source_files=[dict(path=str(p), sha256=sha(p)) for p in sources],
                   spatial_labels_verified=True, automatic_grounding_promotion=False,
                   train_binding_ready=count["train_reviewed_bindings"] > 0,
                   answer_source="official_training_QA_not_model_predictions",
                   limitation="Single-frame candidate review; neither temporal binding nor missing-object negatives are inferred.")
    write(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("label-root", "point-root", "panoptic-root", "metadata-root", "train-manifest", "dev-manifest",
                 "scene-metadata", "sequence-metadata", "train-scene-tokens", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--qa", nargs="+", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--point-cloud-range", nargs=6, type=float, default=[-51.2, -51.2, -5, 51.2, 51.2, 3])
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
