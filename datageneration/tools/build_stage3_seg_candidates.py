#!/usr/bin/env python3
"""Build auditable QA/class/frame/box candidates, never segmentation masks.

Standalone CPU/stdlib tool. It deliberately imports no B3 loader or model code.
See ../docs/stage3_seg_candidates.md for schemas and evidence boundaries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
from typing import Any


SCHEMA_VERSION = "b4dl-stage3-seg-candidates-v1"
TASK_ALIASES = {"binary": "binary_qa", "comprehensive": "comprehensive_reasoning",
                "temporal": "temporal_understanding"}
# nuScenes detection classes plus wheelchair/stroller/debris from the QA vocabulary.
# A lexical alias is evidence about text only; exact annotation leaves are checked later.
CLASS_CATEGORIES = {
    "car": ("vehicle.car",),
    "truck": ("vehicle.truck",),
    "bus": ("vehicle.bus.bendy", "vehicle.bus.rigid"),
    "trailer": ("vehicle.trailer",),
    "construction_vehicle": ("vehicle.construction",),
    "pedestrian": ("human.pedestrian.adult", "human.pedestrian.child",
                   "human.pedestrian.construction_worker", "human.pedestrian.police_officer"),
    "motorcycle": ("vehicle.motorcycle",),
    "bicycle": ("vehicle.bicycle",),
    "traffic_cone": ("movable_object.trafficcone",),
    "barrier": ("movable_object.barrier",),
    "wheelchair": ("human.pedestrian.wheelchair",),
    "stroller": ("human.pedestrian.stroller",),
    "debris": ("movable_object.debris",),
}
CLASS_PATTERNS = {
    "car": r"cars?|suvs?",
    "truck": r"trucks?",
    "bus": r"bus(?:es)?",
    "trailer": r"trailers?",
    "construction_vehicle": r"construction[ _-]+vehicles?",
    "pedestrian": r"pedestrians?|people|person",
    "motorcycle": r"motorcycles?|motorbikes?",
    "bicycle": r"bicycles?",
    "traffic_cone": r"traffic[ _-]*cones?",
    "barrier": r"barriers?",
    "wheelchair": r"wheelchairs?",
    "stroller": r"strollers?",
    "debris": r"debris",
}
MENTION_PATTERNS = {key: re.compile(r"(?<!\w)(?:" + pattern + r")(?!\w)", re.I)
                    for key, pattern in CLASS_PATTERNS.items()}
UNSAFE_ASSERTION = re.compile(
    r"\b(?:no|not|without|never|absent|absence|may|might|possibly|perhaps|could|would|if|cannot|"
    r"isn't|aren't|wasn't|weren't|doesn't|don't|didn't|can't)\b", re.I)
RANGE_PATTERN = re.compile(
    r"\b(?:(?:from|between)\s+)?frames?\s+(\d+)\s*(?:to|through|until|and|[-–])\s*(?:frames?\s+)?(\d+)", re.I)
SINGLE_FRAME = re.compile(r"\bframes?\s+(\d+)\b", re.I)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def file_info(path: Path) -> dict:
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def task_name(value: str | None) -> str | None:
    return TASK_ALIASES.get(value, value)


def clean_question(text: str) -> str:
    text = text.split("<meta>", 1)[0]
    return " ".join(text.replace("<video>", "").replace("<4DLiDAR>", "").split())


def match_text(text: str) -> str:
    # Preserve punctuation and frame numbers: loose normalization can merge QAs.
    return " ".join(text.split()).casefold()


def split_name(value: str | None) -> str | None:
    return "test" if value == "val" else value


def qa_text(row: dict) -> tuple[str, str]:
    if not isinstance(row, dict):
        raise ValueError("QA record must be an object")
    if "conversations" not in row:
        question, answer = row.get("question"), row.get("answer")
    else:
        turns = row["conversations"]
        if (not isinstance(turns, list) or len(turns) != 2
                or not all(isinstance(t, dict) for t in turns)
                or turns[0].get("from") != "human" or turns[1].get("from") != "gpt"):
            raise ValueError("Expected one human/gpt QA pair; multi-turn attribution is unsupported")
        question, answer = turns[0].get("value"), turns[1].get("value")
    if not isinstance(question, str) or not isinstance(answer, str) or not question.strip() or not answer.strip():
        raise ValueError("QA question and answer must be nonempty strings")
    return clean_question(question), answer


def category_mentions(answer: str) -> list[dict]:
    mentions = []
    for category, pattern in MENTION_PATTERNS.items():
        for match in pattern.finditer(answer):
            # Conservative local clause filter. This is not an NLP grounding model.
            start = max([0] + [m.end() for m in re.finditer(r"[.;!?]|\b(?:but|while|whereas)\b", answer[:match.start()], re.I)])
            end_match = re.search(r"[.;!?]|\b(?:but|while|whereas)\b", answer[match.end():], re.I)
            end = match.end() + end_match.start() if end_match else len(answer)
            prefix = answer[max(start, match.start() - 80):match.start()]
            suffix = answer[match.end():min(end, match.end() + 45)]
            uncertain = bool(UNSAFE_ASSERTION.search(prefix) or
                             re.search(r"^\s+(?:(?:is|are|was|were|do|does|did|has|have|had|can|will)\s+)?"
                                       r"(?:not|never|absent|may|might|could|would|isn't|aren't|wasn't|weren't|"
                                       r"hasn't|haven't|hadn't|cannot|can't|doesn't|don't|didn't)\b", suffix, re.I))
            mentions.append({"category": category, "text": match.group(), "span": list(match.span()),
                             "assertion": "negated_or_uncertain" if uncertain else "positive_mention",
                             "nuscenes_categories": list(CLASS_CATEGORIES[category])})
    return sorted(mentions, key=lambda mention: mention["span"])


def frame_references(text: str, num_frames: int) -> list[int]:
    indices: set[int] = set()
    consumed: list[tuple[int, int]] = []
    for match in RANGE_PATTERN.finditer(text):
        first, last = map(int, match.groups())
        if first > last or last >= num_frames:
            raise ValueError("invalid_frame_reference")
        # A range denotes a closed scene-frame interval, not sampled frame_id IDs.
        indices.update(range(first, last + 1))
        consumed.append(match.span())
    for match in SINGLE_FRAME.finditer(text):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        value = int(match.group(1))
        if value >= num_frames:
            raise ValueError("invalid_frame_reference")
        indices.add(value)
    if re.search(r"\bframes?\s*-\s*\d", text, re.I):
        raise ValueError("invalid_frame_reference")
    return sorted(indices)


def safe_file(root: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError("invalid_data_path")
    relative = Path(filename.replace("\\", "/"))
    if relative.is_absolute() or PureWindowsPath(filename).drive or ".." in relative.parts:
        raise ValueError("unsafe_data_path")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("unsafe_data_path")
    return target


def unique_index(rows: list[dict], key: str, label: str) -> dict:
    result = {}
    for row in rows:
        value = row.get(key)
        if value is None or value in result:
            raise ValueError(f"Missing/duplicate {key} in {label}: {value}")
        result[value] = row
    return result


class Metadata:
    def __init__(self, scenes: list[dict], sequences: list[dict]):
        self.scenes = unique_index(scenes, "scene_token", "scene metadata")
        self.by_id = unique_index(scenes, "scene_id", "scene metadata")
        self.sequences = unique_index(sequences, "sequence_id", "sequence metadata")
        self.frames: dict = defaultdict(dict)
        for scene in scenes:
            paths = scene.get("paths", {}).get("PATH_LIDAR_TOP", {})
            for key, filename in paths.items():
                index = int(key.removeprefix("PATH_"))
                self.frames[scene["scene_token"]][index] = {"lidar_filename": filename}
        for seq in sequences:
            indices, frames = seq.get("indices"), seq.get("frames")
            if (not isinstance(indices, list) or not isinstance(frames, list) or not indices
                    or len(indices) != len(frames) or any(type(i) is not int or i < 0 for i in indices)
                    or indices != sorted(set(indices)) or seq.get("scene_token") not in self.scenes):
                raise ValueError(f"Invalid sequence metadata: {seq.get('sequence_id')}")
            scene = self.scenes[seq["scene_token"]]
            if split_name(seq.get("split")) != split_name(scene.get("split")):
                raise ValueError("Sequence/scene split mismatch")
            for index, frame in zip(indices, frames):
                if index >= scene["num_frames"]:
                    raise ValueError("Sequence frame index outside scene")
                target = self.frames[seq["scene_token"]].setdefault(index, {})
                for dest, src in (("lidar_filename", "PATH_LIDAR_TOP"), ("sample_token", "sample_token"),
                                  ("lidar_sample_data_token", "TOKEN_LIDAR_TOP")):
                    value = frame.get(src)
                    if value:
                        if dest in target and target[dest] != value:
                            raise ValueError(f"Conflicting metadata at {seq['scene_token']}:{index}:{dest}")
                        target[dest] = value

    def resolve_scene(self, row: dict) -> tuple[dict | None, list[str]]:
        by_token = self.scenes.get(row.get("scene_token"))
        by_id = self.by_id.get(row.get("scene_id"))
        if row.get("scene_token") and by_token is None:
            return None, ["unknown_scene_token"]
        if row.get("scene_id") and by_id is None:
            return None, ["unknown_scene_id"]
        if by_token and by_id and by_token != by_id:
            return None, ["scene_id_token_mismatch"]
        scene = by_token or by_id
        if scene is None:
            return None, ["missing_scene_identity"]
        return scene, []

    def resolve_frames(self, row: dict, scene: dict, question: str, answer: str) -> tuple[list[int], str]:
        context = None
        envelope = None
        if "start_index" in row or "end_index" in row:
            first, last = row.get("start_index"), row.get("end_index")
            if type(first) is not int or type(last) is not int or not 0 <= first <= last < scene["num_frames"]:
                raise ValueError("invalid_sequence_envelope")
            envelope = (first, last)
        if row.get("sequence_id"):
            seq = self.sequences.get(row["sequence_id"])
            if not seq or seq["scene_token"] != scene["scene_token"]:
                raise ValueError("sequence_scene_mismatch")
            context = seq["indices"]
        if row.get("feat_indices") is not None:
            indices = row["feat_indices"]
            if (not isinstance(indices, list) or not indices or any(type(i) is not int or i < 0 or i >= scene["num_frames"] for i in indices)
                    or indices != sorted(set(indices)) or (context is not None and context != indices)):
                raise ValueError("invalid_feature_frame_context")
            context = indices
        if context is not None and envelope is not None and envelope != (context[0], context[-1]):
            raise ValueError("sequence_envelope_mismatch")
        for text, source in ((question, "question_frame_reference"), (answer, "answer_frame_reference")):
            indices = frame_references(text, scene["num_frames"])
            if indices:
                if envelope is not None and not all(envelope[0] <= i <= envelope[1] for i in indices):
                    raise ValueError("frame_reference_outside_context")
                if context is not None:
                    if not all(context[0] <= i <= context[-1] for i in indices):
                        raise ValueError("frame_reference_outside_context")
                    indices = [i for i in indices if i in context]
                    if not indices:
                        raise ValueError("no_sampled_frame_in_reference")
                return indices, source
        if context:
            return context, "explicit_sampled_frame_context"
        # start/end describe a sequence envelope, not its actual sampled frames.
        raise ValueError("missing_exact_frame_reference")


class NuScenesTables:
    REQUIRED = ("scene", "sample", "sample_data", "sample_annotation", "instance", "category", "sensor", "calibrated_sensor")

    def __init__(self, root: Path, version: str):
        self.root, self.version = root.resolve(), version
        self.table_dir = safe_file(self.root, version)
        self.missing = [str(self.table_dir / f"{name}.json") for name in self.REQUIRED
                        if not (self.table_dir / f"{name}.json").is_file()]
        self.available = not self.missing
        self.tables, self.by_sample, self.lidar, self._scene_cache = {}, defaultdict(list), {}, {}
        self.sources = []
        if not self.available:
            return
        for name in self.REQUIRED + ("lidarseg", "panoptic"):
            path = self.table_dir / f"{name}.json"
            if path.is_file():
                self.tables[name] = unique_index(load_json(path), "token", name)
                self.sources.append(file_info(path))
        for sd in self.tables["sample_data"].values():
            if not sd.get("is_key_frame"):
                continue
            calibration = self.tables["calibrated_sensor"].get(sd.get("calibrated_sensor_token"), {})
            sensor = self.tables["sensor"].get(calibration.get("sensor_token"), {})
            if sensor.get("channel") == "LIDAR_TOP" and sensor.get("modality") == "lidar":
                if sd["sample_token"] in self.lidar:
                    raise ValueError("Duplicate LIDAR_TOP keyframe for sample")
                self.lidar[sd["sample_token"]] = sd
        for ann in self.tables["sample_annotation"].values():
            self.by_sample[ann["sample_token"]].append(ann)

    def scene_samples(self, token: str) -> list[dict]:
        if token in self._scene_cache:
            return self._scene_cache[token]
        scene = self.tables["scene"].get(token)
        if not scene:
            raise ValueError("scene_not_in_nuscenes")
        current, previous, result, visited = scene["first_sample_token"], "", [], set()
        while current:
            sample = self.tables["sample"].get(current)
            if current in visited or not sample or sample.get("scene_token") != token or sample.get("prev", "") != previous:
                raise ValueError("invalid_nuscenes_sample_chain")
            visited.add(current)
            result.append(sample)
            previous, current = current, sample.get("next", "")
        if len(result) != scene["nbr_samples"] or (result and result[-1]["token"] != scene["last_sample_token"]):
            raise ValueError("nuscenes_scene_length_mismatch")
        self._scene_cache[token] = result
        return result

    def frame_evidence(self, scene: dict, index: int, meta_frame: dict) -> tuple[dict, list[str]]:
        evidence = {"frame_index": index, **meta_frame}
        if not self.available:
            return evidence, ["nuscenes_tables_unavailable"]
        try:
            samples = self.scene_samples(scene["scene_token"])
            if len(samples) != scene["num_frames"]:
                raise ValueError("metadata_nuscenes_scene_length_mismatch")
            sample = samples[index]
            sd = self.lidar.get(sample["token"])
            if not sd:
                raise ValueError("missing_lidar_keyframe")
            for key, actual in (("sample_token", sample["token"]), ("lidar_sample_data_token", sd["token"]),
                                ("lidar_filename", sd["filename"])):
                if meta_frame.get(key) and meta_frame[key] != actual:
                    raise ValueError("metadata_nuscenes_frame_mismatch")
                evidence[key] = actual
            path = safe_file(self.root, sd["filename"])
            evidence["lidar_path"] = str(path)
            if not path.is_file() or not path.stat().st_size:
                raise ValueError("missing_lidar_file")
            if path.stat().st_size % 20:
                raise ValueError("invalid_lidar_file_size")
            evidence["lidar_point_count"] = path.stat().st_size // 20
            point_labels = []
            for table in ("lidarseg", "panoptic"):
                label = self.tables.get(table, {}).get(sd["token"])
                if label:
                    label_path = safe_file(self.root, label["filename"])
                    point_labels.append({"kind": table, "token": label["token"], "path": str(label_path),
                                         "file_exists": label_path.is_file() and label_path.stat().st_size > 0,
                                         "contents_verified": False})
            evidence["point_label_sources"] = point_labels
            return evidence, []
        except ValueError as error:
            return evidence, [str(error)]

    def annotation_evidence(self, sample_token: str, category: str) -> tuple[list[dict], list[str]]:
        annotations, errors, matched, visible = [], [], False, False
        for ann in self.by_sample[sample_token]:
            instance = self.tables["instance"].get(ann.get("instance_token"))
            cat = self.tables["category"].get(instance.get("category_token")) if instance else None
            if not cat:
                errors.append("broken_annotation_category_link")
                continue
            if cat["name"] not in CLASS_CATEGORIES[category]:
                continue
            matched = True
            if ann.get("num_lidar_pts", 0) <= 0:
                continue
            visible = True
            geometry = {key: ann.get(key) for key in ("translation", "size", "rotation")}
            if (any(not isinstance(geometry[key], list) or len(geometry[key]) != length
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in geometry[key])
                    for key, length in (("translation", 3), ("size", 3), ("rotation", 4)))
                    or any(value <= 0 for value in geometry["size"])
                    or not math.isclose(sum(value * value for value in geometry["rotation"]), 1.0, abs_tol=1e-3)):
                errors.append("invalid_annotation_box_geometry")
                continue
            annotations.append({"sample_annotation_token": ann["token"], "instance_token": instance["token"],
                                "category_token": cat["token"], "category_name": cat["name"],
                                "num_lidar_pts": ann["num_lidar_pts"], "annotation_kind": "3d_box",
                                "box_global": geometry})
        if not annotations and not visible:
            errors.append("class_has_no_lidar_points" if matched else "class_annotation_not_found")
        return sorted(annotations, key=lambda ann: ann["sample_annotation_token"]), sorted(set(errors))


def source_rows(paths: list[Path]) -> tuple[dict, list[dict]]:
    lookup, sources = defaultdict(list), []
    for path in paths:
        source = file_info(path)
        sources.append(source)
        rows = load_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"Original QA must be a JSON array: {path}")
        inferred_task = task_name(path.stem) if path.stem not in ("stage2", "stage3") else None
        for index, row in enumerate(rows):
            question, answer = qa_text(row)
            task = task_name(row.get("task")) or inferred_task
            lookup[(task, match_text(question), match_text(answer))].append(
                (row, {**source, "row_index": index, "task": task}))
    return lookup, sources


def input_rows(path: Path, original_qa: list[Path], answer_source: str):
    data, source = load_json(path), file_info(path)
    if isinstance(data, list):
        if answer_source == "prediction":
            raise ValueError("Flat QA input supports --answer-source answer only")
        for index, row in enumerate(data):
            try:
                question, answer = qa_text(row)
                yield {"row": row, "question": question, "answer": answer, "ground_truth": answer,
                       "source": {**source, "row_index": index,
                                  "answer_field": "conversations[1].value" if "conversations" in row else "answer"}, "reasons": []}
            except ValueError as error:
                yield {"row": row if isinstance(row, dict) else {}, "question": "", "answer": "", "ground_truth": "",
                       "source": {**source, "row_index": index}, "reasons": ["invalid_qa_record"], "detail": str(error)}
        return
    if not isinstance(data, dict):
        raise ValueError("Input must be flat QA/conversations or grouped predictions")
    lookup, _ = source_rows(original_qa)
    if answer_source == "answer":
        raise ValueError("Grouped predictions require --answer-source prediction or ground_truth")
    for name, values in data.items():
        if name.startswith("_"):
            continue
        if not isinstance(values, dict):
            raise ValueError(f"Invalid prediction task: {name}")
        arrays = [values.get(field) for field in ("questions", "ground_truths", "predictions")]
        if not all(isinstance(a, list) for a in arrays) or len({len(a) for a in arrays}) != 1:
            raise ValueError(f"Prediction arrays have inconsistent lengths: {name}")
        task = task_name(name)
        for index, (question, gt, prediction) in enumerate(zip(*arrays)):
            if not all(isinstance(value, str) for value in (question, gt, prediction)):
                raise ValueError(f"Non-string prediction at {name}:{index}")
            source_ref = {**source, "task": task, "row_index": index,
                          "answer_field": "predictions" if answer_source == "prediction" else "ground_truths"}
            matched = lookup.get((task, match_text(clean_question(question)), match_text(gt)), [])
            record = {"row": {}, "question": clean_question(question),
                      "answer": prediction if answer_source == "prediction" else gt,
                      "ground_truth": gt, "source": source_ref, "reasons": [],
                      "qa_match_count": len(matched)}
            if len(matched) == 1:
                record["row"], record["linked_qa_source"] = matched[0]
            else:
                record["reasons"].append("ambiguous_original_qa" if matched else "original_qa_not_found")
                record["possible_qa_sources"] = [ref for _, ref in matched]
            yield record


def build_record(record: dict, metadata: Metadata, tables: NuScenesTables, expected_split: str,
                 train_tokens: set[str], require_point_labels: bool = False) -> tuple[dict, list[dict], list[dict]]:
    row, answer = record["row"], record["answer"]
    mentions = category_mentions(answer)
    audit = {key: value for key, value in record.items() if key != "row"}
    audit.update({"schema_version": SCHEMA_VERSION, "mentions": mentions,
                  "mask_status": "not_generated", "mask_verified": False,
                  "grounding_verified": False, "supervision_ready": False})
    audit["qa_metadata"] = {key: row[key] for key in ("scene_id", "scene_token", "sequence_id", "split", "task",
                                                       "start_index", "end_index", "feat_indices", "feat_range", "human_annotation") if key in row}
    reasons = list(record["reasons"])
    positive_classes = sorted({m["category"] for m in mentions if m["assertion"] == "positive_mention"})
    if not mentions:
        reasons.append("no_supported_answer_category")
    elif not positive_classes:
        reasons.append("only_negated_or_uncertain_mentions")
    scene, scene_reasons = metadata.resolve_scene(row)
    reasons.extend(scene_reasons)
    if scene:
        audit["scene_id"], audit["scene_token"] = scene["scene_id"], scene["scene_token"]
        actual_split = split_name(scene.get("split"))
        audit["split"] = actual_split
        if actual_split != expected_split:
            reasons.append("excluded_split")
        if row.get("split") is not None and split_name(row["split"]) != actual_split:
            reasons.append("qa_metadata_split_mismatch")
        if actual_split == "train" and scene["scene_token"] not in train_tokens:
            reasons.append("scene_not_in_training_whitelist")
        try:
            indices, frame_source = metadata.resolve_frames(row, scene, record["question"], answer)
            audit["frame_indices"], audit["frame_source"] = indices, frame_source
            audit["frame_links"] = [{"frame_index": i, **metadata.frames[scene["scene_token"]].get(i, {})} for i in indices]
        except ValueError as error:
            reasons.append(str(error))
    if not tables.available:
        reasons.append("nuscenes_tables_unavailable")
    reasons = sorted(set(reasons))
    accepted, rejected = [], []
    if reasons:
        audit.update({"status": "rejected", "reasons": reasons, "candidate_count": 0})
        return audit, [], [audit]
    for index in audit["frame_indices"]:
        frame, frame_reasons = tables.frame_evidence(scene, index, metadata.frames[scene["scene_token"]].get(index, {}))
        for category in positive_classes:
            candidate = {key: value for key, value in audit.items() if key not in ("frame_links", "frame_indices", "reasons")}
            candidate.update({"category": category, "frame": frame,
                              "association_level": "answer_class_present_in_frame",
                              "eligible_for_training": audit["split"] == "train"})
            failure = list(frame_reasons)
            annotations = []
            if not failure:
                annotations, annotation_reasons = tables.annotation_evidence(frame["sample_token"], category)
                failure.extend(annotation_reasons)
                if require_point_labels and not any(label["file_exists"] for label in frame["point_label_sources"]):
                    failure.append("missing_point_label_file")
            candidate["annotation_candidates"] = annotations
            candidate["multiple_instance_candidates"] = len(annotations) > 1
            candidate["reasons"] = sorted(set(failure))
            identity = [record["source"]["sha256"], record["source"].get("task"), record["source"]["row_index"],
                        record["source"].get("answer_field"), scene["scene_token"], index, category]
            candidate["candidate_id"] = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
            if failure:
                candidate["status"] = "rejected"
                rejected.append(candidate)
            else:
                candidate["status"] = "candidate_pending_mask_and_grounding"
                accepted.append(candidate)
    audit.update({"status": "candidate" if accepted else "rejected", "candidate_count": len(accepted),
                  "reasons": sorted({r for item in rejected for r in item["reasons"]})})
    return audit, accepted, rejected


def run(args: argparse.Namespace) -> dict:
    input_path = args.input.resolve()
    original_qa = sorted({p.resolve() for p in args.original_qa})
    scene_path, sequence_path = args.scene_metadata.resolve(), args.sequence_metadata.resolve()
    metadata = Metadata(load_json(scene_path), load_json(sequence_path))
    tables = NuScenesTables(args.nuscenes_root, args.nuscenes_version)
    whitelist_path = args.train_scene_tokens.resolve()
    train_tokens = set(load_json(whitelist_path))
    inputs = [input_path, scene_path, sequence_path, whitelist_path, *original_qa]
    output_dir = args.output_dir.resolve()
    # Do not overwrite any input or nuScenes table through a chosen output path.
    output_names = ("audit.jsonl", "candidates.jsonl", "rejected.jsonl", "summary.json")
    all_output_names = output_names + tuple(name + ".part" for name in output_names)
    protected = {p.resolve() for p in inputs} | {Path(source["path"]) for source in tables.sources}
    if any((output_dir / name).resolve() in protected for name in all_output_names):
        raise ValueError("Output collides with a source file")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in all_output_names):
        raise ValueError("Output files already exist; choose a fresh --output-dir")
    counts, reason_counts, task_counts, category_counts, row_status = Counter(), Counter(), Counter(), Counter(), Counter()
    handles = {name: (output_dir / (name + ".jsonl.part")).open("w", encoding="utf-8")
               for name in ("audit", "candidates", "rejected")}
    try:
        for record in input_rows(input_path, original_qa, args.answer_source):
            audit, candidates, rejected = build_record(record, metadata, tables, args.expected_split,
                                                       train_tokens, args.require_point_labels)
            counts["input_rows"] += 1
            counts["uniquely_linked_qa_rows"] += int("linked_qa_source" in record)
            counts["rows_with_category_mentions"] += bool(audit["mentions"])
            counts["scene_linked_rows"] += "scene_token" in audit
            counts["frame_linked_rows"] += "frame_indices" in audit
            counts["candidate_rows"] += len(candidates)
            counts["rejected_entries"] += len(rejected)
            counts["training_candidate_rows"] += sum(c["eligible_for_training"] for c in candidates)
            row_status[audit["status"]] += 1
            task_counts[record["source"].get("task") or record["row"].get("task") or "unspecified"] += 1
            reason_counts.update(audit["reasons"])
            category_counts.update(m["category"] for m in audit["mentions"])
            for name, rows in (("audit", [audit]), ("candidates", candidates), ("rejected", rejected)):
                for row in rows:
                    handles[name].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    source_info = [file_info(p) for p in inputs] + tables.sources
    summary = {"schema_version": SCHEMA_VERSION, "counts": dict(counts), "input_row_status": dict(row_status),
               "rejection_reason_rows": dict(reason_counts), "tasks": dict(task_counts),
               "category_mention_counts": dict(category_counts), "sources": source_info,
               "builder": file_info(Path(__file__)), "category_mapping": CLASS_CATEGORIES,
               "config": {"answer_source": args.answer_source, "expected_split": args.expected_split,
                          "require_point_labels": args.require_point_labels, "nuscenes_root": str(tables.root),
                          "nuscenes_version": tables.version},
               "nuscenes_tables_available": tables.available, "missing_nuscenes_tables": tables.missing,
               "mask_status": "not_generated", "mask_verified": False, "supervision_ready": False,
               "scope": "Candidates only; class/box evidence does not verify language grounding or point masks."}
    for name in handles:
        (output_dir / (name + ".jsonl.part")).replace(output_dir / (name + ".jsonl"))
    summary_part = output_dir / "summary.json.part"
    summary_part.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_part.replace(output_dir / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Flat QA/conversations array or grouped predictions.json")
    parser.add_argument("--original-qa", type=Path, nargs="*", default=[], help="Original QA arrays used to link legacy predictions uniquely")
    parser.add_argument("--answer-source", choices=("answer", "prediction", "ground_truth"), default="answer")
    parser.add_argument("--scene-metadata", type=Path, required=True)
    parser.add_argument("--sequence-metadata", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, default=Path(os.environ.get("B4DL_NUSCENES_ROOT", root.parent / "nuScenes")))
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument("--train-scene-tokens", type=Path, default=Path(__file__).parent / "assets/train_scene_tokens.json")
    parser.add_argument("--expected-split", choices=("train", "test"), default="train", help="test is an audit scope, never training data")
    parser.add_argument("--require-point-labels", action="store_true", help="Additionally require an existing lidarseg/panoptic file; contents still unverified")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        summary = run(parse_args())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Candidate builder failed: {error}")
        return 2
    print(json.dumps({"counts": summary["counts"], "rejection_reason_rows": summary["rejection_reason_rows"],
                      "nuscenes_tables_available": summary["nuscenes_tables_available"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
