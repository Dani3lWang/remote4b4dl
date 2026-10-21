"""Reviewed language bindings joined to independently audited, full-frame panoptic data."""

import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from grounded4d.data import read_hashed_file

from . import SCHEMA
from .language import MARKERS, make_prompt, mark_answer


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected an object")
                yield row


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_request(row):
    for key in ("request_id", "scene_token", "sample_data_token", "question"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError("missing input identity: " + key)
    if row.get("split") not in ("train", "internal_dev"):
        raise ValueError("only train/internal_dev requests are supported")
    if any(marker in row["question"] for marker in MARKERS):
        raise ValueError("question contains segmentation supervision")
    feature = row.get("features", {})
    if feature.get("input_policy") not in ("explicit_frames", "whole_scene"):
        raise ValueError(
            "features require explicit_frames/whole_scene input; answer_frames is forbidden"
        )
    frames = feature.get("frame_tokens")
    if (
        not isinstance(frames, list)
        or not frames
        or any(not isinstance(token, str) or not token for token in frames)
        or row["sample_data_token"] not in frames
    ):
        raise ValueError(
            "feature rows must retain explicit frame tokens including the segmentation frame"
        )
    if (
        not isinstance(feature.get("encoder_checkpoint_id"), str)
        or not feature["encoder_checkpoint_id"]
    ):
        raise ValueError("the SST/LiDARCLIP feature encoder identity is required")


def load_features(root, row):
    validate_request(row)
    entry = row["features"]
    values = np.load(io.BytesIO(read_hashed_file(root, entry)), allow_pickle=False)
    if values.ndim != 2 or values.shape[1] != 768 or not np.isfinite(values).all():
        raise ValueError("scene features must be finite [frames,768]")
    indices = entry.get("indices", list(range(len(values))))
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(i) is not int for i in indices)
        or len(set(indices)) != len(indices)
        or min(indices) < 0
        or max(indices) >= len(values)
        or len(indices) != len(entry["frame_tokens"])
    ):
        raise ValueError("explicit feature indices and frame-token rows disagree")
    return torch.from_numpy(values[indices].astype(np.float32, copy=True))


def validate_review(review):
    if review.get("decision") != "accept":
        raise ValueError("only explicit accepted reviews may enter training/evaluation")
    for key in ("reviewer", "reviewed_at", "evidence"):
        if not isinstance(review.get(key), str) or not review[key].strip():
            raise ValueError("missing review provenance: " + key)


def prepare_records(sources, feature_index, reviews=()):
    """Import legacy accepted bindings OR fingerprinted, reviewed generated answers."""
    index, review_index = {}, {}
    for item in feature_index:
        key = (item["scene_token"], item["sample_data_token"])
        if key in index:
            raise ValueError("duplicate scene/frame feature mapping")
        index[key] = item["features"]
    for review in reviews:
        key = review["request_id"]
        if key in review_index:
            raise ValueError("duplicate generated-answer review")
        review_index[key] = review
    result, used_reviews, legacy_groups = [], set(), {}
    for source in sources:
        if source.get("schema_version") == "b4dl-answer-binding-v1":
            if (
                source.get("grounding_verified") is not True
                or source.get("review_status") != "accepted"
            ):
                raise ValueError(
                    "legacy candidates are pending; import binding/train or internal_dev only"
                )
            review = source.get("review", {})
            validate_review(review)
            if (
                review.get("candidate_id") != source.get("candidate_id")
                or review.get("candidate_fingerprint")
                != source.get("candidate_fingerprint")
                or review.get("target_span") != source.get("target_span")
                or review.get("instance_token") != source.get("selected_instance_token")
            ):
                raise ValueError("legacy review identity/span mismatch")
            matches = [
                obj
                for obj in source["objects"]
                if obj["instance_token"] == source["selected_instance_token"]
            ]
            if len(matches) != 1 or matches[0]["mask_row"] != source.get(
                "target_mask_row"
            ):
                raise ValueError(
                    "legacy selected mask/instance is not a unique reviewed candidate"
                )
            start, end = source["target_span"]
            phrase = source["answer"][start:end]
            if source.get("target_phrase") != phrase:
                raise ValueError("legacy answer phrase/span mismatch")
            row = {
                "request_id": source["candidate_id"],
                "scene_token": source["scene_token"],
                "sample_data_token": source["sample_data_token"],
                "split": source["split"],
                "question": source["question"],
                "answer": source["answer"],
                "answer_source": source["answer_source"],
                "answer_model_id": None,
                "targets": [
                    {
                        "span": [start, end],
                        "status": "matched",
                        "instance_token": source["selected_instance_token"],
                    }
                ],
                "reviews": [review],
                "source_fingerprint": source["candidate_fingerprint"],
                "spatial_reference": {
                    "point_file": source["point_file"],
                    "mask_file": source["mask_file"],
                },
            }
        elif source.get("schema_version") == SCHEMA + "-generated":
            review = review_index.get(source["request_id"], {})
            validate_review(review)
            fingerprinted = {
                key: value
                for key, value in source.items()
                if key != "record_fingerprint"
            }
            if (
                source.get("record_fingerprint") != digest(fingerprinted)
                or review.get("record_fingerprint") != source["record_fingerprint"]
            ):
                raise ValueError("reviewed generated answer fingerprint mismatch")
            if not source.get("finished"):
                raise ValueError(
                    "truncated generated answers cannot become training supervision"
                )
            used_reviews.add(source["request_id"])
            row = {
                key: source[key]
                for key in (
                    "request_id",
                    "scene_token",
                    "sample_data_token",
                    "split",
                    "question",
                    "answer",
                )
            }
            row.update(
                answer_source="prediction",
                answer_model_id=source["answer_model_id"],
                targets=review.get("targets"),
                reviews=[review],
                source_fingerprint=source["record_fingerprint"],
            )
        else:
            raise ValueError(
                "unsupported source; require accepted bindings or collected B3 answers"
            )
        key = (row["scene_token"], row["sample_data_token"])
        if key not in index:
            raise ValueError("missing exact scene/frame-to-feature mapping")
        if (
            source.get("schema_version") == SCHEMA + "-generated"
            and index[key] != source["features"]
        ):
            raise ValueError(
                "training features differ from the reviewed B3 generation inputs"
            )
        row.update(
            schema_version=SCHEMA + "-reviewed",
            features=index[key],
            review_status="accepted",
            grounding_verified=True,
        )
        validate_request(row)
        mark_answer(row["answer"], row["targets"])
        if source.get("schema_version") == "b4dl-answer-binding-v1":
            group = digest(
                [
                    row["scene_token"],
                    row["sample_data_token"],
                    row["split"],
                    row["question"],
                    row["answer"],
                    row["answer_source"],
                ]
            )
            if group in legacy_groups:
                previous = legacy_groups[group]
                if previous["spatial_reference"] != row["spatial_reference"]:
                    raise ValueError(
                        "same-answer bindings refer to different spatial source files"
                    )
                previous["targets"].extend(row["targets"])
                previous["reviews"].extend(row["reviews"])
                previous["source_fingerprint"] = digest(
                    [previous["source_fingerprint"], row["source_fingerprint"]]
                )
                previous["request_id"] = digest(
                    [previous["request_id"], row["request_id"]]
                )
                mark_answer(previous["answer"], previous["targets"])
                continue
            legacy_groups[group] = row
        result.append(row)
    if set(review_index) != used_reviews:
        raise ValueError("review references an unknown/out-of-scope generated answer")
    if not result:
        raise ValueError(
            "no reviewed answer supervision; pending candidates are not trainable"
        )
    return result


def verify_feature_frames(records, metadata_root, version):
    """Reject feature context from another scene/split or a camera channel."""
    from grounded4d.prepare import table_rows

    root = Path(metadata_root)
    root = root if root.name == version else root / version
    needed = {token for row in records for token in row["features"]["frame_tokens"]}
    frames = {
        item["token"]: item
        for item in table_rows(root / "sample_data.json")
        if item["token"] in needed
    }
    if set(frames) != needed:
        raise ValueError("feature frame is missing from original sample_data metadata")
    sample_ids = {item["sample_token"] for item in frames.values()}
    samples = {
        item["token"]: item["scene_token"]
        for item in table_rows(root / "sample.json")
        if item["token"] in sample_ids
    }
    calibration_ids = {item["calibrated_sensor_token"] for item in frames.values()}
    calibrations = {
        item["token"]: item["sensor_token"]
        for item in table_rows(root / "calibrated_sensor.json")
        if item["token"] in calibration_ids
    }
    sensors = {
        item["token"]: item["channel"] for item in table_rows(root / "sensor.json")
    }
    for row in records:
        for token in row["features"]["frame_tokens"]:
            frame = frames[token]
            if (
                samples.get(frame["sample_token"]) != row["scene_token"]
                or sensors.get(calibrations.get(frame["calibrated_sensor_token"]))
                != "LIDAR_TOP"
            ):
                raise ValueError(
                    "feature input leaks another scene or is not LIDAR_TOP"
                )
    return {"verified_input_frames": len(needed), "same_scene_lidar_context": True}


class AnswerSegDataset:
    def __init__(self, manifest, spatial, feature_root):
        self.manifest, self.spatial, self.feature_root = (
            Path(manifest),
            spatial,
            Path(feature_root),
        )
        self.records = list(read_jsonl(manifest))
        self.lookup = {}
        for index, clip in enumerate(spatial.records):
            if len(clip["frames"]) != 1:
                raise ValueError(
                    "the first AnswerSeg interface accepts single-frame spatial outputs"
                )
            token = clip["frames"][0]["sample_data_token"]
            if token in self.lookup:
                raise ValueError("duplicate spatial frame")
            self.lookup[token] = index
        seen = set()
        if not self.records:
            raise ValueError("empty reviewed binding manifest")
        for row in self.records:
            validate_request(row)
            if (
                row.get("schema_version") != SCHEMA + "-reviewed"
                or row.get("review_status") != "accepted"
                or row.get("grounding_verified") is not True
            ):
                raise ValueError("pending/unreviewed answer bindings cannot be trained")
            for review in row.get("reviews", []):
                validate_review(review)
            if not row.get("reviews") or not row.get("source_fingerprint"):
                raise ValueError("missing answer/review provenance")
            if row.get("answer_source") not in (
                "dataset_ground_truth",
                "prediction",
            ):
                raise ValueError("unknown answer source")
            if row["answer_source"] == "prediction" and not row.get("answer_model_id"):
                raise ValueError("generated answer model identity is required")
            if row["request_id"] in seen:
                raise ValueError("duplicate binding request")
            seen.add(row["request_id"])
            if row["sample_data_token"] not in self.lookup:
                raise ValueError("binding has no audited spatial frame")
            clip = spatial.records[self.lookup[row["sample_data_token"]]]
            if (
                row["split"] != clip["split"]
                or row["scene_token"] != clip["scene_token"]
            ):
                raise ValueError("binding scene/split disagrees with spatial data")
            reference = row.get("spatial_reference")
            if reference and any(
                reference[key] != clip["frames"][0][key]
                for key in ("point_file", "mask_file")
            ):
                raise ValueError("legacy binding references different point/mask files")
            mark_answer(row["answer"], row["targets"])

    @property
    def scenes(self):
        return {row["scene_token"] for row in self.records}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        frame = self.spatial[self.lookup[row["sample_data_token"]]]["frames"][0]
        for target in row["targets"]:
            if (
                target["status"] == "matched"
                and target["instance_token"] not in frame["instance_tokens"]
            ):
                raise ValueError(
                    "reviewed target is missing from the audited frame labels"
                )
        return dict(
            row,
            prompt=make_prompt(row["question"]),
            features=load_features(self.feature_root, row),
            frame=frame,
        )

    def preflight(self):
        positives = negatives = 0
        for index in range(len(self)):
            item = self[index]
            for target in item["targets"]:
                positives += target["status"] == "matched"
                negatives += target["status"] == "no_target"
        return {
            "records": len(self),
            "scenes": sorted(self.scenes),
            "positive_targets": positives,
            "explicit_no_targets": negatives,
            "feature_encoder_ids": sorted(
                {row["features"]["encoder_checkpoint_id"] for row in self.records}
            ),
            "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "interface_validated": True,
            "language_grounding_accepted": False,
            "supervision_sources": dict(
                Counter(
                    row.get("supervision_kind", "human_reviewed_binding")
                    for row in self.records
                )
            ),
        }
