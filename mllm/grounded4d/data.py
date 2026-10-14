"""Reviewed clip manifests; preserve raw point order and refuse label promotion."""
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset


SCHEMA = "b4dl-grounded4d-v1"
LABEL_KINDS = {"panoptic": "panoptic_gt", "box-pseudo": "box_geometry_pseudo_mask"}


def read_hashed_file(root, entry):
    """Hash and decode the same bytes, including symlink/absolute-path checks."""
    root = Path(root).resolve()
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"file escapes declared data root: {path}")
    expected = entry.get("sha256", "")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"missing/invalid SHA-256 for {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return raw


def load_scene_tokens(path):
    tokens = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) or not t for t in tokens):
        raise ValueError("scene whitelist must be a nonempty JSON string array")
    return set(tokens)


class Grounded4DDataset(Dataset):
    def __init__(self, manifest, nuscenes_root, label_root, config,
                 scene_whitelist, split="train", supervision="panoptic"):
        config.validate()
        if split not in ("train", "internal_dev"):
            raise ValueError("training accepts only train/internal_dev from official train scenes")
        if supervision not in LABEL_KINDS:
            raise ValueError("unknown supervision mode")
        self.manifest = Path(manifest).resolve()
        self.manifest_sha256 = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.nuscenes_root, self.label_root = Path(nuscenes_root), Path(label_root)
        self.config, self.supervision = config, supervision
        self.records = []
        seen = set()
        for line_number, line in enumerate(self.manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != SCHEMA:
                raise ValueError(f"line {line_number}: expected {SCHEMA}, not a candidate/pseudo-mask audit")
            clip_id = row.get("clip_id")
            if not isinstance(clip_id, str) or not clip_id or clip_id in seen:
                raise ValueError("clip IDs must be nonempty and unique")
            seen.add(clip_id)
            if row.get("split") != split or row.get("scene_token") not in scene_whitelist:
                raise ValueError(f"{clip_id}: split mismatch or non-training scene")
            if row.get("label_kind") != LABEL_KINDS[supervision]:
                raise ValueError(f"{clip_id}: label kind does not match explicit supervision mode")
            if row.get("annotation_scope") != "all_configured_instances":
                raise ValueError("set prediction needs complete frame-instance supervision, not QA-selected candidates")
            frames = row.get("frames")
            if not isinstance(frames, list) or not frames:
                raise ValueError(f"{clip_id}: no frames")
            if row.get("coordinate_frame") != "anchor_lidar":
                raise ValueError("all frames must declare a common anchor_lidar coordinate system")
            if len(frames) > 1 and row.get("track_verified") is not True:
                raise ValueError("multi-frame clips require reviewed cross-frame instance identity")
            timestamps = [frame.get("timestamp_us") for frame in frames]
            if any(type(t) is not int or t < 0 for t in timestamps):
                raise ValueError("timestamps must be nonnegative integer microseconds")
            if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
                raise ValueError("frames must be in strictly increasing time order")
            frame_ids = [frame.get("sample_data_token") for frame in frames]
            if any(not isinstance(t, str) or not t for t in frame_ids) or len(set(frame_ids)) != len(frame_ids):
                raise ValueError("LiDAR frame tokens must be nonempty and unique within a clip")
            self.records.append(row)
        if not self.records:
            raise ValueError("empty training manifest; zero candidates cannot start training")

    @property
    def scenes(self):
        return {row["scene_token"] for row in self.records}

    def __len__(self):
        return len(self.records)

    def _frame(self, row, frame):
        raw = read_hashed_file(self.nuscenes_root, frame["point_file"])
        if not raw or len(raw) % 20:
            raise ValueError("LiDAR must be nonempty original float32 XYZ/intensity/ring")
        points = np.frombuffer(raw, dtype="<f4").reshape(-1, 5).copy()
        if not np.isfinite(points).all():
            raise ValueError("nonfinite LiDAR values")
        ring = points[:, 4]
        if ((ring < 0) | (ring > 31) | (ring != np.floor(ring))).any():
            raise ValueError("expected original nuScenes HDL-32 ring values")
        transform = np.asarray(frame.get("lidar_to_anchor"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("lidar_to_anchor must be an explicit finite 4x4 rigid transform")
        rotation = transform[:3, :3]
        if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
            raise ValueError("lidar_to_anchor is not a rigid rotation/translation")
        points[:, :3] = points[:, :3] @ rotation.T + transform[:3, 3]
        if not np.isfinite(points).all():
            raise ValueError("coordinate transform produced nonfinite LiDAR values")
        raw_masks = read_hashed_file(self.label_root, frame["mask_file"])
        with np.load(io.BytesIO(raw_masks), allow_pickle=False) as archive:
            labels = archive["labels"].copy()
            tokens = archive["instance_tokens"].tolist()
            if str(archive["label_kind"].item()) != row["label_kind"]:
                raise ValueError("NPZ label kind disagrees with manifest; pseudo masks cannot be promoted")
            if str(archive["point_file_sha256"].item()) != frame["point_file"]["sha256"]:
                raise ValueError("masks do not correspond to this raw point file")
            if not np.array_equal(archive["point_indices"], np.arange(len(points))):
                raise ValueError("mask point order differs from original LiDAR rows")
            if self.supervision == "panoptic":
                if frame.get("mask_verified") is not True or archive["mask_verified"].item() is not True:
                    raise ValueError("panoptic supervision requires independently verified point masks")
        classes = np.asarray(frame.get("class_ids"))
        if (labels.ndim != 2 or labels.shape != (len(tokens), len(points))
                or labels.dtype.kind not in "ib" or not np.isin(labels, [-1, 0, 1]).all()):
            raise ValueError("labels must be integer [instances, original_points] with values -1/0/1")
        if (len(set(tokens)) != len(tokens) or any(not isinstance(t, str) or not t for t in tokens)):
            raise ValueError("instance_tokens must identify unique instances, not panoptic numeric IDs")
        if classes.shape != (len(tokens),) or (len(tokens) and classes.dtype.kind not in "iu"):
            raise ValueError("class_ids must be one integer per instance")
        if ((classes < 0) | (classes >= len(self.config.class_names))).any():
            raise ValueError("class_ids outside the configured class mapping")
        if self.supervision == "panoptic" and ((labels == 1).sum(0) > 1).any():
            raise ValueError("panoptic instance masks overlap")
        bounds = np.asarray(self.config.point_cloud_range)
        valid = ((points[:, :3] >= bounds[:3]) & (points[:, :3] < bounds[3:])).all(1)
        if not valid.any() or (len(tokens) and not ((labels[:, valid] == 1).any(1)).all()):
            raise ValueError("no in-range points for frame or supervised instance")
        references = frame.get("references")
        reference_features, reference_tokens = None, []
        if references is not None:
            if not isinstance(references, dict):
                raise ValueError("references must be a reviewed correspondence object")
            if references.get("grounding_verified") is not True:
                raise ValueError("reference loss requires reviewed phrase-to-instance correspondence")
            if references.get("answer_source") not in ("ground_truth", "prediction"):
                raise ValueError("reference embeddings must record ground_truth/prediction answer source")
            for field in ("question", "answer", "encoder_checkpoint_id"):
                if not isinstance(references.get(field), str) or not references[field]:
                    raise ValueError(f"references must record {field}")
            features_raw = read_hashed_file(self.label_root, references["feature_file"])
            reference_features = np.load(io.BytesIO(features_raw), allow_pickle=False)
            targets = references.get("targets", [])
            if (reference_features.shape != (len(targets), self.config.reference_dim)
                    or not np.isfinite(reference_features).all()):
                raise ValueError("reference embeddings have invalid shape/values")
            for target in targets:
                token, status = target.get("instance_token"), target.get("status")
                if not isinstance(target.get("phrase"), str) or not target["phrase"]:
                    raise ValueError("reference targets must retain their original phrase text")
                if status == "matched" and token in tokens:
                    reference_tokens.append(token)
                elif status == "no_target" and token is None:
                    reference_tokens.append(None)
                else:
                    raise ValueError("ambiguous/occluded references cannot be silently labeled NOOBJ")
            reference_features = torch.as_tensor(reference_features.copy(), dtype=torch.float32)
        return dict(points=torch.from_numpy(points[:, :4].copy()),
                    labels=torch.as_tensor(labels, dtype=torch.float32),
                    class_ids=torch.as_tensor(classes, dtype=torch.long),
                    instance_tokens=tokens, timestamp_us=frame["timestamp_us"],
                    sample_data_token=frame["sample_data_token"],
                    reference_features=reference_features, reference_tokens=reference_tokens)

    def __getitem__(self, index):
        row = self.records[index]
        return dict(clip_id=row["clip_id"], scene_token=row["scene_token"],
                    frames=[self._frame(row, frame) for frame in row["frames"]])

    def preflight(self):
        frame_count, instance_count, reference_count = 0, 0, 0
        encoder_ids = set()
        max_instances, max_detection_targets = 0, 0
        versions = {row.get("dataset_version", "unspecified") for row in self.records}
        if len(versions) != 1:
            raise ValueError("dataset versions cannot be mixed in one manifest")
        for index, row in enumerate(self.records):
            clip = self[index]
            class_per_token = {}
            last_seen = {}
            for frame, source in zip(clip["frames"], row["frames"]):
                frame_count += 1
                instance_count += len(frame["instance_tokens"])
                max_instances = max(max_instances, len(frame["instance_tokens"]))
                new_targets = sum(token not in last_seen or
                                  (frame["timestamp_us"] - last_seen[token]) / 1e6 > self.config.track_lifetime_seconds
                                  for token in frame["instance_tokens"])
                max_detection_targets = max(max_detection_targets, new_targets)
                last_seen.update({token: frame["timestamp_us"] for token in frame["instance_tokens"]})
                reference_count += len(frame["reference_tokens"])
                if source.get("references"):
                    encoder_ids.add(source["references"]["encoder_checkpoint_id"])
                for token, class_id in zip(frame["instance_tokens"], frame["class_ids"].tolist()):
                    if token in class_per_token and class_per_token[token] != class_id:
                        raise ValueError("instance class changes within a clip")
                    class_per_token[token] = class_id
        if len(encoder_ids) > 1:
            raise ValueError("reference embeddings from different encoder checkpoints cannot be mixed")
        return dict(clips=len(self), scenes=len(self.scenes), frames=frame_count,
                    dataset_version=next(iter(versions)),
                    max_instances_per_frame=max_instances, max_detection_targets=max_detection_targets,
                    instances=instance_count, references=reference_count,
                    reference_encoder_ids=sorted(encoder_ids), manifest_sha256=self.manifest_sha256,
                    label_kind=LABEL_KINDS[self.supervision])
