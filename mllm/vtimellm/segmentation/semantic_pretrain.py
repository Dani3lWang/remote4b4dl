"""nuScenes lidarseg data and model pieces for spatial-encoder pretraining."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .config import ReasonSegConfig
from .spatial_encoder import SparseUNetPointEncoder


class NuScenesLidarsegDataset(Dataset):
    """Read key-frame point clouds and lidarseg labels by official scene split."""

    def __init__(
        self,
        dataroot: str,
        *,
        split: str,
        version: str = "v1.0-trainval",
        exclude_scenes: Optional[str] = None,
        max_samples: int = 0,
    ):
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes

        self.dataroot = Path(dataroot).resolve()
        nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=False)
        if not getattr(nusc, "lidarseg", None):
            raise RuntimeError("nuScenes lidarseg metadata is unavailable")
        split_key = f"mini_{split}" if "mini" in version else split
        scene_names = set(create_splits_scenes()[split_key])
        excluded = _load_excluded_scenes(exclude_scenes)
        scenes = {record["token"]: record for record in nusc.scene}
        records = []
        for sample in nusc.sample:
            scene = scenes[sample["scene_token"]]
            if scene["name"] not in scene_names:
                continue
            if scene["name"] in excluded or scene["token"] in excluded:
                continue
            lidar_token = sample["data"]["LIDAR_TOP"]
            sample_data = nusc.get("sample_data", lidar_token)
            lidarseg = nusc.get("lidarseg", lidar_token)
            records.append(
                (
                    sample["token"],
                    self.dataroot / sample_data["filename"],
                    self.dataroot / lidarseg["filename"],
                )
            )
        if max_samples:
            records = records[:max_samples]
        if not records:
            raise ValueError(f"no lidarseg samples found for {split_key}")
        self.records = records
        mapping = getattr(nusc, "lidarseg_name2idx_mapping", None) or {}
        self.num_classes = max((int(value) for value in mapping.values()), default=31) + 1

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        sample_token, point_path, label_path = self.records[index]
        points = np.fromfile(point_path, dtype=np.float32)
        if points.size % 5:
            raise RuntimeError(f"invalid nuScenes point file: {point_path}")
        points = points.reshape(-1, 5)[:, :4]
        labels = np.fromfile(label_path, dtype=np.uint8).astype(np.int64)
        if len(points) != len(labels):
            raise RuntimeError(
                f"point/lidarseg mismatch for {sample_token}: {len(points)} != {len(labels)}"
            )
        return {
            "sample_token": sample_token,
            "points": torch.from_numpy(points.copy()),
            "labels": torch.from_numpy(labels),
        }


class LidarsegCollator:
    def __call__(self, samples: Sequence[dict]):
        counts = [len(sample["points"]) for sample in samples]
        return {
            "points": torch.cat([sample["points"] for sample in samples], dim=0),
            "labels": torch.cat([sample["labels"] for sample in samples], dim=0),
            "point_batch_indices": torch.cat(
                [
                    torch.full((count,), batch_index, dtype=torch.long)
                    for batch_index, count in enumerate(counts)
                ],
                dim=0,
            ),
            "sample_tokens": [sample["sample_token"] for sample in samples],
        }


@dataclass
class SpatialPretrainOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    valid_mask: torch.Tensor


class SpatialPretrainModel(nn.Module):
    def __init__(
        self,
        config: ReasonSegConfig,
        num_classes: int,
        *,
        ignore_label: int = 0,
        point_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.point_encoder = point_encoder or SparseUNetPointEncoder(config)
        self.classifier = nn.Linear(config.point_feature_dim, num_classes)
        self.num_classes = num_classes
        self.ignore_label = ignore_label

    def forward(self, *, points, point_batch_indices, labels) -> SpatialPretrainOutput:
        encoding = self.point_encoder(points, point_batch_indices)
        logits = self.classifier(encoding.point_features)
        valid = (
            encoding.point_valid_mask
            & labels.ne(self.ignore_label)
            & labels.ge(0)
            & labels.lt(self.num_classes)
        )
        if not valid.any():
            raise RuntimeError("lidarseg batch has no valid in-range labeled points")
        loss = F.cross_entropy(logits[valid].float(), labels[valid].long())
        return SpatialPretrainOutput(loss, logits, labels, valid)


def confusion_matrix(output: SpatialPretrainOutput, num_classes: int) -> torch.Tensor:
    predictions = output.logits[output.valid_mask].argmax(dim=-1)
    labels = output.labels[output.valid_mask].long()
    values = labels * num_classes + predictions
    return torch.bincount(values, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def semantic_metrics(confusion: torch.Tensor) -> dict[str, float]:
    confusion = confusion.double()
    intersection = confusion.diag()
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
    present = union > 0
    iou = torch.zeros_like(union)
    iou[present] = intersection[present] / union[present]
    accuracy = intersection.sum() / confusion.sum().clamp_min(1.0)
    return {
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        "point_accuracy": float(accuracy),
    }


def _load_excluded_scenes(path: Optional[str]) -> set[str]:
    if not path:
        return set()
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(values, dict):
        values = values.get("scene_tokens") or values.get("scenes") or []
    return {str(value) for value in values}
