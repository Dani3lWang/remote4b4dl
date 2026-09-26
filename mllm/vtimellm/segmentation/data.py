"""Manifest schema and batching for nuScenes reasoning segmentation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from vtimellm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from vtimellm.conversation import conv_templates
from vtimellm.mm_utils import tokenizer_image_token

from .config import ReasonSegConfig


@dataclass(frozen=True)
class TargetInstance:
    panoptic_id: int
    class_id: int
    class_name: str


@dataclass(frozen=True)
class ReasonSegRecord:
    sample_token: str
    scene_token: str
    split: str
    lidar_path: str
    panoptic_path: str
    query: str
    answer: str
    targets: tuple[TargetInstance, ...]

    @classmethod
    def from_dict(cls, values: Dict[str, object]) -> "ReasonSegRecord":
        return cls(
            sample_token=str(values["sample_token"]),
            scene_token=str(values["scene_token"]),
            split=str(values["split"]),
            lidar_path=str(values["lidar_path"]),
            panoptic_path=str(values["panoptic_path"]),
            query=str(values["query"]),
            answer=str(values["answer"]),
            targets=tuple(TargetInstance(**target) for target in values.get("targets", [])),
        )


class ReasonSegDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        *,
        dataroot: Optional[str] = None,
        config: Optional[ReasonSegConfig] = None,
        require_reachable: bool = False,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.dataroot = Path(dataroot).resolve() if dataroot else self.manifest_path.parent
        self.config = config or ReasonSegConfig()
        self.require_reachable = require_reachable
        self.records = _load_jsonl(self.manifest_path)
        if not self.records:
            raise ValueError(f"empty ReasonSeg manifest: {self.manifest_path}")
        for index, record in enumerate(self.records):
            if len(record.targets) > self.config.max_objects:
                raise ValueError(
                    f"record {index} has {len(record.targets)} targets; maximum is "
                    f"{self.config.max_objects}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        lidar_path = _resolve(self.dataroot, record.lidar_path)
        panoptic_path = _resolve(self.dataroot, record.panoptic_path)
        points = np.fromfile(lidar_path, dtype=np.float32)
        if points.size % 5:
            raise RuntimeError(f"invalid nuScenes point file: {lidar_path}")
        points = points.reshape(-1, 5)[:, :4]
        with np.load(panoptic_path) as values:
            if "data" not in values:
                raise RuntimeError(f"panoptic archive has no 'data' array: {panoptic_path}")
            panoptic = np.asarray(values["data"]).reshape(-1)
        if panoptic.shape[0] != points.shape[0]:
            raise RuntimeError(
                f"point/panoptic length mismatch for {record.sample_token}: "
                f"{points.shape[0]} != {panoptic.shape[0]}"
            )

        object_count = max(1, len(record.targets))
        masks = np.zeros((object_count, points.shape[0]), dtype=np.bool_)
        loc_masks = np.zeros_like(masks)
        classes = np.zeros((object_count,), dtype=np.int64)
        object_valid = np.zeros((object_count,), dtype=np.bool_)
        xyz = points[:, :3]
        for target_index, target in enumerate(record.targets):
            mask = panoptic == target.panoptic_id
            if not mask.any():
                raise RuntimeError(
                    f"target panoptic id {target.panoptic_id} is absent in {record.sample_token}"
                )
            if self.require_reachable:
                lower, upper = self.config.point_cloud_range[:3], self.config.point_cloud_range[3:]
                inside = ((xyz[mask] >= lower) & (xyz[mask] < upper)).all(axis=1)
                if not inside.any():
                    raise ValueError(
                        f"unreachable target {target.panoptic_id} in {record.sample_token}; "
                        "prepare_audited_manifests.py must filter the whole record before training"
                    )
            center = xyz[mask].mean(axis=0)
            radius = np.linalg.norm(xyz[mask] - center, axis=1).max()
            loc_mask = np.linalg.norm(xyz - center, axis=1) <= (
                radius + self.config.coarse_radius_margin_m
            )
            masks[target_index] = mask
            loc_masks[target_index] = loc_mask
            classes[target_index] = target.class_id
            object_valid[target_index] = True
        return {
            "sample_token": record.sample_token,
            "query": record.query,
            "answer": record.answer,
            "points": torch.from_numpy(points.copy()),
            "target_masks": torch.from_numpy(masks),
            "target_loc_masks": torch.from_numpy(loc_masks),
            "target_classes": torch.from_numpy(classes),
            "object_valid_mask": torch.from_numpy(object_valid),
        }


class ReasonSegCollator:
    def __init__(self, tokenizer, *, model_max_length: int = 2048, template: str = "v1"):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.template = template

    def __call__(self, samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
        tokenized = [self._tokenize(sample["query"], sample["answer"]) for sample in samples]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [value[0] for value in tokenized],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )[:, : self.model_max_length]
        labels = torch.nn.utils.rnn.pad_sequence(
            [value[1] for value in tokenized],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        point_counts = [sample["points"].shape[0] for sample in samples]
        points = torch.cat([sample["points"] for sample in samples], dim=0)
        point_batch_indices = torch.cat(
            [
                torch.full((count,), index, dtype=torch.long)
                for index, count in enumerate(point_counts)
            ],
            dim=0,
        )
        max_points = max(point_counts)
        max_objects = max(sample["target_masks"].shape[0] for sample in samples)
        target_masks = torch.zeros(
            len(samples), max_objects, max_points, dtype=torch.bool
        )
        target_loc_masks = torch.zeros_like(target_masks)
        target_classes = torch.zeros(len(samples), max_objects, dtype=torch.long)
        object_valid = torch.zeros(len(samples), max_objects, dtype=torch.bool)
        for batch_index, sample in enumerate(samples):
            object_count, point_count = sample["target_masks"].shape
            target_masks[batch_index, :object_count, :point_count] = sample["target_masks"]
            target_loc_masks[batch_index, :object_count, :point_count] = sample[
                "target_loc_masks"
            ]
            target_classes[batch_index, :object_count] = sample["target_classes"]
            object_valid[batch_index, :object_count] = sample["object_valid_mask"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "points": points,
            "point_batch_indices": point_batch_indices,
            "target_masks": target_masks,
            "target_loc_masks": target_loc_masks,
            "target_classes": target_classes,
            "object_valid_mask": object_valid,
            "sample_tokens": [sample["sample_token"] for sample in samples],
        }

    def _tokenize(self, query: str, answer: str) -> tuple[torch.Tensor, torch.Tensor]:
        conversation = conv_templates[self.template].copy()
        user_message = "<4DLiDAR>\n<video>\n" + str(query).strip()
        conversation.append_message(conversation.roles[0], user_message)
        conversation.append_message(conversation.roles[1], None)
        prompt_only = conversation.get_prompt()
        conversation.messages[-1][-1] = str(answer).strip()
        full_prompt = conversation.get_prompt()
        full_ids = tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        prompt_ids = tokenizer_image_token(
            prompt_only,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        labels = full_ids.clone()
        labels[: min(len(prompt_ids), len(labels))] = IGNORE_INDEX
        return full_ids, labels


def _load_jsonl(path: Path) -> List[ReasonSegRecord]:
    records: List[ReasonSegRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(ReasonSegRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise RuntimeError(f"invalid manifest line {line_number} in {path}: {exc}") from exc
    return records


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
