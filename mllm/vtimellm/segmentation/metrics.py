"""Metrics that count misses, extra masks and token failures."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class SegmentationMetricAccumulator:
    num_classes: int
    sample_count: int = 0
    cumulative_intersection: float = 0.0
    cumulative_union: float = 0.0
    matched_ious: List[float] = field(default_factory=list)
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    count_correct: int = 0
    no_object_samples: int = 0
    no_object_false_positive: int = 0
    token_failures: int = 0
    confusion: np.ndarray = field(init=False)

    def __post_init__(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(
        self,
        predicted_masks: np.ndarray,
        predicted_classes: np.ndarray,
        target_masks: np.ndarray,
        target_classes: np.ndarray,
        *,
        token_status: str = "ok",
        iou_match_threshold: float = 0.5,
    ) -> None:
        predicted_masks = np.asarray(predicted_masks, dtype=bool)
        target_masks = np.asarray(target_masks, dtype=bool)
        predicted_classes = np.asarray(predicted_classes, dtype=np.int64)
        target_classes = np.asarray(target_classes, dtype=np.int64)
        self.sample_count += 1
        if token_status not in ("ok", "no_object"):
            self.token_failures += 1
        if len(predicted_masks) == len(target_masks):
            self.count_correct += 1
        if len(target_masks) == 0:
            self.no_object_samples += 1
            if len(predicted_masks):
                self.no_object_false_positive += 1

        ious, intersections, unions = pairwise_mask_iou(predicted_masks, target_masks)
        matches = optimal_iou_matching(ious)
        matched_predictions = set()
        matched_targets = set()
        for predicted_index, target_index in matches:
            iou = float(ious[predicted_index, target_index])
            self.matched_ious.append(iou)
            self.cumulative_intersection += intersections[predicted_index, target_index]
            self.cumulative_union += unions[predicted_index, target_index]
            if iou >= iou_match_threshold:
                self.true_positive += 1
                matched_predictions.add(predicted_index)
                matched_targets.add(target_index)
                predicted_class = int(predicted_classes[predicted_index])
                target_class = int(target_classes[target_index])
                if 0 <= predicted_class < self.num_classes and 0 <= target_class < self.num_classes:
                    self.confusion[target_class, predicted_class] += 1
            else:
                self.false_positive += 1
                self.false_negative += 1
        self.false_positive += len(predicted_masks) - len(matched_predictions)
        self.false_negative += len(target_masks) - len(matched_targets)

    def compute(self) -> Dict[str, float]:
        precision = self.true_positive / max(1, self.true_positive + self.false_positive)
        recall = self.true_positive / max(1, self.true_positive + self.false_negative)
        class_f1 = []
        for class_index in range(self.num_classes):
            true_positive = self.confusion[class_index, class_index]
            false_positive = self.confusion[:, class_index].sum() - true_positive
            false_negative = self.confusion[class_index, :].sum() - true_positive
            denominator = 2 * true_positive + false_positive + false_negative
            if denominator:
                class_f1.append(float(2 * true_positive / denominator))
        return {
            "cIoU": self.cumulative_intersection / max(1.0, self.cumulative_union),
            "gIoU": float(np.mean(self.matched_ious)) if self.matched_ious else 0.0,
            "instance_precision@0.5": precision,
            "instance_recall@0.5": recall,
            "count_accuracy": self.count_correct / max(1, self.sample_count),
            "no_object_false_positive_rate": self.no_object_false_positive
            / max(1, self.no_object_samples),
            "token_failure_rate": self.token_failures / max(1, self.sample_count),
            "class_macro_f1": float(np.mean(class_f1)) if class_f1 else 0.0,
        }


def pairwise_mask_iou(
    predicted_masks: np.ndarray,
    target_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted_masks = np.asarray(predicted_masks, dtype=bool)
    target_masks = np.asarray(target_masks, dtype=bool)
    point_count = 0
    if predicted_masks.ndim == 2:
        point_count = predicted_masks.shape[1]
    if target_masks.ndim == 2:
        point_count = target_masks.shape[1]
    predicted_masks = predicted_masks.reshape(len(predicted_masks), point_count)
    target_masks = target_masks.reshape(len(target_masks), point_count)
    intersections = np.logical_and(
        predicted_masks[:, None, :], target_masks[None, :, :]
    ).sum(axis=-1)
    unions = np.logical_or(
        predicted_masks[:, None, :], target_masks[None, :, :]
    ).sum(axis=-1)
    ious = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=unions > 0,
    )
    return ious, intersections, unions


def optimal_iou_matching(ious: np.ndarray) -> List[Tuple[int, int]]:
    """Exact maximum-IoU one-to-one matching for at most eight objects."""

    ious = np.asarray(ious, dtype=np.float64)
    predicted_count, target_count = ious.shape
    if not predicted_count or not target_count:
        return []
    transpose = predicted_count > target_count
    matrix = ious.T if transpose else ious
    row_count, column_count = matrix.shape

    @lru_cache(maxsize=None)
    def solve(row: int, used_columns: int):
        if row == row_count:
            return 0.0, ()
        best_score, best_pairs = solve(row + 1, used_columns)
        for column in range(column_count):
            if used_columns & (1 << column):
                continue
            score, pairs = solve(row + 1, used_columns | (1 << column))
            score += float(matrix[row, column])
            if score > best_score:
                best_score = score
                best_pairs = ((row, column),) + pairs
        return best_score, best_pairs

    _, pairs = solve(0, 0)
    if transpose:
        return [(column, row) for row, column in pairs]
    return list(pairs)
