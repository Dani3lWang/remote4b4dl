"""Scoring-only query diagnostics, never inputs to the model or standard predictions."""
from collections import Counter

import numpy as np

from .metrics import match_sets, pairwise_iou


REASONS = ("hit", "mask_below_iou", "no_object", "wrong_class", "score_filtered", "query_competition")


def query_diagnostics(gt_masks, gt_classes, masks, classes, scores, retained, class_count,
                      points, roi, evaluable, memory_points, threshold=0.5, memory_indices=None):
    gt_masks, masks = np.asarray(gt_masks, bool), np.asarray(masks, bool)
    gt_classes, classes = np.asarray(gt_classes), np.asarray(classes)
    scores, retained = np.asarray(scores), np.asarray(retained, bool)
    if len(classes) != len(masks) or scores.shape != classes.shape or retained.shape != classes.shape:
        raise ValueError("query diagnostic arrays disagree")
    iou = pairwise_iou(gt_masks[:, evaluable], masks[:, evaluable])
    gt_count, queries = iou.shape
    _, geometry_pairs = match_sets(iou, np.zeros(gt_count), np.zeros(queries), threshold)
    _, class_pairs = match_sets(iou, gt_classes, classes, threshold)
    _, retained_pairs = match_sets(iou[:, retained], gt_classes, classes[retained], threshold)
    hits = {target for target, _, _ in retained_pairs}
    indices = np.asarray(memory_indices) if memory_indices is not None else np.flatnonzero(roi)
    if memory_indices is None and len(indices) > memory_points:
        raise ValueError("provide exact memory indices from the forward device for subsampled memory")
    point_counts = gt_masks[:, evaluable].sum(1)
    rows = []
    for target in range(gt_count):
        good_geometry = iou[target] >= threshold
        good_class = good_geometry & (classes == gt_classes[target])
        if target in hits:
            reason = "hit"
        elif not good_geometry.any():
            reason = "mask_below_iou"
        elif not good_class.any():
            reason = "no_object" if (classes[good_geometry] == class_count).all() else "wrong_class"
        elif not (good_class & retained).any():
            reason = "score_filtered"
        else:
            reason = "query_competition"
        best = int(iou[target].argmax()) if queries else None
        size = int(point_counts[target])
        center = np.asarray(points)[gt_masks[target] & evaluable, :2].mean(0)
        same_class = iou[target, classes == gt_classes[target]]
        kept_class = iou[target, retained & (classes == gt_classes[target])]
        rows.append(dict(gt_index=target, class_id=int(gt_classes[target]), point_count=size,
                         point_bucket="1-5" if size <= 5 else "6-20" if size <= 20 else "21-100" if size <= 100 else "101+",
                         horizontal_centroid_distance=float(np.linalg.norm(center)),
                         memory_foreground_points=int(gt_masks[target, indices].sum()), reason=reason,
                         best_geometry_iou=float(iou[target, best]) if best is not None else 0.0,
                         best_geometry_query=best,
                         best_geometry_predicted_class=int(classes[best]) if best is not None else None,
                         best_geometry_score=float(scores[best]) if best is not None else None,
                         best_class_iou=float(same_class.max()) if len(same_class) else 0.0,
                         best_retained_class_iou=float(kept_class.max()) if len(kept_class) else 0.0))
    return dict(gt_instances=gt_count, total_queries=queries, retained_queries=int(retained.sum()),
                stage_distinct_hits=dict(all_query_geometry=len(geometry_pairs), all_query_class_aware=len(class_pairs),
                                         retained_class_aware=len(retained_pairs)),
                reasons={reason: sum(row["reason"] == reason for row in rows) for reason in REASONS}, gt=rows)


class QueryDiagnosticTotals:
    def __init__(self):
        self.frames, self.gt = 0, 0
        self.hits = Counter()
        self.reasons = Counter()
        self.buckets = {}

    def add(self, diagnostic):
        self.frames += 1
        self.gt += diagnostic["gt_instances"]
        self.hits.update(diagnostic["stage_distinct_hits"])
        self.reasons.update(diagnostic["reasons"])
        for row in diagnostic["gt"]:
            bucket = self.buckets.setdefault(row["point_bucket"], Counter())
            bucket["gt_instances"] += 1
            bucket[row["reason"]] += 1
            bucket["zero_memory_foreground"] += int(row["memory_foreground_points"] == 0)

    def result(self):
        return dict(role="scoring_only_all_query_diagnostic_not_standard_acceptance", frames=self.frames,
                    gt_instances=self.gt, stage_distinct_hits=dict(self.hits),
                    stage_recall={key: value / self.gt if self.gt else None for key, value in self.hits.items()},
                    reasons={key: self.reasons[key] for key in REASONS}, point_buckets=self.buckets,
                    definitions=dict(stage_hits="class_aware_or_geometry_only_maximum_cardinality_one_to_one",
                                     per_gt_best_iou="independent_query_maximum_may_reuse_queries_not_recall",
                                     reasons="disjoint_cascade_mask_class_score_competition; retained GT hits fixed by standard matching",
                                     memory_coverage="direct_GT_foreground_in_uniform_memory_only; not a causal attribution"))
