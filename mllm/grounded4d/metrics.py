"""Class-aware set-mask metrics; missing objects and false positives count."""
import numpy as np

from .criterion import hungarian


def pairwise_iou(target_masks, predicted_masks):
    targets, predictions = np.asarray(target_masks, dtype=bool), np.asarray(predicted_masks, dtype=bool)
    if targets.ndim != 2 or predictions.ndim != 2 or targets.shape[1] != predictions.shape[1]:
        raise ValueError("instance masks must share a two-dimensional point space")
    intersection = targets.astype(np.float32) @ predictions.astype(np.float32).T
    union = targets.sum(1)[:, None] + predictions.sum(1)[None] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float64), where=union > 0)


def match_sets(iou, target_classes, predicted_classes, threshold=0.5):
    """Separate total-IoU matching from maximum-cardinality threshold matching."""
    iou = np.asarray(iou, dtype=np.float64)
    targets, predictions = np.asarray(target_classes), np.asarray(predicted_classes)
    if iou.shape != (len(targets), len(predictions)) or not np.isfinite(iou).all():
        raise ValueError("IoU matrix and class arrays disagree")
    if (iou < 0).any() or (iou > 1).any() or not 0 < threshold <= 1:
        raise ValueError("IoUs and matching threshold must lie in [0,1]")
    count, columns = iou.shape
    same_class = targets[:, None] == predictions[None]
    allowed = np.where(same_class, iou, 0.0)
    # One dummy column per target permits every GT to remain unmatched.
    cost = np.zeros((count, columns + count))
    cost[:, :columns] = -allowed
    assignment = hungarian(cost)
    gt_ious = np.zeros(count)
    for target, prediction in enumerate(assignment):
        if prediction < columns:
            gt_ious[target] = allowed[target, prediction]
    cost[:, :columns] = 1.0
    eligible = same_class & (iou >= threshold)
    # Cardinality takes precedence over total IoU; the fractional tie-break
    # across all pairs is less than one additional true-positive match.
    cost[:, :columns][eligible] = -(1.0 + iou[eligible] / (count + 1))
    assignment = hungarian(cost)
    pairs = [(target, prediction, float(iou[target, prediction]))
             for target, prediction in enumerate(assignment)
             if prediction < columns and eligible[target, prediction]]
    return gt_ious, pairs


def frame_counts(target_masks, target_classes, predicted_masks, predicted_classes, class_count, threshold=0.5):
    targets, predictions = np.asarray(target_classes), np.asarray(predicted_classes)
    iou = pairwise_iou(target_masks, predicted_masks)
    gt_ious, pairs = match_sets(iou, targets, predictions, threshold)
    geometry_ious, geometry_pairs = match_sets(iou, np.zeros(len(targets)), np.zeros(len(predictions)), threshold)
    classes = []
    for category in range(class_count):
        gt_count, pred_count = int((targets == category).sum()), int((predictions == category).sum())
        hits = sum(targets[target] == category for target, _, _ in pairs)
        classes.append(dict(gt=gt_count, predictions=pred_count, tp=int(hits),
                            fp=pred_count - int(hits), fn=gt_count - int(hits),
                            gt_iou_sum=float(gt_ious[targets == category].sum())))
    result = dict(gt=len(targets), predictions=len(predictions), tp=len(pairs),
                  fp=len(predictions) - len(pairs), fn=len(targets) - len(pairs),
                  gt_iou_sum=float(gt_ious.sum()), geometry_gt_iou_sum=float(geometry_ious.sum()),
                  geometry_tp=len(geometry_pairs), per_class=classes)
    return result, dict(gt_matched_ious=gt_ious.tolist(), iou50_matches=pairs)


class MetricTotals:
    def __init__(self, class_names):
        self.class_names = tuple(class_names)
        self.frames = 0
        self.totals = {key: 0 for key in ("gt", "predictions", "tp", "fp", "fn", "gt_iou_sum",
                                         "geometry_gt_iou_sum", "geometry_tp")}
        self.classes = [{key: 0 for key in ("gt", "predictions", "tp", "fp", "fn", "gt_iou_sum")}
                        for _ in class_names]

    def add(self, record):
        self.frames += 1
        for key in self.totals:
            self.totals[key] += record[key]
        for accumulated, current in zip(self.classes, record["per_class"]):
            for key in accumulated:
                accumulated[key] += current[key]

    @staticmethod
    def summarize(counts):
        gt, predictions, tp = counts["gt"], counts["predictions"], counts["tp"]
        return dict(gt_instances=gt, predicted_instances=predictions, true_positives=tp,
                    false_positives=counts["fp"], false_negatives=counts["fn"],
                    precision_iou50=tp / predictions if predictions else 0.0,
                    recall_iou50=tp / gt if gt else None,
                    f1_iou50=2 * tp / (gt + predictions) if gt + predictions else None,
                    mean_gt_mask_iou=counts["gt_iou_sum"] / gt if gt else None)

    def result(self):
        per_class = {name: self.summarize(counts) for name, counts in zip(self.class_names, self.classes)}
        supported = [value for value in per_class.values() if value["gt_instances"]]
        return dict(frames=self.frames, **self.summarize(self.totals), per_class=per_class,
                    macro_recall_iou50=float(np.mean([value["recall_iou50"] for value in supported])) if supported else None,
                    macro_mean_gt_mask_iou=float(np.mean([value["mean_gt_mask_iou"] for value in supported])) if supported else None,
                    geometry_only_diagnostic=dict(
                        recall_iou50=self.totals["geometry_tp"] / self.totals["gt"] if self.totals["gt"] else None,
                        mean_gt_mask_iou=self.totals["geometry_gt_iou_sum"] / self.totals["gt"] if self.totals["gt"] else None))
