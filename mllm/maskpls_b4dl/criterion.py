"""MaskPLS set loss adapted for complete, tiny-instance-preserving nuScenes masks.

Derived from PRBonn/MaskPLS (MIT; see NOTICE.md). GT is used here only.
"""
import torch
from torch.nn import functional as F

from grounded4d.criterion import hungarian
from .config import NO_OBJECT
from .data import build_targets


def mask_binary_cross_entropy(prediction, truth, reduction="point_mean"):
    """Balance positive/negative means per matched instance, without removing negatives."""
    if reduction == "point_mean":
        return F.binary_cross_entropy_with_logits(prediction, truth)
    if reduction != "instance_balanced":
        raise ValueError("unknown mask BCE reduction")
    if prediction.ndim != 2 or prediction.shape != truth.shape or not len(truth):
        raise ValueError("balanced mask BCE requires nonempty [instances, points] targets")
    if not ((truth == 0) | (truth == 1)).all():
        raise ValueError("balanced mask BCE expects binary sampled targets")
    positive = truth.eq(1)
    positive_count = positive.sum(1)
    if (positive_count == 0).any():
        raise ValueError("balanced mask BCE requires positive support for every instance")
    negative = ~positive
    negative_count = negative.sum(1)
    losses = F.binary_cross_entropy_with_logits(prediction, truth, reduction="none")
    positive_mean = (losses * positive).sum(1) / positive_count
    negative_mean = (losses * negative).sum(1) / negative_count.clamp_min(1)
    # An all-positive sampled target has one nonempty group, not half a loss.
    return ((positive_mean + negative_mean) / (1 + negative_count.gt(0).to(losses.dtype))).mean()


def sample_supervised_points(labels, valid, budget, per_instance):
    """Shared sample with guaranteed foreground support for every supervised mask."""
    available = valid.nonzero(as_tuple=True)[0]
    if not len(available):
        raise ValueError("frame has no supervised points")
    if len(available) <= budget:
        return available
    if budget < len(labels):
        raise ValueError("point budget cannot cover all target instances")
    count = min(per_instance, max(1, budget // max(2, 2 * len(labels))))
    chosen = []
    for target in labels:
        positive = ((target == 1) & valid).nonzero(as_tuple=True)[0]
        if not len(positive):
            raise ValueError("target has no foreground; do not convert it to NOOBJ")
        chosen.append(positive[torch.randperm(len(positive), device=positive.device)[:count]])
    selected = torch.unique(torch.cat(chosen)) if chosen else available[:0]
    remaining = available[~torch.isin(available, selected)]
    fill = remaining[torch.randperm(len(remaining), device=remaining.device)[:budget - len(selected)]]
    return torch.cat((selected, fill))


def lovasz_softmax(logits, labels):
    """Present-class Lovasz-Softmax; ignore points are removed by the caller."""
    if not len(labels):
        return logits.sum() * 0
    probability = logits.softmax(-1)
    losses = []
    for category in labels.unique().tolist():
        foreground = labels.eq(category).to(probability.dtype)
        errors, order = (foreground - probability[:, category]).abs().sort(descending=True)
        sorted_fg = foreground[order]
        total = sorted_fg.sum()
        intersection = total - sorted_fg.cumsum(0)
        union = total + (1 - sorted_fg).cumsum(0)
        gradient = 1 - intersection / union.clamp_min(1)
        gradient = torch.cat((gradient[:1], gradient[1:] - gradient[:-1]))
        losses.append(torch.dot(errors, gradient))
    return torch.stack(losses).mean()


class MaskPLSCriterion:
    def __init__(self, config):
        self.config = config

    @torch.no_grad()
    def match(self, output, labels, classes, indices):
        if len(labels) > len(output["mask_logits"]):
            raise ValueError("query capacity below target count")
        if not len(labels):
            return []
        prediction = output["mask_logits"][:, indices].float()
        truth = labels[:, indices].float()
        positive, negative = F.softplus(-prediction), F.softplus(prediction)
        bce = (positive @ truth.T + negative @ (1 - truth).T) / len(indices)
        probability = prediction.sigmoid()
        dice = 1 - (2 * probability @ truth.T + 1) / (
            probability.sum(1)[:, None] + truth.sum(1)[None] + 1)
        classification = output["class_logits"].float().softmax(-1)[:, classes]
        cost = 5 * bce + 5 * dice - 2 * classification
        return hungarian(cost.T.cpu().numpy())

    def layer_loss(self, output, labels, classes, indices, assignment):
        target_classes = torch.full((len(output["class_logits"]),), NO_OBJECT,
                                    dtype=torch.long, device=classes.device)
        if assignment:
            target_classes[assignment] = classes
        weights = output["class_logits"].new_ones(NO_OBJECT + 1)
        weights[-1] = 0.1
        classification = F.cross_entropy(output["class_logits"].float(), target_classes, weight=weights.float())
        zero = output["mask_logits"].sum() * 0
        if assignment:
            prediction = output["mask_logits"][assignment][:, indices].float()
            truth = labels[:, indices].float()
            bce = mask_binary_cross_entropy(prediction, truth, self.config.mask_bce_reduction)
            probability = prediction.sigmoid()
            dice = (1 - (2 * (probability * truth).sum(1) + 1) /
                    (probability.sum(1) + truth.sum(1) + 1)).mean()
        else:
            bce, dice = zero, zero
        return dict(classification=classification, mask_bce=bce, mask_dice=dice)

    def __call__(self, output, frame):
        labels, classes, semantic_valid = build_targets(frame, output["valid_points"])
        mask_valid = semantic_valid
        if self.config.mask_ignore_policy == "background":
            # Upstream binary instance masks contain zero at ignored semantic points.
            # They are not any admitted Thing/Stuff instance, but remain in the input
            # and attention. Excluding them from every mask loss leaves unconstrained
            # foreground (notably ego returns). Do not change semantic supervision,
            # evaluation eligibility, raw point order, or label-free inference.
            labels = labels.clamp_min(0)
            mask_valid = output["valid_points"]
        match_indices = sample_supervised_points(labels, mask_valid, self.config.match_points,
                                                  self.config.points_per_instance)
        loss_indices = sample_supervised_points(labels, mask_valid, self.config.loss_points,
                                                 self.config.points_per_instance)
        parts = {}
        assignment = None
        # Every decoder output, including initial query prediction, is supervised.
        # Match each layer independently, matching the upstream training design.
        for index, layer in enumerate(output["auxiliary"] + [output]):
            assignment = self.match(layer, labels, classes, match_indices)
            current = self.layer_loss(layer, labels, classes, loss_indices, assignment)
            for key, value in current.items():
                parts[key] = parts.get(key, 0) + value
        # Upstream sums auxiliary layer losses; do not silently average them away.
        semantic = output["semantic_logits"][semantic_valid].float()
        semantic_labels = frame["semantic_labels"][semantic_valid]
        parts["semantic_ce"] = F.cross_entropy(semantic, semantic_labels)
        parts["semantic_lovasz"] = lovasz_softmax(semantic, semantic_labels)
        total = (2 * parts["classification"] + 5 * parts["mask_bce"] + 5 * parts["mask_dice"]
                 + 2 * parts["semantic_ce"] + 6 * parts["semantic_lovasz"])
        return total, parts, assignment
