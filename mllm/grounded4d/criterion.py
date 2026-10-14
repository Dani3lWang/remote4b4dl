"""Fixed track correspondence, detection set matching and explicit null references."""
import numpy as np
import torch
from torch.nn import functional as F


def hungarian(cost):
    """Minimum-cost rectangular assignment (target rows <= query columns).

    Uses NumPy only; empty targets are valid, insufficient queries are an error.
    Returns one distinct query column for each target row.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError("assignment cost must be a finite matrix")
    rows, columns = cost.shape
    if rows > columns:
        raise ValueError("more target instances than available detection queries")
    u, v = np.zeros(rows + 1), np.zeros(columns + 1)
    p, way = np.zeros(columns + 1, dtype=int), np.zeros(columns + 1, dtype=int)
    for row in range(1, rows + 1):
        p[0], column = row, 0
        minimum = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[column] = True
            current_row = p[column]
            delta, following = np.inf, 0
            for candidate in range(1, columns + 1):
                if not used[candidate]:
                    value = cost[current_row - 1, candidate - 1] - u[current_row] - v[candidate]
                    if value < minimum[candidate]:
                        minimum[candidate], way[candidate] = value, column
                    if minimum[candidate] < delta:
                        delta, following = minimum[candidate], candidate
            u[p[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            column = following
            if p[column] == 0:
                break
        while column:
            following = way[column]
            p[column] = p[following]
            column = following
    result = np.empty(rows, dtype=int)
    for column in range(1, columns + 1):
        if p[column]:
            result[p[column] - 1] = column - 1
    return result.tolist()


LOSS_COMPONENTS = ("mask_class", "identity", "reference", "mask_bce", "mask_dice",
                   "classification", "foreground_bce", "background_bce")


def mask_losses(logits, labels, valid, bce_mode="mean", return_details=False):
    if bce_mode not in ("mean", "foreground_balanced"):
        raise ValueError("unknown mask BCE mode: " + bce_mode)
    usable = valid & (labels >= 0)
    if not usable.any():
        raise ValueError("mask has no supervised in-range points")
    prediction, target = logits[usable], labels[usable]
    terms = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
    foreground, background = target == 1, target == 0
    zero = prediction.sum() * 0
    foreground_bce = terms[foreground].mean() if foreground.any() else zero
    background_bce = terms[background].mean() if background.any() else zero
    if bce_mode == "mean":
        # Preserve the original reduction for the control recipe.
        bce = F.binary_cross_entropy_with_logits(prediction, target)
    else:
        present = int(foreground.any()) + int(background.any())
        bce = (foreground_bce + background_bce) / present
    probability = prediction.sigmoid()
    dice = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
    if return_details:
        return bce, dice, dict(foreground_bce=foreground_bce, background_bce=background_bce)
    return bce, dice


class Grounded4DCriterion:
    def __init__(self, config, bce_mode="mean"):
        self.config = config
        if bce_mode not in ("mean", "foreground_balanced"):
            raise ValueError("unknown mask BCE mode: " + bce_mode)
        self.bce_mode = bce_mode

    @torch.no_grad()
    def assignment(self, output, frame):
        tokens = frame["instance_tokens"]
        tracks = output["track_tokens"]
        mapping = {index: tracks.index(token) for index, token in enumerate(tokens) if token in tracks}
        remaining = [index for index in range(len(tokens)) if index not in mapping]
        if not remaining:
            return mapping
        logits = output["mask_logits"][len(tracks):]
        classes = output["class_logits"][len(tracks):].softmax(-1)
        indices = output["valid_points"].nonzero(as_tuple=True)[0]
        if len(indices) > self.config.match_points:
            indices = indices[torch.linspace(0, len(indices) - 1, self.config.match_points,
                                            device=indices.device).long()]
        costs = []
        for target_index in remaining:
            # Always retain target foreground in matching, even for tiny instances.
            positive = ((frame["labels"][target_index] == 1) & output["valid_points"]).nonzero(as_tuple=True)[0]
            selected = torch.unique(torch.cat((indices, positive)))
            label = frame["labels"][target_index, selected]
            selected, label = selected[label >= 0], label[label >= 0]
            probability = logits[:, selected].sigmoid()
            dice = 1 - (2 * (probability * label).sum(-1) + 1) / (probability.sum(-1) + label.sum() + 1)
            bce = F.binary_cross_entropy_with_logits(logits[:, selected], label.expand(len(logits), -1),
                                                    reduction="none").mean(-1)
            costs.append(self.config.mask_bce_weight * bce + self.config.mask_dice_weight * dice
                         - classes[:, frame["class_ids"][target_index]])
        chosen = hungarian(torch.stack(costs).cpu().numpy())
        mapping.update({target: query + len(tracks) for target, query in zip(remaining, chosen)})
        return mapping

    def _mask_class_loss(self, output, frame, assignment):
        class_targets = torch.full((len(output["mask_logits"]),), len(self.config.class_names),
                                   dtype=torch.long, device=output["mask_logits"].device)
        zero = output["mask_logits"].sum() * 0
        bce, dice = zero, zero
        details = dict(foreground_bce=zero, background_bce=zero)
        for target_index, query_index in assignment.items():
            class_targets[query_index] = frame["class_ids"][target_index]
            item_bce, item_dice, item_details = mask_losses(output["mask_logits"][query_index],
                                            frame["labels"][target_index], output["valid_points"],
                                            self.bce_mode, return_details=True)
            bce, dice = bce + item_bce, dice + item_dice
            for key in details:
                details[key] = details[key] + item_details[key]
        weights = output["class_logits"].new_ones(len(self.config.class_names) + 1)
        weights[-1] = self.config.no_object_weight
        classification = F.cross_entropy(output["class_logits"], class_targets, weight=weights)
        denominator = max(1, len(assignment))
        value = classification + self.config.mask_bce_weight * bce / denominator + self.config.mask_dice_weight * dice / denominator
        components = dict(classification=classification, mask_bce=bce / denominator, mask_dice=dice / denominator,
                          **{key: value / denominator for key, value in details.items()})
        return value, components

    def __call__(self, output, frame):
        assignment = self.assignment(output, frame)
        detection, components = self._mask_class_loss(output, frame, assignment)
        for auxiliary in output["auxiliary"]:
            value, current = self._mask_class_loss(
                dict(**auxiliary, valid_points=output["valid_points"]), frame, assignment)
            detection = detection + value
            for key in components:
                components[key] = components[key] + current[key]
        detection /= len(output["auxiliary"]) + 1
        components = {key: value / (len(output["auxiliary"]) + 1) for key, value in components.items()}
        identity = output["mask_logits"].sum() * 0
        for target_index, query_index in assignment.items():
            other = torch.zeros_like(output["valid_points"])
            for index in range(len(frame["labels"])):
                if index != target_index:
                    other |= frame["labels"][index] == 1
            other &= output["valid_points"] & (frame["labels"][target_index] == 0)
            if other.any():
                identity = identity + F.softplus(output["mask_logits"][query_index, other]).mean()
        identity /= max(1, len(assignment))
        reference = output["mask_logits"].sum() * 0
        if frame["reference_tokens"]:
            token_queries = {frame["instance_tokens"][target]: query for target, query in assignment.items()}
            targets = [len(output["mask_logits"]) if token is None else token_queries[token]
                       for token in frame["reference_tokens"]]
            reference = F.cross_entropy(output["reference_logits"], torch.tensor(targets, device=reference.device))
        total = detection + self.config.identity_weight * identity + self.config.reference_weight * reference
        return total, dict(mask_class=detection, identity=identity, reference=reference, **components), assignment
