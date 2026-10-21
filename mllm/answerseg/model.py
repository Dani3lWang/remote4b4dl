"""Train online B3 states against actual exported MaskPLS instance candidates."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .language import LanguageBridge, mark_answer, training_sequence


class BindingHead(nn.Module):
    def __init__(self, language_dim, query_dim=256):
        super().__init__()
        self.text = nn.Sequential(
            nn.Linear(language_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )
        self.object = nn.Linear(query_dim, query_dim)
        self.no_referent = nn.Parameter(torch.randn(query_dim) * 0.02)

    def forward(self, states, queries):
        candidates = torch.cat(
            (self.object(queries.float()), self.no_referent[None]), 0
        )
        return (
            F.normalize(self.text(states.float()), dim=-1)
            @ F.normalize(candidates, dim=-1).T
            / 0.1
        )


def candidate_assignment(masks, labels, valid, minimum_iou):
    """Geometry-only coverage diagnostic; uncovered positives NEVER become NOOBJ."""
    from scipy.optimize import linear_sum_assignment

    if not math.isfinite(minimum_iou) or not 0 < minimum_iou <= 1:
        raise ValueError("coverage IoU must lie in (0,1]")
    result = {}
    if not len(masks) or not len(labels):
        return result
    # Never materialize [candidates, targets, points]: real frames can contain
    # >100 targets and tens of thousands of points alongside the 7B model.
    prediction = (masks & valid[None]).float()
    target = (labels.eq(1) & valid[None]).float()
    supervised = (labels.ge(0) & valid[None]).float()
    intersection = prediction @ target.T
    union = (prediction @ supervised.T + target.sum(-1)[None] - intersection).clamp_min(
        1
    )
    iou = (intersection / union).detach().cpu().numpy()
    # Maximize number above threshold, then IoU. A plain max-IoU sum can lose coverage.
    score = (iou >= minimum_iou).astype(np.float64) * (min(iou.shape) + 1) + iou
    rows, columns = linear_sum_assignment(-score)
    for candidate, target_index in zip(rows, columns):
        if iou[candidate, target_index] >= minimum_iou:
            result[int(target_index)] = int(candidate)
    return result


class AnswerSegModel(nn.Module):
    def __init__(
        self,
        language,
        tokenizer,
        tokens,
        spatial,
        *,
        stage="binding",
        tune_spatial=False,
        minimum_iou=0.5,
        language_weight=1.0,
        binding_weight=1.0,
        mask_weight=1.0,
    ):
        super().__init__()
        if stage not in ("binding", "joint") or (stage == "binding" and tune_spatial):
            raise ValueError("spatial tuning requires the explicit joint stage")
        for weight in (language_weight, binding_weight, mask_weight):
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("loss weights must be positive and finite")
        if not math.isfinite(minimum_iou) or not 0 < minimum_iou <= 1:
            raise ValueError("minimum_iou must lie in (0,1]")
        self.language, self.spatial = language, spatial
        self.bridge = LanguageBridge(language, tokenizer, tokens)
        self.tokens, self.stage, self.tune_spatial = tokens, stage, tune_spatial
        self.minimum_iou = minimum_iou
        self.loss_weights = (language_weight, binding_weight, mask_weight)
        self.head = BindingHead(language.config.hidden_size)
        if stage == "binding":
            self.language.requires_grad_(False)
        self.spatial.requires_grad_(tune_spatial)

    def train(self, mode=True):
        super().train(mode)
        if self.stage == "binding":
            self.language.eval()
        if not self.tune_spatial:
            self.spatial.eval()
        return self

    def candidates(self, points):
        from maskpls_b4dl.model import extract_instances

        with torch.set_grad_enabled(self.training and self.tune_spatial):
            output = self.spatial(points)
        # Export decisions are discrete; the selected live queries/logits retain gradients.
        with torch.no_grad():
            instances = extract_instances(output, self.spatial.config)
        queries = output["queries"][instances["query_indices"]]
        return output, instances, queries

    def forward(self, example):
        marked, targets = mark_answer(example["answer"], example["targets"])
        ids, labels = training_sequence(
            self.bridge.tokenizer, example["prompt"], marked
        )
        with torch.set_grad_enabled(self.stage == "joint" and self.training):
            language_loss, states, kinds = self.bridge.forward(
                ids, labels, example["features"]
            )
        if len(states) != len(targets):
            raise ValueError(
                "reviewed targets and assistant marker hidden states disagree"
            )
        frame = example["frame"]
        output, instances, queries = self.candidates(frame["points"])
        logits = self.head(states, queries)
        assignment = candidate_assignment(
            instances["masks"],
            frame["labels"],
            output["valid_points"],
            self.minimum_iou,
        )
        losses, mask_losses, covered, positives = [], [], 0, 0
        tokens = frame["instance_tokens"]
        for index, target in enumerate(targets):
            if target["status"] == "no_target":
                if int(kinds[index]) != self.tokens.noobj:
                    raise ValueError("NOOBJ marker mismatch")
                losses.append(
                    F.cross_entropy(
                        logits[index : index + 1],
                        logits.new_tensor([len(queries)], dtype=torch.long),
                    )
                )
                continue
            positives += 1
            if target["instance_token"] not in tokens:
                raise ValueError(
                    "reviewed instance is absent from this original-order frame"
                )
            gt_index = tokens.index(target["instance_token"])
            if gt_index not in assignment:
                continue
            covered += 1
            losses.append(
                F.cross_entropy(
                    logits[index : index + 1],
                    logits.new_tensor([assignment[gt_index]], dtype=torch.long),
                )
            )
            supervised = frame["labels"][gt_index].ge(0) & output["valid_points"]
            truth = frame["labels"][gt_index, supervised].float()
            probability = (
                logits[index].softmax(-1)[:-1]
                @ output["mask_logits"][instances["query_indices"]][:, supervised]
                .float()
                .sigmoid()
            )
            bce = F.binary_cross_entropy(probability.clamp(1e-6, 1 - 1e-6), truth)
            dice = 1 - (2 * (probability * truth).sum() + 1) / (
                probability.sum() + truth.sum() + 1
            )
            mask_losses.append(bce + dice)
        zero = logits.sum() * 0
        binding_loss = torch.stack(losses).mean() if losses else zero
        mask_loss = torch.stack(mask_losses).mean() if mask_losses else zero
        language_term = language_loss if self.stage == "joint" else zero
        weights = self.loss_weights
        loss = (
            weights[0] * language_term
            + weights[1] * binding_loss
            + weights[2] * mask_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite answer segmentation loss")
        return {
            "loss": loss,
            "language_loss": language_loss.detach(),
            "binding_loss": binding_loss.detach(),
            "mask_loss": mask_loss.detach(),
            "positive_targets": positives,
            "covered_targets": covered,
            "supervised_bindings": len(losses),
            "candidates": len(queries),
            "skipped_uncovered": positives - covered,
        }

    @torch.no_grad()
    def predict(
        self, prompt, features, points, *, max_new_tokens=256, selection_threshold=0.5
    ):
        """Prediction accepts only input features/points: no masks, GT answers or identities."""
        if self.training:
            raise ValueError("call eval() before free generation")
        if not math.isfinite(selection_threshold) or not 0 < selection_threshold <= 1:
            raise ValueError("selection_threshold must lie in (0,1]")
        generated = self.bridge.generate(prompt, features, max_new_tokens)
        if not generated["finished"]:
            return {
                "answer": generated["answer"],
                "status": "generation_truncated",
                "objects": [],
                "masks": [],
            }
        _, states, kinds = self.bridge.forward(
            generated["output_ids"], generated["labels"], features
        )
        if not len(states):
            return {
                "answer": generated["answer"],
                "status": "missing_seg_token",
                "objects": [],
                "masks": [],
            }
        if len(states) > 8:
            return {
                "answer": generated["answer"],
                "status": "too_many_seg_tokens",
                "objects": [],
                "masks": [],
            }
        _, instances, queries = self.candidates(points)
        probabilities = self.head(states, queries).softmax(-1)
        objects, masks = [], []
        for index, kind in enumerate(kinds.tolist()):
            confidence, selected = probabilities[index].max(0)
            if kind == self.tokens.noobj:
                reason = "explicit_no_target"
            elif int(selected) == len(queries):
                reason = "predicted_no_referent"
            elif float(confidence) < selection_threshold:
                reason = "low_binding_confidence"
            else:
                reason = "selected"
            mask = torch.zeros(len(points), dtype=torch.bool, device=points.device)
            record = {
                "seg_index": index,
                "status": reason,
                "binding_confidence": float(confidence),
                "query_index": None,
                "class_id": None,
                "track_id": None,
            }
            if reason == "selected":
                candidate = int(selected)
                mask = instances["masks"][candidate]
                record.update(
                    query_index=int(instances["query_indices"][candidate]),
                    class_id=int(instances["class_ids"][candidate]),
                    spatial_score=float(instances["scores"][candidate]),
                )
            objects.append(record)
            masks.append(mask)
        return {
            "answer": generated["answer"],
            "status": "ok",
            "objects": objects,
            "masks": masks,
        }
