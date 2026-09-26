"""MORE3D-style object routing with a Reason3D coarse-to-fine decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .config import ReasonSegConfig


@dataclass
class ReasonSegHeadOutput:
    loc_logits: torch.Tensor
    mask_logits: torch.Tensor
    class_logits: torch.Tensor
    object_features: torch.Tensor
    loss: Optional[torch.Tensor] = None
    losses: Optional[Dict[str, torch.Tensor]] = None


class QueryMaskDecoder(nn.Module):
    """Decode one or more language-derived queries against point features."""

    def __init__(self, config: ReasonSegConfig):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=config.point_feature_dim,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.decoder_layers)
        self.query_norm = nn.LayerNorm(config.point_feature_dim)
        self.memory_norm = nn.LayerNorm(config.point_feature_dim)
        self.mask_query = nn.Linear(config.point_feature_dim, config.point_feature_dim)
        self.mask_memory = nn.Linear(config.point_feature_dim, config.point_feature_dim)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        point_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if queries.ndim != 3 or memory.ndim != 3:
            raise ValueError("queries and memory must be [B, K, C] and [B, N, C]")
        if point_valid_mask.shape != memory.shape[:2]:
            raise ValueError("point_valid_mask shape does not match memory")
        decoded = self.decoder(
            self.query_norm(queries),
            self.memory_norm(memory),
            memory_key_padding_mask=~point_valid_mask.bool(),
        )
        logits = torch.einsum(
            "bkc,bnc->bkn",
            self.mask_query(decoded),
            self.mask_memory(memory),
        ) / (memory.shape[-1] ** 0.5)
        logits = logits.masked_fill(~point_valid_mask[:, None, :].bool(), -1e4)
        return decoded, logits


class HierarchicalMaskDecoder(nn.Module):
    """Predict a soft coarse region, fine masks and independent classes."""

    def __init__(self, config: ReasonSegConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.loc_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.point_feature_dim),
            nn.GELU(),
            nn.Linear(config.point_feature_dim, config.point_feature_dim),
        )
        self.seg_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.point_feature_dim),
            nn.GELU(),
            nn.Linear(config.point_feature_dim, config.point_feature_dim),
        )
        self.region_decoder = QueryMaskDecoder(config)
        self.mask_decoder = QueryMaskDecoder(config)
        self.location_prior = nn.Sequential(
            nn.Linear(1, config.point_feature_dim),
            nn.GELU(),
            nn.Linear(config.point_feature_dim, config.point_feature_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.point_feature_dim * 2, config.point_feature_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.point_feature_dim, config.num_classes),
        )

    def forward(
        self,
        point_features: torch.Tensor,
        point_valid_mask: torch.Tensor,
        loc_hidden_states: torch.Tensor,
        seg_hidden_states: torch.Tensor,
        object_valid_mask: Optional[torch.Tensor] = None,
        target_masks: Optional[torch.Tensor] = None,
        target_loc_masks: Optional[torch.Tensor] = None,
        target_classes: Optional[torch.Tensor] = None,
    ) -> ReasonSegHeadOutput:
        if loc_hidden_states.shape != seg_hidden_states.shape:
            raise ValueError("LOC and SEG hidden states must have identical shapes")
        if loc_hidden_states.ndim != 3:
            raise ValueError("LOC and SEG hidden states must be [B, K, H]")
        batch_size, object_count, _ = loc_hidden_states.shape
        if object_count > self.config.max_objects:
            raise ValueError(
                f"received {object_count} objects; configured maximum is {self.config.max_objects}"
            )
        if point_features.shape[:2] != point_valid_mask.shape:
            raise ValueError("point_valid_mask shape does not match point_features")
        if point_features.shape[0] != batch_size:
            raise ValueError("point feature batch does not match token hidden-state batch")

        if object_valid_mask is None:
            object_valid_mask = torch.ones(
                (batch_size, object_count), dtype=torch.bool, device=point_features.device
            )
        else:
            object_valid_mask = object_valid_mask.bool()

        loc_queries = self.loc_projection(loc_hidden_states)
        _, loc_logits = self.region_decoder(
            loc_queries, point_features, point_valid_mask
        )

        # Each object gets a different continuous region prior.  Flatten B*K
        # so the same decoder weights process variable object counts.
        point_count, feature_dim = point_features.shape[1], point_features.shape[2]
        if self.config.use_loc_prior:
            loc_probabilities = torch.sigmoid(loc_logits)
            prior = self.location_prior(loc_probabilities.unsqueeze(-1))
            fine_memory = point_features[:, None, :, :] + prior
        else:
            # 关掉先验后 K 个物体共享同一份 memory，区分它们的只剩 seg_query——
            # 这正是这次消融要问的问题。LOC 头仍然训练、loc 损失仍然算，好让
            # loc 指标保持可比，只是不再影响精细解码。
            fine_memory = point_features[:, None, :, :].expand(
                batch_size, object_count, point_count, feature_dim
            )
        fine_memory = fine_memory.reshape(
            batch_size * object_count, point_count, feature_dim
        )
        fine_valid = point_valid_mask[:, None, :].expand(
            batch_size, object_count, point_count
        ).reshape(batch_size * object_count, point_count)
        seg_queries = self.seg_projection(seg_hidden_states).reshape(
            batch_size * object_count, 1, self.config.point_feature_dim
        )
        decoded, mask_logits = self.mask_decoder(seg_queries, fine_memory, fine_valid)
        decoded = decoded.reshape(batch_size, object_count, self.config.point_feature_dim)
        mask_logits = mask_logits.reshape(batch_size, object_count, point_features.shape[1])

        mask_probabilities = torch.sigmoid(mask_logits)
        valid_weights = point_valid_mask[:, None, :].to(mask_probabilities.dtype)
        weights = mask_probabilities * valid_weights
        pooled = torch.einsum("bkn,bnc->bkc", weights, point_features)
        pooled = pooled / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        class_logits = self.classifier(torch.cat([decoded, pooled], dim=-1))

        invalid_objects = ~object_valid_mask
        loc_logits = loc_logits.masked_fill(invalid_objects[:, :, None], -1e4)
        mask_logits = mask_logits.masked_fill(invalid_objects[:, :, None], -1e4)

        losses = None
        total_loss = None
        if target_masks is not None or target_loc_masks is not None or target_classes is not None:
            if target_masks is None or target_loc_masks is None or target_classes is None:
                raise ValueError("mask, coarse-location and class targets must be provided together")
            losses = self.compute_losses(
                loc_logits=loc_logits,
                mask_logits=mask_logits,
                class_logits=class_logits,
                target_loc_masks=target_loc_masks,
                target_masks=target_masks,
                target_classes=target_classes,
                point_valid_mask=point_valid_mask,
                object_valid_mask=object_valid_mask,
            )
            total_loss = (
                self.config.seg_loss_weight * losses["seg"]
                + self.config.loc_loss_weight * losses["loc"]
                + self.config.class_loss_weight * losses["class"]
            )
            losses["total"] = total_loss

        return ReasonSegHeadOutput(
            loc_logits=loc_logits,
            mask_logits=mask_logits,
            class_logits=class_logits,
            object_features=decoded,
            loss=total_loss,
            losses=losses,
        )

    def compute_losses(
        self,
        *,
        loc_logits: torch.Tensor,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        target_loc_masks: torch.Tensor,
        target_masks: torch.Tensor,
        target_classes: torch.Tensor,
        point_valid_mask: torch.Tensor,
        object_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        expected_mask_shape = mask_logits.shape
        if target_masks.shape != expected_mask_shape:
            raise ValueError(
                f"target mask shape {tuple(target_masks.shape)} != {tuple(expected_mask_shape)}"
            )
        if target_loc_masks.shape != expected_mask_shape:
            raise ValueError("coarse-location target shape does not match logits")
        if target_classes.shape != class_logits.shape[:2]:
            raise ValueError("class target shape does not match class logits")

        # A positive target with no visible point is not an all-background example.
        object_valid_mask = object_valid_mask & (
            target_masks.bool() & point_valid_mask[:, None, :]
        ).any(dim=-1)
        valid = point_valid_mask[:, None, :] & object_valid_mask[:, :, None]
        seg = _mask_loss(
            mask_logits,
            target_masks,
            valid,
            bce_mode=self.config.bce_mode,
            region_loss=self.config.region_loss,
            tversky_alpha=self.config.tversky_alpha,
            tversky_beta=self.config.tversky_beta,
        )
        loc = _mask_loss(
            loc_logits,
            target_loc_masks,
            valid,
            bce_mode=self.config.bce_mode,
            region_loss=self.config.region_loss,
            tversky_alpha=self.config.tversky_alpha,
            tversky_beta=self.config.tversky_beta,
        )
        if object_valid_mask.any():
            class_loss = F.cross_entropy(
                class_logits[object_valid_mask], target_classes.long()[object_valid_mask]
            )
        else:
            class_loss = class_logits.sum() * 0.0
        return {"seg": seg, "loc": loc, "class": class_loss}


def _mask_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    bce_mode: str,
    region_loss: str,
    tversky_alpha: float,
    tversky_beta: float,
) -> torch.Tensor:
    targets = targets.to(logits.dtype)
    valid_float = valid.to(logits.dtype)
    per_point = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    per_point = per_point * valid_float
    object_valid = valid.any(dim=-1)

    if bce_mode == "balanced":
        # 正例只占帧内点的 3e-4，逐点平均会把正例梯度稀释约 3000 倍；此时"全不
        # 触发"与"在所有同类候选上对冲"的损失差仅 0.004，梯度没有动力去分辨实例。
        # 两侧各自归一再等权平均，使损失量级与正例占比无关。
        positive_count = (targets * valid_float).sum(dim=-1)
        negative_count = valid_float.sum(dim=-1) - positive_count
        positive_term = (per_point * targets).sum(dim=-1) / positive_count.clamp_min(1.0)
        negative_term = (per_point * (1.0 - targets)).sum(dim=-1) / negative_count.clamp_min(1.0)
        per_object = 0.5 * (positive_term + negative_term)
        bce = per_object[object_valid].mean() if object_valid.any() else logits.sum() * 0.0
    else:
        bce = per_point.sum() / valid_float.sum().clamp_min(1.0)

    probabilities = torch.sigmoid(logits) * valid_float
    target_values = targets * valid_float
    intersection = (probabilities * target_values).sum(dim=-1)
    if region_loss == "dice":
        union = probabilities.sum(dim=-1) + target_values.sum(dim=-1)
        per_object_region = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)
    else:
        # dice 恒等于 Tversky(0.5, 0.5) 配平滑常数 0.5，故沿用同一常数：
        # alpha 降低误检代价、beta 抬高漏检代价，把操作点从"沉默"推向"敢触发"。
        false_positive = (probabilities * (1.0 - target_values)).sum(dim=-1)
        false_negative = ((1.0 - probabilities) * target_values).sum(dim=-1)
        denominator = (
            intersection + tversky_alpha * false_positive + tversky_beta * false_negative
        )
        per_object_region = 1.0 - (intersection + 0.5) / (denominator + 0.5)
    region = (
        per_object_region[object_valid].mean()
        if object_valid.any()
        else logits.sum() * 0.0
    )
    return bce + region
