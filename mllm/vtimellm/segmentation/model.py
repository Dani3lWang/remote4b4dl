"""End-to-end wrapper joining B4DL's LLM with point-mask prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from vtimellm.constants import IMAGE_TOKEN_INDEX

from .config import ReasonSegConfig
from .heads import HierarchicalMaskDecoder, ReasonSegHeadOutput
from .spatial_encoder import (
    PointEncoding,
    SceneQueryCompressor,
    SparseUNetPointEncoder,
    pad_point_features,
)
from .tokens import SegmentationTokenIds


@dataclass
class ReasonSegModelOutput:
    loss: Optional[torch.Tensor]
    text_loss: Optional[torch.Tensor]
    head: ReasonSegHeadOutput
    token_status: List[str]
    language_output: object
    point_valid_mask: torch.Tensor


@dataclass
class GeneratedSegmentation:
    output_ids: torch.Tensor
    text: List[str]
    status: List[str]
    mask_probabilities: List[torch.Tensor]
    loc_probabilities: List[torch.Tensor]
    class_ids: List[torch.Tensor]
    class_scores: List[torch.Tensor]
    original_point_indices: List[torch.Tensor]


class ReasonSegModel(nn.Module):
    """B4DL language model plus a full-resolution point segmentation branch."""

    def __init__(
        self,
        language_model: nn.Module,
        token_ids: SegmentationTokenIds,
        config: ReasonSegConfig,
        point_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        config.validate()
        self.language_model = language_model
        self.token_ids = token_ids
        self.reasonseg_config = config
        self.point_encoder = point_encoder or SparseUNetPointEncoder(config)
        self.scene_compressor = SceneQueryCompressor(config)
        self.mask_head = HierarchicalMaskDecoder(config)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        points: torch.Tensor,
        point_batch_indices: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        target_masks: Optional[torch.Tensor] = None,
        target_loc_masks: Optional[torch.Tensor] = None,
        target_classes: Optional[torch.Tensor] = None,
        target_object_valid_mask: Optional[torch.Tensor] = None,
    ) -> ReasonSegModelOutput:
        encoding, padded_features, point_valid, scene_tokens, _ = self.encode_points(
            points, point_batch_indices
        )
        language_output = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=scene_tokens,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        final_hidden = language_output.hidden_states[-1]
        loc_hidden, seg_hidden, object_valid, statuses = extract_object_hidden_states(
            final_hidden_states=final_hidden,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_token_length=scene_tokens.shape[1],
            token_ids=self.token_ids,
            max_objects=self.reasonseg_config.max_objects,
            padding_side=getattr(self.language_model.config, "tokenizer_padding_side", "right"),
            max_expanded_length=getattr(
                self.language_model.config, "tokenizer_model_max_length", None
            ),
        )
        if target_masks is not None:
            failures = [status for status in statuses if status not in ("ok", "no_object")]
            if failures:
                raise RuntimeError(
                    "invalid LOC/SEG supervision tokens in training batch: "
                    + ", ".join(failures)
                )
            target_masks = _align_object_dimension(target_masks, object_valid.shape[1])
            target_loc_masks = _align_object_dimension(
                target_loc_masks, object_valid.shape[1]
            )
            target_classes = _align_object_dimension(
                target_classes, object_valid.shape[1]
            )
            if target_object_valid_mask is not None:
                expected_valid = _align_object_dimension(
                    target_object_valid_mask, object_valid.shape[1]
                ).bool()
                if not torch.equal(expected_valid, object_valid):
                    raise RuntimeError(
                        "answer LOC/SEG count does not match manifest target count"
                    )
        head_output = self.mask_head(
            point_features=padded_features,
            point_valid_mask=point_valid,
            loc_hidden_states=loc_hidden,
            seg_hidden_states=seg_hidden,
            object_valid_mask=object_valid,
            target_masks=target_masks,
            target_loc_masks=target_loc_masks,
            target_classes=target_classes,
        )
        text_loss = getattr(language_output, "loss", None)
        loss = head_output.loss
        if text_loss is not None:
            weighted_text = self.reasonseg_config.text_loss_weight * text_loss
            loss = weighted_text if loss is None else loss + weighted_text
        return ReasonSegModelOutput(
            loss=loss,
            text_loss=text_loss,
            head=head_output,
            token_status=statuses,
            language_output=language_output,
            point_valid_mask=point_valid,
        )

    def encode_points(
        self,
        points: torch.Tensor,
        point_batch_indices: torch.Tensor,
    ) -> tuple[PointEncoding, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        encoding = self.point_encoder(points, point_batch_indices)
        padded_features, point_valid, original_indices = pad_point_features(encoding)
        scene_tokens = self.scene_compressor(padded_features, point_valid)
        return encoding, padded_features, point_valid, scene_tokens, original_indices

    @torch.inference_mode()
    def generate_and_segment(
        self,
        *,
        tokenizer,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        points: torch.Tensor,
        point_batch_indices: torch.Tensor,
        threshold: float = 0.5,
        **generation_kwargs,
    ) -> GeneratedSegmentation:
        _, padded_features, point_valid, scene_tokens, original_indices = self.encode_points(
            points, point_batch_indices
        )
        output_ids = self.language_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=scene_tokens,
            **generation_kwargs,
        )
        full_attention = torch.ones_like(output_ids, dtype=attention_mask.dtype)
        language_output = self.language_model(
            input_ids=output_ids,
            attention_mask=full_attention,
            images=scene_tokens,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        loc_hidden, seg_hidden, object_valid, statuses = extract_object_hidden_states(
            final_hidden_states=language_output.hidden_states[-1],
            input_ids=output_ids,
            attention_mask=full_attention,
            image_token_length=scene_tokens.shape[1],
            token_ids=self.token_ids,
            max_objects=self.reasonseg_config.max_objects,
            padding_side=getattr(self.language_model.config, "tokenizer_padding_side", "right"),
            max_expanded_length=getattr(
                self.language_model.config, "tokenizer_model_max_length", None
            ),
        )
        head = self.mask_head(
            point_features=padded_features,
            point_valid_mask=point_valid,
            loc_hidden_states=loc_hidden,
            seg_hidden_states=seg_hidden,
            object_valid_mask=object_valid,
        )
        mask_probabilities = torch.sigmoid(head.mask_logits)
        loc_probabilities = torch.sigmoid(head.loc_logits)
        class_probabilities = torch.softmax(head.class_logits.float(), dim=-1)
        class_scores, class_ids = class_probabilities.max(dim=-1)

        masks: List[torch.Tensor] = []
        locs: List[torch.Tensor] = []
        classes: List[torch.Tensor] = []
        scores: List[torch.Tensor] = []
        for batch_index, indices in enumerate(original_indices):
            object_mask = object_valid[batch_index]
            point_count = int(indices.numel())
            masks.append(
                (mask_probabilities[batch_index, object_mask, :point_count] >= threshold).cpu()
            )
            locs.append(loc_probabilities[batch_index, object_mask, :point_count].cpu())
            classes.append(class_ids[batch_index, object_mask].cpu())
            scores.append(class_scores[batch_index, object_mask].cpu())
        # output_ids still carries the IMAGE_TOKEN_INDEX (-200) prompt placeholder, which is
        # not a real vocabulary id and makes sentencepiece raise IndexError on decode.
        text = tokenizer.batch_decode(
            [ids[ids >= 0].tolist() for ids in output_ids], skip_special_tokens=False
        )
        return GeneratedSegmentation(
            output_ids=output_ids,
            text=text,
            status=statuses,
            mask_probabilities=masks,
            loc_probabilities=locs,
            class_ids=classes,
            class_scores=scores,
            original_point_indices=[indices.cpu() for indices in original_indices],
        )


def extract_object_hidden_states(
    *,
    final_hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_length: int,
    token_ids: SegmentationTokenIds,
    max_objects: int,
    padding_side: str = "right",
    max_expanded_length: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Map text-token indices through B4DL's ``<video>`` embedding expansion."""

    batch_size, _, hidden_size = final_hidden_states.shape
    loc_positions: List[List[int]] = []
    seg_positions: List[List[int]] = []
    statuses: List[str] = []
    for batch_index in range(batch_size):
        valid_ids = input_ids[batch_index][attention_mask[batch_index].bool()].tolist()
        expanded_index = 0
        sample_loc: List[int] = []
        sample_seg: List[int] = []
        has_noobj = False
        for token_id in valid_ids:
            if token_id == IMAGE_TOKEN_INDEX:
                expanded_index += image_token_length
                continue
            if token_id == token_ids.loc:
                sample_loc.append(expanded_index)
            elif token_id == token_ids.seg:
                sample_seg.append(expanded_index)
            elif token_id == token_ids.noobj:
                has_noobj = True
            expanded_index += 1
        expanded_length = min(
            expanded_index,
            max_expanded_length if max_expanded_length is not None else expanded_index,
        )
        sample_loc = [position for position in sample_loc if position < expanded_length]
        sample_seg = [position for position in sample_seg if position < expanded_length]
        if padding_side == "left":
            offset = final_hidden_states.shape[1] - expanded_length
            sample_loc = [position + offset for position in sample_loc]
            sample_seg = [position + offset for position in sample_seg]
        if has_noobj and not sample_loc and not sample_seg:
            status = "no_object"
        elif len(sample_loc) != len(sample_seg):
            status = f"token_pair_mismatch:loc={len(sample_loc)},seg={len(sample_seg)}"
        elif not sample_seg:
            status = "missing_segmentation_tokens"
        elif len(sample_seg) > max_objects:
            status = f"too_many_objects:{len(sample_seg)}>{max_objects}"
        else:
            status = "ok"
        # Token failures are observable model failures.  Do not decode a
        # convenient subset of mismatched/random tokens because that hides the
        # failure and makes repeated evaluation nondiagnostic.
        pair_count = (
            min(len(sample_loc), len(sample_seg), max_objects)
            if status == "ok"
            else 0
        )
        loc_positions.append(sample_loc[:pair_count])
        seg_positions.append(sample_seg[:pair_count])
        statuses.append(status)

    object_count = max(1, max((len(value) for value in seg_positions), default=0))
    loc_hidden = final_hidden_states.new_zeros(batch_size, object_count, hidden_size)
    seg_hidden = final_hidden_states.new_zeros(batch_size, object_count, hidden_size)
    object_valid = torch.zeros(
        batch_size, object_count, dtype=torch.bool, device=final_hidden_states.device
    )
    for batch_index, (locs, segs) in enumerate(zip(loc_positions, seg_positions)):
        count = min(len(locs), len(segs))
        if count:
            loc_hidden[batch_index, :count] = final_hidden_states[batch_index, locs[:count]]
            seg_hidden[batch_index, :count] = final_hidden_states[batch_index, segs[:count]]
            object_valid[batch_index, :count] = True
    return loc_hidden, seg_hidden, object_valid, statuses


def _align_object_dimension(values: Optional[torch.Tensor], object_count: int):
    if values is None:
        return None
    if values.shape[1] < object_count:
        pad_shape = list(values.shape)
        pad_shape[1] = object_count - values.shape[1]
        padding = values.new_zeros(pad_shape)
        return torch.cat([values, padding], dim=1)
    return values[:, :object_count]
