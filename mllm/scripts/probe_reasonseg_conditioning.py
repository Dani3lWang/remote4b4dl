#!/usr/bin/env python3
"""Does the predicted mask actually depend on the language side?

Feeds one LiDAR frame through the ReasonSeg checkpoint with several query
variants that differ only in the class word (or only in the answer's class, or
only in the ordinal), plus a control pair that keeps the query and swaps the
frame.  A head that ignores the language side produces near-identical masks
for every variant; a head that uses it must track the class it was asked for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

MLLM_ROOT = Path(__file__).resolve().parents[1]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))

from vtimellm.segmentation.config import ReasonSegConfig
from vtimellm.segmentation.data import ReasonSegCollator, ReasonSegDataset
from vtimellm.segmentation.loader import load_reasonseg_model
from vtimellm.segmentation.spatial_encoder import PointEncoding


class GeometryStubEncoder(torch.nn.Module):
    """Deterministic CPU stand-in for the spconv U-Net.

    spconv's implicit-gemm path needs CUDA, so the probe swaps the encoder for a
    fixed random projection of the raw point coordinates.  Features stay a
    function of the frame (so the frame-control variant is meaningful) but carry
    no learned content, which is exactly what isolates the language path: any
    change between class-word variants has to come from the text side.
    """

    def __init__(self, config: ReasonSegConfig, input_dim: int = 4, seed: int = 1234):
        super().__init__()
        self.config = config
        self.voxel_size = config.voxel_size
        self.input_dim = input_dim
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "projection",
            torch.randn(input_dim, config.point_feature_dim, generator=generator) * 0.5,
        )
        direction = torch.randn(config.point_feature_dim, generator=generator)
        self.register_buffer("oracle_direction", direction / direction.norm())
        self.loaded_keys = 0
        self.oracle_mask = None
        self.oracle_scale = 8.0

    def set_oracle_mask(self, mask):
        self.oracle_mask = None if mask is None else mask.bool()

    def load_state_dict(self, state_dict, strict=True, assign=False):  # noqa: D102
        self.loaded_keys = len(state_dict)
        return torch.nn.modules.module._IncompatibleKeys([], [])

    def forward(self, points: torch.Tensor, batch_indices: torch.Tensor) -> PointEncoding:
        features = torch.tanh(points[:, : self.input_dim] @ self.projection)
        if self.oracle_mask is not None:
            # Marks the target object's points with a distinct direction.  A
            # constant offset would be annihilated by the head's LayerNorm, so
            # the marker has to differ from point to point in feature space.
            marker = self.oracle_mask[:, None].to(features.dtype) * self.oracle_direction[None, :]
            features = features + self.oracle_scale * marker
        return PointEncoding(
            point_features=features,
            point_valid_mask=torch.ones(points.shape[0], dtype=torch.bool, device=points.device),
            batch_indices=batch_indices,
            voxel_coordinates=points[:, :3],
            point_to_voxel=torch.arange(points.shape[0], device=points.device),
        )



def iou(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    pred = pred.bool() & valid
    target = target.bool() & valid
    union = (pred | target).sum().item()
    if union == 0:
        return float("nan")
    return float((pred & target).sum().item()) / union


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--pretrain-mm-mlp-adapter", required=True)
    parser.add_argument("--b3-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, default=-1,
                        help="manifest row to probe; -1 picks the first frame whose class "
                             "appears in --swap-classes and has the most points")
    parser.add_argument("--control-index", type=int, default=-1)
    parser.add_argument("--swap-classes", default="truck,bus,traffic_cone")
    parser.add_argument("--stub-encoder", action="store_true",
                        help="replace the spconv encoder with a deterministic CPU stand-in "
                             "(required without a free CUDA device)")
    parser.add_argument("--oracle-features", action="store_true",
                        help="add the target object's mask to the stub features, handing the "
                             "head the spatial answer it would otherwise have to learn")
    parser.add_argument("--min-gt-points", type=int, default=0,
                        help="pick a frame whose primary target has at least this many points")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(min(64, torch.get_num_threads()))
    config = ReasonSegConfig()
    dataset = ReasonSegDataset(args.manifest, dataroot=args.dataroot, config=config)

    loader_args = SimpleNamespace(
        model_base=args.model_base,
        pretrain_mm_mlp_adapter=args.pretrain_mm_mlp_adapter,
        model_max_length=2048,
    )
    tokenizer, model, _ = load_reasonseg_model(
        loader_args,
        b3_checkpoint=args.b3_checkpoint,
        reasonseg_config=config,
        segmentation_checkpoint=args.checkpoint,
        trainable=False,
        point_encoder=GeometryStubEncoder(config) if args.stub_encoder else None,
    )
    model = model.float().eval()
    collator = ReasonSegCollator(tokenizer, model_max_length=2048)

    rows = dataset.records
    swap_classes = [name.strip() for name in args.swap_classes.split(",") if name.strip()]
    index = args.index
    if index < 0:
        best = (-1, -1)
        for candidate, record in enumerate(rows[:400]):
            if not record.targets:
                continue
            if swap_classes and record.targets[0].class_name not in swap_classes:
                continue
            gt_points = int(dataset[candidate]["target_masks"][0].sum().item())
            if args.min_gt_points and gt_points < args.min_gt_points:
                continue
            if gt_points > best[1]:
                best = (candidate, gt_points)
        index = best[0]
        if index < 0:
            raise SystemExit(f"no manifest row matched swap_classes={swap_classes}")
    record = rows[index]
    original_class = record.targets[0].class_name if record.targets else "unknown"

    control_index = args.control_index
    if control_index < 0:
        control_index = index + max(1, len(rows) // 400)
    control_record = rows[control_index % len(rows)]

    if args.oracle_features:
        swap_class = next((name for name in swap_classes if name != original_class), None)
        variants = [
            ("original (no marker)", record.query, record.answer, index, False),
            ("original + marker on GT", record.query, record.answer, index, True),
        ]
        if swap_class:
            variants.append((
                f"class={swap_class} + marker on GT",
                record.query.replace(original_class, swap_class),
                record.answer.replace(original_class, swap_class),
                index,
                True,
            ))
        variants.append((
            "other frame + marker",
            control_record.query,
            control_record.answer,
            control_index,
            True,
        ))
    else:
        variants = [("original", record.query, record.answer, index, False)]
        for name in swap_classes:
            if name == original_class:
                continue
            variants.append((
                f"query+answer:class={name}",
                record.query.replace(original_class, name),
                record.answer.replace(original_class, name),
                index,
                False,
            ))
        answer_only_class = next((name for name in swap_classes if name != original_class), None)
        if answer_only_class:
            variants.append((
                f"query=original,answer:class={answer_only_class}",
                record.query,
                record.answer.replace(original_class, answer_only_class),
                index,
                False,
            ))
        if "nearest" in record.query:
            variants.append((
                "ordinal=nearest->farthest",
                record.query.replace("nearest", "farthest"),
                record.answer.replace("nearest", "farthest"),
                index,
                False,
            ))
        variants.append(("control:same query, other frame", control_record.query,
                         control_record.answer, control_index, False))

    results = []
    reference = {}
    for name, query, answer, row_index, use_marker in variants:
        sample = dataset[row_index]
        sample["query"] = query
        sample["answer"] = answer
        batch = collator([sample])
        if isinstance(model.point_encoder, GeometryStubEncoder):
            masks = batch["target_masks"]
            model.point_encoder.set_oracle_mask(
                (masks[0][0] if masks.shape[1] else None) if use_marker else None
            )
        with torch.inference_mode():
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                points=batch["points"],
                point_batch_indices=batch["point_batch_indices"],
            )
        logits = output.head.mask_logits[0]
        loc_logits = output.head.loc_logits[0]
        point_valid = output.point_valid_mask[0]
        gt_target = batch["target_masks"][0]
        object_valid = batch["object_valid_mask"][0].bool()
        slot = 0
        valid_logits = logits[slot][point_valid]
        entry = {
            "variant": name,
            "query": query,
            "answer": answer,
            "row": row_index,
            "status": output.token_status[0],
            "positives": int(object_valid.sum().item()),
            "mean_mask_probability": round(float(torch.sigmoid(valid_logits).mean()), 4),
            "mask_area_at_threshold": int(((torch.sigmoid(logits[slot]) >= args.threshold) & point_valid).sum().item()),
            "logit_std": round(float(valid_logits.std()), 3),
            "logit_min": round(float(valid_logits.min()), 3),
            "logit_max": round(float(valid_logits.max()), 3),
            "mean_loc_probability": round(float(torch.sigmoid(loc_logits[slot])[point_valid].mean()), 4),
        }
        if object_valid.numel() and object_valid[0]:
            predicted = (torch.sigmoid(logits[slot]) >= args.threshold) & point_valid
            loc_predicted = (torch.sigmoid(loc_logits[slot]) >= args.threshold) & point_valid
            gt = gt_target[slot] & point_valid
            entry["iou_ceiling_predict_all"] = round(float(gt.sum().item()) / max(1, int(point_valid.sum().item())), 4)
            entry["iou_vs_frame_gt"] = round(iou(predicted, gt_target[slot], point_valid), 4)
            entry["iou_vs_frame_gt_loc"] = round(iou(loc_predicted, batch["target_loc_masks"][0][slot],
                                                     point_valid), 4)
            same_frame = row_index == index
            entry["iou_vs_reference"] = round(iou(predicted, reference["mask"], point_valid), 4) \
                if same_frame and "mask" in reference else None
            entry["cosine_query_feature"] = round(float(
                torch.nn.functional.cosine_similarity(
                    output.head.object_features[0][slot].float(), reference["query_feature"], dim=0
                )), 4) if same_frame and "query_feature" in reference else None
            if same_frame:
                reference.setdefault("mask", predicted)
                reference.setdefault("query_feature", output.head.object_features[0][slot].float())
        results.append(entry)

    ground_truth = {
        "row": index,
        "class": original_class,
        "query": record.query,
        "answer": record.answer,
    }
    control = {"row": control_index % len(rows), "query": control_record.query,
               "answer": control_record.answer}
    payload = {"checkpoint": args.checkpoint, "target": ground_truth, "control": control,
               "variants": results}
    print(json.dumps(payload, indent=1, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
