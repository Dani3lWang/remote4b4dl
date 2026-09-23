#!/usr/bin/env python3
"""Phase 2.0 grounding diagnosis for ReasonSeg, teacher-forcing forward.

Answers one question before any architecture change is paid for: is the mask
head's probability map informative but badly thresholded, or uninformative?

Per ground-truth object it measures
  * coarse LOC quality: IoU, recall@0.5, and containment of the GT mask inside
    the predicted coarse region;
  * fine-mask ranking quality: AUC of sigmoid probabilities (GT-positive vs
    GT-negative points), mean probability on each side;
  * threshold-free top-K recall@0.5 (K = GT point count) plus a random-K
    baseline, and the same top-K restricted to the predicted LOC region;
  * context: ordinal bucket from the query and the number of same-class
    instances in the frame (from the panoptic ids).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

MLLM_ROOT = Path(__file__).resolve().parents[2]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))

from vtimellm.segmentation.data import ReasonSegCollator, ReasonSegDataset  # noqa: E402
from vtimellm.segmentation.loader import load_reasonseg_model  # noqa: E402

ORDINAL_RE = re.compile(r"(\d+)(?:st|nd|rd|th)-nearest")


def ordinal_bucket(query: str) -> str:
    match = ORDINAL_RE.search(query)
    if match is None:
        return "1(no_ordinal)"
    ordinal = int(match.group(1))
    if ordinal <= 1:
        return "1(no_ordinal)"
    if ordinal <= 3:
        return "2-3"
    if ordinal <= 9:
        return "4-9"
    return "10+"


def auc_from_scores(positives: np.ndarray, negatives: np.ndarray) -> float:
    """Rank-based AUC without sklearn; ties get average rank."""
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    scores = np.concatenate([positives, negatives])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # average ranks for ties
    sorted_scores = scores[order]
    index = 0
    while index < sorted_scores.size:
        end = index
        while end + 1 < sorted_scores.size and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        if end > index:
            ranks[order[index:end + 1]] = (index + end + 2) / 2.0
        index = end + 1
    positive_rank_sum = ranks[: positives.size].sum()
    count = positives.size * negatives.size
    return float((positive_rank_sum - positives.size * (positives.size + 1) / 2.0) / count)


def iou_from_masks(predicted: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(predicted, target).sum()
    union = np.logical_or(predicted, target).sum()
    return float(intersection / union) if union else 0.0


def same_class_instance_count(panoptic: np.ndarray, target_panoptic_id: int) -> int:
    """Count instances sharing the target's semantic class (panoptic // 1000)."""
    semantic = target_panoptic_id // 1000
    if semantic <= 0:
        return -1
    ids = np.unique(panoptic[panoptic // 1000 == semantic])
    return int(ids.size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--pretrain-mm-mlp-adapter", required=True)
    parser.add_argument("--b3-checkpoint", required=True)
    parser.add_argument("--seg-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--encoder-dtype", choices=("same", "fp32"), default="fp32",
                        help="spconv 编码器必须留 fp32：整模型 fp16 在 4090 上会崩，"
                             "bf16 会让部分帧特征变 NaN")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    args.stage2 = args.b3_checkpoint
    tokenizer, model, _ = load_reasonseg_model(
        args,
        b3_checkpoint=args.b3_checkpoint,
        segmentation_checkpoint=args.seg_checkpoint,
        trainable=False,
    )
    eval_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                  "fp32": torch.float32}[args.dtype]
    model = model.to(device=device, dtype=eval_dtype).eval()
    point_dtype = eval_dtype
    if args.encoder_dtype == "fp32":
        model.point_encoder.to(torch.float32)
        model.scene_compressor.to(torch.float32)
        point_dtype = torch.float32
        encode_points = model.encode_points

        def encode_points_fp32(points, point_batch_indices):
            encoding, padded, valid, scene, indices = encode_points(
                points.to(torch.float32), point_batch_indices
            )
            return encoding, padded.to(eval_dtype), valid, scene.to(eval_dtype), indices

        model.encode_points = encode_points_fp32
    config = model.reasonseg_config

    dataset = ReasonSegDataset(args.manifest, dataroot=args.dataroot, config=config)
    collator = ReasonSegCollator(tokenizer, model_max_length=args.model_max_length)
    limit = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    with torch.no_grad():
        for index in range(limit):
            sample = dataset[index]
            batch = collator([sample])
            batch = {
                key: (value.to(device) if torch.is_tensor(value) else value)
                for key, value in batch.items()
            }
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                points=batch["points"],
                point_batch_indices=batch["point_batch_indices"],
                target_masks=batch["target_masks"],
                target_loc_masks=batch["target_loc_masks"],
                target_classes=batch["target_classes"],
                target_object_valid_mask=batch["object_valid_mask"],
            )
            head = output.head
            point_valid = output.point_valid_mask[0].bool().cpu().numpy()
            point_count = int(point_valid.sum())
            loc_prob = torch.sigmoid(head.loc_logits)[0].cpu().numpy()
            mask_prob = torch.sigmoid(head.mask_logits)[0].cpu().numpy()
            targets = batch["target_masks"][0].bool().cpu().numpy()
            loc_targets = batch["target_loc_masks"][0].bool().cpu().numpy()
            valid_objects = batch["object_valid_mask"][0].bool().cpu().numpy()

            record = dataset.records[index]
            panoptic = None
            for object_index in range(int(valid_objects.sum())):
                target = targets[object_index][:point_count]
                loc_target = loc_targets[object_index][:point_count]
                positive_probability = mask_prob[object_index][:point_count][target]
                negative_probability = mask_prob[object_index][:point_count][~target]
                target_size = int(target.sum())
                if target_size == 0:
                    continue

                predicted_loc = loc_prob[object_index][:point_count] >= args.threshold
                predicted_mask = mask_prob[object_index][:point_count] >= args.threshold

                order = np.argsort(-mask_prob[object_index][:point_count], kind="stable")
                top_k = np.zeros(point_count, dtype=bool)
                top_k[order[:target_size]] = True
                top_k_in_loc = np.zeros(point_count, dtype=bool)
                candidates = np.flatnonzero(predicted_loc)
                if candidates.size:
                    ranked = candidates[np.argsort(
                        -mask_prob[object_index][:point_count][candidates], kind="stable"
                    )]
                    top_k_in_loc[ranked[:target_size]] = True
                random_k = np.zeros(point_count, dtype=bool)
                random_k[rng.choice(point_count, size=target_size, replace=False)] = True

                if panoptic is None:
                    panoptic_path = Path(args.dataroot) / record.panoptic_path
                    with np.load(panoptic_path) as values:
                        panoptic = np.asarray(values["data"]).reshape(-1)

                target_instance = record.targets[object_index]
                rows.append({
                    "index": index,
                    "sample_token": record.sample_token,
                    "query": record.query,
                    "ordinal_bucket": ordinal_bucket(record.query),
                    "class_id": int(target_instance.class_id),
                    "class_name": target_instance.class_name,
                    "same_class_instances": same_class_instance_count(
                        panoptic, int(target_instance.panoptic_id)
                    ),
                    "point_count": point_count,
                    "target_size": target_size,
                    "predicted_size": int(predicted_mask.sum()),
                    "loc_iou": iou_from_masks(predicted_loc, loc_target),
                    "loc_hit": float(iou_from_masks(predicted_loc, loc_target) >= 0.5),
                    "loc_containment": float(
                        np.logical_and(predicted_loc, target).sum() / target_size
                    ),
                    "auc": auc_from_scores(positive_probability, negative_probability),
                    "mean_prob_positive": float(positive_probability.mean()),
                    "mean_prob_negative": float(negative_probability.mean())
                    if negative_probability.size else float("nan"),
                    "iou_thresholded": iou_from_masks(predicted_mask, target),
                    "hit_thresholded": float(
                        iou_from_masks(predicted_mask, target) >= 0.5
                    ),
                    "iou_top_k": iou_from_masks(top_k, target),
                    "hit_top_k": float(iou_from_masks(top_k, target) >= 0.5),
                    "hit_top_k_in_loc": float(
                        iou_from_masks(top_k_in_loc, target) >= 0.5
                    ),
                    "hit_random_k": float(iou_from_masks(random_k, target) >= 0.5),
                })
            if (index + 1) % 100 == 0:
                print(f"diagnosed {index + 1}/{limit}", flush=True)

    def aggregate(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "auc_mean": float(np.mean([row["auc"] for row in subset])),
            "mean_prob_positive": float(np.mean([row["mean_prob_positive"] for row in subset])),
            "mean_prob_negative": float(np.mean([row["mean_prob_negative"] for row in subset])),
            "loc_iou_mean": float(np.mean([row["loc_iou"] for row in subset])),
            "loc_recall_at_0.5": float(np.mean([row["loc_hit"] for row in subset])),
            "loc_containment_mean": float(np.mean([row["loc_containment"] for row in subset])),
            "recall_thresholded": float(np.mean([row["hit_thresholded"] for row in subset])),
            "recall_top_k": float(np.mean([row["hit_top_k"] for row in subset])),
            "recall_top_k_in_loc": float(np.mean([row["hit_top_k_in_loc"] for row in subset])),
            "recall_random_k": float(np.mean([row["hit_random_k"] for row in subset])),
            "iou_thresholded_mean": float(np.mean([row["iou_thresholded"] for row in subset])),
            "iou_top_k_mean": float(np.mean([row["iou_top_k"] for row in subset])),
            "target_size_median": float(np.median([row["target_size"] for row in subset])),
            "predicted_size_median": float(np.median([row["predicted_size"] for row in subset])),
        }

    by_ordinal: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_ordinal[row["ordinal_bucket"]].append(row)

    def crowd_bucket(row: dict) -> str:
        count = row["same_class_instances"]
        if count < 0:
            return "unknown"
        if count <= 1:
            return "1"
        if count <= 3:
            return "2-3"
        if count <= 7:
            return "4-7"
        return "8+"

    by_crowd: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_crowd[crowd_bucket(row)].append(row)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["class_name"]].append(row)

    report = {
        "overall": aggregate(rows),
        "by_ordinal_bucket": {key: aggregate(by_ordinal[key]) for key in sorted(by_ordinal)},
        "by_same_class_instances": {key: aggregate(by_crowd[key]) for key in sorted(by_crowd)},
        "by_class": {
            key: aggregate(by_class[key])
            for key in sorted(by_class, key=lambda name: -len(by_class[name]))
        },
        "config": {
            "manifest": args.manifest,
            "seg_checkpoint": args.seg_checkpoint,
            "samples": limit,
            "objects": len(rows),
            "threshold": args.threshold,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    args.output.with_suffix(".per_object.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("overall", "by_ordinal_bucket",
                                                    "by_same_class_instances")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
