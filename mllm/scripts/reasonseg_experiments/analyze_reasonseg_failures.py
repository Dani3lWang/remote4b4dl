#!/usr/bin/env python3
"""Decompose ReasonSeg free-generation failures offline from saved npz masks.

Reads an ``evaluate_reasonseg.py`` output directory (predictions.json + masks/)
and answers three questions without touching the GPU:

1. Does recall@0.5 collapse as the ordinal in the query grows ("nearest" vs
   "17th-nearest")?  That separates ordinal grounding from mask quality.
2. What kind of mistake dominates: no prediction, empty mask, right class in
   the wrong place, or a near miss just below IoU 0.5?
3. Is the generated text still a faithful copy of the reference answer, i.e. is
   the language side already saturated?
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

MLLM_ROOT = Path(__file__).resolve().parents[2]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))

from vtimellm.segmentation.metrics import (  # noqa: E402
    optimal_iou_matching,
    pairwise_mask_iou,
)

ORDINAL_RE = re.compile(r"(\d+)(?:st|nd|rd|th)-nearest")
SIDE_RE = re.compile(r"\b(left|right)-side\b")
SPECIAL_TOKENS = ("<LOC>", "<SEG>", "</s>", "<s>")
IOU_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01)


def ordinal_bucket(query: str) -> str:
    match = ORDINAL_RE.search(query)
    if match is None:
        return "no_ordinal"
    ordinal = int(match.group(1))
    if ordinal <= 1:
        return "1"
    if ordinal <= 3:
        return "2-3"
    if ordinal <= 9:
        return "4-9"
    return "10+"


def normalize_text(text: str) -> str:
    for token in SPECIAL_TOKENS:
        text = text.replace(token, " ")
    return " ".join(text.split()).strip().lower()


def iou_bin(value: float) -> str:
    for low, high in zip(IOU_BINS, IOU_BINS[1:]):
        if low <= value < high:
            return f"[{low:.1f},{high:.1f})" if high <= 1.0 else "[0.7,1.0]"
    return f"[{IOU_BINS[-2]:.1f},1.0]"


def classify(
    target_index: int,
    ious: np.ndarray,
    matched: dict[int, int],
    predicted_classes: np.ndarray,
    target_classes: np.ndarray,
    predicted_sizes: np.ndarray,
) -> tuple[str, float]:
    """Return (failure_label, best_iou) for one ground-truth instance."""
    if target_index in matched:
        predicted_index = matched[target_index]
        iou = float(ious[predicted_index, target_index])
        if iou >= 0.5:
            return "hit", iou
        return "near_miss_below_0.5", iou
    if ious.size == 0:
        return "no_prediction_at_all", 0.0
    best_predicted = int(np.argmax(ious[:, target_index]))
    best_iou = float(ious[best_predicted, target_index])
    if predicted_sizes[best_predicted] == 0:
        return "predicted_mask_empty", best_iou
    if best_iou >= 0.2:
        return "overlap_stolen_by_other_target", best_iou
    same_class = int(predicted_classes[best_predicted]) == int(target_classes[target_index])
    return "right_class_wrong_place" if same_class else "wrong_class_wrong_place", best_iou


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        help="reasonseg_config.json，用于把 class id 映射回类名")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    class_names: dict[int, str] = {}
    if args.config and args.config.is_file():
        names = json.loads(args.config.read_text(encoding="utf-8")).get("config", {}).get("class_names")
        class_names = {index: str(name) for index, name in enumerate(names or [])}

    predictions = json.loads((args.eval_dir / "predictions.json").read_text(encoding="utf-8"))

    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"targets": 0, "hits": 0})
    failures: Counter[str] = Counter()
    per_class: dict[int, dict[str, int]] = defaultdict(lambda: {"targets": 0, "hits": 0})
    iou_hist: Counter[str] = Counter()
    sides: dict[str, dict[str, int]] = defaultdict(lambda: {"targets": 0, "hits": 0})
    predicted_sizes: list[int] = []
    target_sizes: list[int] = []
    text_exact = 0
    text_similarity: list[float] = []
    status_counts: Counter[str] = Counter()
    samples_with_no_prediction = 0
    rows: list[dict] = []

    for entry in predictions:
        npz_path = args.eval_dir / entry["mask_file"]
        with np.load(npz_path) as npz:
            predicted = np.asarray(npz["predicted_masks"], dtype=bool)
            predicted_classes = np.asarray(npz["predicted_classes"], dtype=np.int64)
            targets = np.asarray(npz["target_masks"], dtype=bool)
            target_classes = np.asarray(npz["target_classes"], dtype=np.int64)
        if predicted.ndim == 1:
            predicted = predicted.reshape(0, -1) if predicted.size == 0 else predicted.reshape(1, -1)
        if targets.ndim == 1:
            targets = targets.reshape(0, -1) if targets.size == 0 else targets.reshape(1, -1)

        status_counts[entry.get("status", "unknown")] += 1
        predicted_sizes.extend(int(mask.sum()) for mask in predicted)
        target_sizes.extend(int(mask.sum()) for mask in targets)
        if len(predicted) == 0:
            samples_with_no_prediction += 1

        normalized_generated = normalize_text(entry.get("generated", ""))
        normalized_answer = normalize_text(entry.get("answer", ""))
        if normalized_generated == normalized_answer:
            text_exact += 1
        text_similarity.append(
            SequenceMatcher(None, normalized_generated, normalized_answer).ratio()
        )

        ious, _, _ = pairwise_mask_iou(predicted, targets)
        matches = optimal_iou_matching(ious)
        matched = {target_index: predicted_index
                   for predicted_index, target_index in matches}
        pred_size_array = np.asarray([int(mask.sum()) for mask in predicted], dtype=np.int64)

        bucket = ordinal_bucket(entry["query"])
        side_match = SIDE_RE.search(entry["query"])
        side = side_match.group(1) if side_match else "none"
        sample_labels = []
        for target_index in range(len(targets)):
            label, best_iou = classify(
                target_index, ious, matched, predicted_classes, target_classes, pred_size_array
            )
            failures[label] += 1
            iou_hist[iou_bin(best_iou)] += 1
            buckets[bucket]["targets"] += 1
            sides[side]["targets"] += 1
            target_class = int(target_classes[target_index])
            per_class[target_class]["targets"] += 1
            if label == "hit":
                buckets[bucket]["hits"] += 1
                sides[side]["hits"] += 1
                per_class[target_class]["hits"] += 1
            sample_labels.append(label)
        rows.append({
            "index": entry["index"],
            "sample_token": entry["sample_token"],
            "query": entry["query"],
            "bucket": bucket,
            "n_predicted": int(len(predicted)),
            "n_targets": int(len(targets)),
            "labels": sample_labels,
            "text_exact": normalized_generated == normalized_answer,
        })

    def rate(counter: dict[str, int]) -> float:
        return counter["hits"] / counter["targets"] if counter["targets"] else 0.0

    total_targets = sum(bucket["targets"] for bucket in buckets.values())
    total_hits = sum(bucket["hits"] for bucket in buckets.values())
    report = {
        "eval_dir": str(args.eval_dir),
        "samples": len(predictions),
        "targets": total_targets,
        "recall_at_0.5": total_hits / total_targets if total_targets else 0.0,
        "status_counts": dict(status_counts),
        "samples_with_zero_predictions": samples_with_no_prediction,
        "by_ordinal_bucket": {
            name: {**buckets[name], "recall": rate(buckets[name])}
            for name in ("1", "2-3", "4-9", "10+", "no_ordinal") if name in buckets
        },
        "by_side": {
            name: {**sides[name], "recall": rate(sides[name])} for name in sorted(sides)
        },
        "by_class": {
            (class_names.get(int(key), str(key))): {**value, "recall": rate(value)}
            for key, value in sorted(per_class.items(), key=lambda item: -item[1]["targets"])
        },
        "failure_labels": dict(failures.most_common()),
        "matched_iou_histogram": {
            name: iou_hist[name] for name in
            ("[0.0,0.1)", "[0.1,0.2)", "[0.2,0.3)", "[0.3,0.4)", "[0.4,0.5)", "[0.5,0.7)", "[0.7,1.0]")
            if name in iou_hist
        },
        "mask_size_points": {
            "predicted_mean": float(np.mean(predicted_sizes)) if predicted_sizes else 0.0,
            "predicted_median": float(np.median(predicted_sizes)) if predicted_sizes else 0.0,
            "predicted_zero_count": int(sum(1 for size in predicted_sizes if size == 0)),
            "target_mean": float(np.mean(target_sizes)) if target_sizes else 0.0,
            "target_median": float(np.median(target_sizes)) if target_sizes else 0.0,
        },
        "text": {
            "exact_match_rate": text_exact / len(predictions) if predictions else 0.0,
            "mean_similarity": float(np.mean(text_similarity)) if text_similarity else 0.0,
            "min_similarity": float(np.min(text_similarity)) if text_similarity else 0.0,
        },
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        (args.output.with_suffix(".per_sample.jsonl")).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
