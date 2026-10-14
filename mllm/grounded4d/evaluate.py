"""Single-frame predicted-instance evaluation; GT is used only for scoring."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from .config import Grounded4DConfig
from .data import Grounded4DDataset, load_scene_tokens
from .diagnostics import QueryDiagnosticTotals, query_diagnostics
from .metrics import MetricTotals, frame_counts
from .model import Grounded4DModel
from .train import CHECKPOINT_SCHEMA, atomic_json, validate_stage


def draw_frame(path, points, labels, prediction_masks, prediction_classes, valid, class_names, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Display only. All quantitative metrics use every original evaluable point.
    indices = np.flatnonzero(valid)
    indices = indices[::max(1, int(np.ceil(len(indices) / 16000)))]
    xyz = points[indices, :3]
    palette = plt.get_cmap("tab10")(np.arange(len(class_names)))
    gt_colors = np.tile([0.75, 0.75, 0.75, 0.2], (len(indices), 1))
    pred_colors = gt_colors.copy()
    for category, mask in labels:
        gt_colors[mask[indices]] = palette[category]
    # Query order only resolves overlapping colors in the viewer; it does not
    # change any predicted mask or the instance-set metric computation.
    for category, mask in zip(prediction_classes, prediction_masks):
        pred_colors[mask[indices]] = palette[category]
    figure = plt.figure(figsize=(13, 9))
    for row, (name, colors) in enumerate((("Ground truth", gt_colors), ("Prediction", pred_colors))):
        bev = figure.add_subplot(2, 2, 2 * row + 1)
        bev.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=1.4, linewidths=0, rasterized=True)
        bev.set(xlabel="x (m)", ylabel="y (m)", title=name + " | BEV", xlim=(-52, 52), ylim=(-52, 52))
        bev.set_aspect("equal")
        ax = figure.add_subplot(2, 2, 2 * row + 2, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=0.6, linewidths=0, rasterized=True)
        ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=name + " | 3D",
               xlim=(-52, 52), ylim=(-52, 52), zlim=(-5, 3))
        ax.view_init(elev=28, azim=-60)
        ax.set_box_aspect((1, 1, 0.2))
    handles = [Patch(color=palette[index], label=name) for index, name in enumerate(class_names)]
    figure.legend(handles=handles, loc="lower center", ncol=5, fontsize=8)
    figure.suptitle(title, fontsize=10)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    figure.savefig(path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def run(args):
    if not 0 <= args.score_threshold <= 1 or not 0 < args.mask_threshold < 1 or not 0 < args.iou_threshold <= 1:
        raise ValueError("invalid prediction or IoU threshold")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; a data audit is not model evaluation")
    checkpoint_raw = args.checkpoint.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_raw).hexdigest()
    checkpoint = torch.load(io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA or metadata["training_config"]["stage"] != "spatial":
        raise ValueError("this evaluator accepts spatial checkpoints only; it is not free tracking or QA evaluation")
    config = Grounded4DConfig.from_dict(metadata["config"])
    if config.encoder != "spconv":
        raise ValueError("debug encoders cannot be evaluated as production spatial models")
    for name in ("model.py", "backbone.py"):
        actual = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        if metadata["source_sha256"].get(name) != actual:
            raise ValueError("checkpoint/model source identity differs: " + name)
    training_scenes, dev_scenes = load_scene_tokens(args.train_scene_tokens), load_scene_tokens(args.dev_scene_tokens)
    if training_scenes & dev_scenes:
        raise ValueError("training/development scene overlap")
    train_probe = args.split == "train_probe"
    split = "train" if train_probe else "internal_dev"
    dataset = Grounded4DDataset(args.manifest, args.nuscenes_root, args.label_root, config,
                               training_scenes if train_probe else dev_scenes, split,
                               "panoptic", args.panoptic_root, args.min_target_points)
    validate_stage(dataset, "spatial")
    preflight = dataset.preflight()
    if preflight["dataset_version"] != metadata["data"]["train"]["dataset_version"]:
        raise ValueError("checkpoint/evaluation dataset versions differ")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    (output / "qualitative").mkdir()
    identity = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
                    checkpoint_completed_epoch=checkpoint["completed_epoch"], checkpoint_global_step=checkpoint["global_step"],
                    checkpoint_initialization=metadata.get("initialization"), training_data=metadata["data"],
                    evaluation_data=preflight, class_names=list(config.class_names),
                    evaluation_role="training_seen_diagnostic" if train_probe else "internal_development",
                    checkpoint_training_config=metadata["training_config"],
                    query_diagnostics_enabled=args.query_diagnostics,
                    evaluated_training_manifest=dataset.manifest_sha256 == metadata["data"]["train"]["manifest_sha256"],
                    protocol="single_frame_predicted_instance_sets_class_aware_iou_matching",
                    model_inputs=["points", "timestamp_us"], gt_used_only_for_scoring=True,
                    prediction=dict(score_threshold=args.score_threshold, mask_threshold=args.mask_threshold,
                                    iou_threshold=args.iou_threshold, no_object_filter="argmax_including_no_object",
                                    nms=False, masks_may_overlap=True),
                    metric_definitions=dict(mean_gt_mask_iou="maximum_sum_class_aware_one_to_one_IoU_all_GT_denominator_unmatched_zero",
                                            recall_iou50="maximum_cardinality_class_aware_one_to_one_hits_over_all_GT",
                                            precision_iou50="hits_over_all_predictions_with_evaluable_points",
                                            geometry_only="diagnostic_matching_without_category_requirement",
                                            ignored_points="ROI_and_official_ignore_classes_excluded_from_scoring"),
                    scope=("training_seen_diagnostic_not_generalization_or_final_acceptance" if train_probe else
                           "official_mini_val_internal_development_not_full_nuscenes_or_answer_grounding_acceptance"),
                    evaluation_source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                              for path in Path(__file__).parent.glob("*.py")})
    atomic_json(output / "evaluation_config.json", identity)
    model = Grounded4DModel(config).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    grouped = {}
    for index, row in enumerate(dataset.records):
        grouped.setdefault(row["scene_token"], []).append(index)
    visual_indices = {indices[index] for indices in grouped.values() for index in (0, len(indices) // 2)}
    total, scenes = MetricTotals(config.class_names), {}
    diagnostics = QueryDiagnosticTotals() if args.query_diagnostics else None
    discarded_no_evaluable_points = 0
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as records:
        for index, row in enumerate(dataset.records):
            frame = dataset[index]["frames"][0]
            # Explicit geometry-only input. No labels, classes, GT identity,
            # references or GT-derived track memory enter the forward pass.
            model_output = model(dict(points=frame["points"].cuda(), timestamp_us=frame["timestamp_us"]), tracks=())
            if not torch.isfinite(model_output["mask_logits"]).all() or not torch.isfinite(model_output["class_logits"]).all():
                raise FloatingPointError("nonfinite model prediction")
            probability = model_output["class_logits"].softmax(-1)
            scores, classes = probability.max(-1)
            selected = (classes < len(config.class_names)) & (scores >= args.score_threshold)
            query_indices = selected.nonzero(as_tuple=True)[0]
            masks = model_output["mask_logits"][selected].sigmoid() >= args.mask_threshold
            roi = model_output["valid_points"].cpu().numpy()
            masks = masks.cpu().numpy() & roi[None]
            classes, scores = classes[selected].cpu().numpy(), scores[selected].cpu().numpy()
            query_indices = query_indices.cpu().numpy()
            labels = frame["labels"].numpy()
            evaluable = roi & (labels >= 0).all(0)
            has_points = masks[:, evaluable].any(1)
            discarded_no_evaluable_points += int((~has_points).sum())
            masks, classes, scores, query_indices = masks[has_points], classes[has_points], scores[has_points], query_indices[has_points]
            gt_masks, gt_classes = labels == 1, frame["class_ids"].numpy()
            counts, assignments = frame_counts(gt_masks[:, evaluable], gt_classes, masks[:, evaluable], classes,
                                                len(config.class_names), args.iou_threshold)
            total.add(counts)
            scenes.setdefault(row["scene_token"], MetricTotals(config.class_names)).add(counts)
            token = frame["sample_data_token"]
            key = hashlib.sha256(token.encode()).hexdigest()
            artifact = output / "frames" / (key + ".npz")
            with artifact.open("xb") as handle:
                np.savez_compressed(handle, predicted_mask_bits=np.packbits(masks, axis=1),
                                    gt_mask_bits=np.packbits(gt_masks, axis=1),
                                    evaluable_point_bits=np.packbits(evaluable), point_count=len(evaluable),
                                    predicted_classes=classes, predicted_scores=scores, query_indices=query_indices,
                                    gt_classes=gt_classes, gt_instance_tokens=np.asarray(frame["instance_tokens"]))
            record = dict(clip_id=row["clip_id"], scene_token=row["scene_token"], sample_data_token=token,
                          original_point_file=row["frames"][0]["point_file"], counts=counts, **assignments,
                          query_indices=query_indices.tolist(), predicted_classes=classes.tolist(),
                          predicted_scores=scores.tolist(), mask_artifact=str(artifact.relative_to(output)),
                          gt_point_counts=gt_masks[:, evaluable].sum(1).tolist(),
                          predicted_point_counts=masks[:, evaluable].sum(1).tolist(),
                          evaluable_point_count=int(evaluable.sum()),
                          mask_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
            if diagnostics is not None:
                all_masks = (model_output["mask_logits"].sigmoid() >= args.mask_threshold).cpu().numpy() & roi[None]
                all_classes = probability.argmax(-1).cpu().numpy()
                all_scores = probability.max(-1).values.cpu().numpy()
                retained = np.zeros(len(all_classes), dtype=bool)
                retained[query_indices] = True
                memory_indices = model_output["valid_points"].nonzero(as_tuple=True)[0]
                if len(memory_indices) > config.memory_points:
                    memory_indices = memory_indices[torch.linspace(0, len(memory_indices) - 1, config.memory_points,
                                                                   device=memory_indices.device).long()]
                diagnostic = query_diagnostics(gt_masks, gt_classes, all_masks, all_classes, all_scores, retained,
                                               len(config.class_names), frame["points"].numpy(), roi, evaluable,
                                               config.memory_points, args.iou_threshold, memory_indices.cpu().numpy())
                if diagnostic["stage_distinct_hits"]["retained_class_aware"] != counts["tp"]:
                    raise AssertionError("query diagnostics changed standard hit accounting")
                diagnostics.add(diagnostic)
                query_artifact = output / "frames" / (key + ".queries.npz")
                with query_artifact.open("xb") as handle:
                    np.savez_compressed(handle, all_query_mask_bits=np.packbits(all_masks, axis=1),
                                        point_count=len(evaluable), class_probabilities=probability.cpu().numpy(),
                                        all_query_classes=all_classes, all_query_scores=all_scores, retained=retained)
                record["query_diagnostic"] = diagnostic
                record["query_artifact"] = str(query_artifact.relative_to(output))
                record["query_artifact_sha256"] = hashlib.sha256(query_artifact.read_bytes()).hexdigest()
            records.write(json.dumps(record, allow_nan=False) + "\n")
            records.flush()
            if index in visual_indices:
                title = f"{row['scene_token']} / {token}\nGT={counts['gt']}, Pred={counts['predictions']}, TP={counts['tp']}, FP={counts['fp']}, FN={counts['fn']}"
                draw_frame(output / "qualitative" / (key + ".png"), frame["points"].numpy(),
                           list(zip(gt_classes, gt_masks)), masks, classes, evaluable, config.class_names, title)
            if (index + 1) % 10 == 0:
                print(json.dumps(dict(event="evaluation_progress", completed_frames=index + 1, total_frames=len(dataset))), flush=True)
    result = dict(identity=identity, metrics=total.result(), per_scene={key: value.result() for key, value in scenes.items()},
                  discarded_predictions_with_no_evaluable_points=discarded_no_evaluable_points,
                  expected_frames=len(dataset), evaluated_frames=total.frames,
                  evaluation_complete=total.frames == len(dataset), model_performance_accepted=False)
    atomic_json(output / "metrics.json", result)
    if diagnostics is not None:
        atomic_json(output / "query_diagnostics.json", dict(identity=identity, **diagnostics.result()))
    print(json.dumps(dict(event="evaluation_complete", output=str(output), metrics=result["metrics"])), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=["internal_dev", "train_probe"], default="internal_dev",
                        help="Explicit training-set diagnosis; train labels are never relabeled as development data")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--nuscenes-root", required=True, type=Path)
    parser.add_argument("--label-root", required=True, type=Path)
    parser.add_argument("--panoptic-root", required=True, type=Path)
    parser.add_argument("--train-scene-tokens", required=True, type=Path)
    parser.add_argument("--dev-scene-tokens", required=True, type=Path)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--query-diagnostics", action="store_true",
                        help="Save all-query masks and disjoint missed-GT diagnostics; does not change predictions")
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
