"""Collect B3 answers, import reviews, train extensions and run free segmentation."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import SCHEMA
from .data import (
    AnswerSegDataset,
    digest,
    load_features,
    prepare_records,
    read_jsonl,
    validate_request,
    verify_feature_frames,
    write_json,
)
from .language import LanguageBridge, make_prompt
from .runtime import (
    build_model,
    file_sha256,
    implementation_identity,
    load_b3,
    load_bundle,
    optimizer_for,
    save_bundle,
    source_identity,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "prepare", "preflight", "train", "predict", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
        if name in ("collect", "train", "predict", "evaluate"):
            command.add_argument("--runtime", type=Path, required=True)
            command.add_argument("--device", default="cuda:0")
        if name in ("prepare", "preflight", "train", "evaluate"):
            command.add_argument("--data", type=Path, required=True)
        if name in ("collect", "predict"):
            command.add_argument("--requests", type=Path, required=True)
            command.add_argument("--feature-root", type=Path, required=True)
        if name in ("collect", "predict", "evaluate"):
            command.add_argument("--max-new-tokens", type=int, default=256)
        if name in ("predict", "evaluate"):
            command.add_argument("--bundle", type=Path, required=True)
            command.add_argument("--selection-threshold", type=float, default=0.5)
        if name == "predict":
            command.add_argument("--point-root", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--source", type=Path, required=True)
            command.add_argument("--feature-index", type=Path, required=True)
            command.add_argument("--reviews", type=Path)
            command.add_argument(
                "--split", choices=("train", "internal_dev"), default="train"
            )
        if name == "train":
            command.add_argument(
                "--stage", choices=("binding", "joint"), default="binding"
            )
            command.add_argument("--tune-spatial", action="store_true")
            command.add_argument("--epochs", type=int, required=True)
            command.add_argument("--lr", type=float, default=1e-4)
            command.add_argument("--minimum-iou", type=float, default=0.5)
            command.add_argument("--lora-rank", type=int, default=16)
            command.add_argument("--lora-alpha", type=int, default=32)
            command.add_argument("--seed", type=int, default=2026)
            command.add_argument(
                "--init-bundle",
                type=Path,
                help="Explicit new-stage initialization; optimizer resets.",
            )
    return parser.parse_args(argv)


def make_dataset(spec, *, development=False, manifest=None):
    from grounded4d.data import load_scene_tokens
    from maskpls_b4dl.config import MaskPLSConfig
    from maskpls_b4dl.data import MaskPLSDataset
    from maskpls_b4dl.protocol import verify_source_split
    from maskpls_b4dl.runner import load_native

    prefix = "dev_" if development else ""
    # Spatial admission uses the EXACT checkpoint architecture/ROI, never a default ROI.
    raw = load_native(spec["spatial_checkpoint"])
    config = MaskPLSConfig.from_dict(raw["metadata"]["config"])
    spatial = MaskPLSDataset(
        spec[prefix + "spatial_manifest"],
        spec["nuscenes_root"],
        spec["label_root"],
        config.data_config(),
        load_scene_tokens(spec[prefix + "scene_tokens"]),
        split="internal_dev" if development else "train",
        panoptic_root=spec["panoptic_root"],
    )
    policy = verify_source_split(spatial, spec["metadata_root"])
    dataset = AnswerSegDataset(
        manifest or spec[prefix + "manifest"], spatial, spec["feature_root"]
    )
    version = spatial.records[0]["dataset_version"]
    policy = dict(
        policy,
        feature_inputs=verify_feature_frames(
            dataset.records, spec["metadata_root"], version
        ),
    )
    return dataset, policy


def preflight(spec):
    train, train_policy = make_dataset(spec)
    dev, dev_policy = make_dataset(spec, development=True)
    if train.spatial.scenes & dev.spatial.scenes:
        raise ValueError("training and development spatial scenes overlap")
    encoder_ids = {
        row["features"]["encoder_checkpoint_id"]
        for data in (train, dev)
        for row in data.records
    }
    if len(encoder_ids) != 1:
        raise ValueError(
            "training/development features must use one identified SST encoder"
        )
    # Read every binding's source points/GT/features now, before allocating B3/GPU.
    return (
        train,
        dev,
        {
            "train": train.preflight(),
            "internal_dev": dev.preflight(),
            "source_split": {"train": train_policy, "internal_dev": dev_policy},
            "model_performance_accepted": False,
        },
    )


def move_example(example, device):
    result = dict(example)
    result["frame"] = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in example["frame"].items()
    }
    result["features"] = example["features"].to(device)
    return result


def collect(args, runtime):
    identity = source_identity(runtime, include_spatial=False)
    tokenizer, language = load_b3(runtime, args.device)
    language.eval()
    bridge = LanguageBridge(language, tokenizer, None)
    seen, count = set(), 0
    output = args.output_dir / "generated.jsonl.part"
    with output.open("x", encoding="utf-8") as handle:
        for request in read_jsonl(args.requests):
            validate_request(request)
            if request["request_id"] in seen:
                raise ValueError("duplicate generation request")
            seen.add(request["request_id"])
            features = load_features(args.feature_root, request)
            generated = bridge.generate(
                make_prompt(request["question"]), features, args.max_new_tokens
            )
            record = {
                key: request[key]
                for key in (
                    "request_id",
                    "scene_token",
                    "sample_data_token",
                    "split",
                    "question",
                    "features",
                )
            }
            record.update(
                schema_version=SCHEMA + "-generated",
                answer=generated["answer"],
                answer_model_id=digest(identity),
                model_identity=identity,
                finished=generated["finished"],
                review_status="pending",
                eligible_for_binding_training=False,
            )
            record["record_fingerprint"] = digest(record)
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    if not count:
        raise ValueError("empty question requests")
    output.replace(args.output_dir / "generated.jsonl")
    return {
        "complete": True,
        "generated": count,
        "answer_model_id": digest(identity),
        "binding_training_ready": False,
        "review_required": True,
    }


def prepare(args, spec):
    records = prepare_records(
        read_jsonl(args.source),
        read_jsonl(args.feature_index),
        read_jsonl(args.reviews) if args.reviews else (),
    )
    if any(row["split"] != args.split for row in records):
        raise ValueError(
            "prepare one original scene split per output; never relabel development as train"
        )
    part = args.output_dir / "reviewed.jsonl.part"
    with part.open("x", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    dataset, policy = make_dataset(
        spec, development=args.split == "internal_dev", manifest=part
    )
    report = dataset.preflight()
    part.replace(args.output_dir / "reviewed.jsonl")
    return {"complete": True, "report": report, "source_split": policy}


def initialize_stage(model, directory, identity):
    """Binding -> joint migration preserves the trained selector, starts new LoRA/optimizer."""
    directory = Path(directory)
    metadata = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
    if metadata["options"]["stage"] != "binding" or model.stage != "joint":
        raise ValueError("only explicit binding-to-joint initialization is implemented")
    if (
        metadata["source_identity"] != identity
        or metadata["implementation_sha256"] != implementation_identity()
        or metadata["extension_sha256"] != file_sha256(directory / "extension.pt")
    ):
        raise ValueError("binding-stage source/extension identity mismatch")
    if vars(model.tokens) != metadata["tokens"]:
        raise ValueError("stage transition changed the restored B3 vocabulary")
    payload = torch.load(
        directory / "extension.pt", map_location="cpu", weights_only=True
    )
    # Read only the small selector; never allocate a second 7B language model on GPU.
    head = {
        key.removeprefix("head."): value
        for key, value in payload["extension"]["parameters"].items()
        if key.startswith("head.")
    }
    model.head.load_state_dict(head, strict=True)


def train(args, runtime, spec):
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if (
        Path(runtime["spatial_checkpoint"]).resolve()
        != Path(spec["spatial_checkpoint"]).resolve()
    ):
        raise ValueError(
            "data admission and model must use the same spatial checkpoint"
        )
    train_data, dev_data, report = preflight(spec)
    write_json(args.output_dir / "preflight.json", report)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    identity = source_identity(runtime)
    model, options = build_model(
        runtime,
        args.device,
        stage=args.stage,
        tune_spatial=args.tune_spatial,
        minimum_iou=args.minimum_iou,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    if args.init_bundle:
        initialize_stage(model, args.init_bundle, identity)
    optimizer = optimizer_for(model, args.lr)
    step, binding_supervision = 0, 0
    history = args.output_dir / "training.jsonl"
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train_data)))
        random.shuffle(order)
        for index in order:
            example = move_example(train_data[index], args.device)
            optimizer.zero_grad(set_to_none=True)
            result = model(example)
            binding_supervision += result["supervised_bindings"]
            update = args.stage == "joint" or result["supervised_bindings"] > 0
            if update:
                result["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                step += 1
            else:
                norm = torch.tensor(0.0)
            record = {
                key: float(value.detach()) if isinstance(value, torch.Tensor) else value
                for key, value in result.items()
            }
            record.update(
                epoch=epoch + 1,
                global_step=step,
                request_id=example["request_id"],
                optimizer_updated=update,
                gradient_norm=float(norm),
            )
            with history.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        if step == 0 or binding_supervision == 0:
            raise ValueError(
                "zero supervised optimizer updates: spatial candidates did not cover reviewed targets"
            )
        model.eval()
        dev_totals = Counter()
        with torch.no_grad():
            for index in range(len(dev_data)):
                result = model(move_example(dev_data[index], args.device))
                for key in (
                    "positive_targets",
                    "covered_targets",
                    "supervised_bindings",
                    "skipped_uncovered",
                ):
                    dev_totals[key] += result[key]
                dev_totals["language_loss_sum"] += float(result["language_loss"])
        training = {
            "completed_epoch": epoch + 1,
            "global_step": step,
            "seed": args.seed,
            "data": report,
            "teacher_forced_development": dict(dev_totals),
            "init_bundle_sha256": file_sha256(args.init_bundle / "bundle.json")
            if args.init_bundle
            else None,
            "optimizer_reset": True,
            "free_generation_accepted": False,
        }
        save_bundle(
            args.output_dir / f"epoch-{epoch + 1}",
            model,
            options,
            identity,
            training,
            optimizer,
        )
    return {
        "training_workflow_complete": True,
        "completed_epoch": args.epochs,
        "global_step": step,
        "bundle": str(args.output_dir / f"epoch-{args.epochs}"),
        "model_performance_accepted": False,
        "free_generation_accepted": False,
    }


def input_points(root, request):
    from grounded4d.data import read_hashed_file

    raw = read_hashed_file(root, request["point_file"])
    if not raw or len(raw) % 20:
        raise ValueError("expected original float32 XYZ/intensity/ring LiDAR rows")
    values = np.frombuffer(raw, dtype="<f4").reshape(-1, 5)
    if (
        not np.isfinite(values).all()
        or (values[:, 4] < 0).any()
        or (values[:, 4] > 31).any()
        or (values[:, 4] != np.floor(values[:, 4])).any()
    ):
        raise ValueError("invalid original nuScenes points")
    return torch.from_numpy(values[:, :4].copy())


def save_prediction(directory, request, prediction, point_sha256, identity):
    stem = digest([request["request_id"], request["sample_data_token"]])
    masks = prediction.pop("masks")
    count = len(masks[0]) if masks else request["point_count"]
    arrays = (
        np.stack([mask.cpu().numpy() for mask in masks])
        if masks
        else np.empty((0, count), dtype=bool)
    )
    np.savez_compressed(
        directory / (stem + ".npz"),
        masks=arrays,
        point_indices=np.arange(count),
        point_sha256=np.asarray(point_sha256),
    )
    record = dict(
        prediction,
        schema_version=SCHEMA + "-prediction",
        request_id=request["request_id"],
        sample_data_token=request["sample_data_token"],
        scene_token=request["scene_token"],
        mask_file=stem + ".npz",
        point_sha256=point_sha256,
        point_order="original_nuscenes_rows",
        model_identity={
            key: identity[key]
            for key in (
                "source_identity",
                "implementation_sha256",
                "extension_sha256",
                "tokens",
            )
        },
        tracking_accepted=False,
    )
    for obj in record["objects"]:
        query = obj["query_index"]
        obj["object_id"] = (
            f"{request['sample_data_token']}:q{query}" if query is not None else None
        )
    write_json(directory / (stem + ".json"), record)


def predict(args, runtime):
    model, metadata = load_bundle(args.bundle, runtime, args.device)
    model.eval()
    statuses, seen = Counter(), set()
    for request in read_jsonl(args.requests):
        validate_request(request)
        if request["request_id"] in seen:
            raise ValueError("duplicate prediction request")
        seen.add(request["request_id"])
        points = input_points(args.point_root, request).to(args.device)
        prediction = model.predict(
            make_prompt(request["question"]),
            load_features(args.feature_root, request),
            points,
            max_new_tokens=args.max_new_tokens,
            selection_threshold=args.selection_threshold,
        )
        statuses[prediction["status"]] += 1
        save_prediction(
            args.output_dir,
            dict(request, point_count=len(points)),
            prediction,
            request["point_file"]["sha256"],
            metadata,
        )
    if not seen:
        raise ValueError("empty prediction requests")
    return {
        "complete": True,
        "records": len(seen),
        "statuses": dict(statuses),
        "model_performance_accepted": False,
    }


def score_prediction(prediction, example, totals):
    """All positive targets stay in the denominator, including parse/missing-token failures."""
    from scipy.optimize import linear_sum_assignment

    gt = [
        example["frame"]["labels"][
            example["frame"]["instance_tokens"].index(target["instance_token"])
        ]
        for target in example["targets"]
        if target["status"] == "matched"
    ]
    predicted = [
        mask.cpu()
        for mask, obj in zip(prediction["masks"], prediction["objects"])
        if obj["status"] == "selected"
    ]
    totals["records"] += 1
    totals["positive_targets"] += len(gt)
    totals["predicted_masks"] += len(predicted)
    if not gt:
        totals["no_target_records"] += 1
        totals["correct_no_target_records"] += int(
            prediction["status"] == "ok"
            and bool(prediction["objects"])
            and all(
                obj["status"] == "explicit_no_target" for obj in prediction["objects"]
            )
        )
    iou = np.zeros((len(gt), len(predicted)))
    for row, truth in enumerate(gt):
        valid = truth.ge(0).cpu()
        target = truth.eq(1).cpu()
        for column, mask in enumerate(predicted):
            union = ((mask | target) & valid).sum()
            iou[row, column] = float(
                ((mask & target) & valid).sum() / union.clamp_min(1)
            )
    hits = 0
    if iou.size:
        score = (iou >= 0.5) * (min(iou.shape) + 1) + iou
        rows, columns = linear_sum_assignment(-score)
        hits = int((iou[rows, columns] >= 0.5).sum())
        totals["best_target_iou_sum"] += float(iou.max(1).sum())
    totals["true_positives"] += hits
    totals["false_positives"] += len(predicted) - hits
    totals["false_negatives"] += len(gt) - hits


def evaluate(args, runtime, spec):
    if (
        Path(runtime["spatial_checkpoint"]).resolve()
        != Path(spec["spatial_checkpoint"]).resolve()
    ):
        raise ValueError("evaluation admission and runtime spatial checkpoint differ")
    _, dataset, report = preflight(spec)
    model, metadata = load_bundle(args.bundle, runtime, args.device)
    model.eval()
    totals, statuses = Counter(), Counter()
    for index in range(len(dataset)):
        example = dataset[index]
        # Only these three inputs enter prediction; GT is read later by score_prediction.
        prediction = model.predict(
            example["prompt"],
            example["features"],
            example["frame"]["points"].to(args.device),
            max_new_tokens=args.max_new_tokens,
            selection_threshold=args.selection_threshold,
        )
        statuses[prediction["status"]] += 1
        score_prediction(prediction, example, totals)
        save_prediction(
            args.output_dir,
            dict(example, point_count=len(example["frame"]["points"])),
            prediction,
            example["frame"]["point_sha256"],
            metadata,
        )
    value = dict(totals)
    value.update(
        recall_at_0_5=totals["true_positives"] / totals["positive_targets"]
        if totals["positive_targets"]
        else None,
        precision_at_0_5=totals["true_positives"] / totals["predicted_masks"]
        if totals["predicted_masks"]
        else None,
        mean_best_target_iou=totals["best_target_iou_sum"] / totals["positive_targets"]
        if totals["positive_targets"]
        else None,
        no_target_accuracy=totals["correct_no_target_records"]
        / totals["no_target_records"]
        if totals["no_target_records"]
        else None,
        statuses=dict(statuses),
        data=report,
        complete=True,
        evaluation_scope="reviewed internal_dev bindings; B3/spatial exposure must be audited separately",
        model_performance_accepted=False,
    )
    return value


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runtime = (
        json.loads(args.runtime.read_text(encoding="utf-8-sig"))
        if hasattr(args, "runtime")
        else None
    )
    spec = (
        json.loads(args.data.read_text(encoding="utf-8-sig"))
        if hasattr(args, "data")
        else None
    )
    if args.command == "collect":
        result = collect(args, runtime)
    elif args.command == "prepare":
        result = prepare(args, spec)
    elif args.command == "preflight":
        _, _, result = preflight(spec)
    elif args.command == "train":
        result = train(args, runtime, spec)
    elif args.command == "predict":
        result = predict(args, runtime)
    else:
        result = evaluate(args, runtime, spec)
    write_json(args.output_dir / "completion.json", result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result
