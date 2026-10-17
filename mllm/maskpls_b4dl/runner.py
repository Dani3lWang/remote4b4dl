"""Single-GPU MaskPLS data audit, smoke, training, inference and evaluation."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version as package_version
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from grounded4d.data import load_scene_tokens
from grounded4d.metrics import MetricTotals
from grounded4d.taxonomy import THING_CLASSES
from grounded4d.train import atomic_json, rng_state, restore_rng, to_device
from .config import MaskPLSConfig, SCHEMA, UPSTREAM_COMMIT
from .criterion import MaskPLSCriterion
from .data import MaskPLSDataset
from .model import MaskPLS, extract_instances, load_official, sha256_file
from .objects import save_prediction
from .protocol import implementation_digest, score_frame, verify_source_split


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "smoke", "train", "evaluate", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--seed", type=int, default=2026)
        if name != "predict":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--scene-tokens", type=Path, required=True)
            command.add_argument("--split", choices=("train", "internal_dev"), default="train")
            command.add_argument("--nuscenes-root", type=Path, required=True)
            command.add_argument("--label-root", type=Path, required=True)
            command.add_argument("--panoptic-root", type=Path, required=True)
            command.add_argument("--metadata-root", type=Path, required=True)
            command.add_argument("--official-val", action="store_true",
                                 help="Full official val is evaluation only; never training/early stopping.")
        if name != "preflight":
            weights = command.add_mutually_exclusive_group()
            weights.add_argument("--official-checkpoint", type=Path)
            if name == "train":
                weights.add_argument("--resume", type=Path)
                weights.add_argument("--warm-start", type=Path,
                                     help="Initialize a new run from native weights; reset optimizer and schedule.")
                command.add_argument("--warm-start-source-sha256",
                                     help="Required source implementation digest for an audited warm start.")
            else:
                weights.add_argument("--checkpoint", type=Path)
        if name in ("evaluate", "predict"):
            command.add_argument("--save-all-queries", action="store_true")
        if name == "train":
            command.add_argument("--validation-manifest", type=Path, required=True)
            command.add_argument("--dev-scene-tokens", type=Path, required=True)
            command.add_argument("--epochs", type=int, required=True)
            command.add_argument("--accumulate", type=int, default=4)
            command.add_argument("--lr", type=float, default=1e-4)
            command.add_argument("--gradient-clip", type=float, default=1.0)
        if name == "predict":
            command.add_argument("--point-file", type=Path, required=True)
            command.add_argument("--sample-token", required=True)
            command.add_argument("--sample-data-token", required=True)
            command.add_argument("--timestamp-us", type=int)
    return parser.parse_args(argv)


def load_native(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    metadata = value.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA:
        raise ValueError("not a B4DL MaskPLS checkpoint; use --official-checkpoint for upstream weights")
    if metadata.get("implementation_sha256") != implementation_digest():
        raise ValueError("checkpoint implementation differs; use its recorded code version")
    return value


def load_warm_start(path, config, source_sha256):
    """Explicit loss-policy migration, never an implicit bypass of exact resume."""
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_sha256)):
        raise ValueError("warm start requires the audited source implementation SHA-256")
    value = torch.load(path, map_location="cpu", weights_only=True)
    metadata = value.get("metadata", {})
    if (metadata.get("schema_version") != SCHEMA
            or metadata.get("implementation_sha256") != source_sha256
            or metadata.get("upstream_commit") != UPSTREAM_COMMIT):
        raise ValueError("warm-start source identity mismatch")
    old = MaskPLSConfig.from_dict(metadata["config"]).to_dict()
    new = config.to_dict()
    changed = {key for key in new if old[key] != new[key]}
    if changed - {"mask_ignore_policy", "mask_bce_reduction", "no_object_weight"}:
        raise ValueError("warm start may change only explicit loss policies")
    return value


def check_warm_start_data(source, target):
    # This migration is intentionally narrow: no unseen change in training scenes,
    # no prior training exposure to a newly claimed held-out validation split.
    for key in ("data", "internal_dev_data"):
        if key not in source or source[key] != target.get(key):
            raise ValueError("warm start requires identical training and development data")


def make_dataset(args, config, *, development=False):
    manifest = args.validation_manifest if development else args.manifest
    whitelist = args.dev_scene_tokens if development else args.scene_tokens
    dataset = MaskPLSDataset(manifest, args.nuscenes_root, args.label_root, config.data_config(),
                             load_scene_tokens(whitelist),
                             split="internal_dev" if development else args.split,
                             supervision="panoptic", panoptic_root=args.panoptic_root,
                             min_target_points=1)
    policy = verify_source_split(dataset, args.metadata_root,
                                 official_val=args.official_val and not development)
    report = dataset.preflight(require_capacity=args.command in ("preflight", "train", "smoke")
                               and not args.official_val)
    return dataset, dict(**report, source_split=policy)


def base_metadata(config):
    return dict(schema_version=SCHEMA, config=config.to_dict(), upstream_commit=UPSTREAM_COMMIT,
                implementation_sha256=implementation_digest(),
                prediction_inputs="raw XYZ/intensity only", min_instance_points=1,
                language_grounding_accepted=False, tracking_accepted=False,
                model_performance_accepted=False)


def log_event(path, event):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


@torch.no_grad()
def development_loss(model, criterion, dataset, device):
    model.eval()
    total = 0.0
    for index in range(len(dataset)):
        frame = to_device(dataset[index]["frames"][0], device)
        loss, _, _ = criterion(model(frame["points"]), frame)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite development loss")
        total += float(loss)
    return dict(role="loss_diagnostic_not_mask_metrics", frames=len(dataset), loss=total / len(dataset))


def train(model, dataset, development, args, metadata, checkpoint, device):
    if args.epochs < 1 or args.accumulate < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("positive epochs, accumulation and learning rate required")
    if not np.isfinite(args.gradient_clip) or args.gradient_clip <= 0:
        raise ValueError("positive gradient clip required")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
    recipe = dict(lr=args.lr, accumulate=args.accumulate, gradient_clip=args.gradient_clip,
                  seed=args.seed, precision="FP32", batch_size=1, scheduler="StepLR(80,0.1)")
    metadata["recipe"] = recipe
    epoch_start, step = 0, 0
    if checkpoint:
        if checkpoint["metadata"] != metadata:
            raise ValueError("resume changes config, code, dataset, split or training recipe")
        epoch_start, step = checkpoint["completed_epoch"], checkpoint["global_step"]
        if args.epochs <= epoch_start:
            raise ValueError("resume target epochs must exceed the completed epoch")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng"])
        # A rejected resume must not overwrite the previous experiment's status.
        args._output_owned = True
        completion = args.output_dir / "completion.json"
        if completion.exists():
            completion.replace(args.output_dir / f"completion_before_resume_{time.time_ns()}.json")
    atomic_json(args.output_dir / "status.json", dict(status="running", target_epochs=args.epochs,
                                                     resume_from_epoch=epoch_start))
    criterion = MaskPLSCriterion(model.config)
    for epoch in range(epoch_start, args.epochs):
        model.train()
        order = torch.randperm(len(dataset)).tolist()
        total_loss = 0.0
        for offset in range(0, len(order), args.accumulate):
            window = order[offset:offset + args.accumulate]
            optimizer.zero_grad(set_to_none=True)
            parts = Counter()
            for index in window:
                frame = to_device(dataset[index]["frames"][0], device)
                output = model(frame["points"])
                loss, components, _ = criterion(output, frame)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                (loss / len(window)).backward()
                total_loss += float(loss.detach())
                for key, value in components.items():
                    parts[key] += float(value.detach()) / len(window)
                del output, loss, components
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            log_event(args.output_dir / "steps.jsonl",
                      dict(epoch=epoch + 1, global_step=step, gradient_norm=float(norm), **parts))
        diagnostic = development_loss(model, criterion, development, device)
        scheduler.step()
        state = dict(metadata=metadata, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), rng=rng_state(),
                     completed_epoch=epoch + 1, global_step=step)
        temporary = args.output_dir / "last.pt.part"
        torch.save(state, temporary)
        temporary.replace(args.output_dir / "last.pt")
        progress = dict(completed_epoch=epoch + 1, global_step=step, train_loss=total_loss / len(dataset),
                        internal_dev=diagnostic,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
                        model_performance_accepted=False)
        atomic_json(args.output_dir / "progress.json", progress)
        log_event(args.output_dir / "epochs.jsonl", progress)
    return dict(status="completed", completed_epoch=max(epoch_start, args.epochs), global_step=step,
                checkpoint=str(args.output_dir / "last.pt"),
                checkpoint_sha256=sha256_file(args.output_dir / "last.pt"), model_performance_accepted=False)


@torch.no_grad()
def evaluate(model, dataset, args, identity, device):
    model.eval()
    totals = MetricTotals(THING_CLASSES)
    geometry, class_hits, best_sum = 0, 0, 0.0
    buckets = {name: Counter() for name in ("1-10", "11-49", "50+")}
    for index in range(len(dataset)):
        frame = to_device(dataset[index]["frames"][0], device)
        output = model(frame["points"])
        instances = extract_instances(output, model.config)
        counts, matches, diagnostic = score_frame(frame, output, instances, model.config.mask_threshold)
        totals.add(counts)
        geometry += diagnostic["raw_query_geometry_hits"]
        class_hits += diagnostic["raw_query_class_aware_hits"]
        best_sum += diagnostic["best_geometry_iou_sum"]
        for name, value in diagnostic["point_buckets"].items():
            buckets[name].update(value)
        save_prediction(args.output_dir / "predictions", frame, output, instances, identity,
                        args.save_all_queries)
        log_event(args.output_dir / "frame_metrics.jsonl",
                  dict(sample_data_token=frame["sample_data_token"], **counts,
                       matching=matches, candidate_diagnostic=diagnostic))
        atomic_json(args.output_dir / "progress.json",
                    dict(completed_frames=index + 1, expected_frames=len(dataset), partial=True))
    metrics = totals.result()
    count = metrics["gt_instances"]
    metrics["candidate_diagnostic"] = dict(
        role="oracle_scoring_not_answer_selection",
        raw_query_geometry_recall=geometry / count if count else None,
        raw_query_class_aware_recall=class_hits / count if count else None,
        mean_best_geometry_iou=best_sum / count if count else None)
    metrics["point_buckets"] = {name: dict(**value, recall=value["tp"] / value["gt"] if value["gt"] else None)
                               for name, value in buckets.items()}
    metrics.update(identity=identity, complete=True, scope="all admitted manifest frames",
                   benchmark_pq=False, language_grounding_accepted=False, tracking_accepted=False,
                   noobj_query_capability="not evaluated; manifest is spatial instance supervision")
    atomic_json(args.output_dir / "metrics.json", metrics)
    atomic_json(args.output_dir / "progress.json",
                dict(completed_frames=len(dataset), expected_frames=len(dataset), partial=False))
    return metrics


def raw_frame(args, device):
    if args.timestamp_us is not None and args.timestamp_us < 0:
        raise ValueError("timestamp_us must be nonnegative")
    raw = args.point_file.read_bytes()
    if not raw or len(raw) % 20:
        raise ValueError("expected original float32 nuScenes [N,5] LiDAR file")
    points = np.frombuffer(raw, dtype="<f4").reshape(-1, 5).copy()
    if (not np.isfinite(points).all() or (points[:, 4] < 0).any() or (points[:, 4] > 31).any()
            or (points[:, 4] != np.floor(points[:, 4])).any()):
        raise ValueError("invalid original nuScenes points/rings")
    return dict(points=torch.from_numpy(points[:, :4].copy()).to(device),
                point_sha256=hashlib.sha256(raw).hexdigest(),
                sample_token=args.sample_token, sample_data_token=args.sample_data_token,
                timestamp_us=args.timestamp_us)


def run(args):
    is_train = args.command == "train"
    warm_start = getattr(args, "warm_start", None)
    warm_source_sha = getattr(args, "warm_start_source_sha256", None)
    if warm_source_sha and not warm_start:
        raise ValueError("--warm-start-source-sha256 requires --warm-start")
    if args.command in ("train", "smoke") and (args.split != "train" or args.official_val):
        raise ValueError("training/smoke cannot use official validation or internal-dev")
    checkpoint_path = getattr(args, "resume", None) or getattr(args, "checkpoint", None)
    checkpoint = load_native(checkpoint_path) if checkpoint_path else None
    if args.config:
        config = MaskPLSConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8-sig")))
        if checkpoint and config.to_dict() != checkpoint["metadata"]["config"]:
            raise ValueError("explicit config differs from checkpoint")
    elif checkpoint:
        config = MaskPLSConfig.from_dict(checkpoint["metadata"]["config"])
    else:
        config = MaskPLSConfig()
    warm_checkpoint = load_warm_start(warm_start, config, warm_source_sha) if warm_start else None
    if args.command in ("evaluate", "predict") and not (checkpoint or args.official_checkpoint):
        raise ValueError("inference requires trained weights; random initialization is not an evaluation")
    # Frozen official inference must keep the original 100 query rows.
    if args.command in ("evaluate", "predict") and args.official_checkpoint and config.num_queries != 100:
        raise ValueError("official frozen inference requires num_queries=100; expansion is training-only")
    resume = getattr(args, "resume", None)
    if resume:
        if args.output_dir.resolve() != resume.resolve().parent or resume.name != "last.pt":
            raise ValueError("resume must use last.pt in its existing experiment directory")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        args._output_owned = True
    metadata = base_metadata(config)
    dataset = development = None
    if args.command != "predict":
        dataset, report = make_dataset(args, config)
        metadata["data"] = report
        if is_train:
            development, dev_report = make_dataset(args, config, development=True)
            if dataset.scenes & development.scenes:
                raise ValueError("train and internal-dev scenes overlap")
            if report["dataset_version"] != dev_report["dataset_version"]:
                raise ValueError("train/dev dataset versions differ")
            metadata["internal_dev_data"] = dev_report
            if warm_checkpoint:
                check_warm_start_data(warm_checkpoint["metadata"], metadata)
        if not resume:
            atomic_json(args.output_dir / "preflight.json", metadata)
    if args.command == "preflight":
        return metadata
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production MaskPLS requires CUDA and a matching MinkowskiEngine build")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = MaskPLS(config).to(device)
    metadata["runtime"] = dict(torch=str(torch.__version__), torch_cuda=torch.version.cuda,
                               minkowski_engine=package_version("MinkowskiEngine"),
                               knn_backend=config.knn_backend,
                               pykeops=package_version("pykeops") if config.knn_backend == "keops" else None,
                               gpu=torch.cuda.get_device_name(device))
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        metadata["initialization"] = checkpoint["metadata"].get("initialization")
    elif warm_checkpoint:
        model.load_state_dict(warm_checkpoint["model"], strict=True)
        metadata["initialization"] = dict(
            kind="native_warm_start", checkpoint_sha256=sha256_file(warm_start),
            source_implementation_sha256=warm_source_sha,
            source_config=warm_checkpoint["metadata"]["config"],
            source_completed_epoch=warm_checkpoint["completed_epoch"],
            source_global_step=warm_checkpoint["global_step"],
            source_initialization=warm_checkpoint["metadata"].get("initialization"),
            pretrained_official_train_exposure=warm_checkpoint["metadata"].get(
                "pretrained_official_train_exposure", False), optimizer_reset=True)
    elif args.official_checkpoint:
        metadata["initialization"] = load_official(model, args.official_checkpoint,
                                                   allow_query_expansion=is_train or args.command == "smoke")
    else:
        metadata["initialization"] = dict(kind="random", seed=args.seed)
    initialization = metadata["initialization"] or {}
    metadata["pretrained_official_train_exposure"] = bool(
        initialization.get("pretrained_official_train_exposure", False)
        if initialization.get("kind") == "native_warm_start"
        else "checkpoint_sha256" in initialization)
    if is_train:
        return train(model, dataset, development, args, metadata, checkpoint, device)
    identity = dict(**metadata, weights_sha256=sha256_file(checkpoint_path or args.official_checkpoint)
                    if checkpoint_path or args.official_checkpoint else None)
    if args.command == "smoke":
        model.train()
        frame = to_device(dataset[0]["frames"][0], device)
        started = time.monotonic()
        output = model(frame["points"])
        loss, parts, _ = MaskPLSCriterion(config)(output, frame)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite smoke loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        torch.cuda.synchronize(device)
        result = dict(status="forward_backward_completed", frames=1, loss=float(loss.detach()),
                      gradient_norm=float(norm), seconds=time.monotonic() - started,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
                      identity=identity, model_performance_accepted=False)
        atomic_json(args.output_dir / "smoke.json", result)
        return result
    if args.command == "evaluate":
        return evaluate(model, dataset, args, identity, device)
    model.eval()
    frame = raw_frame(args, device)
    with torch.no_grad():
        output = model(frame["points"])
        save_prediction(args.output_dir / "predictions", frame, output,
                        extract_instances(output, config), identity, args.save_all_queries)
    return dict(status="predicted", frames=1, output_dir=str(args.output_dir),
                grounding_verified=False, track_verified=False)


def main():
    args = parse_args()
    try:
        result = run(args)
    except Exception as error:
        # Failure is not completion; preserve committed checkpoints and partial per-frame artifacts.
        if getattr(args, "_output_owned", False):
            atomic_json(args.output_dir / "failure.json",
                        dict(status="failed", error_type=type(error).__name__, error=str(error)))
            atomic_json(args.output_dir / "status.json", dict(status="failed"))
        raise
    atomic_json(args.output_dir / "completion.json", result)
    atomic_json(args.output_dir / "status.json", dict(status="completed", command=args.command))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
