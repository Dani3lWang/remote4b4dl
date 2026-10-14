"""Standalone clip training; never launches B3 or silently substitutes labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .config import Grounded4DConfig
from .criterion import Grounded4DCriterion
from .data import Grounded4DDataset, load_scene_tokens
from .model import Grounded4DModel


CHECKPOINT_SCHEMA = "b4dl-grounded4d-checkpoint-v1"
OFFICIAL_TRAIN_WHITELIST = Path(__file__).resolve().parents[2] / "datageneration/tools/assets/train_scene_tokens.json"


def atomic_json(path, value):
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    part.replace(path)


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG topology does not match current host")
        torch.cuda.set_rng_state_all(state["cuda"])


def to_device(frame, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in frame.items()}


def clip_loss(model, criterion, clip, device, stage="joint"):
    """One differentiable ordered clip, with memory reset at every clip boundary."""
    tracks, totals = [], []
    parts = {key: 0.0 for key in ("mask_class", "identity", "reference")}
    for source_frame in clip["frames"]:
        frame = to_device(source_frame, device)
        if stage in ("spatial", "temporal"):
            frame["reference_features"], frame["reference_tokens"] = None, []
        output = model(frame, tracks)
        loss, losses, assignment = criterion(output, frame)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss in {clip['clip_id']}/{frame['sample_data_token']}")
        totals.append(loss)
        for key in parts:
            parts[key] += float(losses[key].detach()) / len(clip["frames"])
        tracks = model.update_memory(frame, output, assignment, tracks)
    return torch.stack(totals).mean(), parts


@torch.no_grad()
def validation_loss(model, criterion, dataset, device, stage):
    """Teacher-forced identity diagnostic, explicitly not free-generation metrics."""
    model.eval()
    total = 0.0
    for index in range(len(dataset)):
        loss, _ = clip_loss(model, criterion, dataset[index], device, stage)
        total += float(loss)
    return dict(protocol="teacher_forced_gt_track_assignment", clips=len(dataset),
                mean_loss=total / len(dataset), model_performance_accepted=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--nuscenes-root", required=True, type=Path)
    parser.add_argument("--label-root", required=True, type=Path)
    parser.add_argument("--train-scene-tokens", type=Path, default=OFFICIAL_TRAIN_WHITELIST)
    parser.add_argument("--supervision", choices=["panoptic", "box-pseudo"], default="panoptic")
    parser.add_argument("--stage", choices=["spatial", "temporal", "grounding", "joint"], default="spatial")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path, help="An existing checkpoint produced by this trainer")
    parser.add_argument("--init-from", type=Path, help="Compatible model weights for a NEW stage/data run")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args(argv)


def validate_stage(dataset, stage):
    if stage == "spatial" and any(len(row["frames"]) != 1 for row in dataset.records):
        raise ValueError("spatial stage requires single-frame clips; no silent frame dropping")
    if stage == "temporal" and not any(len(row["frames"]) > 1 for row in dataset.records):
        raise ValueError("temporal stage requires reviewed multi-frame clips")
    if stage in ("grounding", "joint"):
        if any(not any((frame.get("references") or {}).get("targets") for frame in row["frames"])
               for row in dataset.records):
            raise ValueError("grounding/joint stage requires reviewed references in every clip")


def run(args):
    if any(value <= 0 for value in (args.epochs, args.gradient_accumulation, args.log_every)):
        raise ValueError("epochs/gradient-accumulation/log-every must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in (args.learning_rate, args.gradient_clip)):
        raise ValueError("learning-rate/gradient-clip must be finite and positive")
    config = Grounded4DConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8-sig"))) if args.config else Grounded4DConfig()
    config.validate()
    if args.resume and args.init_from:
        raise ValueError("resume and init-from have different contracts and cannot be combined")
    whitelist = load_scene_tokens(args.train_scene_tokens)
    train = Grounded4DDataset(args.train_manifest, args.nuscenes_root, args.label_root, config,
                             whitelist, supervision=args.supervision)
    validate_stage(train, args.stage)
    preflight = dict(train=train.preflight())
    development = None
    if args.validation_manifest:
        development = Grounded4DDataset(args.validation_manifest, args.nuscenes_root, args.label_root,
                                       config, whitelist, "internal_dev", args.supervision)
        validate_stage(development, args.stage)
        if train.scenes & development.scenes:
            raise ValueError("training and internal-dev scenes overlap")
        preflight["internal_dev"] = development.preflight()
        if preflight["train"]["reference_encoder_ids"] and preflight["internal_dev"]["reference_encoder_ids"]:
            if preflight["train"]["reference_encoder_ids"] != preflight["internal_dev"]["reference_encoder_ids"]:
                raise ValueError("train/dev reference encoder identity differs")
    print(json.dumps(dict(event="data_preflight", **preflight), ensure_ascii=False), flush=True)
    if args.preflight_only:
        return preflight
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; data audit is not a completed training run")
    if config.encoder == "spconv" and device.type != "cuda":
        raise ValueError("production sparse encoder requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = Grounded4DModel(config).to(device)
    criterion = Grounded4DCriterion(config)
    initialization = None
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=True)
        if (checkpoint["metadata"].get("schema_version") != CHECKPOINT_SCHEMA
                or checkpoint["metadata"]["config"] != config.to_dict()):
            raise ValueError("initialization checkpoint architecture/class mapping mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        initialization = dict(sha256=hashlib.sha256(args.init_from.read_bytes()).hexdigest(),
                              completed_epoch=checkpoint["completed_epoch"])
    if args.stage == "grounding":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.reference_projection.requires_grad_(True)
        model.no_referent.requires_grad_(True)
    elif args.freeze_encoder:
        model.encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=args.learning_rate)
    training_config = dict(learning_rate=args.learning_rate, gradient_accumulation=args.gradient_accumulation,
                           gradient_clip=args.gradient_clip, seed=args.seed, supervision=args.supervision,
                           stage=args.stage, freeze_encoder=args.freeze_encoder)
    metadata = dict(schema_version=CHECKPOINT_SCHEMA, config=config.to_dict(),
                    training_config=training_config, data=preflight,
                    debug_encoder=config.encoder == "point_mlp_debug",
                    initialization=initialization,
                    protocol="supervised_masks_gt_track_assignment_optional_cached_phrase_alignment",
                    source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in Path(__file__).parent.glob("*.py")})
    completed_epoch, global_step = 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        previous = checkpoint["metadata"]
        if (previous.get("schema_version") != CHECKPOINT_SCHEMA
                or previous["config"] != metadata["config"]
                or previous["training_config"] != training_config
                or previous["data"] != preflight):
            raise ValueError("resume checkpoint architecture, supervision or data identity mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        metadata["initialization"] = previous.get("initialization")
        completed_epoch, global_step = checkpoint["completed_epoch"], checkpoint["global_step"]
        restore_rng(checkpoint["rng"])
        if args.epochs <= completed_epoch:
            raise ValueError("requested epochs are already complete in the resume checkpoint")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        run_file = output / "run_config.json"
        if not args.resume or not run_file.is_file():
            raise ValueError("output directory must be new/empty unless explicitly resuming its own run")
        previous_run = json.loads(run_file.read_text(encoding="utf-8"))
        # JSON converts tuple config fields to lists.
        if previous_run != json.loads(json.dumps(metadata)):
            raise ValueError("output directory belongs to a different run or code revision")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "run_config.json", metadata)
    log_path = output / "training.jsonl"
    for epoch in range(completed_epoch, args.epochs):
        model.train()
        if args.stage == "grounding":
            model.eval()
            model.reference_projection.train()
        elif args.freeze_encoder:
            model.encoder.eval()
        order = list(range(len(train)))
        random.shuffle(order)
        epoch_loss = 0.0
        for start in range(0, len(order), args.gradient_accumulation):
            window = order[start:start + args.gradient_accumulation]
            optimizer.zero_grad(set_to_none=True)
            accumulated = {key: 0.0 for key in ("mask_class", "identity", "reference")}
            for index in window:
                loss, parts = clip_loss(model, criterion, train[index], device, args.stage)
                (loss / len(window)).backward()
                epoch_loss += float(loss.detach())
                for key in accumulated:
                    accumulated[key] += parts[key] / len(window)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            global_step += 1
            event = dict(event="optimizer_step", epoch=epoch + 1, global_step=global_step,
                         gradient_norm=float(norm), **accumulated)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, allow_nan=False) + "\n")
            if global_step % args.log_every == 0:
                print(json.dumps(event), flush=True)
        record = dict(event="epoch_complete", completed_epoch=epoch + 1, global_step=global_step,
                      mean_train_loss=epoch_loss / len(train))
        if development is not None:
            record["internal_dev_diagnostic"] = validation_loss(model, criterion, development, device, args.stage)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        checkpoint = dict(metadata=metadata, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          completed_epoch=epoch + 1, global_step=global_step, rng=rng_state())
        part = output / "last.pt.part"
        torch.save(checkpoint, part)
        part.replace(output / "last.pt")
        atomic_json(output / "progress.json", record)
        print(json.dumps(record), flush=True)
    return dict(completed_epochs=args.epochs, global_step=global_step, checkpoint=str(output / "last.pt"),
                model_performance_accepted=False)


def main():
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
