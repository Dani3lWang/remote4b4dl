#!/usr/bin/env python3
"""Pretrain the ReasonSeg sparse 3-D encoder with nuScenes lidarseg labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


MLLM_ROOT = Path(__file__).resolve().parents[1]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))

from vtimellm.segmentation.checkpoint import save_spatial_encoder_checkpoint
from vtimellm.segmentation.config import ReasonSegConfig
from vtimellm.segmentation.semantic_pretrain import (
    LidarsegCollator,
    NuScenesLidarsegDataset,
    SpatialPretrainModel,
    confusion_matrix,
    semantic_metrics,
)


def main() -> int:
    args = build_parser().parse_args()
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed)
    config = ReasonSegConfig()
    common = {
        "dataroot": args.dataroot,
        "version": args.version,
        "exclude_scenes": args.exclude_scenes,
        "label_source": args.label_source,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
    }
    train_dataset = NuScenesLidarsegDataset(
        split="train", max_samples=args.max_train_samples, **common
    )
    val_dataset = NuScenesLidarsegDataset(
        split="val", max_samples=args.max_val_samples, **common
    )
    if train_dataset.num_classes != val_dataset.num_classes:
        raise RuntimeError("train and validation lidarseg class mappings differ")
    model = SpatialPretrainModel(
        config,
        train_dataset.num_classes,
        ignore_label=args.ignore_label,
    )
    collator = LidarsegCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    best_miou = -1.0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            output = model(**_model_batch(batch))
            accelerator.backward(output.loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if step % args.logging_steps == 0:
                accelerator.print(
                    json.dumps(
                        {"epoch": epoch, "step": step, "loss": float(output.loss.detach())}
                    )
                )
        metrics = validate(accelerator, model, val_loader, train_dataset.num_classes)
        accelerator.print(json.dumps({"epoch": epoch, **metrics}, indent=2))
        unwrapped = accelerator.unwrap_model(model)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_spatial_encoder_checkpoint(
                unwrapped.point_encoder,
                str(Path(args.output_dir) / "spatial-last"),
                config=config,
                epoch=epoch,
                validation_miou=metrics["miou"],
                num_classes=train_dataset.num_classes,
            )
            if metrics["miou"] > best_miou:
                save_spatial_encoder_checkpoint(
                    unwrapped.point_encoder,
                    str(Path(args.output_dir) / "spatial-best"),
                    config=config,
                    epoch=epoch,
                    validation_miou=metrics["miou"],
                    num_classes=train_dataset.num_classes,
                )
        best_miou = max(best_miou, metrics["miou"])
        accelerator.wait_for_everyone()
    return 0


@torch.inference_mode()
def validate(accelerator, model, loader, num_classes: int) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros(
        num_classes, num_classes, dtype=torch.long, device=accelerator.device
    )
    for batch in loader:
        output = model(**_model_batch(batch))
        confusion += confusion_matrix(output, num_classes)
    confusion = accelerator.reduce(confusion, reduction="sum")
    return semantic_metrics(confusion.cpu())


def _model_batch(batch):
    return {
        "points": batch["points"],
        "point_batch_indices": batch["point_batch_indices"],
        "labels": batch["labels"],
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--exclude-scenes")
    parser.add_argument(
        "--label-source", choices=("auto", "lidarseg", "panoptic"), default="auto"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=int, default=20)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ignore-label", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
