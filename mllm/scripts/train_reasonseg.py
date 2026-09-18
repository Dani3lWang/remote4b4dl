#!/usr/bin/env python3
"""Train the first single-frame B4DL reasoning-segmentation model."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup


MLLM_ROOT = Path(__file__).resolve().parents[1]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))

from vtimellm.segmentation.checkpoint import save_reasonseg_checkpoint
from vtimellm.segmentation.config import ReasonSegConfig
from vtimellm.segmentation.data import ReasonSegCollator, ReasonSegDataset
from vtimellm.segmentation.loader import load_reasonseg_model


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    from accelerate import Accelerator

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    if args.seed is not None:
        from accelerate.utils import set_seed

        set_seed(args.seed)
    config = ReasonSegConfig()
    tokenizer, model, _ = load_reasonseg_model(
        args,
        b3_checkpoint=args.b3_checkpoint,
        reasonseg_config=config,
        segmentation_checkpoint=args.resume_from_checkpoint,
        spatial_checkpoint=args.spatial_checkpoint,
        trainable=True,
    )
    train_dataset = ReasonSegDataset(
        args.train_manifest, dataroot=args.dataroot, config=config
    )
    collator = ReasonSegCollator(tokenizer, model_max_length=args.model_max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    val_loader = None
    if args.validation_manifest:
        val_dataset = ReasonSegDataset(
            args.validation_manifest, dataroot=args.dataroot, config=config
        )
        if args.validation_samples:
            val_dataset = Subset(
                val_dataset,
                range(min(args.validation_samples, len(val_dataset))),
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
    optimizer = build_optimizer(model, args)
    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_steps = updates_per_epoch * args.num_train_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    if val_loader is None:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
    else:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
    start_epoch = 0
    global_step = 0
    best_validation_iou = -1.0
    epochs_without_improvement = 0
    if args.resume_from_checkpoint:
        accelerator_state = Path(args.resume_from_checkpoint) / "accelerator_state"
        trainer_state = Path(args.resume_from_checkpoint) / "trainer_state.json"
        if not accelerator_state.is_dir() or not trainer_state.is_file():
            raise RuntimeError(
                "resume checkpoint must contain accelerator_state and trainer_state.json"
            )
        accelerator.load_state(str(accelerator_state))
        state = json.loads(trainer_state.read_text(encoding="utf-8"))
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_validation_iou = float(state.get("best_validation_iou", -1.0))
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))

    model.train()
    for epoch in range(start_epoch, args.num_train_epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
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
                if output.loss is None or not torch.isfinite(output.loss):
                    raise RuntimeError(f"non-finite ReasonSeg loss: {output.loss}")
                accelerator.backward(output.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.logging_steps == 0:
                    values = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": float(output.loss.detach().float()),
                        "text_loss": float(output.text_loss.detach().float())
                        if output.text_loss is not None
                        else None,
                        **{
                            f"mask_{name}": float(value.detach().float())
                            for name, value in (output.head.losses or {}).items()
                        },
                    }
                    accelerator.print(json.dumps(values, ensure_ascii=False))
                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        accelerator,
                        model,
                        tokenizer,
                        optimizer,
                        scheduler,
                        args,
                        epoch,
                        global_step,
                        best_validation_iou,
                        epochs_without_improvement,
                    )
        validation = None
        if val_loader is not None:
            validation = validate_teacher_forcing(
                accelerator,
                model,
                val_loader,
                threshold=args.validation_threshold,
            )
            accelerator.print(json.dumps({"epoch": epoch, **validation}, ensure_ascii=False))
            improved = (
                validation["teacher_forcing_mean_iou"]
                > best_validation_iou + args.early_stopping_min_delta
            )
            if improved:
                best_validation_iou = validation["teacher_forcing_mean_iou"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        save_checkpoint(
            accelerator,
            model,
            tokenizer,
            optimizer,
            scheduler,
            args,
            epoch,
            global_step,
            best_validation_iou,
            epochs_without_improvement,
        )
        if validation is not None and improved:
            save_checkpoint(
                accelerator,
                model,
                tokenizer,
                optimizer,
                scheduler,
                args,
                epoch,
                global_step,
                best_validation_iou,
                epochs_without_improvement,
                destination_name="checkpoint-best",
            )
        if (
            val_loader is not None
            and args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            accelerator.print(
                f"early stopping after {epochs_without_improvement} epochs without improvement"
            )
            break
    return 0


@torch.inference_mode()
def validate_teacher_forcing(accelerator, model, loader, *, threshold: float):
    """Measure masks from ground-truth LOC/SEG tokens, separate from generation."""

    model.eval()
    totals = torch.zeros(4, dtype=torch.float64, device=accelerator.device)
    token_failures = torch.zeros(1, dtype=torch.float64, device=accelerator.device)
    sample_count = torch.zeros(1, dtype=torch.float64, device=accelerator.device)
    for batch in loader:
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
        predicted = torch.sigmoid(output.head.mask_logits) >= threshold
        target = batch["target_masks"].bool()
        object_valid = batch["object_valid_mask"].bool()
        if object_valid.shape[1] != predicted.shape[1]:
            object_valid = object_valid[:, : predicted.shape[1]]
            target = target[:, : predicted.shape[1]]
        point_valid = output.point_valid_mask[:, None, :]
        valid = point_valid & object_valid[:, :, None]
        intersection = (predicted & target & valid).sum(dim=-1).double()
        union = ((predicted | target) & valid).sum(dim=-1).double()
        present = object_valid & union.gt(0)
        totals[0] += (intersection[present] / union[present]).sum()
        totals[1] += present.sum()
        totals[2] += intersection.sum()
        totals[3] += union.sum()
        token_failures += sum(
            status not in ("ok", "no_object") for status in output.token_status
        )
        sample_count += len(output.token_status)
    totals = accelerator.reduce(totals, reduction="sum")
    token_failures = accelerator.reduce(token_failures, reduction="sum")
    sample_count = accelerator.reduce(sample_count, reduction="sum")
    model.train()
    return {
        "teacher_forcing_mean_iou": float(totals[0] / totals[1].clamp_min(1.0)),
        "teacher_forcing_global_iou": float(totals[2] / totals[3].clamp_min(1.0)),
        "teacher_forcing_objects": int(totals[1].item()),
        "teacher_forcing_token_failure_rate": float(
            token_failures[0] / sample_count[0].clamp_min(1.0)
        ),
    }


def build_optimizer(model, args):
    lora_and_tokens = []
    new_modules = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name or name.endswith(".delta") or name.endswith(".rows"):
            lora_and_tokens.append(parameter)
        else:
            new_modules.append(parameter)
    if not lora_and_tokens:
        raise RuntimeError("no trainable segmentation LoRA/token parameters were found")
    if not new_modules:
        raise RuntimeError("no trainable spatial/decoder parameters were found")
    return torch.optim.AdamW(
        [
            {"params": lora_and_tokens, "lr": args.lora_learning_rate},
            {"params": new_modules, "lr": args.learning_rate},
        ],
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )


def _prune_old_checkpoints(output_dir: Path, keep: int) -> None:
    if keep <= 0 or not output_dir.is_dir():
        return
    steps = []
    for child in output_dir.iterdir():
        name = child.name
        if child.is_dir() and name.startswith("checkpoint-"):
            suffix = name[len("checkpoint-"):]
            if suffix.isdigit():
                steps.append((int(suffix), child))
    steps.sort(key=lambda item: item[0], reverse=True)
    for _, child in steps[keep:]:
        shutil.rmtree(child, ignore_errors=True)


def save_checkpoint(
    accelerator,
    model,
    tokenizer,
    optimizer,
    scheduler,
    args,
    epoch,
    global_step,
    best_validation_iou,
    epochs_without_improvement,
    destination_name=None,
):
    accelerator.wait_for_everyone()
    destination = Path(args.output_dir) / (
        destination_name or f"checkpoint-{global_step}"
    )
    if not getattr(args, "no_resume_state", False):
        accelerator.save_state(str(destination / "accelerator_state"))
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_reasonseg_checkpoint(
            unwrapped,
            tokenizer,
            str(destination),
            b3_checkpoint=args.b3_checkpoint,
            step=global_step,
            epoch=float(epoch),
            spatial_checkpoint=args.spatial_checkpoint,
        )
        (destination / "trainer_state.json").write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_validation_iou": best_validation_iou,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if destination_name is None:
            _prune_old_checkpoints(
                Path(args.output_dir),
                keep=int(getattr(args, "keep_last", 2)),
            )
    accelerator.wait_for_everyone()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--pretrain-mm-mlp-adapter", required=True)
    parser.add_argument("--b3-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--spatial-checkpoint")
    parser.add_argument("--validation-manifest")
    parser.add_argument("--num-train-epochs", type=int, default=20)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--no-resume-state", action="store_true")
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--validation-threshold", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
