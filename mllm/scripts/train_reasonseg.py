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
    if getattr(args, "dropout", None) is not None:
        config.dropout = float(args.dropout)
    for flag, field in (
        ("bce_mode", "bce_mode"),
        ("region_loss", "region_loss"),
        ("tversky_alpha", "tversky_alpha"),
        ("tversky_beta", "tversky_beta"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            setattr(config, field, value)
    if args.resume_from_checkpoint and args.eval_checkpoint:
        raise RuntimeError(
            "--eval-checkpoint and --resume-from-checkpoint are mutually exclusive; "
            "the former loads weights only, the latter also requires accelerator_state"
        )
    tokenizer, model, _ = load_reasonseg_model(
        args,
        b3_checkpoint=args.b3_checkpoint,
        reasonseg_config=config,
        segmentation_checkpoint=args.resume_from_checkpoint or args.eval_checkpoint,
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
    full_val_dataset = None
    if args.validation_manifest:
        val_dataset = ReasonSegDataset(
            args.validation_manifest, dataroot=args.dataroot, config=config
        )
        full_val_dataset = val_dataset
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
    if args.validate_only:
        return _validate_dtype_variants(
            accelerator, model, full_val_dataset, collator, args
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


def _first_param_dtype(module) -> str:
    for param in module.parameters():
        return str(param.dtype)
    return "none"


def _make_val_loader(dataset, collator, args, limit: int):
    if limit:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collator,
    )


class _NonFiniteProbe:
    """Counts encoder batches that emit non-finite values."""

    def __init__(self) -> None:
        self.batches = 0
        self.nonfinite_batches = 0
        self.nonfinite_elements = 0

    def reset(self) -> None:
        self.batches = self.nonfinite_batches = self.nonfinite_elements = 0

    def __call__(self, _module, _inputs, output) -> None:
        tensors = (
            [output]
            if torch.is_tensor(output)
            else [
                value
                for value in getattr(output, "__dict__", {}).values()
                if torch.is_tensor(value)
            ]
        )
        self.batches += 1
        bad = sum(int((~torch.isfinite(tensor)).sum()) for tensor in tensors)
        if bad:
            self.nonfinite_batches += 1
            self.nonfinite_elements += bad


def _apply_dtype_variant(accelerator, model, variant: str):
    unwrapped = accelerator.unwrap_model(model)
    original_encode = unwrapped.encode_points

    def restore() -> None:
        unwrapped.__dict__.pop("encode_points", None)

    if variant == "as_is":
        return restore
    if variant == "fp16":
        model.to(torch.float16)
        return restore
    if variant != "fp32":
        raise ValueError(f"unknown dtype variant: {variant}")
    # spconv 只接受 fp32/fp16。这里只把编码器抬回 fp32，两路输出按各自消费者的 dtype 回cast，
    # 使该臂与 fp16 臂之间唯一的差别就是编码器精度。
    lm_dtype = next(unwrapped.language_model.get_input_embeddings().parameters()).dtype
    head_dtype = next(unwrapped.mask_head.parameters()).dtype
    unwrapped.point_encoder.to(torch.float32)
    unwrapped.scene_compressor.to(torch.float32)

    def encode_points_fp32(points, point_batch_indices):
        encoding, padded, valid, scene, indices = original_encode(
            points.to(torch.float32), point_batch_indices
        )
        return encoding, padded.to(head_dtype), valid, scene.to(lm_dtype), indices

    unwrapped.encode_points = encode_points_fp32
    return restore


def _validate_dtype_variants(accelerator, model, val_dataset, collator, args) -> int:
    if val_dataset is None:
        raise RuntimeError("--validate-only requires --validation-manifest")
    variants = [
        name.strip() for name in args.validate_dtype_variants.split(",") if name.strip()
    ]
    if not variants or variants[0] != "as_is":
        raise RuntimeError(
            "--validate-dtype-variants must start with as_is; later arms mutate weight dtype"
        )
    unwrapped = accelerator.unwrap_model(model)
    accelerator.print(
        json.dumps(
            {
                "dtype_snapshot": {
                    name: _first_param_dtype(module)
                    for name, module in (
                        ("point_encoder", unwrapped.point_encoder),
                        ("scene_compressor", unwrapped.scene_compressor),
                        ("mask_head", unwrapped.mask_head),
                        ("language_model", unwrapped.language_model),
                    )
                }
            },
            ensure_ascii=False,
        )
    )
    probe = _NonFiniteProbe()
    unwrapped.point_encoder.register_forward_hook(probe)
    for position, variant in enumerate(variants):
        restore = _apply_dtype_variant(accelerator, model, variant)
        limit = args.validation_samples
        if position and args.variant_validation_samples:
            limit = args.variant_validation_samples
        probe.reset()
        # 必须像训练里 prepare(val_loader) 那样把 batch 搬上设备，否则 spconv 会收到 CPU 张量。
        loader = accelerator.prepare(
            _make_val_loader(val_dataset, collator, args, limit)
        )
        metrics = validate_teacher_forcing(
            accelerator, model, loader, threshold=args.validation_threshold,
            extra_thresholds=_parse_thresholds(args.validation_thresholds),
        )
        metrics.update(
            {
                "variant": variant,
                "requested_samples": limit if limit else len(val_dataset),
                "point_encoder_batches": probe.batches,
                "point_encoder_nonfinite_batches": probe.nonfinite_batches,
                "point_encoder_nonfinite_elements": probe.nonfinite_elements,
                "observed_dtypes": {
                    name: _first_param_dtype(module)
                    for name, module in (
                        ("point_encoder", unwrapped.point_encoder),
                        ("language_model", unwrapped.language_model),
                        ("mask_head", unwrapped.mask_head),
                    )
                },
            }
        )
        restore()
        accelerator.print(json.dumps(metrics, ensure_ascii=False))
    return 0


@torch.inference_mode()
def _parse_thresholds(text: str) -> list:
    return [float(value) for value in str(text or "").split(",") if value.strip()]


def validate_teacher_forcing(accelerator, model, loader, *, threshold: float,
                             extra_thresholds=()):
    """Measure masks from ground-truth LOC/SEG tokens, separate from generation.

    ``extra_thresholds`` are scored in the same forward pass: the mask head's
    sigmoid rarely reaches 0.5, so the binarisation cut dominates the reported
    IoU and must be swept instead of assumed.
    """

    model.eval()
    thresholds = [float(threshold)]
    thresholds += [float(value) for value in extra_thresholds
                   if float(value) != float(threshold)]
    totals = torch.zeros(5 * len(thresholds), dtype=torch.float64, device=accelerator.device)
    token_failures = torch.zeros(1, dtype=torch.float64, device=accelerator.device)
    sample_count = torch.zeros(1, dtype=torch.float64, device=accelerator.device)
    for batch in loader:
        # 验证不需要梯度；不关掉的话每批都会建 autograd 图，且 probabilities
        # 跨阈值循环存活会让相邻两批的图同时驻留（实测 +7.7G，4090 上 OOM）。
        with torch.no_grad():
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
            probabilities = torch.sigmoid(output.head.mask_logits)
            target = batch["target_masks"].bool()
            object_valid = batch["object_valid_mask"].bool()
            if object_valid.shape[1] != probabilities.shape[1]:
                object_valid = object_valid[:, : probabilities.shape[1]]
                target = target[:, : probabilities.shape[1]]
            point_valid = output.point_valid_mask[:, None, :]
            valid = point_valid & object_valid[:, :, None]
            # 分母只能由数据与编码器有效性决定，绝不能含 predicted。原本写的是
            # object_valid & union.gt(0)：GT 全部越界的物体在模型恰好触发时也会被
            # 计入（IoU 必为 0），于是触发越多的模型背的不可能项越多——val_thin 上
            # 对照报 2325 个物体而 A2 报 2329 个就是这么来的。三项指标统一只在
            # scorable 上求和，避免 mean IoU 修好了而 global IoU 仍随触发量漂移。
            scorable = object_valid & (target & valid).any(dim=-1)
            for offset, cut in enumerate(thresholds):
                predicted = probabilities >= cut
                intersection = (predicted & target & valid).sum(dim=-1).double()
                union = ((predicted | target) & valid).sum(dim=-1).double()
                base = 5 * offset
                # scorable 保证 union >= |target & valid| > 0，不会除零。
                per_object_iou = intersection[scorable] / union[scorable]
                totals[base] += per_object_iou.sum()
                totals[base + 1] += scorable.sum()
                totals[base + 2] += intersection[scorable].sum()
                totals[base + 3] += union[scorable].sum()
                # mean IoU 对阈值平坦不代表硬计数平坦：recall@0.5 数的是越过
                # 0.5 边界的实例数，阈值可能把它们整体推过去而均值几乎不动。
                totals[base + 4] += (per_object_iou >= 0.5).sum()
            token_failures += sum(
                status not in ("ok", "no_object") for status in output.token_status
            )
            sample_count += len(output.token_status)
    totals = accelerator.reduce(totals, reduction="sum")
    token_failures = accelerator.reduce(token_failures, reduction="sum")
    sample_count = accelerator.reduce(sample_count, reduction="sum")
    model.train()

    def summarize(offset: int) -> dict:
        base = 5 * offset
        objects = totals[base + 1].clamp_min(1.0)
        return {
            "teacher_forcing_mean_iou": float(totals[base] / objects),
            "teacher_forcing_global_iou": float(
                totals[base + 2] / totals[base + 3].clamp_min(1.0)
            ),
            "teacher_forcing_objects": int(totals[base + 1].item()),
            "teacher_forcing_recall_at_0.5": float(totals[base + 4] / objects),
            "teacher_forcing_hits_at_0.5": int(totals[base + 4].item()),
        }

    result = summarize(0)
    result["teacher_forcing_token_failure_rate"] = float(
        token_failures[0] / sample_count[0].clamp_min(1.0)
    )
    if len(thresholds) > 1:
        result["threshold_sweep"] = {
            f"{value:g}": summarize(offset) for offset, value in enumerate(thresholds)
        }
    return result


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
            suffix = name[len("checkpoint-") :]
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
    parser.add_argument("--eval-checkpoint")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--validate-dtype-variants",
        default="as_is",
        help="逗号分隔的精度臂，按顺序执行：as_is（不加任何 cast，忠实复现训练配置）、"
             "fp16（整模型转 fp16，复现评测侧被 spconv NaN 污染的配置）、"
             "fp32（仅点云编码器/压缩器转 fp32）。as_is 必须排在最前，后续臂会改权重 dtype。",
    )
    parser.add_argument("--variant-validation-samples", type=int, default=0)
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
    parser.add_argument(
        "--validation-thresholds",
        default="",
        help="逗号分隔的额外二值化阈值，与 --validation-threshold 在同一次前向内一起评分，"
             "结果出现在 threshold_sweep 字段；仅 validate-only 复算时需要",
    )
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--bce-mode",
        choices=("plain", "balanced"),
        default=None,
        help="balanced 把掩码/粗定位 BCE 的正负两侧各自归一后等权平均；正例占比 "
             "3e-4 时 plain 会让“沉默”与“在所有同类候选上对冲”的损失几乎相同",
    )
    parser.add_argument(
        "--region-loss",
        choices=("dice", "tversky"),
        default=None,
        help="tversky 配 --tversky-alpha/--tversky-beta 降低误检代价、抬高漏检代价；"
             "dice 等价于 alpha=beta=0.5",
    )
    parser.add_argument("--tversky-alpha", type=float, default=None)
    parser.add_argument("--tversky-beta", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
