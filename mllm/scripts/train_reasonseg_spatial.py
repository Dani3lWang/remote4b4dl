#!/usr/bin/env python3
"""Pretrain the ReasonSeg sparse 3-D encoder with nuScenes lidarseg labels.

``--validate-only --spatial-checkpoint <dir|*.pt>`` skips encoder training and runs a
**linear probe** instead: freeze the encoder, train a fresh linear classifier for
``--probe-epochs``, then re-score lidarseg. This is how the semantic quality of an
encoder that has been fine-tuned on the *instance* objective (oracle probe v3/v4) gets
compared against the pretrained one.

It cannot reproduce the ``validation_miou`` recorded inside a checkpoint, for two
independent reasons — the classifier head was never saved (``spatial_encoder.pt`` holds
only the 52 ``point_encoder`` tensors), and commit c51c644 (2026-09-22) changed
``split="val"`` from the official nuScenes val (6,019 samples) to a leak-free internal
partition of official-train scenes (2,806 samples). Probe numbers are only comparable
to other probes run under an identical protocol. See ``validate_only`` for details.
"""

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

from vtimellm.segmentation.checkpoint import (
    load_spatial_encoder_checkpoint,
    save_spatial_encoder_checkpoint,
)
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
    if args.validate_only:
        return validate_only(accelerator, args, config)
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


def validate_only(accelerator, args, config) -> int:
    """线性探针：冻结编码器、只训一个新的线性分类头，再复算 lidarseg 指标。

    **为什么必须训头、不能直接验证**：`save_spatial_encoder_checkpoint` 只存
    `point_encoder.state_dict()`（52 个键，无任何 `classifier.*`），预训练时那个
    `nn.Linear(point_feature_dim, num_classes)` 分类头**从未被保存**。所以档内记录的
    `validation_miou = 0.2875778349554002` 用现存产物**无法复算**。直接拿随机初始化的
    头去"验证"会得到一个看着正常、实际无意义的数——实测 miou 0.19633 /
    point_accuracy 0.89530，后者只是预测塌到多数类（地面）的频率，前者由随机头碰巧
    的偏好决定，换个随机种子就完全变样（同一份权重在官方 val 前 100 条上量到的是
    miou 0.00706 / accuracy 0.0348）。**这是个静默陷阱，所以本入口实现为线性探针。**

    得到的 miou 是"特征的语义可分性"，只能在**完全相同的协议**下横向比较（同 seed、
    同轮数、同 lr、同划分、同精度），例如：预训练档 vs 被实例目标微调过的 oracle 探针
    v3/v4 档。**不能与档内记录的 0.2876 直接比**——除了头缺失，验证集也换过：09-22 的
    c51c644 把 `split="val"` 从 nuScenes 官方 val（6,019 条）改成了官方 train 场景内的
    防泄漏内部划分（2,806 条）。

    探针协议：编码器 `requires_grad_(False)` 且**永久 eval**（BN 用它自己的 running
    stats，不被 batch size=1 的噪声统计污染），只优化分类头；建议 `--mixed-precision no`
    与原预训练的 fp32 数值一致（spconv 在 fp16 下会硬崩、bf16 出过 NaN）。
    """
    if not args.spatial_checkpoint:
        raise SystemExit("--validate-only 需要 --spatial-checkpoint")
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
        config, train_dataset.num_classes, ignore_label=args.ignore_label
    )
    metadata = load_encoder_weights(model.point_encoder, args.spatial_checkpoint, config)
    if metadata is not None:
        saved_classes = int(metadata.get("num_classes", val_dataset.num_classes))
        if saved_classes != val_dataset.num_classes:
            raise RuntimeError(
                f"档内 num_classes={saved_classes} 与验证集标签映射 "
                f"{val_dataset.num_classes} 不一致，混淆矩阵口径会对不上"
            )
    # 编码器冻结且永久 eval：model.train() 会把它一起切回 train 模式，故每轮重钉。
    model.point_encoder.requires_grad_(False)
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
        model.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    trainable = sum(p.numel() for p in model.classifier.parameters())
    accelerator.print(
        f"linear probe: encoder frozen ({sum(p.numel() for p in model.point_encoder.parameters()):,} "
        f"params), classifier trainable ({trainable:,} params), "
        f"train {len(train_dataset)} / val {len(val_dataset)}"
    )

    per_epoch = []
    for epoch in range(args.probe_epochs):
        model.train()
        model.point_encoder.eval()
        running = 0.0
        steps = 0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            output = model(**_model_batch(batch))
            accelerator.backward(output.loss)
            optimizer.step()
            running += float(output.loss.detach())
            steps += 1
            if step % args.logging_steps == 0:
                accelerator.print(
                    json.dumps({"probe_epoch": epoch, "step": step,
                                "loss": float(output.loss.detach())})
                )
        metrics = validate(accelerator, model, val_loader, train_dataset.num_classes)
        entry = {"probe_epoch": epoch, "train_loss": running / max(steps, 1), **metrics}
        per_epoch.append(entry)
        accelerator.print(json.dumps(entry, indent=2))

    final = per_epoch[-1]
    payload = {
        "protocol": "linear probe (encoder frozen + eval, classifier trained from scratch)",
        "spatial_checkpoint": str(args.spatial_checkpoint),
        "checkpoint_epoch": None if metadata is None else metadata.get("epoch"),
        "recorded_miou_in_checkpoint": None if metadata is None
        else metadata.get("validation_miou"),
        "comparable_to_recorded_miou": False,
        "why_not_comparable": (
            "档内只有 point_encoder 权重、分类头从未保存；且 c51c644 之后 val 从官方 "
            "6,019 条换成内部防泄漏划分 2,806 条。本数只能与同协议的其他编码器比。"
        ),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "num_classes": val_dataset.num_classes,
        "split_args": {
            "version": args.version,
            "exclude_scenes": args.exclude_scenes,
            "label_source": args.label_source,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
        },
        "probe": {
            "epochs": args.probe_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "mixed_precision": args.mixed_precision,
        },
        "final": final,
        "history": per_epoch,
    }
    if accelerator.is_main_process:
        text = json.dumps(payload, indent=2)
        print(text)
        if args.validate_output:
            args.validate_output.parent.mkdir(parents=True, exist_ok=True)
            args.validate_output.write_text(text + "\n", encoding="utf-8")
    return 0


def load_encoder_weights(point_encoder, path: str, config):
    """载入空间编码器权重，返回档内元数据（裸 state_dict 时返回 None）。

    支持两种输入：标准档目录（含 `spatial_config.json` + `spatial_encoder.pt`，走带
    voxel 契约校验的 loader）；裸 `.pt` state_dict —— 这一支是给 oracle 探针落盘的
    `report_encoder.pt` 用的，它没有元数据，所以拿不到档内记录的 num_classes 与 miou，
    可比性得靠调用方保证划分一致。
    """
    source = Path(path)
    if source.is_dir():
        return load_spatial_encoder_checkpoint(point_encoder, str(source), config=config)
    state = torch.load(source, map_location="cpu", weights_only=True)
    point_encoder.load_state_dict(state, strict=True)
    return None


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
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="不训练编码器，改用**线性探针**量它的语义质量：冻结编码器、从零训一个线性"
             "分类头 --probe-epochs 轮后复算 lidarseg 指标。需要 --spatial-checkpoint。"
             "结果只能与同协议的其他编码器横向比，**不能与档内记录的 miou 比**"
             "（分类头从未被保存，且验证集在 c51c644 后换过）",
    )
    parser.add_argument(
        "--spatial-checkpoint",
        help="标准档目录（含 spatial_config.json + spatial_encoder.pt）或裸 state_dict 的 .pt",
    )
    parser.add_argument("--probe-epochs", type=int, default=2)
    parser.add_argument("--validate-output", type=Path, help="可选：把探针结果写成 json")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
