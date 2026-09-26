#!/usr/bin/env python3
"""Oracle centre/class diagnostic under one encoder, decoder and training budget.

Report measured per-object top-K IoU/recall alongside thresholded masks and AUC.
K uses the GT size and is diagnostic only, not deployable prediction. AUC is a
pairwise ranking average: 1-K/N is neither a sufficient top-K success criterion
nor evidence of an architectural upper bound. A failed finite probe cannot rule
out single-frame instance segmentation. Select on dev only; evaluate test once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

MLLM_ROOT = Path(__file__).resolve().parents[2]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vtimellm.segmentation.checkpoint import load_spatial_encoder_checkpoint  # noqa: E402
from vtimellm.segmentation.config import ReasonSegConfig  # noqa: E402
from vtimellm.segmentation.data import ReasonSegDataset  # noqa: E402
from vtimellm.segmentation.heads import QueryMaskDecoder, _mask_loss  # noqa: E402
from vtimellm.segmentation.spatial_encoder import (  # noqa: E402
    SparseUNetPointEncoder,
    pad_point_features,
)

from diagnose_reasonseg_grounding import auc_from_scores, iou_from_masks  # noqa: E402


class OracleQueryNet(nn.Module):
    """centre (3, range-normalised) + one-hot class -> decoder query."""

    def __init__(self, config: ReasonSegConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.net = nn.Sequential(
            nn.Linear(3 + config.num_classes, config.point_feature_dim),
            nn.GELU(),
            nn.Linear(config.point_feature_dim, config.point_feature_dim),
        )

    def forward(self, centers: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        onehot = torch.zeros(
            centers.shape[0], self.num_classes, dtype=centers.dtype, device=centers.device
        )
        onehot.scatter_(1, class_ids[:, None], 1.0)
        return self.net(torch.cat([centers, onehot], dim=-1))


def normalize_center(center: torch.Tensor, config: ReasonSegConfig) -> torch.Tensor:
    lower = torch.tensor(config.point_cloud_range[:3], dtype=center.dtype)
    upper = torch.tensor(config.point_cloud_range[3:], dtype=center.dtype)
    return (center - lower) / (upper - lower) * 2.0 - 1.0


def oracle_inputs(sample: dict, config: ReasonSegConfig, device) -> tuple | None:
    """GT centres (CPU->device) and class ids for every valid, non-empty target."""
    masks = sample["target_masks"]
    classes = sample["target_classes"]
    valid = sample["object_valid_mask"]
    xyz = sample["points"][:, :3]
    centers, keep = [], []
    for index in range(int(valid.sum())):
        mask = masks[index]
        if not bool(mask.any()):
            continue
        centers.append(normalize_center(xyz[mask].mean(dim=0), config))
        keep.append(index)
    if not keep:
        return None
    return torch.stack(centers).to(device), classes[keep].to(device), keep


def decode_per_object(query_net, decoder, centers, class_ids, features, point_valid):
    """Mirror HierarchicalMaskDecoder: one query per object against its own memory.

    Returns [object_count, point_count]. QueryMaskDecoder emits [B, K, N] with K=1
    here; squeezing inside this helper keeps both call sites from having to track
    that extra axis, which otherwise silently slices the wrong dimension.
    """
    object_count = centers.shape[0]
    point_count, feature_dim = features.shape[1], features.shape[2]
    queries = query_net(centers, class_ids).reshape(object_count, 1, feature_dim)
    memory = features.expand(object_count, point_count, feature_dim)
    valid_memory = point_valid.expand(object_count, point_count)
    _, logits = decoder(queries, memory, valid_memory)
    logits = logits.squeeze(1)
    if logits.shape != (object_count, point_count):
        raise RuntimeError(
            f"decoder returned {tuple(logits.shape)}, expected "
            f"{(object_count, point_count)}; the K=1 axis must be squeezed or the "
            "eval path silently slices the wrong dimension"
        )
    return logits


def pin_encoder_bn_eval(encoder) -> int:
    """把编码器里所有 BN 钉在 eval 模式，返回被钉住的模块数。

    batch size = 1（单帧）时 BN 的 train 模式用逐帧统计量归一化，并把 running
    stats 往那个噪声方向拽——v3 实测 6 轮漂了 18.71%。这一项**即使权重一个都不
    更新也会改变特征**，所以它会让"冻结 vs 解冻"不再是单变量对照。钉在 eval
    之后归一化一律用预训练的 running stats，"解冻"才真的只意味着"权重可学"。
    每次调用 encoder.train() 之后都要重新钉一遍。
    """
    pinned = 0
    for module in encoder.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            pinned += 1
    return pinned


def encoder_weight_delta(current, initial) -> dict:
    """编码器相对预训练权重的变化量，按卷积权重 / BN 仿射 / BN 统计量分开报。

    必须分开：BN running stats 只要编码器进 train 模式就会漂移，与"有没有学到
    东西"无关；只有卷积权重与 BN 仿射参数的变化才代表特征本身被改造了。v3 整体
    ‖ΔW‖/‖W‖ = 18.28% 看着不小，拆开才发现卷积权重只动了 3.52%——即编码器其实
    几乎没被微调，"点特征能否变得实例判别"根本没被检验到。
    """
    buckets: dict = {}
    for key, value in current.items():
        if not torch.is_tensor(value) or not torch.is_floating_point(value):
            continue
        if key not in initial or initial[key].shape != value.shape:
            continue
        if "running_mean" in key or "running_var" in key:
            name = "bn_stats"
        elif key.endswith(".bias") or value.dim() == 1:
            name = "bn_affine"
        elif key.endswith(".weight"):
            name = "conv_weight"
        else:
            name = "other"
        reference = initial[key].float()
        base = float(reference.norm())
        if base == 0.0:
            continue
        delta = float((value.detach().float().cpu() - reference).norm())
        acc = buckets.setdefault(name, [0.0, 0.0, 0])
        acc[0] += delta * delta
        acc[1] += base * base
        acc[2] += 1
    return {
        name: {
            "relative_delta": (d ** 0.5) / (n ** 0.5),
            "tensors": count,
        }
        for name, (d, n, count) in buckets.items()
    }


def evaluate(dataset, encoder, query_net, decoder, config, device, limit) -> dict:
    """Final-epoch metrics on the encoder's in-range points.

    Scoring uses the `point_valid` mask directly rather than a `[:point_count]`
    prefix slice. The prefix convention — used by diagnose_reasonseg_grounding.py —
    is only equivalent when out-of-range points sit at the tail of the file, and on
    val_thin they do not: out-of-range points are 6.2% of a frame but only 5.9% of
    them fall in the tail, so the prefix keeps ~2035 forced-zero points and drops
    ~2035 valid ones, and truncates 2.4% of objects to empty.
    """
    was_training = query_net.training
    was_encoder_training = encoder.training
    query_net.eval()
    decoder.eval()
    encoder.eval()
    rows = []
    skipped_unreachable = 0
    try:
        with torch.no_grad():
            for index in range(min(limit, len(dataset))):
                sample = dataset[index]
                prepared = oracle_inputs(sample, config, device)
                if prepared is None:
                    continue
                centers, class_ids, keep = prepared
                batch_indices = torch.zeros(
                    sample["points"].shape[0], dtype=torch.long, device=device
                )
                encoding = encoder(sample["points"].to(device, dtype=torch.float32),
                                   batch_indices)
                features, point_valid, _ = pad_point_features(encoding)
                logits = decode_per_object(
                    query_net, decoder, centers, class_ids, features, point_valid
                )
                probabilities = torch.sigmoid(logits).cpu().numpy()
                in_range = point_valid[0].bool().cpu().numpy()
                for row, k in enumerate(keep):
                    target = sample["target_masks"][k].numpy()[in_range]
                    if not target.any():
                        # GT 全在编码器视野外，无法评分；单独计数而不是静默丢弃。
                        skipped_unreachable += 1
                        continue
                    scores = probabilities[row][in_range]
                    predicted = scores >= 0.5
                    iou = iou_from_masks(predicted, target)
                    # top-K（K = GT 点数）是免幅值的上界：它把"排序够不够好"与
                    # "阈值/尺寸校准够不够好"分开。首跑只报 recall@0.5，结果 AUC
                    # 0.978 与 recall 0.045 互相矛盾却无法归因，就是因为缺这一项。
                    target_size = int(target.sum())
                    order = np.argsort(-scores, kind="stable")
                    top_k = np.zeros(scores.shape[0], dtype=bool)
                    top_k[order[:target_size]] = True
                    rows.append({
                        "index": index,
                        "sample_token": sample["sample_token"],
                        "class_name": dataset.records[index].targets[k].class_name,
                        "target_size": target_size,
                        "predicted_size": int(predicted.sum()),
                        "iou": iou,
                        "hit": float(iou >= 0.5),
                        "iou_top_k": iou_from_masks(top_k, target),
                        "hit_top_k": float(iou_from_masks(top_k, target) >= 0.5),
                        "auc": auc_from_scores(scores[target], scores[~target]),
                        "mean_prob_positive": float(scores[target].mean()),
                    })
    finally:
        if was_training:
            query_net.train()
            decoder.train()
        if was_encoder_training:
            encoder.train()
    if not rows:
        return {"objects": 0, "skipped_unreachable": skipped_unreachable}
    return {
        "objects": len(rows),
        "object_metrics": rows,
        "by_class": {
            name: {
                "objects": sum(row["class_name"] == name for row in rows),
                "recall_top_k": float(np.mean([row["hit_top_k"] for row in rows if row["class_name"] == name])),
                "iou_top_k_mean": float(np.mean([row["iou_top_k"] for row in rows if row["class_name"] == name])),
            }
            for name in sorted({row["class_name"] for row in rows})
        },
        "skipped_unreachable": skipped_unreachable,
        "recall_at_0.5": float(np.mean([row["hit"] for row in rows])),
        "recall_top_k": float(np.mean([row["hit_top_k"] for row in rows])),
        "iou_mean": float(np.mean([row["iou"] for row in rows])),
        "iou_top_k_mean": float(np.mean([row["iou_top_k"] for row in rows])),
        "auc_mean": float(np.mean([row["auc"] for row in rows])),
        "mean_prob_positive": float(np.mean([row["mean_prob_positive"] for row in rows])),
        "target_size_median": float(np.median([row["target_size"] for row in rows])),
        "predicted_size_median": float(np.median([row["predicted_size"] for row in rows])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        help="微调空间编码器而不是冻结它；编码器用 --encoder-lr 这个更低的独立学习率",
    )
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument(
        "--freeze-encoder-bn",
        action="store_true",
        help="解冻编码器时把 BN 钉在 eval 模式（归一化用预训练 running stats、且不再更新），"
             "使 batch size=1 下的 BN 统计量漂移不再成为第二个变量",
    )
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--eval-records", type=int, default=600)
    parser.add_argument("--region-loss", choices=("dice", "tversky"), default="tversky")
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--bce-mode", choices=("plain", "balanced"), default="plain")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    config = ReasonSegConfig()
    config.validate()

    # 编码器必须留 fp32：spconv 内部转 fp16 在 4090 上硬崩，bf16 出 NaN。
    encoder = SparseUNetPointEncoder(config)
    load_spatial_encoder_checkpoint(encoder, args.spatial_checkpoint, config=config)
    encoder = encoder.to(device=device, dtype=torch.float32).eval()
    encoder.requires_grad_(args.unfreeze_encoder)
    # 变化量的基准：预训练权重原样留一份在 CPU，每轮与它比，直接盯住"到底学没学"。
    initial_encoder_state = {
        key: value.detach().float().cpu().clone()
        for key, value in encoder.state_dict().items()
        if torch.is_floating_point(value)
    }
    pinned_bn = (
        pin_encoder_bn_eval(encoder)
        if args.unfreeze_encoder and args.freeze_encoder_bn else 0
    )

    query_net = OracleQueryNet(config).to(device)
    decoder = QueryMaskDecoder(config).to(device)
    head_params = list(query_net.parameters()) + list(decoder.parameters())
    if args.unfreeze_encoder:
        # 分组学习率：编码器是 lidarseg 语义预训练来的，用比头低一个量级的 lr 微调，
        # 否则语义特征会在几百步内被实例目标冲掉。
        encoder_params = list(encoder.parameters())
        param_groups = [
            {"params": head_params, "lr": args.learning_rate},
            {"params": encoder_params, "lr": args.encoder_lr},
        ]
        trainable = head_params + encoder_params
    else:
        encoder_params = []
        param_groups = [{"params": head_params, "lr": args.learning_rate}]
        trainable = head_params
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    train_dataset = ReasonSegDataset(
        args.train_manifest, dataroot=args.dataroot, config=config, require_reachable=True
    )
    train_limit = args.max_train_records or len(train_dataset)
    dev_dataset = ReasonSegDataset(args.dev_manifest, dataroot=args.dataroot, config=config)
    test_dataset = ReasonSegDataset(args.test_manifest, dataroot=args.dataroot, config=config)
    print(f"train records: {train_limit} / dev: {len(dev_dataset)} / test: {len(test_dataset)}",
          flush=True)
    encoder_state = (
        f"unfrozen, {sum(p.numel() for p in encoder_params):,} params @ lr {args.encoder_lr}"
        if args.unfreeze_encoder
        else "frozen"
    )
    print(f"trainable params: {sum(p.numel() for p in trainable):,} "
          f"(head {sum(p.numel() for p in head_params):,}; encoder {encoder_state})",
          flush=True)

    # 选轮快照要还原的模块：解冻时编码器也在其中（见下面 best_state 处）
    snapshot_modules = [query_net, decoder]
    if args.unfreeze_encoder:
        snapshot_modules.append(encoder)

    history = []
    best = None
    best_state = None
    encoder_grad_norm = None
    encoder_grad_params = 0
    for epoch in range(args.epochs):
        query_net.train()
        decoder.train()
        if args.unfreeze_encoder:
            encoder.train()
            if pinned_bn:
                # train() 会把 BN 一起切回 train 模式，必须重新钉住
                pin_encoder_bn_eval(encoder)
        started = time.time()
        running = 0.0
        scored = 0
        # 每轮重新洗牌：首跑按固定顺序遍历且 lr 3e-4，末轮 loss 从 0.779 跳到
        # 1.332 再回落到 1.029，而评的正是这个最差末轮。
        order = np.random.permutation(train_limit)
        for step in range(train_limit):
            sample = train_dataset[int(order[step])]
            prepared = oracle_inputs(sample, config, device)
            if prepared is None:
                continue
            centers, class_ids, keep = prepared
            batch_indices = torch.zeros(
                sample["points"].shape[0], dtype=torch.long, device=device
            )
            points = sample["points"].to(device, dtype=torch.float32)
            if args.unfreeze_encoder:
                encoding = encoder(points, batch_indices)
            else:
                with torch.no_grad():
                    encoding = encoder(points, batch_indices)
            features, point_valid, _ = pad_point_features(encoding)
            logits = decode_per_object(
                query_net, decoder, centers, class_ids, features, point_valid
            )

            object_count = len(keep)
            point_count = features.shape[1]
            object_valid = torch.ones(1, object_count, dtype=torch.bool, device=device)
            target = torch.zeros(
                1, object_count, point_count, dtype=torch.bool, device=device
            )
            for row, k in enumerate(keep):
                width = min(point_count, sample["target_masks"][k].shape[0])
                target[0, row, :width] = sample["target_masks"][k][:width].to(device)
            valid_points = point_valid[:, None, :] & object_valid[:, :, None]

            loss = _mask_loss(
                logits[None, :, :],
                target,
                valid_points,
                bce_mode=args.bce_mode,
                region_loss=args.region_loss,
                tversky_alpha=args.tversky_alpha,
                tversky_beta=args.tversky_beta,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.unfreeze_encoder and step == 0:
                # 硬校验梯度真的到了编码器：解冻臂最大的风险是"看着在微调、其实
                # 一步没走"，那样它会与冻结臂逐位相同却报成一个新的数。
                grads = [p.grad for p in encoder_params if p.grad is not None]
                encoder_grad_params = len(grads)
                encoder_grad_norm = (
                    float(torch.norm(torch.stack([g.norm() for g in grads])))
                    if grads else 0.0
                )
            # 分开裁剪：head 仍按冻结臂那样裁到 1.0，编码器独立裁。合成一个全局
            # 裁剪会让编码器那一大堆参数的范数反过来压小 head 的有效步长，破坏
            # "只改冻结与否"这个单变量。
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            if args.unfreeze_encoder:
                torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
            optimizer.step()
            running += float(loss.detach())
            scored += 1
            if (step + 1) % args.log_every == 0:
                print(f"ep{epoch} step{step + 1}/{train_limit} "
                      f"loss={running / max(scored, 1):.4f} "
                      f"elapsed={time.time() - started:.0f}s", flush=True)
        mean_loss = running / max(scored, 1)
        dev_metrics = evaluate(dev_dataset, encoder, query_net, decoder, config,
                               device, args.eval_records)
        entry = {"epoch": epoch, "train_loss": mean_loss, "records": scored,
                 "dev": dev_metrics}
        if args.unfreeze_encoder:
            entry["encoder_grad_params"] = encoder_grad_params
            entry["encoder_grad_norm"] = encoder_grad_norm
            entry["encoder_delta"] = encoder_weight_delta(
                encoder.state_dict(), initial_encoder_state
            )
        history.append(entry)
        if args.unfreeze_encoder:
            delta = entry["encoder_delta"]
            encoder_note = (
                f" | enc grad {encoder_grad_norm:.3e}/{encoder_grad_params}t"
                f" | dW conv {delta.get('conv_weight', {}).get('relative_delta', 0.0):.3%}"
                f" bn {delta.get('bn_affine', {}).get('relative_delta', 0.0):.3%}"
                f" stats {delta.get('bn_stats', {}).get('relative_delta', 0.0):.3%}"
            )
        else:
            encoder_note = ""
        print(f"epoch {epoch} loss {mean_loss:.4f}{encoder_note} | dev recall@0.5 "
              f"{dev_metrics.get('recall_at_0.5', float('nan')):.4f} topK "
              f"{dev_metrics.get('recall_top_k', float('nan')):.4f} AUC "
              f"{dev_metrics.get('auc_mean', float('nan')):.4f} IoU "
              f"{dev_metrics.get('iou_mean', float('nan')):.4f} size "
              f"{dev_metrics.get('predicted_size_median', float('nan')):.0f}/"
              f"{dev_metrics.get('target_size_median', float('nan')):.0f} "
              f"({time.time() - started:.0f}s)", flush=True)
        # 用 dev 的 iou_mean 选轮次：recall@0.5 在几百个物体上太粗、抖动大。
        # test 只在选定轮次上评一次，不参与任何选择。
        if dev_metrics.get("iou_mean") is not None and (
            best is None or dev_metrics["iou_mean"] > best["dev"]["iou_mean"]
        ):
            best = entry
            # 解冻后编码器权重也逐轮变化，选轮快照必须连它一起存；否则 test 评的
            # 会是"末轮编码器 + 选定轮的头"这个从未存在过的组合。
            best_state = [
                {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                for module in snapshot_modules
            ]

    if best_state is not None:
        for module, state in zip(snapshot_modules, best_state):
            module.load_state_dict({k: v.to(device) for k, v in state.items()})
    print(f"selected epoch {best['epoch']} by dev iou_mean "
          f"{best['dev']['iou_mean']:.4f}", flush=True)
    dev = best["dev"]
    test = evaluate(test_dataset, encoder, query_net, decoder, config, device,
                    args.eval_records)
    report = {
        "config": {
            "train_manifest": args.train_manifest,
            "dev_manifest": args.dev_manifest,
            "test_manifest": args.test_manifest,
            "epochs": args.epochs,
            "train_records": train_limit,
            "eval_records": args.eval_records,
            "region_loss": args.region_loss,
            "tversky_alpha": args.tversky_alpha,
            "tversky_beta": args.tversky_beta,
            "bce_mode": args.bce_mode,
            "learning_rate": args.learning_rate,
            "encoder_frozen": not args.unfreeze_encoder,
            "encoder_lr": args.encoder_lr if args.unfreeze_encoder else None,
            "encoder_bn_pinned_eval": pinned_bn if args.unfreeze_encoder else 0,
            "oracle": "gt_center_normalized + one_hot_class",
            "selection": "epoch with best dev iou_mean; test evaluated once at that epoch",
        },
        "selected_epoch": best["epoch"],
        "history": history,
        "dev": dev,
        "test": test,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.unfreeze_encoder:
        # 选定轮的编码器权重必须落盘：解冻臂若赢了，下一步（量 lidarseg 语义质量
        # 是否被实例目标冲掉、把它插回完整模型）都要用这份权重，否则得重跑 3 小时。
        encoder_path = args.output.with_name(args.output.stem + "_encoder.pt")
        torch.save(snapshot_modules[-1].state_dict(), encoder_path)
        print(f"selected-epoch encoder weights -> {encoder_path}", flush=True)
    print(json.dumps({"dev": dev, "test": test}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
