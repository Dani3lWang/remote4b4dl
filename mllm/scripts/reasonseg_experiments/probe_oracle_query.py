#!/usr/bin/env python3
"""Phase 2.3 oracle-query probe: how accurate can masks get with perfect localization?

Bypasses the language model entirely. The query handed to the fine decoder is
built from the ground-truth object centre (normalised by point_cloud_range) plus a
one-hot class — perfect localization, no mask leakage. Everything downstream is
the real thing: the same spconv point encoder, the same QueryMaskDecoder, the same
`_mask_loss`, and the same per-object memory expansion the real head uses, so the
only difference from a full model run is where the query comes from.

This measures a CEILING, not a model. Read it on two axes, because the first run
showed a single recall@0.5 threshold cannot discriminate: it reported 0.045 (which
the original trichotomy mapped to "encoder is the bottleneck") alongside AUC 0.978,
which flatly contradicts that verdict. recall@0.5 conflates ranking quality with
size calibration, so it is the wrong discriminator here.

  * AUC — ranking quality given perfect localization. Compare against the
    full model's 0.858: the gap is what the LM query pathway costs.
  * recall_top_k (K = GT point count) — magnitude-free, so it isolates ranking
    from threshold/size calibration.

The decisive threshold is derived a priori, not fitted to any observation: for
top-K recall to work with K ~= 11 positives among N ~= 32500 in-range points you
need AUC >= 1 - K/N ~= 0.99966 (Phase 2.0 measured the full model at 0.858, where
each positive has ~4938 negatives ranked above it). So:

  AUC >= 0.995      -> features + decoder are close to sufficient; the remaining
                       gap is the LM query pathway
  0.90 <= AUC < 0.995 -> informative but not separable at the point level. Read it
                       as "expected negatives ranked above a positive" =
                       (1 - AUC) * N: at AUC 0.858 that is ~4615, at 0.978 ~715,
                       and top-K needs it down to ~K = 11. So oracle localization
                       buys a 6.5x reduction but still leaves a 65x shortfall —
                       the limit is encoder resolution / receptive field, and
                       query-side work alone cannot close it.
  AUC < 0.90        -> the frozen encoder's features cannot support instance masks

Independently of the above, a predicted/target size median ratio outside
[0.7, 1.5] is its own defect: it means the decoder cannot calibrate extent even
when told exactly where the object is.

The centre+class oracle is deliberately the weakest useful oracle: it states
"segment the car at (x, y, z)" and nothing more. Feeding the GT extent as well
would leak the answer's scale and inflate the ceiling.

Epoch selection uses dev iou_mean (recall@0.5 is too coarse over a few hundred
objects); the test manifest is evaluated once at the selected epoch and never
influences training or selection.
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
    query_net.eval()
    decoder.eval()
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
    if not rows:
        return {"objects": 0, "skipped_unreachable": skipped_unreachable}
    return {
        "objects": len(rows),
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

    # 编码器冻结且必须留 fp32：spconv 内部转 fp16 在 4090 上硬崩，bf16 出 NaN。
    encoder = SparseUNetPointEncoder(config)
    load_spatial_encoder_checkpoint(encoder, args.spatial_checkpoint, config=config)
    encoder = encoder.to(device=device, dtype=torch.float32).eval()
    encoder.requires_grad_(False)

    query_net = OracleQueryNet(config).to(device)
    decoder = QueryMaskDecoder(config).to(device)
    trainable = list(query_net.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)

    train_dataset = ReasonSegDataset(
        args.train_manifest, dataroot=args.dataroot, config=config
    )
    train_limit = args.max_train_records or len(train_dataset)
    dev_dataset = ReasonSegDataset(args.dev_manifest, dataroot=args.dataroot, config=config)
    test_dataset = ReasonSegDataset(args.test_manifest, dataroot=args.dataroot, config=config)
    print(f"train records: {train_limit} / dev: {len(dev_dataset)} / test: {len(test_dataset)}",
          flush=True)
    print(f"trainable params: {sum(p.numel() for p in trainable):,}", flush=True)

    history = []
    best = None
    best_state = None
    for epoch in range(args.epochs):
        query_net.train()
        decoder.train()
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
            with torch.no_grad():
                encoding = encoder(sample["points"].to(device, dtype=torch.float32),
                                   batch_indices)
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
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
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
        history.append(entry)
        print(f"epoch {epoch} loss {mean_loss:.4f} | dev recall@0.5 "
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
            best_state = (
                {k: v.detach().cpu().clone() for k, v in query_net.state_dict().items()},
                {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()},
            )

    if best_state is not None:
        query_net.load_state_dict({k: v.to(device) for k, v in best_state[0].items()})
        decoder.load_state_dict({k: v.to(device) for k, v in best_state[1].items()})
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
    print(json.dumps({"dev": dev, "test": test}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
