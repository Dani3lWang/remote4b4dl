#!/usr/bin/env python3
"""多任务微调空间编码器：语义（lidarseg）与实例（oracle query）两个目标一起训。

为什么要做：线性探针（同协议，train 25,324 / val 2,806、冻结编码器 + 从零训头 2 轮、
fp32）量出 v4 的实例微调把编码器的语义质量砍掉了 28.3%：

    预训练 spatial-best   miou 0.30785   (dW conv  —)
    v3  解冻 6ep lr1e-5   miou 0.30560   (dW conv  3.52%)   −0.7%
    v4  解冻 20ep lr5e-5  miou 0.22073   (dW conv 19.97%)  −28.3%

语义损失与 dW conv 严格同向 ⇒ **实例目标与语义质量在打架**。而完整模型里同一个编码器
还要负责把场景压成 LM 的 scene token，所以 v4 那个把 instance recall@0.5 从 0.0962
抬到 0.2476 的编码器**不能替换 spatial-tv**。本脚本回答的就是：这两个目标能不能共存
于同一份特征。

设计与 v4 的关系：**唯一变量是多了语义损失这一项**。其余全部逐字对齐 v4——同 oracle
query（GT 中心 + one-hot 类别，完全绕过 7B LM）、同 manifest、同 seed（故同洗牌顺序）、
同 20 epoch、编码器 lr 5e-5、实例头 lr 1e-4、BN 钉在 eval、Tversky(0.3,0.7)+plain、
同按 dev iou_mean 选轮、test 只评一次。所以 v4 与本跑的差就是"加不加语义损失"。

每步做两次编码器前向（语义帧与实例帧是不同的点云，无法共享），损失相加：
    total = --semantic-weight * CE_semantic + --instance-weight * mask_loss

三组参数各自独立学习率、各自独立裁剪（合成一个全局裁剪会让某一组的范数反过来压小
另一组的有效步长，破坏可控性）：
    编码器            --encoder-lr       5e-5（与 v4 同）
    实例头            --learning-rate    1e-4（与 v4 同）
    语义分类头        --semantic-head-lr 1e-3（从零训，与预训练/线性探针同）

**语义质量的裁定数不能用训练期那个头的 val miou** —— 它是与编码器联合训出来的，偏乐观。
训练期逐轮报的 `semantic_val_miou` 只当趋势看；裁定必须拿落盘的编码器快照跑
`run_linear_probe_semantic.sh` 那套**与 §6.8 逐字相同的线性探针协议**复量。

每 --snapshot-every 轮落盘一份编码器快照，于是一次运行就能画出"语义 vs 实例"的
权衡曲线，不必为每个点重跑 9 小时。这是从 v4 学到的：那次只存了选定轮，事后想知道
"哪一轮的权衡最好"已经拿不回来了。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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
from vtimellm.segmentation.semantic_pretrain import (  # noqa: E402
    LidarsegCollator,
    NuScenesLidarsegDataset,
    SpatialPretrainModel,
    confusion_matrix,
    semantic_metrics,
)
from vtimellm.segmentation.spatial_encoder import (  # noqa: E402
    SparseUNetPointEncoder,
    pad_point_features,
)

from probe_oracle_query import (  # noqa: E402
    OracleQueryNet,
    decode_per_object,
    encoder_weight_delta,
    evaluate,
    oracle_inputs,
    pin_encoder_bn_eval,
)


def infinite(loader):
    """语义侧数据比实例侧多（25,324 vs 7,436），按实例步数取用、耗尽即重开。"""
    while True:
        for batch in loader:
            yield batch


@torch.inference_mode()
def semantic_validate(model, loader, num_classes: int, device, limit: int) -> dict:
    """联合训练那个语义头在内部 val 上的 miou——只当趋势，不作裁定数（见模块 docstring）。"""
    was_training = model.training
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    try:
        for index, batch in enumerate(loader):
            if limit and index >= limit:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            output = model(
                points=batch["points"],
                point_batch_indices=batch["point_batch_indices"],
                labels=batch["labels"],
            )
            confusion += confusion_matrix(output, num_classes)
    finally:
        if was_training:
            model.train()
    return semantic_metrics(confusion.cpu())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True, help="实例侧（ReasonSeg）训练清单")
    parser.add_argument("--dev-manifest", required=True, help="实例侧 dev，用于选轮")
    parser.add_argument("--test-manifest", required=True, help="实例侧 test，只在选定轮评一次")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="实例头 lr")
    parser.add_argument("--encoder-lr", type=float, default=5e-5)
    parser.add_argument("--semantic-head-lr", type=float, default=1e-3)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--instance-weight", type=float, default=1.0)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--exclude-scenes")
    parser.add_argument("--label-source", choices=("auto", "lidarseg", "panoptic"), default="auto")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--ignore-label", type=int, default=0)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--eval-records", type=int, default=400)
    parser.add_argument("--semantic-val-records", type=int, default=0, help="0 = 全部 2,806 条")
    parser.add_argument("--region-loss", choices=("dice", "tversky"), default="tversky")
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--bce-mode", choices=("plain", "balanced"), default="plain")
    parser.add_argument("--snapshot-every", type=int, default=4)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("spconv 需要 CUDA")
    torch.cuda.set_device(device)

    config = ReasonSegConfig()
    config.validate()

    # 编码器必须 fp32：spconv 内部转 fp16 在 4090 上硬崩，bf16 出 NaN。
    encoder = SparseUNetPointEncoder(config)
    load_spatial_encoder_checkpoint(encoder, args.spatial_checkpoint, config=config)
    encoder = encoder.to(device=device, dtype=torch.float32)
    encoder.requires_grad_(True)
    pinned_bn = pin_encoder_bn_eval(encoder)
    initial_encoder_state = {
        key: value.detach().float().cpu().clone()
        for key, value in encoder.state_dict().items()
        if torch.is_floating_point(value)
    }

    # ---- 语义侧 ----
    common = {
        "dataroot": args.dataroot,
        "version": args.version,
        "exclude_scenes": args.exclude_scenes,
        "label_source": args.label_source,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
    }
    sem_train = NuScenesLidarsegDataset(split="train", max_samples=0, **common)
    sem_val = NuScenesLidarsegDataset(split="val", max_samples=0, **common)
    if sem_train.num_classes != sem_val.num_classes:
        raise RuntimeError("train/val lidarseg 类别映射不一致")
    num_classes = sem_train.num_classes
    # 注入共享编码器：这样语义 CE 与预训练时的实现逐位同源，不是另写一份。
    semantic_model = SpatialPretrainModel(
        config, num_classes, ignore_label=args.ignore_label, point_encoder=encoder
    ).to(device)

    collator = LidarsegCollator()
    workers = args.dataloader_num_workers
    sem_train_loader = DataLoader(
        sem_train, batch_size=1, shuffle=True, num_workers=workers,
        pin_memory=False, collate_fn=collator,
    )
    sem_val_loader = DataLoader(
        sem_val, batch_size=1, shuffle=False, num_workers=workers,
        pin_memory=False, collate_fn=collator,
    )

    # ---- 实例侧 ----
    query_net = OracleQueryNet(config).to(device)
    decoder = QueryMaskDecoder(config).to(device)
    inst_train = ReasonSegDataset(args.train_manifest, dataroot=args.dataroot, config=config)
    inst_dev = ReasonSegDataset(args.dev_manifest, dataroot=args.dataroot, config=config)
    inst_test = ReasonSegDataset(args.test_manifest, dataroot=args.dataroot, config=config)
    train_limit = args.max_train_records or len(inst_train)

    instance_params = list(query_net.parameters()) + list(decoder.parameters())
    semantic_head_params = list(semantic_model.classifier.parameters())
    encoder_params = list(encoder.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": instance_params, "lr": args.learning_rate},
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": semantic_head_params, "lr": args.semantic_head_lr},
        ],
        weight_decay=0.01,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"实例 train {train_limit} / dev {len(inst_dev)} / test {len(inst_test)}；"
        f"语义 train {len(sem_train)} / val {len(sem_val)}；num_classes {num_classes}",
        flush=True,
    )
    print(
        f"trainable: encoder {sum(p.numel() for p in encoder_params):,} @ {args.encoder_lr} | "
        f"instance head {sum(p.numel() for p in instance_params):,} @ {args.learning_rate} | "
        f"semantic head {sum(p.numel() for p in semantic_head_params):,} @ {args.semantic_head_lr}"
        f" | BN pinned eval {pinned_bn}",
        flush=True,
    )
    print(
        f"loss = {args.semantic_weight} * CE_semantic + {args.instance_weight} * mask_loss",
        flush=True,
    )

    history = []
    best = None
    best_state = None
    # 选轮快照要还原的模块。解冻后编码器权重逐轮变化，必须一起存，否则 test 评的会是
    # "末轮编码器 + 选定轮的头"这个从未存在过的组合。
    snapshot_modules = [query_net, decoder, encoder, semantic_model.classifier]
    sem_iter = infinite(sem_train_loader)
    for epoch in range(args.epochs):
        query_net.train()
        decoder.train()
        encoder.train()
        semantic_model.train()
        # train() 会把 BN 一起切回 train 模式，必须重新钉住
        pin_encoder_bn_eval(encoder)

        started = time.time()
        running = {"total": 0.0, "sem": 0.0, "inst": 0.0}
        scored = 0
        encoder_grad_norm = None
        order = np.random.permutation(train_limit)
        for step in range(train_limit):
            # ---- 语义支 ----
            sem_batch = next(sem_iter)
            sem_batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sem_batch.items()
            }
            sem_output = semantic_model(
                points=sem_batch["points"],
                point_batch_indices=sem_batch["point_batch_indices"],
                labels=sem_batch["labels"],
            )
            loss_sem = sem_output.loss

            # ---- 实例支 ----
            sample = inst_train[int(order[step])]
            prepared = oracle_inputs(sample, config, device)
            if prepared is None:
                # 该帧没有可监督的物体：只回传语义支，不要让这一步白跑
                loss_inst = torch.zeros((), device=device)
            else:
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
                object_count = len(keep)
                point_count = features.shape[1]
                object_valid = torch.ones(1, object_count, dtype=torch.bool, device=device)
                target = torch.zeros(1, object_count, point_count, dtype=torch.bool, device=device)
                for row, k in enumerate(keep):
                    width = min(point_count, sample["target_masks"][k].shape[0])
                    target[0, row, :width] = sample["target_masks"][k][:width].to(device)
                valid_points = point_valid[:, None, :] & object_valid[:, :, None]
                loss_inst = _mask_loss(
                    logits[None, :, :], target, valid_points,
                    bce_mode=args.bce_mode, region_loss=args.region_loss,
                    tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta,
                )

            loss = args.semantic_weight * loss_sem + args.instance_weight * loss_inst
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0:
                # 硬校验梯度真的到了编码器（防空洞消融）
                grads = [p.grad for p in encoder_params if p.grad is not None]
                encoder_grad_norm = (
                    float(torch.norm(torch.stack([g.norm() for g in grads]))) if grads else 0.0
                )
            # 三组分别裁剪，避免某一组的范数压小另一组的有效步长
            torch.nn.utils.clip_grad_norm_(instance_params, 1.0)
            torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
            torch.nn.utils.clip_grad_norm_(semantic_head_params, 1.0)
            optimizer.step()

            running["total"] += float(loss.detach())
            running["sem"] += float(loss_sem.detach())
            running["inst"] += float(loss_inst.detach())
            scored += 1
            if (step + 1) % args.log_every == 0:
                print(
                    f"ep{epoch} step{step + 1}/{train_limit} "
                    f"total={running['total'] / scored:.4f} "
                    f"sem={running['sem'] / scored:.4f} "
                    f"inst={running['inst'] / scored:.4f} "
                    f"elapsed={time.time() - started:.0f}s",
                    flush=True,
                )

        dev_metrics = evaluate(inst_dev, encoder, query_net, decoder, config, device,
                               args.eval_records)
        sem_metrics = semantic_validate(
            semantic_model, sem_val_loader, num_classes, device, args.semantic_val_records
        )
        delta = encoder_weight_delta(encoder.state_dict(), initial_encoder_state)
        entry = {
            "epoch": epoch,
            "train_loss_total": running["total"] / max(scored, 1),
            "train_loss_semantic": running["sem"] / max(scored, 1),
            "train_loss_instance": running["inst"] / max(scored, 1),
            "records": scored,
            "encoder_grad_norm": encoder_grad_norm,
            "encoder_delta": delta,
            "dev": dev_metrics,
            "semantic_val": sem_metrics,
        }
        history.append(entry)
        print(
            f"epoch {epoch} total {entry['train_loss_total']:.4f} "
            f"(sem {entry['train_loss_semantic']:.4f} / inst {entry['train_loss_instance']:.4f})"
            f" | enc grad {encoder_grad_norm:.3e}"
            f" | dW conv {delta.get('conv_weight', {}).get('relative_delta', 0.0):.3%}"
            f" stats {delta.get('bn_stats', {}).get('relative_delta', 0.0):.3%}"
            f" | dev R@0.5 {dev_metrics.get('recall_at_0.5', float('nan')):.4f}"
            f" topK {dev_metrics.get('recall_top_k', float('nan')):.4f}"
            f" AUC {dev_metrics.get('auc_mean', float('nan')):.4f}"
            f" IoU {dev_metrics.get('iou_mean', float('nan')):.4f}"
            f" | sem val miou {sem_metrics['miou']:.5f} acc {sem_metrics['point_accuracy']:.5f}"
            f" ({time.time() - started:.0f}s)",
            flush=True,
        )

        if dev_metrics.get("iou_mean") is not None and (
            best is None or dev_metrics["iou_mean"] > best["dev"]["iou_mean"]
        ):
            best = entry
            best_state = [
                {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                for m in snapshot_modules
            ]
        if args.snapshot_every and (epoch + 1) % args.snapshot_every == 0:
            path = args.output_dir / f"encoder_ep{epoch}.pt"
            torch.save(encoder.state_dict(), path)
            print(f"snapshot -> {path}", flush=True)
        # 每轮都把 history 落盘：9 小时的跑，中途崩了不该什么都拿不到
        (args.output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    if best_state is not None:
        for module, state in zip(snapshot_modules, best_state):
            module.load_state_dict({k: v.to(device) for k, v in state.items()})
    print(f"selected epoch {best['epoch']} by dev iou_mean {best['dev']['iou_mean']:.4f}",
          flush=True)
    torch.save(encoder.state_dict(), args.output_dir / "encoder_selected.pt")
    test = evaluate(inst_test, encoder, query_net, decoder, config, device, args.eval_records)
    report = {
        "config": {
            "spatial_checkpoint": args.spatial_checkpoint,
            "train_manifest": args.train_manifest,
            "dev_manifest": args.dev_manifest,
            "test_manifest": args.test_manifest,
            "epochs": args.epochs,
            "train_records": train_limit,
            "eval_records": args.eval_records,
            "semantic_train_samples": len(sem_train),
            "semantic_val_samples": len(sem_val),
            "num_classes": num_classes,
            "encoder_lr": args.encoder_lr,
            "instance_head_lr": args.learning_rate,
            "semantic_head_lr": args.semantic_head_lr,
            "semantic_weight": args.semantic_weight,
            "instance_weight": args.instance_weight,
            "region_loss": args.region_loss,
            "tversky_alpha": args.tversky_alpha,
            "tversky_beta": args.tversky_beta,
            "bce_mode": args.bce_mode,
            "encoder_bn_pinned_eval": pinned_bn,
            "seed": args.seed,
            "oracle": "gt_center_normalized + one_hot_class",
            "selection": "epoch with best dev iou_mean; test evaluated once at that epoch",
            "semantic_val_note": (
                "训练期这个 miou 来自与编码器联合训练的头，偏乐观，只当趋势；"
                "裁定须用 encoder_selected.pt / encoder_ep*.pt 跑与 §6.8 相同的线性探针协议"
            ),
        },
        "selected_epoch": best["epoch"],
        "history": history,
        "dev": best["dev"],
        "test": test,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"dev": best["dev"], "test": test}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
