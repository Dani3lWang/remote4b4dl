"""Run with PYTHONPATH=mllm python -m b4dl_core --help."""
import argparse
from collections import Counter
from pathlib import Path

from .data import build_dataset
from .evaluation import compare_reports, evaluate, opposite_pairs, prior_predictions
from .inputs import GeometryDataset, collate_frames, control_plan
from .io import read_json, read_jsonl, sha256, write_json, write_jsonl


def preflight(dataset):
    for ident in dataset.manifest["scenes"]:
        dataset.scene(ident)
    controls = {}
    for name in ("identity", "zero", "repeat_first", "shuffle_features", "cross_scene"):
        plan = control_plan(dataset, name)
        hashes, checked = [], set()
        for request, row in zip(dataset.requests, plan):
            ident = request["scene_id"]
            if row["status"] != "ready" or ident in checked:
                continue
            checked.add(ident)
            import numpy as np
            original = dataset.model_input(request)
            changed = dataset.model_input(request, row)
            if original["prompt"] != changed["prompt"] or any(not np.array_equal(original[k], changed[k]) for k in ("frame_times", "frame_geometry", "frame_valid_mask")):
                raise ValueError("feature control changed prompt or geometry")
            hashes.append({"scene_id": ident, "features_changed": not np.array_equal(original["features"], changed["features"])})
        controls[name] = {"status_counts": dict(Counter(p["status"] for p in plan)), "scene_checks": hashes}
    return {"dataset_id": dataset.manifest["dataset_id"], "baseline_checkpoint": None,
            "status": "engineering_preflight_passed", "questions": len(dataset.requests),
            "scene_counts": dict(Counter(s["split"] for s in dataset.manifest["scenes"].values())),
            "counts": dict(Counter(r["split"] + "/" + r["task"] for r in dataset.requests)),
            "opposite_pairs": len(opposite_pairs(dataset.requests, dataset.targets)), "controls": controls}


def smoke(dataset):
    import torch
    from .temporal import FrameTemporalAdapter, TemporalConfig, TemporalProjector
    torch.manual_seed(42)
    torch.set_num_threads(2)
    items, used = [], set()
    for r in dataset.requests:
        if r["scene_id"] not in used:
            items.append(dataset.model_input(r))
            used.add(r["scene_id"])
        if len(items) == 2:
            break
    batch = collate_frames(items)
    model = FrameTemporalAdapter(TemporalConfig())
    projector = torch.nn.Linear(768, 64)
    seam = TemporalProjector(projector, model)
    with torch.no_grad():
        actual = seam(**batch)
        expected = projector(batch["features"])
        if not torch.equal(actual, expected):
            raise ValueError("zero initialization changed original projector output")
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    target = batch["features"] * .5
    losses = []
    for _ in range(2):
        optimizer.zero_grad()
        value = model(**batch)
        loss = (value[batch["frame_valid_mask"]] - target[batch["frame_valid_mask"]]).square().mean()
        loss.backward()
        losses.append(float(loss.detach()))
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("nonfinite gradient")
        optimizer.step()
    grad = float(model.blocks[0].attention.in_proj_weight.grad.abs().sum())
    if grad <= 0:
        raise ValueError("no gradient after residual warmup")
    return {"status": "cpu_engineering_smoke_passed", "dataset_id": dataset.manifest["dataset_id"],
            "model_weights": "random_init_discarded", "baseline_checkpoint": None,
            "parameters": sum(p.numel() for p in model.parameters()), "batch_shape": list(batch["features"].shape),
            "zero_init_projector_exact": True, "synthetic_loss": losses,
            "attention_gradient_after_warmup": grad, "torch": torch.__version__,
            "note": "Two CPU steps use an artificial target to check gradients. No task training or quality claim."}


def main():
    parser = argparse.ArgumentParser(description="Independent B4DL geometry preparation; no baseline loader or training job")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("tables", "scene-metadata", "features", "splits", "output"):
        prepare.add_argument("--" + name, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--max-per-label", type=int, default=256)
    prepare.add_argument("--scene-ids", nargs="+")
    for name in ("preflight", "smoke", "controls", "evaluate", "prior", "audit"):
        sub = commands.add_parser(name)
        sub.add_argument("--dataset", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--split", choices=("train", "internal_dev", "test"), required=name in ("evaluate", "prior"))
        if name == "audit":
            sub.add_argument("--tables", required=True)
        if name == "controls":
            sub.add_argument("--condition", choices=("identity", "zero", "repeat_first", "shuffle_features", "cross_scene"), required=True)
        if name == "evaluate":
            sub.add_argument("--predictions", required=True)
            sub.add_argument("--run", required=True, help="Explicit run identity JSON")
        if name == "prior":
            sub.add_argument("--policy", choices=("train_majority", "always_yes", "always_no", "always_a"), default="train_majority")
    sub = commands.add_parser("compare")
    sub.add_argument("--base", required=True)
    sub.add_argument("--candidate", required=True)
    sub.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; choose a new path")
    if args.command == "prepare":
        report = build_dataset(args.tables, args.scene_metadata, args.features, args.splits, output,
                               seed=args.seed, max_per_label=args.max_per_label, scene_ids=args.scene_ids)
        print(report["dataset_id"], report["counts"])
        return
    if args.command == "compare":
        report = compare_reports(read_json(args.base), read_json(args.candidate))
    else:
        dataset = GeometryDataset(args.dataset, args.split)
        if args.command == "audit":
            from .audit import audit_supervision
            report = audit_supervision(dataset, args.tables)
        elif args.command == "preflight":
            report = preflight(dataset)
        elif args.command == "smoke":
            report = smoke(dataset)
        elif args.command == "controls":
            write_jsonl(output, control_plan(dataset, args.condition))
            print(str(output))
            return
        elif args.command == "prior":
            train = GeometryDataset(args.dataset, "train")
            predictions = prior_predictions(train, dataset, args.policy)
            run = {"dataset_id": dataset.manifest["dataset_id"], "split": args.split, "condition": "identity",
                   "model_id": args.policy, "checkpoint_sha256": None, "prediction_kind": "prior"}
            report = evaluate(dataset, predictions, run)
        else:
            report = evaluate(dataset, read_jsonl(args.predictions), read_json(args.run))
            report["source_sha256"] = {"predictions": sha256(args.predictions), "run": sha256(args.run)}
    write_json(output, report)
    print(str(output))


if __name__ == "__main__":
    main()
