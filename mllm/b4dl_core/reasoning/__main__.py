"""CLI for data, SFT, GRPO, controlled inference and six-task regression."""
import argparse
import json
import math
from pathlib import Path

from ..io import read_json, read_jsonl, sha256, write_json
from .data import ReasoningDataset, audit_data, build_data


def recipe_for(path, mode=None, engineering=False, max_updates=None):
    recipe = read_json(path)
    expected = {"schema", "seed", "lora_rank", "sft_lr", "grpo_lr", "group_size", "grad_accum",
                "max_new_tokens", "clip", "beta", "process_weight", "mode", "save_every", "engineering", "max_updates"}
    if set(recipe) != expected or recipe.pop("schema") != "b4dl-reasoning-recipe-v1":
        raise ValueError("invalid reasoning recipe schema")
    if mode is not None:
        recipe["mode"] = mode
    if engineering:
        recipe["engineering"] = True
        recipe["max_updates"] = max_updates if max_updates is not None else 50
    elif max_updates is not None:
        raise ValueError("a bounded run must be marked --engineering")
    for name in ("seed", "lora_rank", "group_size", "grad_accum", "max_new_tokens", "save_every"):
        if type(recipe[name]) is not int or recipe[name] < (0 if name == "seed" else 1):
            raise ValueError("invalid integer setting: " + name)
    for name in ("sft_lr", "grpo_lr", "clip", "beta", "process_weight"):
        if type(recipe[name]) not in (int, float) or not math.isfinite(recipe[name]) or recipe[name] < 0:
            raise ValueError("invalid numeric setting: " + name)
    if (not 0 < recipe["sft_lr"] < 1 or not 0 < recipe["grpo_lr"] < 1 or not 0 < recipe["clip"] < 1
            or recipe["group_size"] < 2 or recipe["mode"] not in ("outcome", "scalar", "process")
            or type(recipe["engineering"]) is not bool):
        raise ValueError("invalid GRPO configuration")
    if recipe["engineering"]:
        if type(recipe["max_updates"]) is not int or not 1 <= recipe["max_updates"] <= 50:
            raise ValueError("engineering runs must have 1-50 updates")
    elif recipe["max_updates"] is not None:
        raise ValueError("formal runs use the complete single-pass budget")
    return recipe


def _dataset_args(parser):
    parser.add_argument("--dataset", required=True, help="reasoning dataset directory")
    parser.add_argument("--geometry-dataset", required=True, help="parent geometry dataset directory")


def _model_args(parser):
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-lock", default="/root/autodl-tmp/.b4dl_gpu0_eval.lock")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-data")
    for name in ("geometry-dataset", "tables", "support", "output"):
        build.add_argument("--"+name, required=True)
    audit = sub.add_parser("audit")
    _dataset_args(audit)
    audit.add_argument("--tables", required=True); audit.add_argument("--support", required=True)
    for stage in ("sft", "grpo"):
        p = sub.add_parser(stage)
        _dataset_args(p); _model_args(p)
        p.add_argument("--recipe", required=True); p.add_argument("--output", required=True)
        p.add_argument("--resume"); p.add_argument("--engineering", action="store_true")
        p.add_argument("--max-updates", type=int)
        if stage == "grpo":
            p.add_argument("--sft-checkpoint", required=True)
            p.add_argument("--mode", choices=("outcome", "scalar", "process"))
    infer = sub.add_parser("infer")
    _dataset_args(infer); _model_args(infer)
    infer.add_argument("--checkpoint", required=True); infer.add_argument("--output", required=True)
    infer.add_argument("--split", choices=("train", "internal_dev", "test"), default="test")
    infer.add_argument("--condition", choices=("identity", "zero", "repeat_first", "shuffle_features", "cross_scene"), default="identity")
    infer.add_argument("--max-new-tokens", type=int, default=512); infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--resume", action="store_true")
    evaluate = sub.add_parser("evaluate")
    _dataset_args(evaluate)
    evaluate.add_argument("--split", choices=("train", "internal_dev", "test"), required=True)
    evaluate.add_argument("--predictions", required=True); evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--allow-incomplete", action="store_true")
    compare = sub.add_parser("compare")
    for name in ("base", "candidate", "output"): compare.add_argument("--"+name, required=True)
    compare.add_argument("--allow-incomplete", action="store_true")
    compare.add_argument("--intervention", action="store_true", help="compare identity with a control for the same checkpoint")
    six_prep = sub.add_parser("prepare-six-task")
    for name in ("project-root", "geometry-dataset", "historical-protocol", "output"):
        six_prep.add_argument("--"+name, required=True)
    six_infer = sub.add_parser("six-task-infer")
    _dataset_args(six_infer); _model_args(six_infer)
    source = six_infer.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint"); source.add_argument("--original-b3", action="store_true")
    for name in ("protocol", "output"): six_infer.add_argument("--"+name, required=True)
    six_infer.add_argument("--resume", action="store_true")
    six_metrics = sub.add_parser("six-task-metrics")
    _dataset_args(six_metrics)
    for name in ("protocol", "predictions", "output"): six_metrics.add_argument("--"+name, required=True)
    six_compare = sub.add_parser("six-task-compare")
    for name in ("base", "candidate", "output"): six_compare.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    if args.command == "build-data":
        result = build_data(args.geometry_dataset, args.tables, args.support, args.output)
    elif args.command == "audit":
        result = audit_data(ReasoningDataset(args.dataset, args.geometry_dataset), args.tables, args.support)
    elif args.command in ("sft", "grpo"):
        from ..b3 import device_lease
        from .trainer import train
        recipe = recipe_for(args.recipe, getattr(args, "mode", None), args.engineering, args.max_updates)
        if args.device == "cpu" and not recipe["engineering"]:
            raise ValueError("formal training requires CUDA; CPU is for engineering verification")
        dataset = ReasoningDataset(args.dataset, args.geometry_dataset, "train")
        with device_lease(args.device, args.gpu_lock):
            result = train(dataset, args.baseline, args.preparation, args.output, args.command, recipe,
                           device=args.device, sft_checkpoint=getattr(args, "sft_checkpoint", None), resume=args.resume)
    elif args.command in ("infer", "six-task-infer"):
        from ..b3 import device_lease
        from .evaluation import infer, infer_six_task, load_model, load_original_model
        dataset = ReasoningDataset(args.dataset, args.geometry_dataset, args.split if args.command == "infer" else None)
        with device_lease(args.device, args.gpu_lock):
            if args.command == "six-task-infer" and args.original_b3:
                tokenizer, model = load_original_model(dataset, args.baseline, args.preparation, args.device)
                checkpoint_metadata = None
            else:
                tokenizer, model, checkpoint_metadata = load_model(dataset, args.baseline, args.preparation, args.checkpoint, args.device)
            if args.command == "infer":
                result = infer(dataset, model, tokenizer, args.checkpoint, args.output, args.condition, args.seed,
                               args.max_new_tokens, args.resume, checkpoint_metadata)
            else:
                result = infer_six_task(dataset, model, tokenizer, args.baseline if args.original_b3 else args.checkpoint,
                                        args.protocol, args.output, args.resume, "original_b3" if args.original_b3 else "reasoning",
                                        baseline=args.baseline, preparation=args.preparation,
                                        checkpoint_metadata=checkpoint_metadata)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        dataset = ReasoningDataset(args.dataset, args.geometry_dataset, args.split)
        prediction_path = Path(args.predictions)
        run = read_json(prediction_path.parent/"run.json")
        if not args.allow_incomplete:
            receipt = read_json(prediction_path.parent/"completion.json")
            if (receipt.get("status") != "reasoning_inference_complete" or receipt.get("run") != run
                    or receipt.get("requests") != len(dataset.requests)
                    or receipt.get("predictions_sha256") != sha256(prediction_path)):
                raise ValueError("prediction completion receipt mismatch")
        rows = read_jsonl(prediction_path)
        result = evaluate(dataset, rows, run)
        result["complete_prediction_set"] = len(rows) == len(dataset.requests) and not args.allow_incomplete
        if Path(args.output).exists(): raise FileExistsError(args.output)
        write_json(args.output, result)
    elif args.command == "compare":
        from .evaluation import compare_reports
        first, second = read_json(args.base), read_json(args.candidate)
        if not args.allow_incomplete and (not first.get("complete_prediction_set") or not second.get("complete_prediction_set")):
            raise ValueError("comparison requires completed prediction sets; use --allow-incomplete only for diagnostics")
        result = compare_reports(first, second, mode="intervention" if args.intervention else "models")
        result["complete_prediction_sets"] = bool(first.get("complete_prediction_set") and second.get("complete_prediction_set"))
        if Path(args.output).exists(): raise FileExistsError(args.output)
        write_json(args.output, result)
    elif args.command == "prepare-six-task":
        from ..inputs import GeometryDataset
        from .evaluation import prepare_free_six_task
        result = prepare_free_six_task(args.project_root, GeometryDataset(args.geometry_dataset), args.historical_protocol, args.output)
    elif args.command == "six-task-metrics":
        from .evaluation import six_task_metrics
        dataset = ReasoningDataset(args.dataset, args.geometry_dataset)
        result = six_task_metrics(dataset, args.protocol, args.predictions, args.output)
    else:
        from .evaluation import compare_six_task_metrics
        result = compare_six_task_metrics(args.base, args.candidate)
        if Path(args.output).exists(): raise FileExistsError(args.output)
        write_json(args.output, result)
    display = {key: value for key, value in result.items() if key != "rows"} if isinstance(result, dict) else result
    print(json.dumps(display, ensure_ascii=False, allow_nan=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
