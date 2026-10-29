"""Full-denominator CoT evaluation, controls and free six-task regression."""
from collections import Counter, defaultdict
from pathlib import Path

from ..evaluation import opposite_pairs, scene_interval
from ..inputs import control_plan
from ..io import digest, read_json, read_jsonl, sha256, unique_index, write_json, write_jsonl
from ..resume import (append_jsonl, bind_run, output_lease, recover_jsonl,
                      validate_predictions, verify_completion)
from . import REWARD_VERSION
from .rewards import score_trace


def load_model(dataset, baseline, preparation, checkpoint, device):
    import torch
    from .policy import activate_adapter, add_reference_adapter, load_checkpoint
    from .trainer import implementation_identity, prepare_model
    meta = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if meta.get("schema") != "b4dl-reasoning-checkpoint-v1" or meta.get("stage") not in ("sft", "grpo"):
        raise ValueError("invalid reasoning model checkpoint")
    tokenizer, model, options = prepare_model(dataset, baseline, preparation, device, meta["recipe"]["lora_rank"])
    common = {"baseline_sha256": sha256(baseline), "preparation_sha256": sha256(preparation),
              "reasoning_dataset_id": dataset.manifest["dataset_id"], "implementation": implementation_identity(),
              "reward_version": REWARD_VERSION, "options": options}
    if meta["identity"]["common"] != common:
        raise ValueError("inference assets do not match checkpoint")
    if meta["stage"] == "grpo":
        add_reference_adapter(model)
    load_checkpoint(checkpoint, model, meta["identity"], meta["stage"])
    if meta["stage"] == "grpo":
        activate_adapter(model, "default")
    model.eval()
    return tokenizer, model, meta


def load_original_model(dataset, baseline, preparation, device):
    """Load the verified original B3 as a six-task comparison baseline."""
    import torch
    from ..b3 import load_original, verify_sst_binding
    from ..integration import attach
    lock, prep = read_json(baseline), read_json(preparation)
    if lock["schema"] != "b4dl-core-baseline-v1" or prep["baseline_checkpoint"] != lock["components"]["stage2"]["path"]:
        raise ValueError("preparation/baseline mismatch")
    verify_sst_binding(lock, dataset.geometry)
    if device.startswith("cuda") and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA BF16 is required for matched six-task comparison")
    tokenizer, original = load_original(lock, device)
    if device.startswith("cuda"):
        original.to(dtype=torch.bfloat16)
    model, _ = attach(original, "A0", lora_rank=0)
    model.eval()
    return tokenizer, model


def infer(dataset, model, tokenizer, checkpoint, output, condition="identity", seed=42,
          max_new_tokens=512, resume=False):
    from .trainer import sample_one
    if not dataset.requests or len({r["split"] for r in dataset.requests}) != 1 or type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("inference requires one nonempty split")
    output = Path(output)
    if output.exists() and not resume or not output.exists() and resume:
        raise ValueError("choose a new output or explicitly resume the same output")
    output.mkdir(parents=True, exist_ok=resume)
    controls = {r["request_id"]: r for r in control_plan(dataset.geometry, condition, seed)}
    run = {"schema": "b4dl-reasoning-inference-v1", "dataset_id": dataset.manifest["dataset_id"],
           "split": dataset.requests[0]["split"], "condition": condition, "seed": seed,
           "checkpoint_sha256": sha256(checkpoint), "max_new_tokens": max_new_tokens,
           "decoding": "greedy", "reward_version": REWARD_VERSION}
    from ..integration import language_model
    from ..runtime import runtime_identity
    run["runtime"] = runtime_identity(language_model(model))
    run["device"] = str(language_model(model).get_input_embeddings().weight.device)
    run["dtype"] = str(language_model(model).get_input_embeddings().weight.dtype)
    path = output/"predictions.jsonl"
    completion = {"status": "reasoning_inference_complete", "requests": len(dataset.requests), "run": run}
    with output_lease(output):
        bind_run(output/"run.json", run, resume, [path, output/"completion.json"])
        if verify_completion(output, completion, {"predictions_sha256": path}):
            validate_predictions(dataset.requests, read_jsonl(path), complete=True)
            return completion
        existing = recover_jsonl(path, lambda rows: validate_predictions(dataset.requests, rows))
        for request in dataset.requests[len(existing):]:
            control = controls[request["request_id"]]
            if control["status"] != "ready":
                prediction = {"text": "", "status": "unavailable", "generated_tokens": 0,
                              "generation_seconds": None, "unavailable_reason": control["status"]}
            else:
                sample = sample_one(model, tokenizer, dataset.model_input(request, control), max_new_tokens,
                                    do_sample=False)
                prediction = {"text": sample["text"], "status": sample["status"], "generated_tokens": len(sample["ids"]),
                              "generation_seconds": sample.get("generation_seconds")}
            prediction.update(request_id=request["request_id"], input_fingerprint=request["input_fingerprint"])
            append_jsonl(path, [prediction])
        validate_predictions(dataset.requests, read_jsonl(path), complete=True)
        completion["predictions_sha256"] = sha256(path)
        write_json(output/"completion.json", completion)
    return completion


def _mean(values):
    return sum(values)/len(values) if values else None


def summarize(rows):
    errors = defaultdict(list)
    steps = defaultdict(list)
    for row in rows:
        for key, error in row["absolute_errors"].items():
            errors[key].append(error)
        for index, value in enumerate(row["step_scores"]):
            steps[str(index)].append(value)
    timed = [r for r in rows if r["generation_seconds"] is not None]
    seconds = sum(r["generation_seconds"] for r in timed)
    return {"n": len(rows), "accuracy": _mean([float(r["correct"]) for r in rows]),
            "valid_rate": _mean([float(r["valid"]) for r in rows]),
            "whole_trace_pass_rate": _mean([float(r["valid"] and r["correct"] and all(s > 0 for s in r["step_scores"])) for r in rows]),
            "mean_step_score": {k: _mean(v) for k, v in sorted(steps.items())},
            "step_pass_rate": {k: _mean([float(score > 0) for score in v]) for k, v in sorted(steps.items())},
            "step_pass_definition": "positive grounded step score; numeric error strictly below 1 m",
            "numeric_errors_valid_only": {k: {"n": len(v), "mae": _mean(v)} for k, v in sorted(errors.items())},
            "mean_generated_tokens": _mean([r["generated_tokens"] for r in rows]),
            "timed_predictions": len(timed), "mean_generation_seconds": _mean([r["generation_seconds"] for r in timed]),
            "generated_tokens_per_second": sum(r["generated_tokens"] for r in timed)/seconds if seconds > 0 else None,
            "status_counts": dict(Counter(r["status"] for r in rows))}


def evaluate(dataset, predictions, run, repetitions=2000):
    if run["dataset_id"] != dataset.manifest["dataset_id"] or {r["split"] for r in dataset.requests} != {run["split"]}:
        raise ValueError("evaluation run/dataset identity mismatch")
    indexed = unique_index(predictions, "request_id")
    if set(indexed) - {r["request_id"] for r in dataset.requests}:
        raise ValueError("unknown prediction ID")
    rows = []
    for request in dataset.requests:
        pred = indexed.get(request["request_id"])
        if pred is not None and (pred.get("input_fingerprint") != request["input_fingerprint"]
                                 or pred.get("status") not in ("ok", "truncated", "error", "unavailable")
                                 or not isinstance(pred.get("text"), str)
                                 or type(pred.get("generated_tokens")) is not int or pred["generated_tokens"] < 0):
            raise ValueError("prediction identity or status mismatch")
        elapsed = pred.get("generation_seconds") if pred else None
        if elapsed is not None and (type(elapsed) not in (int, float) or not 0 <= elapsed < float("inf")):
            raise ValueError("invalid generation timing")
        status = pred["status"] if pred else "missing"
        score = score_trace(request, dataset.targets[request["request_id"]], pred["text"] if pred else "", status)
        rows.append({"request_id": request["request_id"], "scene_id": request["scene_id"],
                     "task": request["task"], "truth": dataset.targets[request["request_id"]]["answer"],
                     "status": status, "generated_tokens": pred.get("generated_tokens", 0) if pred else 0,
                     "generation_seconds": elapsed, **score})
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    report = {"schema": "b4dl-reasoning-evaluation-v1", "run": run, "summary": summarize(rows),
              "by_task": {task: {**summarize(values), "scene_bootstrap_accuracy": scene_interval(values, repetitions=repetitions)}
                          for task, values in sorted(by_task.items())},
              "scene_bootstrap_accuracy": scene_interval(rows, repetitions=repetitions), "rows": rows}
    if hasattr(dataset, "geometry"):
        index = unique_index(rows, "request_id")
        parents = [dataset.parent[r["request_id"]] for r in dataset.requests]
        pairs = opposite_pairs(parents, dataset.geometry.targets)
        report["opposite_pairs"] = {"n": len(pairs), "both_correct_rate": _mean([
            float(index[p["yes_id"]]["correct"] and index[p["no_id"]]["correct"]) for p in pairs])}
        orbits = defaultdict(list)
        for request in dataset.requests:
            if request["task"] == "distance": orbits[request["semantic_id"]].append(request)
        values = []
        for requests in orbits.values():
            if len(requests) != 4 or {r["option_rotation"] for r in requests} != set(range(4)):
                raise ValueError("incomplete distance option rotation")
            chosen = [r["options"].get(index[r["request_id"]]["answer"]) for r in requests]
            values.append({"semantic_consistent": None not in chosen and len(set(chosen)) == 1,
                           "all_correct": all(index[r["request_id"]]["correct"] for r in requests)})
        report["option_rotations"] = {"n": len(values),
            "semantic_consistency": _mean([float(v["semantic_consistent"]) for v in values]),
            "all_correct_rate": _mean([float(v["all_correct"]) for v in values])}
        report["excluded_questions"] = sum(dataset.parent[ident]["split"] == run["split"] for ident in dataset.exclusions)
        report["input_scope"] = dataset.manifest["input_scope"]
    return report


def compare_reports(base, candidate, repetitions=2000):
    keys = ("dataset_id", "split", "condition", "seed", "max_new_tokens", "decoding", "reward_version", "runtime", "device", "dtype")
    if any(base["run"].get(key) != candidate["run"].get(key) for key in keys):
        raise ValueError("paired comparisons require identical inputs and conditions")
    left, right = (unique_index(report["rows"], "request_id") for report in (base, candidate))
    if not left or set(left) != set(right):
        raise ValueError("paired comparison denominators differ")
    rows = []
    for ident, first in left.items():
        second = right[ident]
        if (first["scene_id"], first["truth"], first["task"]) != (second["scene_id"], second["truth"], second["task"]):
            raise ValueError("paired truth differs")
        rows.append({"request_id": ident, "scene_id": first["scene_id"], "task": first["task"],
                     "delta": int(second["correct"])-int(first["correct"])})
    grouped = defaultdict(list)
    for row in rows: grouped[row["task"]].append(row)
    return {"n": len(rows), "accuracy_delta": _mean([r["delta"] for r in rows]),
            "scene_bootstrap_delta": scene_interval(rows, "delta", repetitions),
            "by_task": {k: {"n": len(v), "delta": _mean([r["delta"] for r in v]),
                              "scene_bootstrap_delta": scene_interval(v, "delta", repetitions)} for k, v in grouped.items()},
            "rows": rows}


def prepare_free_six_task(project_root, geometry, historical_protocol, output):
    """Rebuild questions without supplying answer text to the query builder."""
    from ..six_task import validate_protocol
    from .six_query import build_free_query
    old, previous = validate_protocol(historical_protocol, geometry)
    root, output = Path(project_root), Path(output)
    if output.exists():
        raise FileExistsError("choose a new six-task protocol directory")
    data_root = root/"mllm/b4dl_dataset"
    items = read_json(data_root/"test_qa.json")
    if isinstance(items, dict): items = list(items.values())
    ego = read_json(data_root/"ego_metadata.json")
    motion = read_json(data_root/"ego_frame_motion.json")
    rows = []
    for old_row in previous:
        item = items[old_row["source_index"]]
        scene = item.get("scene_id") or item.get("id")
        if scene != old_row["scene_id"] or item["conversations"][1]["value"] != old_row["target"]:
            raise ValueError("source six-task item differs from historical protocol")
        query = build_free_query(item["conversations"][0]["value"], ego, scene, motion)
        fingerprint = digest({"query": query, "scene_bundle_sha256": geometry.manifest["scenes"][scene]["sha256"], "whole_scene": True})
        rows.append({"request_id": digest([old_row["source_index"], old_row["task"], scene, fingerprint])[:24],
                     "source_index": old_row["source_index"], "task": old_row["task"], "scene_id": scene,
                     "query": query, "target": old_row["target"], "input_fingerprint": fingerprint})
    if len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate free six-task IDs")
    output.mkdir(parents=True)
    write_jsonl(output/"requests.jsonl", rows)
    protocol = {"schema": "b4dl-reasoning-six-task-free-v1", "dataset_id": geometry.manifest["dataset_id"],
                "historical_protocol_sha256": sha256(historical_protocol), "requests": len(rows),
                "task_counts": dict(Counter(r["task"] for r in rows)), "requests_sha256": sha256(output/"requests.jsonl"),
                "free_query_builder_sha256": sha256(Path(__file__).parent/"six_query.py"),
                "generation": old["generation"], "metrics": old["metrics"], "backend_sources": old["backend_sources"],
                "historical_files_sha256": old["files_sha256"],
                "input_protocol": {"whole_scene": True, "answer_frames": False, "oracle_assisted_metadata": False},
                "source_hashes": {str(data_root/name): sha256(data_root/name) for name in ("test_qa.json", "ego_metadata.json", "ego_frame_motion.json")}}
    protocol["protocol_id"] = digest(protocol)
    write_json(output/"manifest.json", protocol)
    return protocol


def load_free_six_task(path, geometry):
    path = Path(path)
    protocol = read_json(path)
    if (protocol["schema"] != "b4dl-reasoning-six-task-free-v1"
            or protocol["protocol_id"] != digest({k: v for k, v in protocol.items() if k != "protocol_id"})
            or protocol["dataset_id"] != geometry.manifest["dataset_id"]
            or protocol["input_protocol"]["oracle_assisted_metadata"]
            or protocol["free_query_builder_sha256"] != sha256(Path(__file__).parent/"six_query.py")
            or sha256(path.parent/"requests.jsonl") != protocol["requests_sha256"]):
        raise ValueError("invalid free six-task protocol")
    for name, expected in {**protocol["source_hashes"], **protocol["historical_files_sha256"]}.items():
        if sha256(name) != expected:
            raise ValueError("free six-task source or metric asset changed: " + name)
    rows = read_jsonl(path.parent/"requests.jsonl")
    if (len(rows) != protocol["requests"] or dict(Counter(r["task"] for r in rows)) != protocol["task_counts"]
            or len({r["request_id"] for r in rows}) != len(rows)):
        raise ValueError("free six-task denominator mismatch")
    for row in rows:
        scene = geometry.manifest["scenes"][row["scene_id"]]
        if scene["split"] != "test" or row["input_fingerprint"] != digest({"query": row["query"], "scene_bundle_sha256": scene["sha256"], "whole_scene": True}):
            raise ValueError("free six-task input mismatch")
    return protocol, rows


def infer_six_task(dataset, model, tokenizer, model_artifact, protocol_path, output, resume=False, model_kind="reasoning"):
    from ..six_task import infer_predictions
    from ..integration import language_model
    from ..runtime import runtime_identity
    protocol, rows = load_free_six_task(protocol_path, dataset.geometry)
    output = Path(output)
    if output.exists() and not resume or not output.exists() and resume:
        raise ValueError("choose a new output or explicitly resume it")
    output.mkdir(parents=True, exist_ok=resume)
    run = {"schema": "b4dl-reasoning-six-task-inference-v1", "protocol_id": protocol["protocol_id"],
           "model_kind": model_kind, "model_artifact_sha256": sha256(model_artifact), "generation": protocol["generation"],
           "runtime": runtime_identity(language_model(model)),
           "device": str(language_model(model).get_input_embeddings().weight.device),
           "dtype": str(language_model(model).get_input_embeddings().weight.dtype), "oracle_assisted_metadata": False}
    with output_lease(output):
        infer_predictions(output, run, rows, dataset.geometry, model, tokenizer, resume)
    return run


def six_task_metrics(dataset, protocol_path, prediction_path, output):
    import math
    import os
    import sys
    from ..six_task import prediction_arrays
    protocol, rows = load_free_six_task(protocol_path, dataset.geometry)
    metrics_config = protocol["metrics"]
    os.environ["NLTK_DATA"] = metrics_config["nltk_data"]
    if metrics_config.get("java_home"):
        os.environ["JAVA_HOME"] = metrics_config["java_home"]
        os.environ["PATH"] = str(Path(metrics_config["java_home"])/"bin") + os.pathsep + os.environ.get("PATH", "")
    import nltk
    nltk.data.path.insert(0, metrics_config["nltk_data"])
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab", "corpora/wordnet", "corpora/omw-1.4"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.data.find(resource + ".zip")
    download = nltk.download
    try:
        nltk.download = lambda *args, **kwargs: False
        from evaluation import evaluate_model as backend
    finally:
        nltk.download = download
    if not all(getattr(backend, key) for key in ("HAS_NLTK", "HAS_ROUGE", "HAS_BERTSCORE", "HAS_COCO_BLEU", "HAS_COCO_METEOR")):
        raise RuntimeError("required six-task metric backend is unavailable")
    B4DLEvaluator = backend.B4DLEvaluator
    if sha256(sys.modules[B4DLEvaluator.__module__].__file__) != protocol["backend_sources"]["evaluator"]["sha256"]:
        raise ValueError("six-task evaluator differs from frozen protocol")
    predictions = read_jsonl(prediction_path)
    completion = read_json(Path(prediction_path).parent/"completion.json")
    if (completion.get("status") != "six_task_inference_complete"
            or completion.get("run", {}).get("protocol_id") != protocol["protocol_id"]
            or completion.get("run") != read_json(Path(prediction_path).parent/"run.json")
            or completion.get("predictions_sha256") != sha256(prediction_path)):
        raise ValueError("six-task inference completion receipt mismatch")
    evaluator = B4DLEvaluator(use_gpt=False, bert_model_path=metrics_config["bert_model_path"], meteor_backend="dual")
    if metrics_config["bertscore_layer"] != 17:
        raise ValueError("six-task comparison requires the frozen RoBERTa layer")
    evaluator.metric_backend["bertscore"]["model_type"] = metrics_config["bert_model_path"]
    evaluator.metric_backend["bertscore"]["num_layers"] = metrics_config["bertscore_layer"]
    def strict_bleu(predictions, ground_truths):
        gts = {i: [gt] for i, gt in enumerate(ground_truths)}
        res = {i: [prediction] for i, prediction in enumerate(predictions)}
        score, _ = backend._CocoBleu(4).compute_score(gts, res)
        return float(score[3])
    def strict_bertscore(predictions, ground_truths):
        _, _, values = backend.bert_score_fn(predictions, ground_truths, lang="en",
            model_type=metrics_config["bert_model_path"], num_layers=metrics_config["bertscore_layer"], verbose=False)
        return float(values.mean().item())
    evaluator.compute_bleu4 = strict_bleu
    evaluator.compute_bertscore = strict_bertscore
    results = evaluator.evaluate_all(prediction_arrays(rows, predictions))
    if set(results) != set(evaluator.ALL_TASKS):
        raise ValueError("six-task results are incomplete")
    for task in evaluator.COMPLEX_TASKS:
        names = ("bleu4", "meteor", "meteor_pycocoevalcap", "rouge_l", "bertscore")
        if any(name not in results[task] or not math.isfinite(results[task][name]) for name in names):
            raise ValueError("required complex-task metric failed: " + task)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    evaluator.save_results(results, evaluator.compute_final_score(results), str(output/"metrics.json"))
    receipt = {"status": "six_task_metrics_complete", "protocol_id": protocol["protocol_id"],
               "requests": len(rows), "prediction_statuses": dict(Counter(r["status"] for r in predictions)),
               "predictions_sha256": sha256(prediction_path), "metrics_sha256": sha256(output/"metrics.json"),
               "oracle_assisted_metadata": False}
    write_json(output/"completion.json", receipt)
    return receipt


def compare_six_task_metrics(base_dir, candidate_dir):
    base_dir, candidate_dir = Path(base_dir), Path(candidate_dir)
    left_receipt, right_receipt = (read_json(path/"completion.json") for path in (base_dir, candidate_dir))
    if (left_receipt.get("status") != "six_task_metrics_complete" or right_receipt.get("status") != "six_task_metrics_complete"
            or left_receipt["protocol_id"] != right_receipt["protocol_id"]
            or left_receipt["requests"] != right_receipt["requests"]
            or left_receipt["metrics_sha256"] != sha256(base_dir/"metrics.json")
            or right_receipt["metrics_sha256"] != sha256(candidate_dir/"metrics.json")):
        raise ValueError("six-task metrics require complete matched protocol receipts")
    first, second = (read_json(path/"metrics.json") for path in (base_dir, candidate_dir))
    if first["metric_backend"] != second["metric_backend"] or set(first["per_task_metrics"]) != set(second["per_task_metrics"]):
        raise ValueError("six-task metric backends or tasks differ")
    changes = {}
    for task, a in first["per_task_metrics"].items():
        b = second["per_task_metrics"][task]
        if set(a) != set(b): raise ValueError("six-task metric names differ")
        changes[task] = {key: {"base": a[key], "candidate": b[key],
                               "delta": b[key]-a[key] if a[key] is not None and b[key] is not None else None} for key in a}
    return {"protocol_id": left_receipt["protocol_id"], "requests": left_receipt["requests"],
            "per_task_metrics": changes, "metric_backend": first["metric_backend"]}
