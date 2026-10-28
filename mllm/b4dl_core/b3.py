"""Explicit B3 integration entry: verify, bounded engineering train, and infer."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from .inputs import GeometryDataset, control_plan
from .integration import (attach, batch_inputs, language_model, load_checkpoint, padding_side,
                          save_checkpoint, trainable_state)
from .io import digest, read_json, read_jsonl, sha256, write_json, write_jsonl
from .resume import (append_jsonl, bind_run, output_lease, recover_jsonl,
                     validate_predictions, verify_completion)
from .runtime import runtime_identity
from .temporal import TemporalConfig


def verify_baseline(lock):
    for key, component in lock["components"].items():
        root = Path(component["path"])
        actual = {}
        if root.is_file():
            actual[root.name] = sha256(root)
            artifact = actual[root.name]
        else:
            if not root.is_dir():
                raise FileNotFoundError(root)
            files = sorted(p for p in root.rglob("*") if p.is_file() and
                p.suffix in (".json", ".bin", ".safetensors", ".model", ".txt") and
                not p.name.startswith(("optimizer", "scheduler", "training_args")))
            actual = {p.relative_to(root).as_posix(): sha256(p) for p in files}
            artifact = hashlib.sha256(json.dumps(actual, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if actual != component["files_sha256"] or artifact != component["artifact_sha256"]:
            raise ValueError("baseline files changed: " + key)
        print("verified baseline component:", key, flush=True)


@contextmanager
def device_lease(device, lock_path, minimum_free_mib=20000):
    if torch.device(device).type != "cuda":
        yield
        return
    import fcntl
    if device not in ("cuda", "cuda:0"):
        raise ValueError("current GPU lease supports only cuda:0")
    with Path(lock_path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            values = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True).strip().split(",")
            if int(values[0]) < minimum_free_mib or int(values[1]) > 20:
                raise RuntimeError("GPU not sufficiently idle for B3 integration")
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_original(lock, device):
    from .runtime import check_runtime
    check_runtime(require_cuda=torch.device(device).type == "cuda")
    from vtimellm.model.builder import load_pretrained_model
    verify_baseline(lock)
    components = lock["components"]
    args = SimpleNamespace(model_base=components["model_base"]["path"],
        pretrain_mm_mlp_adapter=components["pretrain_mm_mlp_adapter"]["path"])
    tokenizer, model, _ = load_pretrained_model(args, components["stage2"]["path"], None)
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype).eval()
    return tokenizer, model


def implementation_id():
    package = Path(__file__).parent
    files = list(package.glob("*.py")) + [package.parent/"vtimellm/model"/name for name in
        ("vtimellm_arch.py", "vtimellm_llama.py", "builder.py")]
    files += [package.parent/"vtimellm"/name for name in ("mm_utils.py", "conversation.py", "constants.py")]
    return digest({p.relative_to(package.parent).as_posix(): sha256(p) for p in files})


def identity_for(args, dataset):
    result = {"baseline_manifest_sha256": sha256(args.baseline), "dataset_id": dataset.manifest["dataset_id"],
              "implementation_sha256": implementation_id()}
    if getattr(args, "experiment", None): result["experiment_sha256"] = sha256(args.experiment)
    return result


def verify_sst_binding(lock, dataset):
    expected = lock.get("sst_provenance")
    actual = dataset.manifest.get("encoder_provenance")
    if not expected or not actual or not dataset.manifest.get("input_scope"):
        raise ValueError("formal training requires a verified SST binding and input-scoped supervision")
    if any(expected.get(key) != actual.get(key) for key in ("sha256", "provenance_id")):
        raise ValueError("baseline and data use different SST provenance")
    if sha256(actual["path"]) != actual["sha256"]:
        raise ValueError("SST provenance receipt changed")


def chosen_config(preparation, arm):
    return replace(TemporalConfig(**preparation["frame_adapter"]), use_geometry=arm == "A3")


class NonfiniteTrainingError(ValueError):
    """A stopped update with JSON-safe numerical evidence, never a skipped sample."""
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def tensor_summary(value):
    value = value.detach()
    finite = torch.isfinite(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "nonfinite_count": int((~finite).sum()),
            "finite_abs_max": float(value[finite].double().abs().max()) if finite.any() else None}


def teacher_loss(model, batch):
    device = language_model(model).get_input_embeddings().weight.device
    with padding_side(model, "right"), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        result = model(**batch, use_cache=False)
    if not torch.isfinite(result.loss):
        raise NonfiniteTrainingError("nonfinite training loss", {
            "stage": "forward_loss", "loss": tensor_summary(result.loss),
            "logits": tensor_summary(result.logits)})
    return result.loss


def optimizer_for(model, lr):
    if not 0 < lr < 1:
        raise ValueError("invalid learning rate")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("optimizer requires nonempty FP32 trainable parameters")
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=.01)


def train_step(model, batches, optimizer, scaler):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for micro, batch in enumerate(batches):
        try:
            loss = teacher_loss(model, batch)
            if not torch.isfinite(loss):
                raise NonfiniteTrainingError("nonfinite training loss", {
                    "stage": "forward_loss", "loss": tensor_summary(loss)})
            losses.append(float(loss.detach()))
            scaler.scale(loss/len(batches)).backward()
        except NonfiniteTrainingError as error:
            error.details.update(micro=micro, finite_micro_losses=losses,
                batch={name: tensor_summary(value) for name, value in batch.items()},
                supervised_tokens=int(batch["labels"].ne(-100).sum()) if "labels" in batch else None)
            optimizer.zero_grad(set_to_none=True)
            raise
    scaler.unscale_(optimizer)
    gradients = {name: p.grad for name, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    bad = {name: tensor_summary(g) for name, g in gradients.items() if not torch.isfinite(g).all()}
    if not gradients or bad:
        optimizer.zero_grad(set_to_none=True)
        raise NonfiniteTrainingError("missing or nonfinite gradient; no optimizer update applied", {
            "stage": "unscaled_gradient", "nonfinite_gradients": bad, "gradient_tensors": len(gradients),
            "finite_micro_losses": losses})
    try:
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    except RuntimeError as error:
        optimizer.zero_grad(set_to_none=True)
        raise NonfiniteTrainingError("nonfinite gradient norm; no optimizer update applied", {
            "stage": "gradient_norm", "finite_micro_losses": losses}) from error
    scaler.step(optimizer)
    scaler.update()
    result = {"loss": sum(losses)/len(losses), "gradient_norm": float(norm), "grad_accum": len(batches)}
    adapter = getattr(language_model(model).get_model(), "frame_adapter", None)
    if adapter is not None:
        gradient = adapter.blocks[0].attention.in_proj_weight.grad
        result["adapter_attention_grad_abs_sum"] = float(gradient.abs().sum()) if gradient is not None else 0.
    return result


def legacy_generate(model, batch, budget):
    base = language_model(model)
    legacy_images = [batch["images"][i, mask].to(base.get_model().mm_projector.weight.dtype)
                     for i, mask in enumerate(batch["frame_valid_mask"])]
    with torch.no_grad(), padding_side(base, "left"):
        _, _, attention, _, embeds, _ = base.prepare_inputs_labels_for_multimodal(
            batch["input_ids"], None, batch["attention_mask"], None, None, legacy_images)
        return base.generate(inputs_embeds=embeds, attention_mask=attention, max_new_tokens=budget,
            do_sample=False, use_cache=True, pad_token_id=base.config.eos_token_id)


def verify_integration(args, preparation, lock, dataset):
    tokenizer, model = load_original(lock, args.device)
    requests, lengths = [], set()
    for r in dataset.requests:
        if r["feature_length"] not in lengths:
            requests.append(r); lengths.add(r["feature_length"])
        if len(requests) == 2: break
    if len(requests) < 2:
        raise ValueError("verification requires two actual scenes with different frame counts")
    items = [dataset.model_input(r) for r in requests]
    inference = batch_inputs(model, tokenizer, items)
    with torch.no_grad():
        expected = legacy_generate(model, inference, args.max_new_tokens)
        legacy = {k: v for k, v in inference.items() if not k.startswith("frame_")}
        legacy["images"] = [inference["images"][i, mask].to(model.get_model().mm_projector.weight.dtype)
                            for i, mask in enumerate(inference["frame_valid_mask"])]
        expected_logits = model(**legacy, use_cache=False).logits.cpu()
        disabled = model(**inference, use_cache=False).logits.cpu()
        torch.testing.assert_close(expected_logits, disabled, rtol=0, atol=0)
    model, options = attach(model, arm="A3", temporal=chosen_config(preparation, "A3"), lora_rank=0)
    with torch.no_grad():
        initialized = model(**inference, use_cache=False).logits.cpu()
        torch.testing.assert_close(expected_logits, initialized, rtol=0, atol=0)
    calls = []
    hook = model.get_model().frame_adapter.register_forward_hook(lambda *unused: calls.append(1))
    actual = model.generate(**inference, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
                            pad_token_id=model.config.eos_token_id)
    hook.remove()
    if len(calls) != 1 or not torch.equal(actual, expected):
        raise ValueError(f"generation mismatch: adapter_calls={len(calls)}, expected={expected.tolist()}, actual={actual.tolist()}")
    for i, item in enumerate(items):
        single = model.generate(**batch_inputs(model, tokenizer, [item]), max_new_tokens=args.max_new_tokens,
                                do_sample=False, use_cache=True, pad_token_id=model.config.eos_token_id)
        if not torch.equal(actual[i, :single.shape[1]], single[0]):
            raise ValueError("batched/individual generation differs")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Real training QA is limited to two engineering optimizer steps.
    training = batch_inputs(model, tokenizer, items[:1], [dataset.targets[requests[0]["request_id"]]])
    optimizer = optimizer_for(model, 1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.device(args.device).type == "cuda", init_scale=1024)
    steps = [train_step(model, [training], optimizer, scaler) for _ in range(2)]
    grad = float(model.get_model().frame_adapter.blocks[0].attention.in_proj_weight.grad.abs().sum())
    if grad <= 0:
        raise ValueError("adapter attention received no gradient after residual warmup")
    identity = identity_for(args, dataset)
    checkpoint = Path(args.output)/"engineering-extension.pt"
    save_checkpoint(checkpoint, model, options, identity, {"step": 2, "engineering_only": True}, optimizer, scaler)
    model.eval()
    with torch.no_grad(): before = model(**inference, use_cache=False).logits.cpu()
    with torch.no_grad(): model.get_model().frame_adapter.output.bias.add_(1.)
    load_checkpoint(checkpoint, model, options, identity, optimizer, scaler)
    with torch.no_grad(): after = model(**inference, use_cache=False).logits.cpu()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    report = {"status": "real_b3_integration_verified", "identity": identity, "arm": "A3", "lora_rank": 0,
        "requests": [r["request_id"] for r in requests], "frame_lengths": [r["feature_length"] for r in requests],
        "disabled_logits_exact": True, "zero_init_logits_exact": True, "legacy_generation_tokens_exact": True,
        "variable_frame_batch_tokens_exact": True, "adapter_calls_per_generation": len(calls),
        "optimizer_steps": steps, "attention_gradient_after_warmup": grad, "reload_logits_exact": True,
        "checkpoint_sha256": sha256(checkpoint), "training_split": "train", "engineering_only": True,
        "task_performance_measured": False, "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None}
    write_json(Path(args.output)/"verification.json", report)
    return report


def batch_indices(total, step, micro, batch_size, accum, seed):
    result = []
    for position in range((step*accum+micro)*batch_size, (step*accum+micro+1)*batch_size):
        epoch, index = divmod(position, total)
        result.append(int(np.random.default_rng(seed+epoch).permutation(total)[index]))
    return result


def train(args, preparation, lock, dataset):
    tokenizer, original = load_original(lock, args.device)
    model, options = attach(original, args.arm, chosen_config(preparation, args.arm), args.lora_rank)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = optimizer_for(model, args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.device(args.device).type == "cuda", init_scale=1024)
    identity = identity_for(args, dataset)
    recipe = {"lr": args.lr, "batch_size": args.batch_size, "grad_accum": args.grad_accum, "seed": args.seed,
              "engineering_only": args.engineering, "split": "train"}
    if getattr(args, "experiment", None): recipe["experiment_sha256"] = sha256(args.experiment)
    start = 0
    if args.resume:
        status = load_checkpoint(args.resume, model, options, identity, optimizer, scaler)
        if status["recipe"] != recipe:
            raise ValueError("resume recipe mismatch")
        start = status["step"]
        if start >= args.steps:
            raise ValueError("resume steps must exceed completed optimizer steps")
    logs = []
    write_json(Path(args.output)/"runtime.json", {"status": "model_loaded_training_ready", "identity": identity,
        "options": options, "recipe": recipe, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "train_examples": len(dataset.requests), "optimizer_steps": args.steps,
        "gpu": torch.cuda.get_device_name() if torch.device(args.device).type == "cuda" else None})
    for step in range(start, args.steps):
        started = time.perf_counter()
        batches, request_context = [], []
        for micro in range(args.grad_accum):
            indices = batch_indices(len(dataset.requests), step, micro, args.batch_size, args.grad_accum, args.seed)
            requests = [dataset.requests[i] for i in indices]
            request_context.append([{"index": i, **{key: r[key] for key in
                ("request_id", "scene_id", "input_fingerprint", "task", "predicate")}} for i, r in zip(indices, requests)])
            batches.append(batch_inputs(model, tokenizer, [dataset.model_input(r) for r in requests],
                                        [dataset.targets[r["request_id"]] for r in requests]))
        try:
            row = {"step": step+1, **train_step(model, batches, optimizer, scaler)}
        except NonfiniteTrainingError as error:
            failure = {"status": "training_failed",
                "attempted_step": step+1, "completed_updates": step, "identity": identity,
                "recipe": recipe, "requests_by_micro": request_context, "diagnostics": error.details,
                "optimizer_update_applied": False}
            write_json(Path(args.output)/"failure.json", failure)
            # Preserve the actual failed model, rather than only the older
            # periodic checkpoint. This artifact is never a resume endpoint.
            snapshot = Path(args.output)/"failure-state.pt"
            save_checkpoint(snapshot, model, options, identity,
                {"diagnostic_only": True, "step": step, "recipe": recipe}, optimizer, scaler)
            failure.update(diagnostic_checkpoint=snapshot.name, diagnostic_checkpoint_sha256=sha256(snapshot))
            write_json(Path(args.output)/"failure.json", failure)
            raise
        row.update(examples_seen=(step+1)*args.batch_size*args.grad_accum,
                   epoch=(step+1)*args.batch_size*args.grad_accum/len(dataset.requests),
                   seconds=time.perf_counter()-started)
        logs.append(row)
        print(json.dumps(row), flush=True)
        with (Path(args.output)/"loss.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row)+"\n")
        if step == start or (step+1) % args.save_every == 0 or step+1 == args.steps:
            save_checkpoint(Path(args.output)/f"step-{step+1}.pt", model, options, identity,
                            {"step": step+1, "recipe": recipe}, optimizer, scaler)
            write_json(Path(args.output)/"latest.json", {"step": step+1, "checkpoint": f"step-{step+1}.pt",
                "identity": identity, "checkpoint_sha256": sha256(Path(args.output)/f"step-{step+1}.pt")})
            keep = getattr(args, "keep_last", None)
            if keep:
                checkpoints = sorted((p for p in Path(args.output).glob("step-*.pt")
                    if p.stem[5:].isdigit() and int(p.stem[5:]) != 1), key=lambda p: int(p.stem[5:]))
                for old in checkpoints[:-keep]: old.unlink()
    report = {"status": "engineering_training_complete" if args.engineering else "training_complete",
              "identity": identity, "options": options, "recipe": recipe, "optimizer_steps": args.steps,
              "resumed_from_step": start, "quality_evaluation_completed": False}
    write_json(Path(args.output)/"completion.json", report)
    return report


def infer(args, preparation, lock, dataset):
    with output_lease(args.output):
        return _infer(args, preparation, lock, dataset)


def _infer(args, preparation, lock, dataset):
    tokenizer, original = load_original(lock, args.device)
    model, options = attach(original, args.arm, chosen_config(preparation, args.arm), args.lora_rank)
    identity = identity_for(args, dataset)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, options, identity)
    model.eval()
    controls = control_plan(dataset, args.condition)
    run = {"dataset_id": identity["dataset_id"], "split": args.split, "condition": args.condition,
           "model_id": lock["baseline_id"]+"/"+args.arm, "prediction_kind": "model",
           "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else sha256(args.baseline)}
    generation = {"max_new_tokens": args.max_new_tokens, "do_sample": False, "use_cache": True,
                  "pad_token_id": tokenizer.eos_token_id}
    manifest = {"schema": "b4dl-core-inference-resume-v1", "run": run, "identity": identity,
        "options": options, "preparation_sha256": digest(preparation), "generation": generation,
        "batch_size": args.batch_size, "seed": getattr(args, "seed", 42),
        "device": str(args.device), "dtype": str(language_model(model).get_input_embeddings().weight.dtype),
        "runtime": runtime_identity(language_model(model)),
        "requests_sha256": digest(dataset.requests), "controls_sha256": digest(controls)}
    manifest["run_id"] = digest(manifest)
    directory = Path(args.output)
    prediction_path, receipt_path = directory/"predictions.jsonl", directory/"input-receipts.jsonl"
    journal_path = directory/"inference-journal.jsonl"
    bind_run(directory/"resume.json", manifest, getattr(args, "resume_inference", False),
             [prediction_path, receipt_path, journal_path, directory/"run.json", directory/"completion.json"])
    if (directory/"run.json").exists():
        if read_json(directory/"run.json") != run:
            raise ValueError("evaluation run identity mismatch")
    else:
        write_json(directory/"run.json", run)
    if not journal_path.exists() and any(p.exists() and p.stat().st_size for p in (prediction_path, receipt_path)):
        raise ValueError("missing inference journal for existing exports")

    def validate_journal(batches):
        offset = 0
        for transaction in batches:
            if not isinstance(transaction, dict) or transaction.get("sha256") != digest({k: v for k, v in transaction.items() if k != "sha256"}):
                raise ValueError("inference journal checksum mismatch")
            entries = transaction["entries"]
            if (transaction["run_id"] != manifest["run_id"] or transaction["start"] != offset
                    or not entries or len(entries) != min(args.batch_size, len(dataset.requests)-offset)):
                raise ValueError("inference journal batch/order mismatch")
            requests = dataset.requests[offset:offset+len(entries)]
            validate_predictions(requests, [e["prediction"] for e in entries], complete=True)
            for entry, control in zip(entries, controls[offset:offset+len(entries)]):
                receipt = entry["receipt"]
                if any(receipt.get(k) != v for k, v in control.items()):
                    raise ValueError("inference receipt/control mismatch")
                if control["status"] == "ready":
                    if entry["prediction"]["status"] not in ("ok", "truncated") or len(receipt.get("feature_array_sha256", "")) != 64:
                        raise ValueError("missing generated prediction or feature receipt")
                elif entry["prediction"]["status"] != "unavailable" or receipt != control:
                    raise ValueError("invalid unavailable prediction/receipt")
            offset += len(entries)

    completion = {"status": "inference_complete", "run": run, "identity": identity,
                  "options": options, "predictions": len(dataset.requests), "run_id": manifest["run_id"]}
    files = {"predictions_sha256": prediction_path, "input_receipts_sha256": receipt_path,
             "journal_sha256": journal_path}
    if verify_completion(directory, completion, files):
        validate_journal(read_jsonl(journal_path))
        validate_predictions(dataset.requests, read_jsonl(prediction_path), complete=True)
        return run
    transactions = recover_jsonl(journal_path, validate_journal)
    predictions = [e["prediction"] for t in transactions for e in t["entries"]]
    receipts = [e["receipt"] for t in transactions for e in t["entries"]]
    # These are compatibility views. The joint journal is the recovery authority.
    write_jsonl(prediction_path, predictions)
    write_jsonl(receipt_path, receipts)
    for start in range(len(predictions), len(dataset.requests), args.batch_size):
        requests = dataset.requests[start:start+args.batch_size]
        batch_controls = controls[start:start+args.batch_size]
        entries = {}
        ready = [(r, c) for r, c in zip(requests, batch_controls) if c["status"] == "ready"]
        if ready:
            items = [dataset.model_input(r, c) for r, c in ready]
            batch = batch_inputs(model, tokenizer, items)
            output = model.generate(**batch, **generation)
            if len(output) != len(ready):
                raise ValueError("generation batch size mismatch")
            for (r, c), item, tokens in zip(ready, items, output.tolist()):
                finished = tokenizer.eos_token_id in tokens
                if finished: tokens = tokens[:tokens.index(tokenizer.eos_token_id)]
                entries[r["request_id"]] = {
                    "prediction": {"request_id": r["request_id"], "input_fingerprint": r["input_fingerprint"],
                        "status": "ok" if finished else "truncated", "text": tokenizer.decode(tokens, skip_special_tokens=False).strip()},
                    "receipt": {**c, "feature_array_sha256": hashlib.sha256(item["features"].tobytes()).hexdigest(),
                        "feature_shape": list(item["features"].shape), "feature_dtype": str(item["features"].dtype)}}
        for r, c in zip(requests, batch_controls):
            if c["status"] != "ready":
                entries[r["request_id"]] = {"prediction": {"request_id": r["request_id"],
                    "input_fingerprint": r["input_fingerprint"], "status": "unavailable", "text": ""}, "receipt": c}
        ordered = [entries[r["request_id"]] for r in requests]
        transaction = {"run_id": manifest["run_id"], "start": start, "entries": ordered}
        transaction["sha256"] = digest(transaction)
        append_jsonl(journal_path, [transaction])
        append_jsonl(prediction_path, [e["prediction"] for e in ordered])
        append_jsonl(receipt_path, [e["receipt"] for e in ordered])
    transactions = read_jsonl(journal_path)
    validate_journal(transactions)
    predictions = [e["prediction"] for t in transactions for e in t["entries"]]
    receipts = [e["receipt"] for t in transactions for e in t["entries"]]
    validate_predictions(dataset.requests, predictions, complete=True)
    if read_jsonl(prediction_path) != predictions or read_jsonl(receipt_path) != receipts:
        raise ValueError("inference exports differ from committed journal")
    completion.update({key: sha256(path) for key, path in files.items()})
    write_json(directory/"completion.json", completion)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "train", "infer"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-lock", default="/root/autodl-tmp/.b4dl_gpu0_eval.lock")
    parser.add_argument("--arm", choices=("A0", "A1", "A2", "A3"), default="A3")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--keep-last", type=int, help="Keep this many recent checkpoints, plus step 1")
    parser.add_argument("--experiment", help="Frozen formal-training experiment manifest")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--engineering", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--resume-inference", action="store_true", help="Resume the same inference directory and identity")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("train", "internal_dev", "test"), default="test")
    parser.add_argument("--condition", choices=("identity", "zero", "repeat_first", "shuffle_features", "cross_scene"), default="identity")
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = 8 if args.command == "verify" else 256
    if min(args.batch_size, args.grad_accum, args.save_every, args.max_new_tokens) < 1 or args.lora_rank < 0:
        parser.error("invalid numerical settings")
    if args.keep_last is not None and args.keep_last < 1: parser.error("keep-last must be positive")
    if args.command == "verify" and (args.arm != "A3" or args.lora_rank):
        parser.error("verify checks A3 with no new LoRA; separate tests cover optional LoRA")
    if args.command == "train" and (args.arm == "A0" or args.steps is None or args.steps < 1):
        parser.error("training requires A1/A2/A3 and explicit positive steps")
    if args.command == "infer" and (args.arm == "A0" and args.checkpoint or args.arm != "A0" and not args.checkpoint):
        parser.error("A0 uses original baseline; A1/A2/A3 inference requires an extension checkpoint")
    if args.command != "infer" and args.checkpoint or args.command != "train" and args.resume:
        parser.error("checkpoint belongs to infer; resume belongs to train")
    if args.resume_inference and args.command != "infer":
        parser.error("resume-inference belongs to infer")
    output = Path(args.output)
    if output.exists() and not args.resume_inference:
        parser.error("output already exists; infer may use --resume-inference")
    if args.resume_inference and not (output/"resume.json").is_file():
        parser.error("resume-inference requires an existing inference identity")
    preparation, lock = read_json(args.preparation), read_json(args.baseline)
    if lock["schema"] != "b4dl-core-baseline-v1" or preparation["baseline_checkpoint"] != lock["components"]["stage2"]["path"]:
        parser.error("preparation/baseline identity mismatch")
    split = args.split if args.command == "infer" else "train"
    dataset = GeometryDataset(args.dataset, split)
    if not dataset.requests: parser.error("empty requested split")
    if args.command == "train":
        if args.engineering:
            if args.steps > 20: parser.error("engineering run limited to 20 total optimizer steps")
        elif (preparation["gates_before_real_training"] or dataset.manifest["usage"] == "mini_engineering_validation"
              or not dataset.manifest["encoder_checkpoint_verified"] or not lock["sst_encoder_checkpoint_verified"]):
            parser.error("formal training requires completed readiness gates, full data and verified SST provenance; use --engineering only for bounded checks")
        else:
            try:
                verify_sst_binding(lock, dataset)
                from .experiment import validate_experiment
                validate_experiment(args, preparation, lock, dataset)
            except (ValueError, OSError) as error:
                parser.error(str(error))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output.mkdir(parents=True, exist_ok=args.resume_inference)
    if not args.resume_inference:
        write_json(output/"invocation.json", vars(args))
    try:
        with device_lease(args.device, args.gpu_lock):
            report = {"verify": verify_integration, "train": train, "infer": infer}[args.command](args, preparation, lock, dataset)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except Exception as error:
        write_json(output/"error.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
