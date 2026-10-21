"""Collect reproducible B3 answers for all official nuScenes mini scenes."""

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from answerseg.data import digest, read_jsonl, write_json
from answerseg.language import LanguageBridge, make_prompt, prompt_ids
from answerseg.runtime import file_sha256, load_b3, source_identity

SCHEMA = "b4dl-mini-free-answers-v1"


def event(name, **values):
    print(json.dumps(dict(event=name, time=time.time(), **values)), flush=True)


def question_range(question, count):
    indices = [
        int(i) for i in re.findall(r"\bframes?\s+(\d+)", question, re.IGNORECASE)
    ]
    if not indices:
        return 0, count - 1, "whole_scene_fallback"
    if min(indices) < 0 or max(indices) >= count:
        raise ValueError("question references a frame outside its mini scene")
    return min(indices), max(indices), "question_frames"


def prepare(args):
    from nuscenes.utils.splits import create_splits_scenes
    from scripts.ego_text import render_metatoken_for_range

    args.output_dir.mkdir(parents=True, exist_ok=False)
    mini = args.nuscenes_root / "v1.0-mini"
    tables = {
        name: json.loads((mini / (name + ".json")).read_text(encoding="utf-8"))
        for name in ("scene", "sample", "sample_data", "sensor", "calibrated_sensor")
    }
    splits = create_splits_scenes()
    mini_scenes = {r["token"]: r for r in tables["scene"]}
    if len(mini_scenes) != 10:
        raise ValueError("expected the ten official mini scenes")
    samples = {r["token"]: r for r in tables["sample"]}
    sensors = {r["token"]: r["channel"] for r in tables["sensor"]}
    calibrations = {
        r["token"]: sensors[r["sensor_token"]] for r in tables["calibrated_sensor"]
    }
    lidar = {
        r["filename"]: r
        for r in tables["sample_data"]
        if calibrations[r["calibrated_sensor_token"]] == "LIDAR_TOP"
        and r["is_key_frame"]
    }
    metadata_path = args.repo_root / "datageneration/data/metadata/scene_metadata.json"
    motion_path = args.repo_root / "mllm/b4dl_dataset/ego_frame_motion.json"
    scenes = {
        r["scene_id"]: r
        for r in json.loads(metadata_path.read_text(encoding="utf-8"))
        if r["scene_token"] in mini_scenes
    }
    motion = json.loads(motion_path.read_text(encoding="utf-8"))
    features = {}
    for scene_id, scene in scenes.items():
        paths = [path for key, path in sorted(scene["paths"]["PATH_LIDAR_TOP"].items())]
        frames = []
        for path in paths:
            row = lidar[path]
            if samples[row["sample_token"]]["scene_token"] != scene["scene_token"]:
                raise ValueError("SST row points to another scene")
            frames.append(row["token"])
        path = args.feature_root / (scene_id + ".npy")
        values = np.load(path, allow_pickle=False)
        if values.shape != (len(frames), 768) or not np.isfinite(values).all():
            raise ValueError("SST row count or values disagree with mini metadata")
        features[scene_id] = {
            "path": path.name,
            "sha256": file_sha256(path),
            "indices": list(range(len(frames))),
            "frame_tokens": frames,
            "input_policy": "whole_scene",
            "encoder_checkpoint_id": "legacy-stage2-SST-cache:original-encoder-unverified",
        }
    sources, requests = {}, []
    for filename in (
        "stage2_conversations.json",
        "stage3_conversations.json",
        "test_qa.json",
    ):
        path = args.repo_root / "mllm/b4dl_dataset" / filename
        sources[str(path)] = file_sha256(path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        for index, row in enumerate(rows):
            scene_id = str(row.get("scene_id", ""))
            if scene_id not in scenes:
                continue
            scene = scenes[scene_id]
            if row["scene_token"] != scene["scene_token"]:
                raise ValueError("QA and mini scene identity disagree")
            # ONLY the human question is read. Reference answers are excluded
            # before metadata construction, request hashing and generation.
            original = row["conversations"][0]
            if original["from"] != "human":
                raise ValueError("expected original human question")
            question = original["value"].split("<meta>")[0]
            for tag in ("<video>", "<4DLiDAR>"):
                question = question.replace(tag, "")
            question = question.strip()
            first, last, policy = question_range(
                question, len(features[scene_id]["frame_tokens"])
            )
            meta = render_metatoken_for_range(motion, scene_id, first, last)
            if not meta:
                raise ValueError("missing meta2 sensor motion for mini scene")
            input_question = "<4DLiDAR>\n<video>\n" + question + "\n<meta> " + meta
            name = mini_scenes[scene["scene_token"]]["name"]
            split = "mini_train" if name in splits["mini_train"] else "mini_val"
            if name not in splits[split]:
                raise ValueError("scene is not an official mini split member")
            key = digest([input_question, features[scene_id]])
            requests.append(
                {
                    "request_id": f"mini:{filename}:{index}",
                    "dataset_version": "v1.0-mini",
                    "mini_split": split,
                    "source": {
                        "path": str(path),
                        "sha256": sources[str(path)],
                        "index": index,
                        "original_split": row.get("split"),
                        "group": filename,
                    },
                    "scene_id": scene_id,
                    "scene_token": scene["scene_token"],
                    "scene_name": name,
                    "question": question,
                    "input_question": input_question,
                    "features": features[scene_id],
                    "metadata_policy": policy,
                    "metadata_frame_range": [first, last],
                    "generation_key": key,
                    "segmentation_frame": None,
                    "frame_binding_status": "requires_review",
                }
            )
    if {r["scene_token"] for r in requests} != set(mini_scenes):
        raise ValueError("mini request coverage is incomplete")
    path = args.output_dir / "requests.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": SCHEMA,
        "requests": len(requests),
        "unique_inputs": len({r["generation_key"] for r in requests}),
        "requests_sha256": file_sha256(path),
        "source_sha256": sources,
        "mini_metadata_sha256": {
            name: file_sha256(mini / (name + ".json")) for name in tables
        },
        "scene_metadata_sha256": file_sha256(metadata_path),
        "ego_motion_sha256": file_sha256(motion_path),
        "scenes": sorted(mini_scenes),
        "mini_splits": dict(Counter(r["mini_split"] for r in requests)),
        "groups": dict(Counter(r["source"]["group"] for r in requests)),
        "answer_frames": False,
        "reference_answers_used": False,
        "feature_encoder_checkpoint_verified": False,
        "duplicate_policy": "identical prompt and feature inputs reuse deterministic greedy output; all source rows retained",
    }
    write_json(args.output_dir / "manifest.json", metadata)
    event("prepared", **metadata)


@torch.no_grad()
def greedy_batch(model, tokenizer, prompts, features, max_new_tokens):
    if not prompts or len(prompts) != len(features):
        raise ValueError("one feature sequence per nonempty prompt batch is required")
    if len({len(f) for f in features}) != 1:
        raise ValueError(
            "batch must have equal visual lengths for cached attention alignment"
        )
    device = model.get_input_embeddings().weight.device
    rows = [prompt_ids(tokenizer, prompt)[0] for prompt in prompts]
    width = max(len(row) for row in rows)
    limits = [
        getattr(model.config, k, None)
        for k in ("max_position_embeddings", "tokenizer_model_max_length")
    ]
    if any(
        limit is not None and width - 1 + len(features[0]) + max_new_tokens > limit
        for limit in limits
    ):
        raise ValueError("generation exceeds the expanded multimodal context")
    ids = torch.full(
        (len(rows), width), tokenizer.eos_token_id, dtype=torch.long, device=device
    )
    attention = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        ids[index, -len(row) :] = row.to(device)
        attention[index, -len(row) :] = 1
    previous = getattr(model.config, "tokenizer_padding_side", None)
    model.config.tokenizer_padding_side = "left"
    try:
        # Expand visual tokens BEFORE generate builds cache_position. Padding
        # mixed text lengths through the legacy image-token generate path would
        # leave text-length cache positions beside longer visual embeddings.
        _, _, expanded_attention, _, embeds, _ = (
            model.prepare_inputs_labels_for_multimodal(
                ids,
                None,
                attention,
                None,
                None,
                torch.stack(features).to(
                    device=device, dtype=model.get_model().mm_projector.weight.dtype
                ),
            )
        )
        if embeds.shape[1] != width - 1 + len(features[0]):
            raise ValueError("visual expansion was truncated or changed unexpectedly")
        output = model.generate(
            inputs_embeds=embeds,
            attention_mask=expanded_attention,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    finally:
        if previous is None:
            delattr(model.config, "tokenizer_padding_side")
        else:
            model.config.tokenizer_padding_side = previous
    if output.shape[0] != len(rows) or output.shape[1] > max_new_tokens:
        raise ValueError("unexpected continuation-only generation shape")
    result = []
    for row in output.tolist():
        finished = tokenizer.eos_token_id in row
        if finished:
            row = row[: row.index(tokenizer.eos_token_id)]
        answer = tokenizer.decode(row, skip_special_tokens=False).strip()
        result.append(
            {
                "answer": answer,
                "finished": finished,
                "generated_token_ids": row,
                "status": "ok" if finished else "truncated",
                "max_new_tokens": max_new_tokens,
            }
        )
    return result


def gpu_snapshot():
    import subprocess

    values = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .splitlines()[0]
    )
    used, free, utilization = [int(v) for v in values.split(",")]
    return {"used_mib": used, "free_mib": free, "utilization": utilization}


def legacy_single(model, tokenizer, prompt, feature, budget):
    generated = LanguageBridge(model, tokenizer, None).generate(prompt, feature, budget)
    ids = generated["labels"][generated["labels"].ne(-100)].tolist()
    if tokenizer.eos_token_id in ids:
        ids = ids[: ids.index(tokenizer.eos_token_id)]
    return {
        "answer": generated["answer"],
        "finished": generated["finished"],
        "generated_token_ids": ids,
        "status": "ok" if generated["finished"] else "truncated",
        "max_new_tokens": budget,
    }


def load_collection_model(runtime, device, cpu_offload_layers):
    if not cpu_offload_layers:
        return load_b3(runtime, device)
    from types import SimpleNamespace

    from accelerate import dispatch_model
    from vtimellm.model.builder import load_pretrained_model

    if not device.startswith("cuda") or "chatglm" in runtime["model_base"].lower():
        raise ValueError("CPU layer offload currently requires CUDA Vicuna/Llama")
    tokenizer, model, _ = load_pretrained_model(
        SimpleNamespace(
            model_base=runtime["model_base"],
            pretrain_mm_mlp_adapter=runtime["pretrain_mm_mlp_adapter"],
        ),
        runtime.get("stage2"),
        runtime.get("stage3"),
    )
    count = len(model.get_model().layers)
    if not 0 < cpu_offload_layers < count:
        raise ValueError("offload layer count must leave at least one CUDA layer")
    # CPU stores FP16 weights. Each offloaded layer is copied to CUDA only
    # while executing, preserving the original computation precision.
    mapping = {
        f"model.layers.{i}": "cpu" if i >= count - cpu_offload_layers else device
        for i in range(count)
    }
    for name in ("model.embed_tokens", "model.norm", "model.mm_projector", "lm_head"):
        mapping[name] = device
    model.config.use_cache = False
    return tokenizer, dispatch_model(model, device_map=mapping)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.inputs / "manifest.json").read_text(encoding="utf-8"))
    request_path = args.inputs / "requests.jsonl"
    if file_sha256(request_path) != manifest["requests_sha256"]:
        raise ValueError("request manifest changed")
    requests = list(read_jsonl(request_path))
    runtime = json.loads(args.runtime.read_text(encoding="utf-8"))
    if runtime.get("language_device", args.device) != args.device:
        raise ValueError("runtime language_device must match the collection device")
    identity = source_identity(runtime, include_spatial=False)
    config = {
        "requests_sha256": manifest["requests_sha256"],
        "model_identity": identity,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "retry_tokens": args.retry_tokens,
        "cpu_offload_layers": args.cpu_offload_layers,
        "do_sample": False,
        "answer_frames": False,
    }
    config_path = args.output_dir / "generation-config.json"
    if (
        config_path.exists()
        and json.loads(config_path.read_text(encoding="utf-8")) != config
    ):
        raise ValueError("resume inputs/model/generation configuration changed")
    write_json(config_path, config)
    cache_path = args.output_dir / "unique-answers.jsonl"
    cache = (
        {r["generation_key"]: r for r in read_jsonl(cache_path)}
        if cache_path.exists()
        else {}
    )
    unique = {r["generation_key"]: r for r in requests}
    if not set(cache).issubset(unique):
        raise ValueError("resume cache has unrelated requests")
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    if args.device.startswith("cuda"):
        gpu = gpu_snapshot()
        required = 12500 if args.cpu_offload_layers else 14500
        if gpu["free_mib"] < required:
            raise RuntimeError(f"need {required} MiB GPU headroom before model load")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        event("gpu_before_load", **gpu)
    event("loading_b3", pending=len(unique) - len(cache), requests=len(requests))
    tokenizer, model = load_collection_model(
        runtime, args.device, args.cpu_offload_layers
    )
    model.eval()
    feature_cache = {}
    for row in unique.values():
        feature = row["features"]
        path = args.feature_root / feature["path"]
        if path.name not in feature_cache:
            if file_sha256(path) != feature["sha256"]:
                raise ValueError("SST input changed")
            values = np.load(path, allow_pickle=False)
            feature_cache[path.name] = torch.from_numpy(
                values.astype(np.float32, copy=True)
            )
    # Real checkpoint equivalence probe before enabling padded batch generation.
    probe = list(unique.values())
    pair = next(
        [a, b]
        for i, a in enumerate(probe)
        for b in probe[i + 1 :]
        if len(a["features"]["indices"]) == len(b["features"]["indices"])
        and a["input_question"] != b["input_question"]
    )
    prompts = [make_prompt(r["input_question"]) for r in pair]
    feats = [feature_cache[r["features"]["path"]] for r in pair]
    batched = greedy_batch(model, tokenizer, prompts, feats, 8)
    serial = [
        greedy_batch(model, tokenizer, [p], [f], 8)[0] for p, f in zip(prompts, feats)
    ]
    legacy = [legacy_single(model, tokenizer, p, f, 8) for p, f in zip(prompts, feats)]
    equivalent = [
        a["generated_token_ids"] == b["generated_token_ids"] == c["generated_token_ids"]
        for a, b, c in zip(batched, serial, legacy)
    ]
    write_json(
        args.output_dir / "batch-equivalence.json",
        {
            "passed": all(equivalent),
            "matches": equivalent,
            "probe_tokens": 8,
            "comparison": "padded batch vs expanded serial vs original B3 serial",
        },
    )
    batch_size = args.batch_size if all(equivalent) else 1

    def generate(prompts, features, budget):
        if not all(equivalent):
            return [
                legacy_single(model, tokenizer, p, f, budget)
                for p, f in zip(prompts, features)
            ]
        return greedy_batch(model, tokenizer, prompts, features, budget)

    groups = defaultdict(list)
    for key, row in unique.items():
        if key not in cache:
            groups[len(row["features"]["indices"])].append(row)
    started = time.monotonic()
    for group in groups.values():
        group.sort(key=lambda r: len(r["input_question"]))
        offset = 0
        while offset < len(group):
            batch = group[offset : offset + batch_size]
            if args.device.startswith("cuda") and gpu_snapshot()["free_mib"] < 1200:
                torch.cuda.empty_cache()
                if gpu_snapshot()["free_mib"] < 1200:
                    raise RuntimeError("GPU headroom below 1200 MiB; progress preserved")
            prompts = [make_prompt(r["input_question"]) for r in batch]
            feats = [feature_cache[r["features"]["path"]] for r in batch]
            try:
                generated = generate(prompts, feats, args.max_new_tokens)
                for i, value in enumerate(generated):
                    if (
                        not value["finished"]
                        and args.retry_tokens > args.max_new_tokens
                    ):
                        value = generate(
                            [prompts[i]],
                            [feats[i]],
                            args.retry_tokens,
                        )[0]
                        generated[i] = value
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    event("reduce_batch_after_oom", batch_size=batch_size)
                    continue
                raise
            with cache_path.open("a", encoding="utf-8") as handle:
                for row, value in zip(batch, generated):
                    result = dict(value, generation_key=row["generation_key"])
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    cache[row["generation_key"]] = result
                handle.flush()
            offset += len(batch)
            write_json(
                args.output_dir / "progress.json",
                {
                    "unique_completed": len(cache),
                    "unique_total": len(unique),
                    "source_rows_completed": sum(
                        r["generation_key"] in cache for r in requests
                    ),
                    "source_rows_total": len(requests),
                    "elapsed_seconds": time.monotonic() - started,
                    "batch_size": batch_size,
                },
            )
            event(
                "progress",
                completed=len(cache),
                total=len(unique),
                source_rows=sum(r["generation_key"] in cache for r in requests),
            )
            time.sleep(0.1)
    rows = []
    reused = set()
    for request in requests:
        key = request["generation_key"]
        record = dict(
            request,
            **{k: v for k, v in cache[key].items() if k != "generation_key"},
            schema_version=SCHEMA,
            model_identity=identity,
            answer_model_id=digest(identity),
            review_status="pending",
            eligible_for_binding_training=False,
            generation_reused=key in reused,
        )
        record["record_fingerprint"] = digest(record)
        rows.append(record)
        reused.add(key)
    path = args.output_dir / "answers.jsonl"
    part = path.with_suffix(".jsonl.part")
    part.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    part.replace(path)
    for split in ("mini_train", "mini_val"):
        (args.output_dir / (split + ".jsonl")).write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n"
                for r in rows
                if r["mini_split"] == split
            ),
            encoding="utf-8",
        )
    with (args.output_dir / "answers-preview.md").open("w", encoding="utf-8") as handle:
        handle.write("# B3 mini 自由回答（待审核）\n\n")
        for r in rows:
            handle.write(
                f"## {r['request_id']} / {r['scene_name']} / {r['mini_split']}\n\nQ: {r['question']}\n\nA: {r['answer']}\n\n状态：{r['status']}\n\n"
            )
    result = {
        "complete": True,
        "requests": len(rows),
        "unique_inputs": len(cache),
        "scene_count": len({r["scene_token"] for r in rows}),
        "mini_splits": dict(Counter(r["mini_split"] for r in rows)),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "answers_sha256": file_sha256(path),
        "answer_frames": False,
        "reference_answers_used": False,
        "all_answers_pending_review": True,
        "binding_training_ready": False,
        "model_identity": identity,
        "batch_equivalence": all(equivalent),
        "effective_batch_size": batch_size,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
        if args.device.startswith("cuda")
        else None,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20
        if args.device.startswith("cuda")
        else None,
    }
    write_json(args.output_dir / "completion.json", result)
    event("complete", **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--repo-root", type=Path, required=True)
    prep.add_argument("--nuscenes-root", type=Path, required=True)
    for command in (prep,):
        command.add_argument("--feature-root", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    execute = sub.add_parser("run")
    for name in ("runtime", "inputs", "feature-root", "output-dir"):
        execute.add_argument("--" + name, type=Path, required=True)
    execute.add_argument("--device", default="cuda:0")
    execute.add_argument("--batch-size", type=int, default=2)
    execute.add_argument("--memory-fraction", type=float, default=0.57)
    execute.add_argument("--cpu-offload-layers", type=int, default=0)
    execute.add_argument("--max-new-tokens", type=int, default=256)
    execute.add_argument("--retry-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
