"""
test_b4dl.py - B4DL end-to-end test / evaluation harness
========================================================
Loads the trained B4DL (VTimeLLM-based) model and its precomputed LiDAR
features, runs autoregressive inference over the B4DL *test* split across all
six tasks, writes per-task predictions, and finally computes the paper metrics
(paper Table 3): Accuracy, mIoU, BLEU-4, METEOR, ROUGE-L, BERTScore and,
optionally, the reference-free GPT-4o score.

This is the "verify model results" test code. It mirrors the inference path
used by mllm/vtimellm/eval/eval.py and mllm/vtimellm/inference.py, but targets
the B4DL 6-task benchmark rather than VTimeLLM's video grounding/captioning
benchmark.

Input test data format
----------------------
A single JSON file (--test_data) which can be either a list or a {id: item}
dict. Each QA item may look like:

    {
      "scene_id": "scene-0001",            # or "id": "scene-0001"
      "task": "temporal_understanding",     # one of the 6 tasks
      "conversations": [
        {"from": "human", "value": "<video>\\n<question>"},
        {"from": "gpt",   "value": "<ground truth answer>"}
      ]
      // optional: "meta": {...}, "start_index"/"end_index", "sequence_id"
    }

If `task` is missing it is read from a per-file directory name expected by the
build_test_split.py helper (--test_dir mode). If the human value lacks the
<video> placeholder, it is prepended automatically (the B4DL model needs the
<video> token to know where the LiDAR features are inserted).

Features
---------
Per-scene LiDAR features produced by
encoders/lidarclip/extract_pc_features.py
are loaded from `--feat_folder/<scene_id>.npy` as a [N_frames, 768] float16
tensor (see mllm/vtimellm/train/dataset.py).

Usage
-----
Run from the `mllm/` directory (so the `vtimellm` package + builder imports
resolve, exactly like eval.py):

    cd mllm
    python ../evaluation/test_b4dl.py \
        --model_base ./base_model/vicuna-v1-5-7b \
        --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
        --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
        --feat_folder ./lidarclip/stage2_features \
        --test_data ../data/test_qa.json \
        --output ../evaluation/predictions.json \
        --metrics_output ../evaluation/evaluation_results.json

    # with GPT scoring (optional, needs OPENAI_API_KEY):
        --use_gpt --gpt_api_key $OPENAI_API_KEY

    # quick smoke test on N samples:
        --max_samples 50

Prediction checkpointing
-------------------------
Predictions are incrementally saved to `<output>.ckpt` every
`--checkpoint_interval` samples (default 50) and after each task completes.
If the run is interrupted, re-run with the same `--output` to resume from
the last checkpoint. The checkpoint is automatically removed on clean exit.
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm

# Make `evaluation.*` and `vtimellm.*` importable regardless of cwd.
# Layout:
#   <repo_root>/                          e.g. /root/.../mmb4dl/
#       mllm/
#           evaluation/test_b4dl.py       <-- this file
#           vtimellm/
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))      # .../mllm/evaluation
_MLLM_DIR = os.path.dirname(_EVAL_DIR)                       # .../mllm
_REPO_ROOT = os.path.dirname(_MLLM_DIR)                      # .../ (actual repo root)
for _p in (_MLLM_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.evaluate_model import B4DLEvaluator       # noqa: E402

# VTimeLLM model / inference (must run from mllm/, as in eval.py)
from vtimellm.model.builder import load_pretrained_model  # noqa: E402
from vtimellm.inference import inference                  # noqa: E402
from vtimellm.utils import disable_torch_init             # noqa: E402

VIDEO_TOKEN = "<video>"          # mllm/vtimellm/constants.DEFAULT_IMAGE_TOKEN


# --------------------------------------------------------------------------
def load_test_items(path: str) -> list:
    """Load --test_data into a flat list of QA items (each has scene_id/task/conv)."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        items = list(data.values())
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unsupported test_data format: {type(data)}")
    # Normalize: accept both `scene_id` and `id` as the feature-file key.
    norm = []
    for it in items:
        scene_id = it.get("scene_id") or it.get("id")
        if scene_id is None or not it.get("conversations"):
            continue
        norm.append(it)
    return norm


def iter_items_by_task(items: list):
    """Yield (task, items_for_task) after canonicalizing task names."""
    by_task = {}
    for it in items:
        task = B4DLEvaluator.canonical_task(it.get("task") or "unknown")
        if task not in B4DLEvaluator.ALL_TASKS:
            # Unknown task - skip with a warning rather than silently mixing.
            continue
        by_task.setdefault(task, []).append(it)
    for task, g in by_task.items():
        yield task, g


def build_query(human_value: str,
                ego_meta: dict = None,
                scene_id: str = None,
                use_4dlidar: bool = True,
                use_meta: bool = True) -> str:
    """Ensure the B4DL/VTimeLLM <video> placeholder is present in the query.

    If ego_meta is provided, prepend the Metatoken prefix
    (<4DLiDAR> + <meta> + ego motion text) before <video>.
    """
    q = human_value

    # Inject metatoken prefix if ego metadata is available
    if ego_meta and scene_id and scene_id in ego_meta:
        meta_parts = []
        if use_4dlidar:
            meta_parts.append("<4DLiDAR>")
        if use_meta:
            motion_text = ego_meta[scene_id]
            meta_parts.append(f"<meta>\n{motion_text}")
        meta_parts.append(VIDEO_TOKEN)
        prefix = "\n".join(meta_parts)

        # Strip any existing metatoken/<video> tags from the original to avoid dupes
        for tag in [VIDEO_TOKEN, "<4DLiDAR>", "<meta>"]:
            q = q.replace(tag + "\n", "").replace(tag + " ", "").replace(tag, "")
        q = prefix + "\n" + q.strip()
    elif VIDEO_TOKEN not in q:
        q = f"{VIDEO_TOKEN}\n{q}"

    return q
    return human_value


def load_features(feat_folder: str, scene_id: str) -> torch.Tensor:
    """Load [N_frames, 768] LiDAR features for a scene as a float16 CPU tensor."""
    path = os.path.join(feat_folder, f"{scene_id}.npy")
    feat = np.load(path)           # cpu ndarray, [N, 768]
    return torch.from_numpy(feat).to(torch.float16)


def _save_checkpoint(results: dict, path: str):
    """Atomically save intermediate predictions so a crash won't lose all progress."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic on POSIX


def run_inference(model, tokenizer, features: torch.Tensor, query: str) -> str:
    """Run the B4DL autoregressive generation for one QA. Mirrors inference()."""
    return inference(model, features, query, tokenizer)


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="B4DL test/evaluation harness")
    parser.add_argument("--model_base", type=str, required=True,
                        help="Path to base LLM (e.g. ./base_model/vicuna-v1-5-7b)")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default=None,
                        help="Stage-1 mm_projector.bin path")
    parser.add_argument("--stage2", type=str, default=None,
                        help="Stage-2 checkpoint directory (LoRA)")
    parser.add_argument("--stage3", type=str, default=None,
                        help="Stage-3 checkpoint directory (LoRA, optional)")
    parser.add_argument("--feat_folder", type=str, required=True,
                        help="Precomputed LiDAR features dir (one <scene_id>.npy per scene)")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Single JSON file with the test QA set (see header)")
    parser.add_argument("--predictions_dir", type=str, default=None,
                        help="Directory of per-task generated_<task>_dataset_*.json "
                             "(alternative to --test_data; uses build_test_split to assemble)")
    parser.add_argument("--scene_metadata", type=str, default=None,
                        help="scene_metadata.json (used with --predictions_dir to "
                             "filter to the test split)")
    parser.add_argument("--nuscenes_root", type=str, default=None,
                        help="Path to nuScenes dataset root (used with "
                             "--predictions_dir + --scene_metadata for official val split)")
    parser.add_argument("--output", type=str, default="predictions.json",
                        help="Where to write per-task predictions JSON")
    parser.add_argument("--metrics_output", type=str, default="evaluation_results.json")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Cap number of samples per task (-1 = use all)")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                        help="Save intermediate predictions every N samples (default 50). "
                             "If interrupted, re-run with same --output to resume.")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--use_gpt", action="store_true",
                        help="Also compute the reference-free GPT-4o score")
    parser.add_argument("--gpt_api_key", type=str,
                        default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--gpt_model", type=str, default="gpt-4o")
    parser.add_argument("--ego_meta", type=str, default=None,
                        help="ego_metadata.json from generate_ego_metadata.py. "
                             "If provided, Metatoken prefix (<4DLiDAR> + <meta>) "
                             "is injected at inference time.")
    parser.add_argument("--no_4dlidar", action="store_true",
                        help="Omit <4DLiDAR> from metatoken prefix (ablation)")
    parser.add_argument("--no_meta", action="store_true",
                        help="Omit <meta> from metatoken prefix (ablation)")
    args = parser.parse_args()

    # Assemble the flat test item list
    if args.test_data is not None:
        items = load_test_items(args.test_data)
    elif args.predictions_dir is not None:
        from evaluation.build_test_split import assemble_test_qa
        items = assemble_test_qa(args.predictions_dir, args.scene_metadata,
                                 nuscenes_root=args.nuscenes_root)
    else:
        raise SystemExit("Provide either --test_data or --predictions_dir.")

    if not items:
        raise SystemExit("No test items loaded. Check --test_data / --predictions_dir.")

    print(f"Loaded {len(items)} test QA items across "
          f"{len({B4DLEvaluator.canonical_task(i.get('task')) for i in items})} tasks.")

    # Load ego metadata for Metatoken (optional)
    ego_meta: dict = None
    if args.ego_meta:
        if os.path.isfile(args.ego_meta):
            with open(args.ego_meta) as f:
                ego_meta = json.load(f)
            scenes_with = len(set(it.get("scene_id") or it.get("id") for it in items)
                              & set(ego_meta.keys()))
            print(f"Loaded ego metadata: {len(ego_meta)} scenes "
                  f"({scenes_with} matched to test items)")
        else:
            print(f"Warning: --ego_meta file not found: {args.ego_meta}")

    # Load model (same path as vtimellm/eval/eval.py)
    disable_torch_init()
    tokenizer, model, _ = load_pretrained_model(
        args, args.stage2, args.stage3)
    model = model.cuda()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model.to(dtype)

    # Optional GPU CLIP / video side is not needed - features are precomputed.
    # Cache features per scene_id to avoid reloading the same .npy repeatedly.
    feat_cache: dict = {}

    # ── Resume from checkpoint if available ──
    results: dict = {}     # canonical_task -> {predictions, ground_truths, questions}
    checkpoint_path = args.output + ".ckpt"
    completed_ids: set = set()  # (task, sample_index) already done
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            saved = json.load(f)
        results = saved.get("results", {})
        for task_data in results.values():
            for i in range(len(task_data.get("predictions", []))):
                completed_ids.add((task_data.get("_task", ""), i))
        print(f"  Loaded {sum(len(v.get('predictions', [])) for v in results.values())} "
              f"completed predictions across {len(results)} tasks.")

    skipped = 0
    checkpoint_interval = getattr(args, 'checkpoint_interval', 50)
    for task, group in iter_items_by_task(items):
        # Determine starting point for this task
        if task in results:
            done_count = len(results[task].get("predictions", []))
            if done_count >= len(group):
                print(f"\n[{task}] {len(group)} samples — already complete, skipping")
                continue
            print(f"\n[{task}] resuming from {done_count}/{len(group)} samples")
            preds = results[task]["predictions"]
            gts = results[task]["ground_truths"]
            qs = results[task]["questions"]
            start_idx = done_count
        else:
            preds, gts, qs = [], [], []
            start_idx = 0

        if args.max_samples and args.max_samples > 0:
            group = group[:args.max_samples]

        pbar = tqdm(group[start_idx:], desc=task, initial=start_idx, total=len(group))
        task_start_time = time.time()
        for i, it in enumerate(pbar):
            sample_idx = start_idx + i
            scene_id = it.get("scene_id") or it.get("id")
            try:
                feat = feat_cache.get(scene_id)
                if feat is None:
                    feat = load_features(args.feat_folder, scene_id).cuda()
                    feat_cache[scene_id] = feat
            except Exception as e:
                skipped += 1
                print(f"  ! missing feature for {scene_id}: {e}")
                continue

            question = it["conversations"][0]["value"]
            gt = it["conversations"][1]["value"]
            query = build_query(
                question,
                ego_meta=ego_meta,
                scene_id=scene_id,
                use_4dlidar=not getattr(args, 'no_4dlidar', False),
                use_meta=not getattr(args, 'no_meta', False),
            )
            try:
                pred = run_inference(model, tokenizer, feat, query)
            except Exception as e:
                skipped += 1
                print(f"  ! inference failed for {scene_id}: {e}")
                continue
            preds.append(pred)
            gts.append(gt)
            qs.append(query.replace(VIDEO_TOKEN, "").strip())

            # Periodic checkpoint save
            if (sample_idx + 1) % checkpoint_interval == 0:
                results[task] = {"predictions": preds, "ground_truths": gts,
                                 "questions": qs, "_task": task}
                _save_checkpoint(results, checkpoint_path)
                elapsed = time.time() - task_start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(group) - sample_idx - 1) / rate if rate > 0 else 0
                pbar.set_postfix({"ckpt": "✓", "rate": f"{rate:.1f}/s",
                                  "eta": f"{eta/60:.0f}m"})

        results[task] = {"predictions": preds, "ground_truths": gts,
                         "questions": qs, "_task": task}
        # Save after each task completes
        _save_checkpoint(results, checkpoint_path)

    if skipped:
        print(f"\nSkipped {skipped} items (missing features or inference errors).")

    # Save final predictions (clean version without _task metadata)
    clean_results = {k: {"predictions": v["predictions"],
                         "ground_truths": v["ground_truths"],
                         "questions": v["questions"]}
                     for k, v in results.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    print(f"\nPredictions written to {args.output}")

    # Remove checkpoint on successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint {checkpoint_path} removed (run completed).")

    # Compute metrics
    evaluator = B4DLEvaluator(use_gpt=args.use_gpt, gpt_api_key=args.gpt_api_key,
                              gpt_model=args.gpt_model)
    per_task = evaluator.evaluate_all(results)
    final = evaluator.compute_final_score(per_task)

    print("\n" + "=" * 50)
    print("Final B4DL scores")
    print("=" * 50)
    for k, v in final.items():
        print(f"{k:12s}: {v:.4f}")

    evaluator.save_results(per_task, final, args.metrics_output)

    # Compare against paper Table 3 (B4DL model)
    print("\n" + "=" * 50)
    print("Paper reference (B4DL, Table 3)")
    print("=" * 50)
    paper = {'accuracy': 0.762, 'miou': 0.311, 'bleu4': 0.095,
             'rouge_l': 0.322, 'meteor': 0.275, 'bertscore': 0.897,
             'gpt_score': 59.513}
    print(f"{'metric':12s} {'paper':<10} {'ours':<10}")
    for k, v in paper.items():
        got = final.get(k)
        print(f"{k:12s} {v:<10.4f} "
              f"{got:<10}" + ("(skipped)" if got is None else f"{got:.4f}"))


if __name__ == "__main__":
    main()