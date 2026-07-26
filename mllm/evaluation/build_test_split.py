"""
build_test_split.py - assemble the B4DL test-set QA file
========================================================
Aggregates the per-task generated test datasets written by
datageneration/generate_dataset.py
    (data/generated_dataset/<task>/generated_<task>_dataset_{i}_{j}.json)
into a single flat list suitable for evaluation/test_b4dl.py, while keeping
only the items whose scene belongs to the B4DL *test* split (the 150 nuScenes
val scenes according to the official nuScenes devkit split).

Two ways to determine the test split:
  1. --nuscenes_root <path> (recommended): uses nuscenes.utils.splits.val
     to get the official 150 val scene names, then cross-references with
     --scene_metadata to obtain the corresponding scene_ids.
  2. --scene_metadata <path> only (legacy): reads the "split" field from
     the metadata JSON and keeps scenes with split == "test".

Output: a list of QA items:
    {scene_id, scene_token, sequence_id, start_index, end_index,
     task, conversations:[{from:"human", value:"<video>\\n<q>"},{from:"gpt",value:<a>}]}

Usage (CLI):
    python build_test_split.py \\
        --predictions_dir ../data/generated_dataset \\
        --nuscenes_root /path/to/nuScenes \\
        --scene_metadata ../data/metadata/scene_metadata.json \\
        --output ../data/test_qa.json

Usage (imported by test_b4dl.py):
    from evaluation.build_test_split import assemble_test_qa
    items = assemble_test_qa(predictions_dir, scene_metadata_path,
                             nuscenes_root="/path/to/nuScenes")
"""

import os
import re
import glob
import json
import argparse
from typing import Optional, List, Dict, Set

# Map file-name task tokens -> canonical B4DL task keys.
TASK_FROM_DIR = {
    "existence": "existence",
    "binary": "binary_qa",
    "time_grounding": "time_grounding",
    "description": "description",
    "temporal": "temporal_understanding",
    "temporal_understanding": "temporal_understanding",
    "comprehensive": "comprehensive_reasoning",
    "comprehensive_reasoning": "comprehensive_reasoning",
}

VIDEO_TOKEN = "<video>"


def _load_test_scene_ids(scene_metadata_path: Optional[str],
                         nuscenes_root: Optional[str] = None) -> Optional[Set[str]]:
    """Return the set of scene_ids belonging to the B4DL test split.

    Priority:
      1. If nuscenes_root + scene_metadata_path are both given, use the
         official nuScenes val split (150 scenes) and cross-reference with
         scene_metadata to get scene_ids.
      2. If only scene_metadata_path is given, read the "split" field
         (split == "test") from the metadata.
      3. If neither is given, return None (keep all items).
    """
    # --- Method 1: nuScenes devkit official split ------------------------
    if nuscenes_root and scene_metadata_path and os.path.isfile(scene_metadata_path):
        try:
            from nuscenes.utils.splits import val as val_scene_names
        except ImportError:
            print("Warning: nuscenes-devkit not installed. "
                  "Falling back to metadata split field.")
        else:
            nusc_scene_json = os.path.join(nuscenes_root, 'v1.0-trainval', 'scene.json')
            if not os.path.isfile(nusc_scene_json):
                print(f"Warning: {nusc_scene_json} not found. "
                      f"Falling back to metadata split field.")
            else:
                with open(nusc_scene_json) as f:
                    nusc_scenes = json.load(f)
                name_to_token = {s['name']: s['token'] for s in nusc_scenes}
                val_tokens = {name_to_token[n] for n in val_scene_names
                              if n in name_to_token}

                with open(scene_metadata_path) as f:
                    meta_scenes = json.load(f)
                token_to_scene_id = {s['scene_token']: s['scene_id']
                                     for s in meta_scenes}

                test_ids = {token_to_scene_id[t] for t in val_tokens
                            if t in token_to_scene_id}
                print(f"nuScenes val scenes: {len(val_scene_names)}, "
                      f"matched to scene_ids: {len(test_ids)}")
                return test_ids

    # --- Method 2: metadata split field ----------------------------------
    if scene_metadata_path and os.path.isfile(scene_metadata_path):
        with open(scene_metadata_path) as f:
            scenes = json.load(f)
        test_ids = {s["scene_id"] for s in scenes if s.get("split") == "test"}
        if test_ids:
            print(f"Metadata split='test' scenes: {len(test_ids)}")
            return test_ids
        else:
            print("Warning: scene_metadata has 0 scenes with split=='test'. "
                  "Use --nuscenes_root to get the official split.")

    return None


def _task_from_path(path: str, dir_task_map: Dict[str, str]) -> Optional[str]:
    parent = os.path.basename(os.path.dirname(path))
    token = TASK_FROM_DIR.get(parent.lower())
    if token:
        return token
    # fall back to parsing the file name: generate_<task>_dataset_*.json
    m = re.match(r"generated_(.*?)_dataset_", os.path.basename(path))
    if m:
        return TASK_FROM_DIR.get(m.group(1).lower())
    return None


def _normalize_item(item: Dict, task: str) -> Dict:
    """Adapt a generate_dataset item to the test_b4dl.py schema."""
    conversations = item.get("conversations", [])
    if not conversations:
        return {}
    scene_id = item.get("scene_id") or item.get("id")
    if scene_id is None:
        return {}

    # First human turn may or may not contain the <video> placeholder.
    # Add it so the B4DL model knows where the LiDAR features are injected.
    first = conversations[0].get("value", "")
    if VIDEO_TOKEN not in first:
        conversations = [{**conversations[0],
                          "value": f"{VIDEO_TOKEN}\n{first}"}] + conversations[1:]

    return {
        "scene_id": scene_id,
        "scene_token": item.get("scene_token"),
        "sequence_id": item.get("sequence_id"),
        "start_index": item.get("start_index"),
        "end_index": item.get("end_index"),
        "task": task,
        "conversations": conversations,
    }


def assemble_test_qa(predictions_dir: str,
                      scene_metadata_path: Optional[str] = None,
                      nuscenes_root: Optional[str] = None) -> List[Dict]:
    """Walk predictions_dir for generated_*_dataset_*.json, tag tasks, and
    optionally filter to the test split.

    When *nuscenes_root* is given together with *scene_metadata_path*, the
    official nuScenes val split (150 scenes) is used to determine test scene_ids.
    Otherwise falls back to the metadata "split" field, or keeps all items."""
    test_ids = _load_test_scene_ids(scene_metadata_path, nuscenes_root)
    if test_ids is None:
        print("Warning: no usable scene_metadata split filter - keeping all items.")

    files = glob.glob(os.path.join(predictions_dir, "**", "generated_*_dataset_*.json"),
                      recursive=True)
    print(f"Found {len(files)} per-task dataset files under {predictions_dir}")

    out: List[Dict] = []
    seen = test_ids is None  # if no filter, accept everything
    counts: Dict[str, int] = {}
    for path in sorted(files):
        task = _task_from_path(path, {})
        if task is None:
            print(f"  skipping (unknown task) {path}")
            continue
        with open(path) as f:
            items = json.load(f)
        kept = 0
        for it in items:
            scene_id = it.get("scene_id") or it.get("id")
            if test_ids is not None and scene_id not in test_ids:
                continue
            norm = _normalize_item(it, task)
            if norm:
                out.append(norm)
                kept += 1
        if kept:
            counts[task] = counts.get(task, 0) + kept
            print(f"  [{task:24s}] {kept:6d} items  <- {os.path.basename(path)}")
    print(f"\nTotal kept QA items: {len(out)}  (per task: {counts})")
    return out


def main():
    p = argparse.ArgumentParser(description="Build the flat B4DL test QA JSON")
    p.add_argument("--predictions_dir", type=str, required=True,
                   help="dir containing generated_<task>_dataset_*.json "
                        "(e.g. data/generated_dataset/<task>/)")
    p.add_argument("--scene_metadata", type=str, default=None,
                   help="scene_metadata.json (used for scene_token->scene_id "
                        "mapping with --nuscenes_root, or split=='test' filter)")
    p.add_argument("--nuscenes_root", type=str, default=None,
                   help="path to nuScenes dataset root (uses official val split "
                        "via nuscenes.utils.splits; must be combined with "
                        "--scene_metadata)")
    p.add_argument("--output", type=str, default="test_qa.json")
    args = p.parse_args()

    items = assemble_test_qa(args.predictions_dir, args.scene_metadata,
                             nuscenes_root=args.nuscenes_root)
    with open(args.output, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"\nWrote test QA set: {args.output}")


if __name__ == "__main__":
    main()