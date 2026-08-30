"""Build the Stage-1 (3D LiDAR understanding) training set from LiDAR-LLM-Nu-Caption.

Auxiliary / reference script. Stage-1 uses the external
`Senqiao/LiDAR-LLM-Nu-Caption` dataset. This converts it into the conversation
format and keeps only the samples whose scene belongs to the B4DL training split
(the same 699 scenes used by stage2/stage3), so the 3D and 4D stages are trained
on exactly the same scenes (no test leakage).

A bundled mapping table (`assets/`) is used, so nuScenes is not required:
- `assets/sample_token_to_scene.json` : sample_token -> scene_token
- `assets/train_scene_tokens.json`    : the 699 training scene_tokens

Usage:
  # auto-download from Hugging Face
  python tools/build_stage1_from_lidarllm.py --output ./stage1_train.json
  # or use an already-downloaded copy (a JSON list, or a path load_dataset accepts)
  python tools/build_stage1_from_lidarllm.py --input /path/to/lidarllm --output ./stage1_train.json
"""

import os
import json
import argparse

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
# 键名与生产文件 mllm/b4dl_dataset/stage1_train.json 逐字段一致：dataset.py 按
# item["scene_id"] 加载特征（{feat_folder}/{scene_id}.npy，stage1 特征按
# sample_token 键控），此前只写 "id" 会触发 KeyError→随机样本兜底（审查 L2）。
# <image> 由 dataset.py 归一为 <video>，与生产文件保持一致。
MODALITY_TOKEN = "<image>"
HF_DATASET = "Senqiao/LiDAR-LLM-Nu-Caption"


def load_rows(input_path):
    """Return an iterable of dict rows with keys: sample_token, question, answer_lidar."""
    if input_path and input_path.endswith(".json"):
        return json.load(open(input_path))
    from datasets import load_dataset
    name = input_path if input_path else HF_DATASET
    return load_dataset(name, split="train")


def to_conversation(row, scene_token, stats):
    answer = row.get("answer_lidar") or row.get("answer") or ""
    if not answer:
        stats["empty_answer"] += 1
    return {
        "scene_id": row["sample_token"],
        "scene_token": scene_token,
        "conversations": [
            {"from": "human", "value": f"{MODALITY_TOKEN}\n{row['question']}"},
            {"from": "gpt", "value": answer},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Build Stage-1 data from LiDAR-LLM-Nu-Caption (filtered to B4DL train scenes)")
    parser.add_argument("--input", type=str, default=None, help="Local LiDAR-LLM copy (.json list, or a path load_dataset accepts). Omit to download from Hugging Face.")
    parser.add_argument("--output", type=str, default="./stage1_train.json", help="Output path")
    args = parser.parse_args()

    samp2scene = json.load(open(os.path.join(ASSETS, "sample_token_to_scene.json")))
    train_scenes = set(json.load(open(os.path.join(ASSETS, "train_scene_tokens.json"))))

    rows = load_rows(args.input)
    out, kept_scenes, dropped = [], set(), 0
    stats = {"empty_answer": 0}
    for row in rows:
        scene = samp2scene.get(row["sample_token"])
        if scene in train_scenes:
            out.append(to_conversation(row, scene, stats))
            kept_scenes.add(scene)
        else:
            dropped += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json.dump(out, open(args.output, "w"))
    print(f"Saved {len(out)} samples over {len(kept_scenes)} train scenes to {args.output} "
          f"(dropped {dropped} non-train-scene samples, empty_answer {stats['empty_answer']})")


if __name__ == "__main__":
    main()
