#!/usr/bin/env python3
"""Create train/val/test scene-level split for B4DL datasets."""
import json
import random
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2_data", default="b4dl_dataset/stage2_conversations.json")
    parser.add_argument("--stage3_data", default="b4dl_dataset/stage3_conversations.json")
    parser.add_argument("--output_dir", default="b4dl_dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    s2 = json.load(open(args.stage2_data))
    s3 = json.load(open(args.stage3_data))

    scenes = sorted(set(d['scene_id'] for d in s2) & set(d['scene_id'] for d in s3))
    random.shuffle(scenes)

    n = len(scenes)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_scenes = set(scenes[:n_train])
    val_scenes = set(scenes[n_train:n_train + n_val])
    test_scenes = set(scenes[n_train + n_val:])

    split = {"train": sorted(train_scenes), "val": sorted(val_scenes), "test": sorted(test_scenes)}
    with open(f"{args.output_dir}/split_scenes.json", "w") as f:
        json.dump(split, f, indent=2)
    print(f"Split saved: {args.output_dir}/split_scenes.json")

    for name, data in [("stage2", s2), ("stage3", s3)]:
        for split_name, scene_set in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
            filtered = [d for d in data if d['scene_id'] in scene_set]
            out_path = f"{args.output_dir}/{name}_{split_name}.json"
            with open(out_path, "w") as f:
                json.dump(filtered, f)
            print(f"  {out_path}: {len(filtered)} items")

    print(f"\nScene counts: train={len(train_scenes)}, val={len(val_scenes)}, test={len(test_scenes)}")


if __name__ == "__main__":
    main()
