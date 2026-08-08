#!/usr/bin/env python3
"""
inject_metatoken.py — 将 Metatoken 前缀注入训练/测试数据
============================================================
根据论文 §4.1 / Appendix C / Figure 6，在每条 QA 的 human question 前面拼接：

    <4DLiDAR>
    <meta>
    The metadata of the first frame is '<ego state at first frame>'
    and the metadata of the last frame is '<ego motion to last frame>'
    <video>
    <original question>

需要先运行 generate_ego_metadata.py 生成 ego_metadata.json。

用法：
    cd mllm

    # 注入训练集
    python scripts/inject_metatoken.py \
        --input ./b4dl_dataset/stage2_train.json \
        --ego_meta ./b4dl_dataset/ego_metadata.json \
        --output ./b4dl_dataset/stage2_train_meta.json

    # 注入测试集
    python scripts/inject_metatoken.py \
        --input ./b4dl_dataset/test_qa.json \
        --ego_meta ./b4dl_dataset/ego_metadata.json \
        --output ./b4dl_dataset/test_qa_meta.json

    # 注入全量 stage2（训练用）
    python scripts/inject_metatoken.py \
        --input ./b4dl_dataset/stage2_conversations.json \
        --ego_meta ./b4dl_dataset/ego_metadata.json \
        --output ./b4dl_dataset/stage2_conversations_meta.json
"""

import os
import sys
import json
import argparse
from typing import Dict, Optional


def parse_args():
    p = argparse.ArgumentParser(
        description="Inject B4DL Metatoken prefix into QA data")
    p.add_argument("--input", type=str, required=True,
                   help="Input JSON (conversations format, list of items with scene_id/task/conversations)")
    p.add_argument("--ego_meta", type=str, required=True,
                   help="ego_metadata.json from generate_ego_metadata.py")
    p.add_argument("--output", type=str, required=True,
                   help="Output JSON path")
    p.add_argument("--no_4dlidar", action="store_true",
                   help="Omit <4DLiDAR> prefix (for ablation: Metatoken only)")
    p.add_argument("--no_meta", action="store_true",
                   help="Omit <meta> + ego text (for ablation: <4DLiDAR> only)")
    return p.parse_args()


def build_metatoken_prefix(scene_id: str,
                           ego_meta: Dict[str, str],
                           include_4dlidar: bool = True,
                           include_meta: bool = True) -> str:
    """Build the full Metatoken prefix string for a given scene.

    Returns the text to prepend before the original question, e.g.:
        "<4DLiDAR>\n<meta>\nThe metadata of the first frame is '...'
         and the metadata of the last frame is '...'\n<video>\n"
    """
    parts = []

    if include_4dlidar:
        parts.append("<4DLiDAR>")

    if include_meta:
        motion_text = ego_meta.get(scene_id, "")
        if motion_text:
            parts.append(f"<meta>\n{motion_text}")
        else:
            # Fallback: minimal metatoken without ego data
            parts.append("<meta>\nNo ego motion metadata available for this scene.")

    # <video> placeholder is already in the original question,
    # but we ensure it appears after the metatoken block
    parts.append("<video>")

    return "\n".join(parts)


def inject_metatoken(items: list,
                     ego_meta: Dict[str, str],
                     include_4dlidar: bool = True,
                     include_meta: bool = True) -> list:
    """Inject Metatoken prefix into the first human message of each QA item."""
    modified = []
    skipped_no_meta = 0

    for item in items:
        scene_id = item.get("scene_id") or item.get("id")
        if not scene_id:
            modified.append(item)
            continue

        prefix = build_metatoken_prefix(
            scene_id, ego_meta, include_4dlidar, include_meta)

        # Inject into the first human conversation turn
        conversations = item.get("conversations", [])
        if conversations and conversations[0].get("from") == "human":
            original_value = conversations[0]["value"]

            # Remove existing <video> or <4DLiDAR> or <meta> prefixes
            # to avoid duplication
            cleaned = original_value
            for tag in ["<video>", "<4DLiDAR>", "<meta>"]:
                cleaned = cleaned.replace(tag + "\n", "").replace(tag + " ", "").replace(tag, "")
            cleaned = cleaned.strip()

            # Assemble: metatoken prefix + cleaned question
            new_value = prefix + "\n" + cleaned
            conversations = [
                {"from": "human", "value": new_value}
            ] + conversations[1:]

        item = dict(item)
        item["conversations"] = conversations
        modified.append(item)

    if skipped_no_meta:
        print(f"  ⚠ {skipped_no_meta} items had no ego metadata; used fallback text.")

    return modified


def main():
    args = parse_args()

    # Load inputs
    print(f"Loading QA data: {args.input}")
    with open(args.input) as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = list(data.values())
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unsupported input format: {type(data)}")
    print(f"  {len(items)} QA items loaded")

    print(f"Loading ego metadata: {args.ego_meta}")
    with open(args.ego_meta) as f:
        ego_meta = json.load(f)
    print(f"  {len(ego_meta)} scene entries")

    # Count coverage
    scenes_in_data = set(
        item.get("scene_id") or item.get("id") for item in items)
    scenes_with_meta = scenes_in_data & set(ego_meta.keys())
    print(f"  Scene coverage: {len(scenes_with_meta)}/{len(scenes_in_data)} "
          f"({100*len(scenes_with_meta)/max(1,len(scenes_in_data)):.0f}%)")

    # Inject
    ablated = []
    if args.no_4dlidar:
        ablated.append("<4DLiDAR>")
    if args.no_meta:
        ablated.append("<meta>")
    ablation_tag = " (ablated: " + ", ".join(ablated) + ")" if ablated else ""

    print(f"Injecting Metatoken prefix{ablation_tag} ...")
    modified = inject_metatoken(
        items, ego_meta,
        include_4dlidar=not args.no_4dlidar,
        include_meta=not args.no_meta,
    )

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(modified, f, ensure_ascii=False)
    print(f"Saved {len(modified)} items to {args.output}")

    # Show example
    if modified:
        ex = modified[0]
        q = ex["conversations"][0]["value"]
        print(f"\nExample output (scene_id={ex.get('scene_id', '?')}):")
        print(f"  {q[:200]}...")


if __name__ == "__main__":
    main()
