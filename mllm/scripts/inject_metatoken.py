#!/usr/bin/env python3
"""
inject_metatoken.py — 将 Metatoken 前缀注入训练/测试数据
============================================================
根据论文 §4.1 / Appendix C / Figure 6，在每条 QA 的 human question 中
拼接 Metatoken 前缀。论文格式（Figure 6）：

    <4DLiDAR> <video> <question>
    <meta> The metadata of the first frame is '<ego state at first frame>'
    and the metadata of the last frame is '<ego state at last frame>'

其中 "The metadata of the first frame is" 和 "and the metadata of the last
frame is" 为 Figure 6 中红色高亮的连接词。

需要先运行 generate_ego_metadata.py 生成 ego_metadata.json。
ego_metadata.json 格式：
    {scene_id: {"first_frame": "...", "last_frame": "..."}}

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

    # 消融：仅 <4DLiDAR> 无 <meta>
    python scripts/inject_metatoken.py ... --no_meta
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
                   help="Input JSON (conversations format)")
    p.add_argument("--ego_meta", type=str, required=True,
                   help="ego_metadata.json from generate_ego_metadata.py")
    p.add_argument("--output", type=str, required=True,
                   help="Output JSON path")
    p.add_argument("--no_4dlidar", action="store_true",
                   help="Omit <4DLiDAR> prefix (for ablation)")
    p.add_argument("--no_meta", action="store_true",
                   help="Omit <meta> + ego text (for ablation)")
    return p.parse_args()


def inject_metatoken(items: list,
                     ego_meta: dict,
                     include_4dlidar: bool = True,
                     include_meta: bool = True) -> list:
    """Inject Metatoken prefix into the first human message of each QA item.

    The format follows paper Figure 6 / Appendix C:
      <4DLiDAR> <video> <question>
      <meta> The metadata of the first frame is '...' and the metadata of the last frame is '...'

    The <video> token is kept as the embedding placeholder (VTimeLLM convention).
    The metatoken block (<meta> + frame descriptions) comes after the question,
    matching the paper's Figure 6 layout.
    """
    modified = []
    no_meta_count = 0

    for item in items:
        scene_id = item.get("scene_id") or item.get("id")
        if not scene_id:
            modified.append(item)
            continue

        conversations = item.get("conversations", [])
        if not conversations or conversations[0].get("from") != "human":
            modified.append(item)
            continue

        original_value = conversations[0]["value"]

        # Remove the entire <meta> block first (tag + all following text on
        # the same line and subsequent lines) so orphaned descriptions don't
        # survive in the cleaned question.
        cleaned = original_value
        if "<meta>" in cleaned:
            cleaned = cleaned.split("<meta>")[0]
        # Then strip remaining structural tags
        for tag in ["<video>", "<4DLiDAR>"]:
            cleaned = cleaned.replace(tag + "\n", "").replace(tag + " ", "").replace(tag, "")
        cleaned = cleaned.strip()

        # Build the new human message:
        # <4DLiDAR>\n<video>\n<question>\n<meta> The metadata of ...
        prefix_parts = []
        if include_4dlidar:
            prefix_parts.append("<4DLiDAR>")
        prefix_parts.append("<video>")
        prefix_parts.append(cleaned)  # the actual question

        if include_meta:
            scene_data = ego_meta.get(scene_id, {})
            if isinstance(scene_data, dict):
                first_text = scene_data.get("first_frame", "")
                last_text = scene_data.get("last_frame", "")
            else:
                first_text = scene_data
                last_text = ""

            if first_text and last_text:
                meta_line = (
                    f"<meta> The metadata of the first frame is '{first_text}' "
                    f"and the metadata of the last frame is '{last_text}'"
                )
            elif first_text:
                meta_line = f"<meta> The metadata of the first frame is '{first_text}'"
            else:
                meta_line = "<meta> No ego motion metadata available for this scene."
                no_meta_count += 1
            prefix_parts.append(meta_line)

        new_value = "\n".join(prefix_parts)
        new_conversations = [
            {"from": "human", "value": new_value}
        ] + conversations[1:]

        item = dict(item)
        item["conversations"] = new_conversations
        modified.append(item)

    if no_meta_count:
        print(f"  ⚠ {no_meta_count} items had no ego metadata; used fallback text.")

    return modified


def main():
    args = parse_args()

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
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(modified, f, ensure_ascii=False)
    print(f"Saved {len(modified)} items to {args.output}")

    # Show example
    if modified:
        ex = modified[0]
        q = ex["conversations"][0]["value"]
        print(f"\nExample output (scene_id={ex.get('scene_id', '?')}):")
        print(f"  {q[:300]}...")


if __name__ == "__main__":
    main()
