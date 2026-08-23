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
ego_metadata.json 格式（per-sequence + per-scene fallback）：
    {
      "scene_id": {"first_frame": "...", "last_frame": "..."},
      "scene_id_0_8": {"first_frame": "...", "last_frame": "..."},
      ...
    }

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
import re
import argparse
from typing import Dict, Optional, List, Tuple


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
    p.add_argument("--frame_ctx", action="store_true",
                   help="Prepend frame-range context to the <meta> line for "
                        "questions that reference frame numbers, matching "
                        "test_b4dl.py --per_sequence: "
                        "\"<meta> This sequence covers frames XXX to XXX. ...\"")
    return p.parse_args()


def parse_frame_numbers(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract the first and last frame numbers from a question text.

    Looks for patterns like "frame 030", "frame 6", "frame 008".
    Returns (first_frame, last_frame) as integers, or (None, None) if
    no frame numbers are found.
    """
    matches = re.findall(r'frame\s+(\d+)', text, re.IGNORECASE)
    if not matches:
        return None, None
    nums = [int(m) for m in matches]
    return min(nums), max(nums)


def lookup_ego_for_qa(ego_meta: dict, scene_id: str,
                      question: str) -> Dict[str, str]:
    """Look up per-sequence ego data using frame numbers from the question.

    Tries per-sequence key f"{scene_id}_{first}_{last}" first (paper Appendix C:
    "the metatoken descriptions of the first and last frames referenced in the
    QA pair are concatenated"), then falls back to per-scene key scene_id.
    """
    first_frame, last_frame = parse_frame_numbers(question)

    if first_frame is not None and last_frame is not None:
        seq_key = f"{scene_id}_{first_frame}_{last_frame}"
        if seq_key in ego_meta:
            return ego_meta[seq_key]
        # Try reversed order (some questions mention larger number first)
        seq_key_rev = f"{scene_id}_{last_frame}_{first_frame}"
        if seq_key_rev in ego_meta:
            return ego_meta[seq_key_rev]

    # Fallback: per-scene
    return ego_meta.get(scene_id, {})


def inject_metatoken(items: list,
                     ego_meta: dict,
                     include_4dlidar: bool = True,
                     include_meta: bool = True,
                     frame_ctx: bool = False) -> list:
    """Inject Metatoken prefix into the first human message of each QA item.

    The format follows paper Figure 6 / Appendix C:
      <4DLiDAR> <video> <question>
      <meta> The metadata of the first frame is '...' and the metadata of the last frame is '...'

    The <video> token is kept as the embedding placeholder (VTimeLLM convention).
    The metatoken block (<meta> + frame descriptions) comes after the question,
    matching the paper's Figure 6 layout.

    Per-sequence ego data is looked up by parsing frame numbers from the
    question text (e.g. "frame 30 and frame 38" → key "scene_id_30_38").
    Falls back to per-scene ego data if no frame numbers are found.
    """
    modified = []
    no_meta_count = 0
    seq_matched = 0
    scene_fallback = 0

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
            scene_data = lookup_ego_for_qa(ego_meta, scene_id, cleaned)
            # Track lookup type
            first_frame, last_frame = parse_frame_numbers(cleaned)
            if first_frame is not None:
                seq_key = f"{scene_id}_{first_frame}_{last_frame}"
                if seq_key in ego_meta or f"{scene_id}_{last_frame}_{first_frame}" in ego_meta:
                    seq_matched += 1
                else:
                    scene_fallback += 1
            else:
                scene_fallback += 1

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

            # Frame-range context prefix (matches test_b4dl.py --per_sequence):
            # tells the model which absolute frame indices the sliced feature
            # tensor corresponds to. Only for questions with frame numbers.
            if frame_ctx:
                fc_first, fc_last = parse_frame_numbers(cleaned)
                if fc_first is not None and fc_last is not None:
                    meta_line = (
                        f"<meta> This sequence covers frames {fc_first:03d} to "
                        f"{fc_last:03d}. {meta_line[len('<meta> '):]}"
                    )
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
    print(f"  Per-sequence match: {seq_matched}, per-scene fallback: {scene_fallback}")

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
        frame_ctx=args.frame_ctx,
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
