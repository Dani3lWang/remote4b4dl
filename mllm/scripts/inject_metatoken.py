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

需要先运行 generate_ego_metadata.py 生成 ego_metadata.json（+ 可选
ego_frame_motion.json）。

论文 Appendix C："the metatoken descriptions of the first and last frames
referenced in the QA pair are concatenated" —— metatoken 描述的是 QA 文本
所引用的首/末帧。两种实现模式：

【legacy 模式】（仅 --ego_meta）
    按问题帧号查 ego_metadata.json 的 per-sequence 键
    f"{scene_id}_{first}_{last}"，只有问题引用恰好等于序列边界时命中
    （训练集约 46%），否则回退 per-scene 条目。

【v2 模式】（--frame_motion ego_frame_motion.json）
    对问题引用的任意 (first, last) 帧对，用逐帧运动表现场渲染真实 ego
    描述（含单帧引用），彻底消除 per-scene 回退错误。

【feat_range / feat_indices】（--sequence_metadata sequence_metadata.json）
    为每条有帧号引用的 QA 计算其"包含序列"（论文输入 S_L = QA 所属序列的
    采样帧集合），写入：
      item["feat_indices"] = [i0, i1, ...]   # 序列的精确采样帧（首选）
      item["feat_range"]   = [s, e]          # 序列首末帧闭区间（兼容/前缀用）
    训练 dataset.py / 评测 test_b4dl.py 优先按 feat_indices 选帧。

用法：
    cd mllm

    # legacy 注入（per-seq 键匹配 + per-scene 回退）
    python scripts/inject_metatoken.py \
        --input ./b4dl_dataset/stage2_full_train.json \
        --ego_meta ./b4dl_dataset/ego_metadata.json \
        --output ./b4dl_dataset/stage2_full_train_seq.json

    # v2 论文对齐注入：任意帧范围真实 metatoken + feat_range 包含序列
    python scripts/inject_metatoken.py \
        --input ./b4dl_dataset/stage2_full_train.json \
        --ego_meta ./b4dl_dataset/ego_metadata.json \
        --frame_motion ./b4dl_dataset/ego_frame_motion.json \
        --sequence_metadata ../encoders/lidarclip/annotations/sequence_metadata.json \
        --output ./b4dl_dataset/stage2_full_train_seqv2.json

    # frame-context 变体（自创干预，非论文内容；前缀用 feat_range 边界）
    python scripts/inject_metatoken.py ... --frame_ctx

    # 消融：仅 <4DLiDAR> 无 <meta>
    python scripts/inject_metatoken.py ... --no_meta
"""

import os
import sys
import json
import re
import argparse
from typing import Dict, Optional, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ego_text import render_metatoken_for_range, build_metatoken_line  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="Inject B4DL Metatoken prefix into QA data")
    p.add_argument("--input", type=str, required=True,
                   help="Input JSON (conversations format)")
    p.add_argument("--ego_meta", type=str, required=True,
                   help="ego_metadata.json from generate_ego_metadata.py")
    p.add_argument("--output", type=str, required=True,
                   help="Output JSON path")
    p.add_argument("--frame_motion", type=str, default=None,
                   help="ego_frame_motion.json from generate_ego_metadata.py "
                        "--frame_motion. Enables v2: render REAL metatoken "
                        "descriptions for arbitrary QA-referenced frame "
                        "ranges (paper Appendix C), eliminating per-scene "
                        "fallback for sub-range / single-frame references.")
    p.add_argument("--sequence_metadata", type=str, default=None,
                   help="sequence_metadata.json. Enables feat_range: for each "
                        "framed QA, write item['feat_range'] = [s, e] of its "
                        "containing sequence (paper: model input S_L is the "
                        "QA's sequence, not the referenced sub-range).")
    p.add_argument("--no_4dlidar", action="store_true",
                   help="Omit <4DLiDAR> prefix (for ablation)")
    p.add_argument("--no_meta", action="store_true",
                   help="Omit <meta> + ego text (for ablation)")
    p.add_argument("--frame_ctx", action="store_true",
                   help="Prepend frame-range context to the <meta> line using "
                        "the feat_range boundaries: "
                        "\"<meta> This sequence covers frames XXX to XXX. ...\" "
                        "(our intervention, NOT part of the paper)")
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


def build_sequence_ranges(sequence_metadata: list) -> Dict[str, List[Tuple[int, int, list]]]:
    """Group sequence ranges by scene_token: {token: [(s0, e0, indices), ...]}.

    `indices` is the sequence's full sampled-frame list (e.g. [0,2,4,6,8]) —
    the paper's S_L input is exactly these frames, not every scene frame in
    the [s, e] interval.
    """
    by_scene: Dict[str, List[Tuple[int, int, list]]] = {}
    for seq in sequence_metadata:
        st = seq.get("scene_token")
        indices = seq.get("indices", [])
        if not st or len(indices) < 1:
            continue
        by_scene.setdefault(st, []).append((indices[0], indices[-1], indices))
    return by_scene


def find_containing_sequence(ranges: List[Tuple[int, int, list]], first: int, last: int
                             ) -> Optional[Tuple[int, int, list]]:
    """Find the sequence that the QA-referenced [first, last] belongs to.

    Sequences within a scene overlap (e.g. [0,8], [6,14], [12,20], ...).
    Prefer the narrowest fully-containing range; if none fully contains the
    reference (GPT-generated frame numbers can fall outside), pick the range
    with the largest overlap; no overlap at all → None.
    """
    containing = [r for r in ranges if r[0] <= first and last <= r[1]]
    if containing:
        return min(containing, key=lambda r: r[1] - r[0])
    best, best_ov = None, 0
    for r in ranges:
        ov = min(r[1], last) - max(r[0], first) + 1
        if ov > best_ov:
            best, best_ov = r, ov
    return best


def inject_metatoken(items: list,
                     ego_meta: dict,
                     include_4dlidar: bool = True,
                     include_meta: bool = True,
                     frame_ctx: bool = False,
                     frame_motion: Optional[Dict[str, List[dict]]] = None,
                     sequence_ranges: Optional[Dict[str, List[Tuple[int, int]]]] = None) -> list:
    """Inject Metatoken prefix into the first human message of each QA item.

    The format follows paper Figure 6 / Appendix C:
      <4DLiDAR> <video> <question>
      <meta> The metadata of the first frame is '...' and the metadata of the last frame is '...'

    The <video> token is kept as the embedding placeholder (VTimeLLM convention).
    The metatoken block (<meta> + frame descriptions) comes after the question,
    matching the paper's Figure 6 layout.

    Metatoken source priority per item (when question references frames):
      1. frame_motion table (v2): real descriptions of the QA-referenced frames
      2. per-sequence ego_metadata key (question refs == sequence boundaries)
      3. per-scene ego_metadata entry (no frame refs / missing data)

    When sequence_ranges is provided, each framed item additionally gets
    item['feat_range'] = [s, e] of its containing sequence — the paper's model
    input S_L (the QA's sequence), consumed by dataset.py / test_b4dl.py for
    feature slicing instead of the QA-referenced sub-range.
    """
    modified = []
    no_meta_count = 0
    stats = {"v2_rendered": 0, "seq_matched": 0, "scene_fallback": 0,
             "no_frame": 0, "feat_range_set": 0, "feat_range_missing": 0}

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

        first_frame, last_frame = parse_frame_numbers(cleaned)
        item = dict(item)  # shallow copy; conversations replaced below

        # ── feat_range / feat_indices: containing sequence ──
        # (paper input S_L = the QA's sequence, i.e. its exact sampled frames)
        feat_range = None
        if sequence_ranges is not None and first_frame is not None:
            ranges = sequence_ranges.get(item.get("scene_token") or "", [])
            seq_range = find_containing_sequence(ranges, first_frame, last_frame)
            if seq_range is not None:
                feat_range = [seq_range[0], seq_range[1]]
                item["feat_range"] = feat_range
                item["feat_indices"] = [int(i) for i in seq_range[2]]
                stats["feat_range_set"] += 1
            else:
                stats["feat_range_missing"] += 1

        if include_meta:
            meta_body = None
            if first_frame is not None:
                if frame_motion is not None:
                    meta_body = render_metatoken_for_range(
                        frame_motion, scene_id, first_frame, last_frame)
                    if meta_body is not None:
                        stats["v2_rendered"] += 1
                if meta_body is None:
                    scene_data = lookup_ego_for_qa(ego_meta, scene_id, cleaned)
                    if isinstance(scene_data, dict):
                        first_text = scene_data.get("first_frame", "")
                        last_text = scene_data.get("last_frame", "")
                    else:
                        first_text = scene_data
                        last_text = ""
                    if first_text and last_text:
                        meta_body = build_metatoken_line(first_text, last_text)
                    elif first_text:
                        meta_body = f"The metadata of the first frame is '{first_text}'"
                    if meta_body is not None:
                        stats["seq_matched" if (
                            first_frame is not None
                            and (f"{scene_id}_{first_frame}_{last_frame}" in ego_meta
                                 or f"{scene_id}_{last_frame}_{first_frame}" in ego_meta)
                        ) else "scene_fallback"] += 1
            else:
                stats["no_frame"] += 1
                scene_data = ego_meta.get(scene_id, {})
                if isinstance(scene_data, dict):
                    first_text = scene_data.get("first_frame", "")
                    last_text = scene_data.get("last_frame", "")
                else:
                    first_text = scene_data
                    last_text = ""
                if first_text and last_text:
                    meta_body = build_metatoken_line(first_text, last_text)
                elif first_text:
                    meta_body = f"The metadata of the first frame is '{first_text}'"

            if meta_body is None:
                meta_body = "No ego motion metadata available for this scene."
                no_meta_count += 1
            meta_line = f"<meta> {meta_body}"

            # Frame-range context prefix (our intervention, NOT in the paper).
            # Uses the feat_range boundaries when available (what the sliced
            # features actually cover), else the QA-referenced numbers.
            if frame_ctx:
                fc = feat_range or ([first_frame, last_frame]
                                    if first_frame is not None else None)
                if fc is not None:
                    meta_line = (f"<meta> This sequence covers frames {fc[0]:03d} to "
                                 f"{fc[1]:03d}. {meta_body}")
            prefix_parts.append(meta_line)

        new_value = "\n".join(prefix_parts)
        new_conversations = [
            {"from": "human", "value": new_value}
        ] + conversations[1:]

        item["conversations"] = new_conversations
        modified.append(item)

    if no_meta_count:
        print(f"  ⚠ {no_meta_count} items had no ego metadata; used fallback text.")
    print(f"  metatoken source: v2_rendered={stats['v2_rendered']}, "
          f"seq_key={stats['seq_matched']}, scene_fallback={stats['scene_fallback']}, "
          f"no_frame={stats['no_frame']}")
    if sequence_ranges is not None:
        print(f"  feat_range: set={stats['feat_range_set']}, "
              f"missing={stats['feat_range_missing']}")

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

    frame_motion = None
    if args.frame_motion:
        print(f"Loading frame motion table: {args.frame_motion}")
        with open(args.frame_motion) as f:
            frame_motion = json.load(f)
        print(f"  {len(frame_motion)} scenes in frame table (v2 mode enabled)")

    sequence_ranges = None
    if args.sequence_metadata:
        print(f"Loading sequence metadata: {args.sequence_metadata}")
        with open(args.sequence_metadata) as f:
            seq_meta = json.load(f)
        sequence_ranges = build_sequence_ranges(seq_meta)
        print(f"  {sum(len(v) for v in sequence_ranges.values())} sequences "
              f"across {len(sequence_ranges)} scenes (feat_range enabled)")

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
        frame_motion=frame_motion,
        sequence_ranges=sequence_ranges,
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
        print(f"\nExample output (scene_id={ex.get('scene_id', '?')}, "
              f"feat_range={ex.get('feat_range')}):")
        print(f"  {q[:300]}...")


if __name__ == "__main__":
    main()
