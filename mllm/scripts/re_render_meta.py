#!/usr/bin/env python
"""B3: re-render the <meta> body of an injected training JSON in place.

ego_text.render_meta_texts was fixed to the paper's relative-to-previous
semantics (per-frame position/speed/turn now vary with the frame). The injected
JSONs carry the old render inline, so this re-renders every sample whose frame
reference can be resolved — same resolution rules as inject_metatoken.py:
question frame numbers, else (task==time_grounding) the GT answer range.
Samples without a resolvable range are left untouched (per-scene fallback).

Usage: python scripts/re_render_meta.py \
    --data b4dl_dataset/stage2_full_train_seqv3_148k.json \
    --frame_motion b4dl_dataset/ego_frame_motion.json
"""
import argparse
import json
import re
import sys
from typing import Optional, Tuple

sys.path.insert(0, "scripts")
from ego_text import render_metatoken_for_range, build_metatoken_line  # noqa: E402


def parse_frame_numbers(text: str) -> Tuple[Optional[int], Optional[int]]:
    matches = re.findall(r"frame\s+(\d+)", text, re.IGNORECASE)
    if not matches:
        return None, None
    nums = [int(m) for m in matches]
    return min(nums), max(nums)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--frame_motion", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.load(open(args.data))
    fm = json.load(open(args.frame_motion))

    n_total = n_render = n_skip = 0
    for it in data:
        n_total += 1
        convs = it.get("conversations", [])
        if not convs:
            continue
        human = convs[0]["value"]
        if "<meta>" not in human:
            continue
        head, _, _old_body = human.partition("<meta>")
        question = head.split("<video>")[-1].split("<4DLiDAR>")[-1].strip()

        first, last = parse_frame_numbers(question)
        if first is None and it.get("task") == "time_grounding" and len(convs) > 1:
            first, last = parse_frame_numbers(convs[1]["value"])
        if first is None:
            n_skip += 1
            continue  # per-scene fallback sample; leave untouched
        body = render_metatoken_for_range(fm, it["scene_id"], first, last)
        if body is None:
            n_skip += 1
            continue
        convs[0]["value"] = head + "<meta> " + body
        n_render += 1

    json.dump(data, open(args.out, "w"), ensure_ascii=False)
    print(f"total={n_total} re-rendered={n_render} skipped={n_skip}")


if __name__ == "__main__":
    main()
