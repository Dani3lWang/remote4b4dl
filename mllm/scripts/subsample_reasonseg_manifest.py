#!/usr/bin/env python3
"""Thin a ReasonSeg manifest to at most --per-frame queries per LiDAR frame.

Keeps scene/frame coverage intact while cutting the ~20 queries-per-frame
redundancy, so a much larger scene set costs the same per-epoch compute.
Selection is deterministic.  Frames whose targets are empty ("no such object"
queries) are only used when a frame has nothing else, otherwise the natural
negative rate would be inflated by the balancing.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def class_of(record: dict) -> str:
    targets = record.get("targets") or []
    return targets[0].get("class_name", "unknown") if targets else "unknown"


def pick(records: list[dict], per_frame: int, counts: Counter, rng: random.Random) -> list[dict]:
    """Choose up to per_frame records from one frame's query pool."""
    positive = [record for record in records if record.get("targets")]
    candidates = positive or list(records)
    chosen: list[dict] = []
    for _ in range(per_frame):
        if not candidates:
            break
        if counts is None:
            record = rng.choice(candidates)
        else:
            lowest = min(counts[class_of(record)] for record in candidates)
            pool = [record for record in candidates if counts[class_of(record)] == lowest]
            record = rng.choice(pool)
            counts[class_of(record)] += 1
        candidates.remove(record)
        chosen.append(record)
    if not chosen and records:
        chosen = [records[0]]
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-frame", type=int, default=1)
    parser.add_argument("--balance", action="store_true",
                        help="spread picks evenly over classes instead of keeping the natural mix")
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    frames: dict[str, list[dict]] = defaultdict(list)
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            frames[record["sample_token"]].append(record)

    counts: Counter = Counter() if args.balance else None  # type: ignore[assignment]
    kept: list[dict] = []
    for sample_token in sorted(frames):
        kept.extend(pick(frames[sample_token], args.per_frame, counts, rng))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    seen: Counter = Counter(class_of(record) for record in kept)
    print(json.dumps({
        "input_lines": sum(len(v) for v in frames.values()),
        "frames": len(frames),
        "scenes": len({r["scene_token"] for r in kept}),
        "output_lines": len(kept),
        "negatives": sum(1 for record in kept if not record.get("targets")),
        "classes": dict(sorted(seen.items())),
    }, indent=1, ensure_ascii=False))
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
