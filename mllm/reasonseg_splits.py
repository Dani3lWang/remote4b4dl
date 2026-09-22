"""Leakage-safe scene splitting shared by ReasonSeg data pipelines."""

from __future__ import annotations

import random
from typing import Iterable, Mapping, Sequence


def canonical_scene_tokens(
    scene_records: Sequence[Mapping[str, object]],
    scene_ids: Iterable[str],
) -> set[str]:
    """Resolve a mixture of nuScenes scene names and tokens to tokens."""

    lookup: dict[str, str] = {}
    for record in scene_records:
        token = str(record["token"])
        lookup[token] = token
        lookup[str(record["name"])] = token
    requested = {str(value).strip() for value in scene_ids if str(value).strip()}
    unknown = requested - set(lookup)
    if unknown:
        raise ValueError(
            "unknown nuScenes scenes: " + ", ".join(sorted(unknown)[:5])
        )
    return {lookup[value] for value in requested}


def partition_development_scenes(
    scene_records: Sequence[Mapping[str, object]],
    *,
    official_train_names: Iterable[str],
    excluded_scenes: Iterable[str] = (),
    validation_fraction: float = 0.1,
    seed: int = 20260917,
) -> dict[str, str]:
    """Split official-train scenes into internal train/val sets.

    nuScenes official ``val`` is the B4DL held-out test set, so it must never
    be reused for ReasonSeg tuning.  The returned mapping only contains
    official-train scene tokens and is deterministic for a fixed seed.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    official_train_names = {str(value) for value in official_train_names}
    excluded_tokens = canonical_scene_tokens(scene_records, excluded_scenes)
    candidates = sorted(
        str(record["token"])
        for record in scene_records
        if str(record["name"]) in official_train_names
        and str(record["token"]) not in excluded_tokens
    )
    if len(candidates) < 2:
        raise ValueError("need at least two non-test development scenes")
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    validation_count = round(len(shuffled) * validation_fraction)
    validation_count = max(1, min(len(shuffled) - 1, validation_count))
    validation_tokens = set(shuffled[:validation_count])
    return {
        token: "val" if token in validation_tokens else "train"
        for token in candidates
    }
