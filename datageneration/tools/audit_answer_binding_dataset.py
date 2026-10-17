#!/usr/bin/env python3
"""Audit saved binding references against saved NPZ labels, without regeneration."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def records(path):
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(root):
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is True
    frame_index, counts, scenes = {}, Counter(), {}
    for split in ("train", "internal_dev"):
        scenes[split] = set()
        for clip in records(root / "spatial" / (split + ".jsonl")):
            assert clip["split"] == split
            scenes[split].add(clip["scene_token"])
            frame = clip["frames"][0]
            token = frame["sample_data_token"]
            assert token not in frame_index
            mask = root / "spatial" / frame["mask_file"]["path"]
            assert mask.resolve().is_relative_to((root / "spatial").resolve())
            assert sha(mask) == frame["mask_file"]["sha256"]
            with np.load(mask, allow_pickle=False) as data:
                tokens = data["instance_tokens"].tolist()
                labels = data["labels"]
                assert labels.shape[0] == len(tokens)
                assert np.array_equal(data["point_indices"], np.arange(labels.shape[1]))
                assert str(data["point_file_sha256"].item()) == frame["point_file"]["sha256"]
                assert np.isin(labels, [-1, 0, 1]).all()
                assert data["mask_verified"].item() is True
                sizes = (labels == 1).sum(1).tolist()
            frame_index[token] = (split, clip["scene_token"], frame, tokens, sizes)
            counts[split + "_frames"] += 1
            counts[split + "_instances"] += len(tokens)
    assert not scenes["train"] & scenes["internal_dev"]
    ids, eligible = set(), {"train": set(), "internal_dev": set()}
    priority = []
    for row in records(root / "binding/candidates.jsonl"):
        assert row["candidate_id"] not in ids
        ids.add(row["candidate_id"])
        split, scene, frame, tokens, sizes = frame_index[row["sample_data_token"]]
        assert row["split"] == split and row["scene_token"] == scene
        assert row["point_file"] == frame["point_file"] and row["mask_file"] == frame["mask_file"]
        assert row["answer_source"] == "dataset_ground_truth"
        start, end = row["mention_span"]
        assert row["answer"][start:end] == row["mention_text"]
        start, end = row["target_span"]
        assert row["answer"][start:end] == row["target_phrase"]
        for obj in row["objects"]:
            index = obj["mask_row"]
            assert tokens[index] == obj["instance_token"] and sizes[index] == obj["point_count"]
            assert frame["class_ids"][index] == obj["class_id"]
        if row["review_status"] == "pending":
            assert row["grounding_verified"] is False and row["selected_instance_token"] is None
            assert row["eligible_for_binding_training"] is False and row["eligible_for_binding_evaluation"] is False
        else:
            assert row["review_status"] == "accepted" and row["grounding_verified"] is True
            assert row["selected_instance_token"] == tokens[row["target_mask_row"]]
            eligible[split].add(row["candidate_id"])
            assert row["eligible_for_binding_training"] == (split == "train")
        counts[split + "_binding_candidates"] += 1
        if split == "train" and len(priority) < 300 and len(row["objects"]) == 1 and len(row["referenced_frame_indices"]) == 1:
            priority.append(row)
    for split in ("train", "internal_dev"):
        accepted = list(records(root / "binding" / (split + ".jsonl")))
        assert {r["candidate_id"] for r in accepted} == eligible[split]
        assert len(accepted) == len(eligible[split])
        counts[split + "_reviewed_bindings"] = len(accepted)
    for key, value in counts.items():
        assert value == summary["counts"].get(key, 0), key
    with (root / "binding/review_first300.jsonl").open("x", encoding="utf-8") as stream:
        for row in priority:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = dict(complete=True, counts=dict(counts), scene_disjoint=True,
                  original_point_axes_verified=True, all_candidate_mask_references_verified=True,
                  pending_candidates_excluded_from_training=True, priority_review_rows=len(priority),
                  priority_selection="first up to 300 train candidates with one frame and one same-class object; NOT verified grounding")
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    print(json.dumps(audit(parser.parse_args().dataset), indent=2))
