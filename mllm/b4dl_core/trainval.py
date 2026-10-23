"""Prepare and audit the full official trainval using verified historical SST inputs."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
try:
    import resource
except ImportError:
    resource = None
from types import SimpleNamespace

from .audit import audit_supervision
from .data import build_dataset
from .inputs import GeometryDataset, control_plan
from .io import digest, iter_array, read_json, sha256, write_json
from .sst_source import seal_provenance


def opposite_pair_capacity(requests, targets):
    # Edges join opposite labels with identical prompts/lengths across different scenes.
    groups = defaultdict(lambda: {"Yes": Counter(), "No": Counter()})
    for request in requests:
        if request["kind"] == "binary":
            key = (request["split"], request["task"], request["predicate"], request["prompt"], request["feature_length"])
            groups[key][targets[request["request_id"]]["answer"]][request["scene_id"]] += 1
    result = Counter()
    for key, labels in groups.items():
        yes, no = sum(labels["Yes"].values()), sum(labels["No"].values())
        combined = labels["Yes"] + labels["No"]
        capacity = min(yes, no, yes+no-max(combined.values(), default=0))
        result[key[0]] += capacity
    return dict(result)


def coverage_report(dataset, tables):
    import numpy as np
    tables = Path(tables)
    categories = {r["token"]: r["name"] for r in iter_array(tables/'category.json')}
    instances = {r["token"]: categories[r["category_token"]] for r in iter_array(tables/'instance.json')}
    scene_counts, rows_per_scene, buckets, classes, semantics = Counter(), Counter(), Counter(), Counter(), defaultdict(set)
    donor_groups = Counter((r["split"], r["frames"]) for r in dataset.manifest["scenes"].values())
    donor_counts, representatives = Counter(), {}
    for request in dataset.requests:
        target = dataset.targets[request["request_id"]]
        split, ident = request["split"], request["scene_id"]
        rows_per_scene[ident] += 1
        buckets[(split, request["task"], request["predicate"], target["answer"])] += 1
        if request["task"] == "existence":
            label = request["question"].split("Is there a ", 1)[1].split(" supported by", 1)[0]
            classes[(split, request["task"], label, target["answer"])] += 1
        else:
            labels = {instances[i] for i in target["provenance"]["instances"]}
            for label in labels: classes[(split, request["task"], label, target["answer"])] += 1
        if request["task"] == "distance": semantics[split].add(request["semantic_id"])
        donor_counts[(split, "ready" if donor_groups[(split, request["feature_length"])] > 1 else "unavailable")] += 1
        representatives.setdefault(ident, request)
    split_scene_tokens = defaultdict(set)
    for ident, record in dataset.manifest["scenes"].items():
        scene_counts[record["split"]] += 1
        split_scene_tokens[record["split"]].add(record["scene_token"])
        dataset.scene(ident)
    for a in split_scene_tokens:
        for b in split_scene_tokens:
            if a != b and split_scene_tokens[a] & split_scene_tokens[b]:
                raise ValueError("physical scene leakage")
    checks = {}
    compact = SimpleNamespace(manifest=dataset.manifest, requests=list(representatives.values()))
    for condition in ("identity", "zero", "repeat_first", "shuffle_features", "cross_scene"):
        changed, unchanged, unavailable = 0, 0, 0
        for request, control in zip(compact.requests, control_plan(compact, condition)):
            if control["status"] != "ready":
                unavailable += 1
                continue
            original, modified = dataset.model_input(request), dataset.model_input(request, control)
            if original["prompt"] != modified["prompt"] or any(not np.array_equal(original[k], modified[k]) for k in
                    ("frame_times", "frame_geometry", "frame_valid_mask")):
                raise ValueError("control changed geometry or prompt")
            if np.array_equal(original["features"], modified["features"]): unchanged += 1
            else: changed += 1
        if condition == "identity" and changed or condition != "identity" and unchanged:
            raise ValueError("feature control was ineffective")
        checks[condition] = {"changed_scenes": changed, "unchanged_scenes": unchanged, "unavailable_scenes": unavailable}
    missing_questions = sorted(set(dataset.manifest["scenes"])-set(rows_per_scene))
    return {"schema": "b4dl-full-trainval-coverage-v1", "dataset_id": dataset.manifest["dataset_id"],
        "status": "coverage_audited", "scenes": dict(scene_counts),
        "frames": sum(r["frames"] for r in dataset.manifest["scenes"].values()),
        "questions": len(dataset.requests), "candidate_questions": dataset.manifest["candidate_questions"],
        "task_counts": dataset.manifest["counts"], "scene_question_counts": dict(rows_per_scene),
        "scenes_without_selected_questions": missing_questions,
        "bucket_counts": [{"split": k[0], "task": k[1], "predicate": k[2], "answer": k[3], "n": n}
                          for k, n in sorted(buckets.items())],
        "class_counts": [{"split": k[0], "task": k[1], "class": k[2], "answer": k[3], "n": n}
                         for k, n in sorted(classes.items())],
        "distance_semantic_questions": {k: len(v) for k, v in semantics.items()},
        "strict_opposite_pairs": opposite_pair_capacity(dataset.requests, dataset.targets),
        "cross_scene_donor_counts": {"/".join(k): v for k, v in sorted(donor_counts.items())},
        "actual_feature_controls": checks, "shortages": dataset.manifest["shortages"],
        "input_scope": dataset.manifest.get("input_scope"), "encoder_verified": dataset.manifest["encoder_checkpoint_verified"],
        "base_model_may_have_seen_internal_dev": True,
        "caveats": ["official validation is the geometry test set; not the hidden nuScenes test split",
                    "internal_dev is held out from new supervision but may have been seen by historical B3/encoder",
                    "distance option rotations are not independent geometric events",
                    "visibility uses oriented-box support, not panoptic labels"],
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 if resource else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "tables", "metadata", "features"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--max-per-label", type=int, default=1000000)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    from nuscenes.utils import splits as split_module
    official = split_module.create_splits_scenes()
    split_path = root/'official-trainval-splits.json'
    if split_path.exists(): parser.error("run already started; use a new run root")
    write_json(split_path, {"version": "v1.0-trainval", "source": "nuscenes-devkit:"+sha256(split_module.__file__),
                            "train": official["train"], "validation": official["val"]})
    print("sealing source provenance", flush=True)
    provenance_path = root/'sst-provenance.json'
    seal_provenance(root/'sst-inventory.json', root/'sst-reproduction.json', provenance_path)
    print("building complete visibility-aware trainval", flush=True)
    manifest = build_dataset(args.tables, args.metadata, args.features, split_path, root/'trainval-v1',
        max_per_label=args.max_per_label, input_support=root/'input-support.jsonl', encoder_provenance=provenance_path)
    print("built", manifest['dataset_id'], manifest['selected_questions'], flush=True)
    dataset = GeometryDataset(root/'trainval-v1')
    print("auditing raw numerical evidence", flush=True)
    write_json(root/'raw-audit.json', audit_supervision(dataset, args.tables))
    print("auditing coverage and actual feature controls", flush=True)
    coverage = coverage_report(dataset, args.tables)
    write_json(root/'coverage.json', coverage)
    expected = {"train": 560, "internal_dev": 140, "test": 150}
    if coverage["scenes"] != expected or coverage["scenes_without_selected_questions"]:
        raise ValueError("complete trainval scene/question coverage not satisfied")
    write_json(root/'completion.json', {"status": "full_trainval_prepared_and_audited", "dataset_id": manifest["dataset_id"],
        "encoder_provenance_id": read_json(provenance_path)["provenance_id"], "questions": coverage["questions"],
        "scenes": coverage["scenes"], "frames": coverage["frames"], "training_started": False,
        "receipts": {name: sha256(root/name) for name in ('sst-provenance.json', 'raw-audit.json', 'coverage.json')}})
    print("full_trainval_prepared_and_audited", flush=True)


if __name__ == '__main__':
    main()
