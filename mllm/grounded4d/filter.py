"""Derive admitted manifests without changing original points or supervision."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import Grounded4DConfig
from .data import Grounded4DDataset, load_scene_tokens, read_hashed_file
from .taxonomy import policy_info
from .train import validate_stage


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    part.replace(path)


def run(args):
    config = Grounded4DConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8-sig")))
    output = args.output_dir.resolve()
    for source in (args.label_root, args.panoptic_root, args.nuscenes_root / "samples",
                   args.nuscenes_root / "sweeps"):
        if output.is_relative_to(source.resolve()):
            raise ValueError("filter output must be outside original point, label and panoptic directories")
    if args.min_target_points < 1:
        raise ValueError("min-target-points must be positive")
    specs = [("train", args.train_manifest, args.train_scene_tokens)]
    if args.validation_manifest:
        specs.append(("internal_dev", args.validation_manifest,
                      args.dev_scene_tokens or args.train_scene_tokens))
    datasets = {}
    for split, manifest, whitelist in specs:
        # Repair ONLY ignore-class background values in a new output mask. All
        # foreground, class, provenance and completeness checks still apply.
        dataset = Grounded4DDataset(manifest, args.nuscenes_root, args.label_root, config,
                                    load_scene_tokens(whitelist), split, "panoptic",
                                    panoptic_root=args.panoptic_root,
                                    min_target_points=args.min_target_points,
                                    normalize_ignored_points=True)
        validate_stage(dataset, args.stage)
        datasets[split] = dataset
    if len(datasets) == 2 and datasets["train"].scenes & datasets["internal_dev"].scenes:
        raise ValueError("training and development scenes overlap; filtering must not conceal split leakage")
    versions = {row.get("dataset_version") for dataset in datasets.values() for row in dataset.records}
    if len(versions) != 1:
        raise ValueError("train/dev nuScenes dataset versions differ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "masks").mkdir()
    (output / "reference_features").mkdir()
    results, reasons = {}, Counter()
    with (output / "rejected.jsonl").open("x", encoding="utf-8") as rejected:
        for split, dataset in datasets.items():
            accepted, scenes = 0, set()
            with (output / (split + ".jsonl.part")).open("x", encoding="utf-8") as manifest:
                for index, source in enumerate(dataset.records):
                    try:
                        clip = dataset[index]
                    except (ValueError, OSError, KeyError) as error:
                        reason = str(error)
                        reasons[reason] += 1
                        rejected.write(json.dumps(dict(split=split, clip_id=source["clip_id"],
                                                       scene_token=source["scene_token"], reason=reason,
                                                       sample_data_tokens=[frame["sample_data_token"]
                                                                           for frame in source["frames"]]),
                                                  ensure_ascii=False) + "\n")
                        continue
                    row = copy.deepcopy(source)
                    row["class_admission"] = dict(**policy_info(config.class_names, args.min_target_points),
                                                   source_category_sha256=dataset.source_category_sha256)
                    for frame_index, (frame, checked) in enumerate(zip(row["frames"], clip["frames"])):
                        key = hashlib.sha256((row["clip_id"] + ":" + str(frame_index)).encode()).hexdigest()
                        relative = "masks/" + key + ".npz"
                        mask_path = output / relative
                        with mask_path.open("xb") as handle:
                            np.savez_compressed(handle, labels=checked["labels"].numpy().astype(np.int8),
                                                instance_tokens=np.asarray(checked["instance_tokens"]),
                                                point_indices=np.arange(len(checked["points"]), dtype=np.int64),
                                                point_file_sha256=np.asarray(frame["point_file"]["sha256"]),
                                                label_kind=np.asarray("panoptic_gt"), mask_verified=np.asarray(True))
                        frame["mask_file"] = dict(path=relative, sha256=digest(mask_path))
                        frame["class_admission"] = checked["class_admission"]
                        if frame.get("references"):
                            feature = frame["references"]["feature_file"]
                            raw = read_hashed_file(args.label_root, feature)
                            relative_feature = "reference_features/" + key + ".npy"
                            with (output / relative_feature).open("xb") as handle:
                                handle.write(raw)
                            frame["references"]["feature_file"] = dict(path=relative_feature,
                                                                         sha256=hashlib.sha256(raw).hexdigest())
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                    accepted += 1
                    scenes.add(row["scene_token"])
            results[split] = dict(input_clips=len(dataset), accepted_clips=accepted,
                                  rejected_clips=len(dataset) - accepted,
                                  input_manifest_sha256=dataset.manifest_sha256)
            write_json(output / ("train_scene_tokens.json" if split == "train" else "dev_scene_tokens.json"),
                       sorted(scenes))
    summary = dict(policy=policy_info(config.class_names, args.min_target_points), stage=args.stage,
                   original_label_root=str(args.label_root.resolve()), output_root=str(output),
                   splits=results, rejection_reasons=dict(reasons),
                   ignored_semantic_points="-1_in_new_masks_original_files_untouched",
                   qa_grounding_promoted=False, temporal_identity_promoted=False,
                   source_sha256={path.name: digest(path) for path in Path(__file__).parent.glob("*.py")})
    if any(not report["accepted_clips"] for report in results.values()):
        write_json(output / "failed_filtering.json", summary)
        raise ValueError("filtering produced an empty split; training is refused")
    preflight = {}
    for split in datasets:
        (output / (split + ".jsonl.part")).replace(output / (split + ".jsonl"))
        whitelist = output / ("train_scene_tokens.json" if split == "train" else "dev_scene_tokens.json")
        checked = Grounded4DDataset(output / (split + ".jsonl"), args.nuscenes_root, output, config,
                                   load_scene_tokens(whitelist), split, "panoptic",
                                   panoptic_root=args.panoptic_root, min_target_points=args.min_target_points)
        preflight[split] = checked.preflight()
        results[split]["output_manifest_sha256"] = checked.manifest_sha256
    summary["preflight"] = preflight
    summary["required_detection_queries"] = max(report["max_detection_targets"] for report in preflight.values())
    summary["query_capacity_sufficient"] = summary["required_detection_queries"] <= config.num_queries
    write_json(output / "filtering_summary.json", summary)
    if not summary["query_capacity_sufficient"]:
        raise ValueError("query capacity is insufficient; increase queries instead of discarding GT instances")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--nuscenes-root", required=True, type=Path)
    parser.add_argument("--panoptic-root", required=True, type=Path)
    parser.add_argument("--label-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-scene-tokens", type=Path,
                        default=Path(__file__).resolve().parents[2] / "datageneration/tools/assets/train_scene_tokens.json")
    parser.add_argument("--dev-scene-tokens", type=Path)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--stage", choices=["spatial", "temporal", "grounding", "joint"], default="spatial")
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
