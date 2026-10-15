"""Experiment identity, official split checks, and point-space evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from grounded4d.metrics import frame_counts, match_sets, pairwise_iou
from grounded4d.taxonomy import THING_CLASSES


def implementation_digest():
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    sources = list(package.glob("*.py"))
    sources += [package.parent / "grounded4d" / name for name in
                ("data.py", "taxonomy.py", "metrics.py", "criterion.py", "config.py", "train.py")]
    for path in sorted(sources):
        digest.update(str(path.relative_to(package.parent)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_source_split(dataset, metadata_root, *, official_val=False):
    """Validate actual scene/sample/frame membership, not just claimed manifest tokens."""
    from nuscenes.utils import splits

    versions = {row["dataset_version"] for row in dataset.records}
    if len(versions) != 1:
        raise ValueError("mixed versions")
    version = next(iter(versions))
    root = Path(metadata_root)
    root = root if root.name == version else root / version
    source = {}
    for name in ("scene", "sample", "sample_data"):
        raw = (root / (name + ".json")).read_bytes()
        source[name] = json.loads(raw)
    scenes = {row["token"]: row for row in source["scene"]}
    samples = {row["token"]: row for row in source["sample"]}
    data = {row["token"]: row for row in source["sample_data"]}
    panoptic_path = dataset.panoptic_root / version / "panoptic.json"
    panoptic_raw = panoptic_path.read_bytes()
    panoptic = {row["sample_data_token"]: row for row in json.loads(panoptic_raw)}
    if version == "v1.0-mini":
        if official_val:
            raise ValueError("--official-val is for full trainval only")
        names = set(splits.mini_train if dataset.records[0]["split"] == "train" else splits.mini_val)
    else:
        names = set(splits.val if official_val else splits.train)
    expected_scenes = {token for token, row in scenes.items() if row["name"] in names}
    if not dataset.scenes.issubset(expected_scenes):
        raise ValueError("manifest scenes do not belong to the declared official split")
    frame_tokens = set()
    for row in dataset.records:
        for frame in row["frames"]:
            token = frame["sample_data_token"]
            if token in frame_tokens:
                raise ValueError("duplicate frame: would bias spatial training/evaluation")
            frame_tokens.add(token)
            sd = data[token]
            sample = samples[frame["sample_token"]]
            if (sd["sample_token"] != frame["sample_token"] or sample["scene_token"] != row["scene_token"]
                    or not sd["is_key_frame"] or sd["timestamp"] != frame["timestamp_us"]
                    or sd["filename"] != frame["original_sample_data_filename"]):
                raise ValueError("manifest frame disagrees with official nuScenes metadata")
            native_filename = sd["filename"].replace("\\", "/")
            declared_point = frame["point_file"]["path"].replace("\\", "/")
            if declared_point != native_filename and not declared_point.endswith("/" + native_filename):
                raise ValueError("point path does not preserve the official frame filename")
            declared_panoptic = (dataset.panoptic_root / frame["source_panoptic_file"]["path"]).resolve()
            expected_panoptic = (dataset.panoptic_root / panoptic[token]["filename"]).resolve()
            if declared_panoptic != expected_panoptic:
                raise ValueError("panoptic source belongs to a different LiDAR frame")
    return dict(dataset_version=version, official_split="val" if official_val else
                ("mini_" + ("train" if dataset.records[0]["split"] == "train" else "val")
                 if version == "v1.0-mini" else "train"),
                metadata_sha256={**{name: hashlib.sha256((root / (name + ".json")).read_bytes()).hexdigest()
                                    for name in source},
                                 "panoptic": hashlib.sha256(panoptic_raw).hexdigest()})


def score_frame(frame, output, instances, mask_threshold):
    valid = (output["valid_points"] & frame["semantic_labels"].ne(0)).cpu().numpy()
    gt = frame["labels"].eq(1).cpu().numpy()[:, valid]
    classes = frame["class_ids"].cpu().numpy()
    predicted = instances["masks"].cpu().numpy()[:, valid]
    predicted_classes = instances["class_ids"].cpu().numpy()
    counts, matches = frame_counts(gt, classes, predicted, predicted_classes, len(THING_CLASSES))
    raw = output["mask_logits"].sigmoid().ge(mask_threshold).cpu().numpy()[:, valid]
    iou = pairwise_iou(gt, raw)
    query_classes = instances["all_classes"].cpu().numpy() - 1
    _, geometric = match_sets(iou, np.zeros(len(gt)), np.zeros(len(raw)))
    _, semantic = match_sets(iou, classes, query_classes)
    hits = {target for target, _, _ in matches["iou50_matches"]}
    sizes = gt.sum(1)
    strata = {}
    for name, lower, upper in (("1-10", 1, 10), ("11-49", 11, 49), ("50+", 50, np.inf)):
        targets = {int(i) for i in np.flatnonzero((sizes >= lower) & (sizes <= upper))}
        strata[name] = dict(gt=len(targets), tp=len(targets & hits))
    diagnostic = dict(role="scoring_only_not_language_grounding",
                      raw_query_geometry_hits=len(geometric),
                      raw_query_class_aware_hits=len(semantic),
                      best_geometry_iou_sum=float(iou.max(1).sum()) if iou.shape[1] else 0.0,
                      point_buckets=strata)
    return counts, matches, diagnostic
