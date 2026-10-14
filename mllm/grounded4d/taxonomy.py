"""Official nuScenes fine labels, challenge classes and instance admission."""
from collections import Counter
import io

import numpy as np


POLICY = "nuscenes-panoptic-instance-admission-v1"
THING_CLASSES = (
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
)
STUFF_CLASSES = ("driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation")
FINE_NAMES = (
    "noise", "animal", "human.pedestrian.adult", "human.pedestrian.child",
    "human.pedestrian.construction_worker", "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer", "human.pedestrian.stroller", "human.pedestrian.wheelchair",
    "movable_object.barrier", "movable_object.debris", "movable_object.pushable_pullable",
    "movable_object.trafficcone", "static_object.bicycle_rack", "vehicle.bicycle",
    "vehicle.bus.bendy", "vehicle.bus.rigid", "vehicle.car", "vehicle.construction",
    "vehicle.emergency.ambulance", "vehicle.emergency.police", "vehicle.motorcycle",
    "vehicle.trailer", "vehicle.truck", "flat.driveable_surface", "flat.other",
    "flat.sidewalk", "flat.terrain", "static.manmade", "static.other", "static.vegetation", "vehicle.ego",
)
FINE_TO_COARSE = (
    "ignore", "ignore", "pedestrian", "pedestrian", "pedestrian", "ignore", "pedestrian",
    "ignore", "ignore", "barrier", "ignore", "ignore", "traffic_cone", "ignore", "bicycle",
    "bus", "bus", "car", "construction_vehicle", "ignore", "ignore", "motorcycle",
    "trailer", "truck", "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
    "ignore", "vegetation", "ignore",
)
LEAF_TO_CLASS = {name: coarse for name, coarse in zip(FINE_NAMES, FINE_TO_COARSE) if coarse in THING_CLASSES}
IGNORE_IDS = tuple(index for index, name in enumerate(FINE_TO_COARSE) if name == "ignore")


def policy_info(class_names, min_target_points):
    return dict(policy=POLICY, fine_label_space="nuscenes_original_32",
                challenge_classes=list(THING_CLASSES + STUFF_CLASSES),
                model_class_names=list(class_names), model_class_ids="zero_based_config_order",
                min_target_points=min_target_points, require_target_in_every_frame=True,
                ignored_raw_semantic_ids=list(IGNORE_IDS), stuff_points="non_target_background")


def validate_category_table(rows):
    mapping = {}
    for row in rows:
        index = row.get("index")
        if type(index) is not int or index in mapping:
            raise ValueError("duplicate_or_invalid_panoptic_category_index")
        mapping[index] = row.get("name")
    if mapping != dict(enumerate(FINE_NAMES)):
        raise ValueError("panoptic_category_table_is_not_official_original_32_mapping")
    return mapping


def decode_panoptic(raw, point_count):
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        encoded = archive["data"].copy()
    if (encoded.ndim != 1 or encoded.dtype.kind not in "iu" or len(encoded) != point_count
            or (encoded < 0).any() or (encoded >= 32000).any()):
        raise ValueError("invalid_original_panoptic_layout_or_semantic_index")
    return encoded.astype(np.int64)


def audit_frame(frame, labels, classes, tokens, valid, encoded, class_names,
                min_target_points, normalize_ignored_points=False):
    """Check masks against original labels; never drop individual GT objects."""
    if not tokens:
        raise ValueError("no_configured_instance_in_frame_roi")
    evidence = frame.get("instance_mapping_evidence")
    if not isinstance(evidence, list) or len(evidence) != len(tokens):
        raise ValueError("missing_per_instance_panoptic_class_evidence")
    ignored = np.isin(encoded // 1000, IGNORE_IDS)
    if normalize_ignored_points:
        if (labels[:, ignored] == 1).any():
            raise ValueError("ignored_semantic_points_are_instance_foreground")
        labels[:, ignored] = -1
    expected_ids = set()
    for panoptic_id in np.unique(encoded[valid]):
        if FINE_TO_COARSE[int(panoptic_id) // 1000] in class_names:
            if int(panoptic_id) % 1000 == 0:
                raise ValueError("configured_thing_has_no_panoptic_instance")
            expected_ids.add(int(panoptic_id))
    seen, counts, point_counts = set(), Counter(), Counter()
    for index, (token, class_id, item) in enumerate(zip(tokens, classes.tolist(), evidence)):
        panoptic_id = item.get("panoptic_id") if isinstance(item, dict) else None
        if (type(panoptic_id) is not int or panoptic_id not in expected_ids
                or panoptic_id in seen or item.get("instance_token") != token):
            raise ValueError("panoptic_instance_evidence_disagrees_with_mask_identity")
        seen.add(panoptic_id)
        class_name = FINE_TO_COARSE[panoptic_id // 1000]
        if class_names[class_id] != class_name:
            raise ValueError("model_class_id_disagrees_with_original_panoptic_semantic_class")
        mask = encoded == panoptic_id
        expected = mask.astype(np.int8)
        expected[ignored] = -1
        if not np.array_equal(labels[index], expected):
            raise ValueError("instance_mask_disagrees_with_original_panoptic_points_or_ignore_labels")
        count = int((mask & valid).sum())
        if count < min_target_points:
            raise ValueError("instance_has_insufficient_in_range_foreground_points")
        counts[class_name] += 1
        point_counts[class_name] += count
    if seen != expected_ids:
        raise ValueError("configured_instances_missing_from_complete_frame_supervision")
    return dict(instance_class_counts=dict(counts), target_point_counts=dict(point_counts),
                in_range_points=int(valid.sum()), ignored_in_range_points=int((ignored & valid).sum()))
