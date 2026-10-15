"""Native nuScenes input: reuse verified masks and add complete stuff/semantic supervision."""
import torch
import numpy as np

from grounded4d.data import Grounded4DDataset, read_hashed_file
from grounded4d.taxonomy import FINE_TO_COARSE, THING_CLASSES, decode_panoptic
from .config import CLASS_NAMES


def semantic_from_panoptic(encoded):
    lookup = np.asarray([CLASS_NAMES.index(name) if name in CLASS_NAMES else 0
                         for name in FINE_TO_COARSE], dtype=np.int64)
    return torch.from_numpy(lookup[np.asarray(encoded) // 1000].copy())


def build_targets(frame, valid):
    """Keep every thing, including one-point instances; add one mask per visible stuff class."""
    labels = frame["labels"]
    classes = frame["class_ids"] + 1
    semantic = frame["semantic_labels"]
    if labels.shape[1] != len(valid) or semantic.shape != valid.shape:
        raise ValueError("targets do not share original point order")
    supervised = valid & semantic.ne(0)
    masks = [m for m in labels]
    ids = classes.tolist()
    for category in range(11, 17):
        foreground = semantic.eq(category)
        if (foreground & valid).any():
            mask = foreground.float()
            mask[semantic.eq(0)] = -1
            masks.append(mask)
            ids.append(category)
    result = torch.stack(masks) if masks else labels.new_empty((0, len(valid)))
    if len(result) and not ((result == 1) & supervised[None]).any(1).all():
        raise ValueError("an instance has no supervised point in the model ROI; rebuild manifest")
    return result, torch.tensor(ids, dtype=torch.long, device=labels.device), supervised


class MaskPLSDataset(Grounded4DDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tuple(self.config.class_names) != THING_CLASSES:
            raise ValueError("MaskPLS requires all ten nuScenes thing classes in canonical order")
        for row in self.records:
            if len(row["frames"]) != 1:
                raise ValueError("MaskPLS spatial training accepts single-frame manifests only")
            transform = np.asarray(row["frames"][0]["lidar_to_anchor"])
            if not np.allclose(transform, np.eye(4), atol=1e-7):
                raise ValueError("MaskPLS expects native single-frame LiDAR coordinates, not an aligned clip")

    def _frame(self, row, source):
        frame = super()._frame(row, source)
        encoded = decode_panoptic(read_hashed_file(self.panoptic_root, source["source_panoptic_file"]),
                                  len(frame["points"]))
        frame["semantic_labels"] = semantic_from_panoptic(encoded)
        frame["point_sha256"] = source["point_file"]["sha256"]
        frame["sample_token"] = source["sample_token"]
        frame["scene_token"] = row["scene_token"]
        return frame

    def preflight(self, require_capacity=True):
        report = super().preflight()
        maximum = 0
        for index in range(len(self)):
            frame = self[index]["frames"][0]
            lower = frame["points"].new_tensor(self.config.point_cloud_range[:3])
            upper = frame["points"].new_tensor(self.config.point_cloud_range[3:])
            valid = ((frame["points"][:, :3] >= lower) & (frame["points"][:, :3] < upper)).all(1)
            labels, _, _ = build_targets(frame, valid)
            maximum = max(maximum, len(labels))
        if require_capacity and maximum > self.config.num_queries:
            raise ValueError(f"need at least {maximum} queries for all thing+stuff masks; never truncate targets")
        return dict(**report, max_panoptic_targets=maximum, min_instance_points=1,
                    semantic_supervision="16 nuScenes classes; original ignore excluded")
