"""Versioned MaskPLS architecture and raw-nuScenes preprocessing contract."""
from dataclasses import asdict, dataclass
import math
from types import SimpleNamespace

from grounded4d.config import Grounded4DConfig
from grounded4d.taxonomy import THING_CLASSES, STUFF_CLASSES

UPSTREAM_COMMIT = "84881ba83f2f2e846cbded677544993d0c8ecd76"
SCHEMA = "b4dl-maskpls-v1"
# Preserve official checkpoint rows: 0=ignore, 1..16=nuScenes, 17=no-object.
CLASS_NAMES = ("ignore",) + THING_CLASSES + STUFF_CLASSES
NO_OBJECT = 17


@dataclass
class MaskPLSConfig:
    num_queries: int = 128
    voxel_size: float = 0.05
    point_cloud_range: tuple = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
    intensity_mode: str = "raw"
    knn_backend: str = "keops"
    knn_chunk_size: int = 512
    checkpoint_decoder: bool = True
    loss_points: int = 16384
    points_per_instance: int = 500
    match_points: int = 8192
    score_threshold: float = 0.25
    mask_threshold: float = 0.5
    overlap_threshold: float = 0.8
    # Semantic ignore and instance-mask background are different contracts.
    # Keep the legacy setting explicit; the corrected recipe uses background.
    mask_ignore_policy: str = "exclude"
    # Preserve the original point-mean objective unless explicitly opted in.
    mask_bce_reduction: str = "point_mean"
    # DETR-style unmatched-query class weight; independent of matching costs.
    no_object_weight: float = 0.1
    # Explicit query experiment; old checkpoints/configs retain learned queries.
    query_initialization: str = "learned"

    def validate(self):
        if self.query_initialization not in ("learned", "zero_learned", "zero_fps"):
            raise ValueError("invalid query_initialization")
        for key in ("num_queries", "knn_chunk_size", "loss_points", "points_per_instance", "match_points"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key + " must be a positive integer")
        if not math.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError("invalid voxel_size")
        if (len(self.point_cloud_range) != 6
                or not all(math.isfinite(v) for v in self.point_cloud_range)
                or any(self.point_cloud_range[i] >= self.point_cloud_range[i + 3] for i in range(3))):
            raise ValueError("invalid ROI")
        for key in ("score_threshold", "mask_threshold", "overlap_threshold"):
            if not 0 < getattr(self, key) <= 1:
                raise ValueError(key + " must lie in (0,1]")
        if self.intensity_mode not in ("raw", "frame_minmax"):
            raise ValueError("intensity_mode must be explicit: raw or frame_minmax")
        if self.knn_backend not in ("keops", "torch"):
            raise ValueError("knn_backend must be keops or torch")
        if type(self.checkpoint_decoder) is not bool:
            raise ValueError("checkpoint_decoder must be boolean")
        if self.mask_ignore_policy not in ("exclude", "background"):
            raise ValueError("mask_ignore_policy must be exclude or background")
        if self.mask_bce_reduction not in ("point_mean", "instance_balanced", "sqrt_positive"):
            raise ValueError("mask_bce_reduction must be point_mean, instance_balanced or sqrt_positive")
        if (isinstance(self.no_object_weight, bool) or not math.isfinite(self.no_object_weight)
                or not 0 < self.no_object_weight <= 1):
            raise ValueError("no_object_weight must lie in (0,1]")

    def to_dict(self):
        value = asdict(self)
        value["point_cloud_range"] = list(self.point_cloud_range)
        return value

    @classmethod
    def from_dict(cls, values):
        value = cls(**values)
        value.point_cloud_range = tuple(value.point_cloud_range)
        value.validate()
        return value

    def data_config(self):
        # Reuse verified native masks and all ten thing classes without changing legacy code.
        return Grounded4DConfig(num_queries=self.num_queries, voxel_size=self.voxel_size,
                                point_cloud_range=self.point_cloud_range)

    def upstream(self):
        backbone = SimpleNamespace(INPUT_DIM=4, CHANNELS=[32, 32, 64, 128, 256, 256, 128, 96, 96],
                                   RESOLUTION=self.voxel_size, KNN_UP=3,
                                   KNN_BACKEND=self.knn_backend, KNN_CHUNK=self.knn_chunk_size)
        decoder = SimpleNamespace(HIDDEN_DIM=256, NHEADS=8, DIM_FFN=1024,
                                  FEATURE_LEVELS=3, DEC_BLOCKS=3, NUM_QUERIES=self.num_queries,
                                  CHECKPOINT=self.checkpoint_decoder,
                                  QUERY_INITIALIZATION=self.query_initialization,
                                  POS_ENC=SimpleNamespace(MAX_FREQ=10000, DIMENSIONALITY=3, BASE=2))
        return backbone, decoder, SimpleNamespace(NUM_CLASSES=17)
