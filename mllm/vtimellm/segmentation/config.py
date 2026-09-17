"""Configuration shared by segmentation training, inference and checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


NUSCENES_THING_CLASSES: List[str] = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
]


@dataclass
class ReasonSegConfig:
    """Architecture and preprocessing contract for the first segmentation model."""

    hidden_size: int = 4096
    multimodal_input_dim: int = 768
    point_feature_dim: int = 256
    decoder_layers: int = 6
    decoder_heads: int = 8
    decoder_ffn_dim: int = 1024
    dropout: float = 0.1
    max_objects: int = 8
    num_scene_queries: int = 128
    voxel_size: float = 0.1
    point_cloud_range: List[float] = field(
        default_factory=lambda: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    )
    coarse_radius_margin_m: float = 2.0
    class_names: List[str] = field(default_factory=lambda: list(NUSCENES_THING_CLASSES))
    text_loss_weight: float = 1.0
    seg_loss_weight: float = 1.0
    loc_loss_weight: float = 0.5
    class_loss_weight: float = 1.0

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, object]) -> "ReasonSegConfig":
        return cls(**values)

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.point_feature_dim <= 0 or self.multimodal_input_dim <= 0:
            raise ValueError("hidden_size, multimodal_input_dim and point_feature_dim must be positive")
        if self.decoder_layers <= 0 or self.decoder_heads <= 0:
            raise ValueError("decoder_layers and decoder_heads must be positive")
        if self.point_feature_dim % self.decoder_heads:
            raise ValueError("point_feature_dim must be divisible by decoder_heads")
        if self.max_objects <= 0 or self.num_scene_queries <= 0:
            raise ValueError("max_objects and num_scene_queries must be positive")
        if len(self.point_cloud_range) != 6:
            raise ValueError("point_cloud_range must contain six values")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique")
        if not self.class_names:
            raise ValueError("class_names must not be empty")
