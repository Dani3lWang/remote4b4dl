"""Architecture and label contracts shared by training and checkpoints."""
from dataclasses import asdict, dataclass
import math


THING_CLASSES = (
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
)


@dataclass
class Grounded4DConfig:
    feature_dim: int = 128
    num_queries: int = 100
    decoder_layers: int = 3
    attention_heads: int = 8
    memory_points: int = 4096
    voxel_size: float = 0.1
    point_cloud_range: tuple = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
    class_names: tuple = THING_CLASSES
    encoder: str = "spconv"
    reference_dim: int = 4096
    track_lifetime_seconds: float = 2.5
    position_sigma: float = 3.0
    no_object_weight: float = 0.1
    mask_bce_weight: float = 5.0
    mask_dice_weight: float = 5.0
    identity_weight: float = 1.0
    reference_weight: float = 1.0
    match_points: int = 8192

    def validate(self):
        for key in ("feature_dim", "num_queries", "decoder_layers", "attention_heads",
                    "memory_points", "reference_dim", "match_points"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.feature_dim % self.attention_heads:
            raise ValueError("feature_dim must be divisible by attention_heads")
        if self.encoder not in ("spconv", "point_mlp_debug"):
            raise ValueError("encoder must be spconv or point_mlp_debug")
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be nonempty and unique")
        bounds = self.point_cloud_range
        if len(bounds) != 6 or not all(math.isfinite(v) for v in bounds):
            raise ValueError("point_cloud_range must contain six finite values")
        if any(bounds[i] >= bounds[i + 3] for i in range(3)):
            raise ValueError("point_cloud_range upper bounds must exceed lower bounds")
        for key in ("voxel_size", "track_lifetime_seconds", "position_sigma",
                    "no_object_weight", "mask_bce_weight", "mask_dice_weight",
                    "identity_weight", "reference_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key in ("point_cloud_range", "class_names"):
            if key in values:
                values[key] = tuple(values[key])
        config = cls(**values)
        config.validate()
        return config
