"""Optional frame adapter. No B3 checkpoint or VTimeLLM import is required."""
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from torch import nn


@dataclass(frozen=True)
class TemporalConfig:
    width: int = 768
    heads: int = 8
    layers: int = 2
    ff_width: int = 1536
    use_geometry: bool = True
    distance_scale_m: float = 10.0
    time_scale_s: float = 4.0

    def __post_init__(self):
        if min(self.width, self.heads, self.layers, self.ff_width) < 1 or self.width % self.heads:
            raise ValueError("invalid attention dimensions")
        if not (0 < self.distance_scale_m < float("inf") and 0 < self.time_scale_s < float("inf")):
            raise ValueError("invalid physical scales")


def validate_frames(features, times, poses, valid, geometry=True):
    if features.ndim != 3 or not features.is_floating_point():
        raise ValueError("features must be [B,T,C] floating point")
    b, t, _ = features.shape
    if t == 0 or times.shape != (b, t) or valid.shape != (b, t) or valid.dtype != torch.bool:
        raise ValueError("time/valid axes must match feature rows")
    if any(x.device != features.device for x in (times, valid)):
        raise ValueError("frame tensors must share a device")
    if not valid.any(dim=1).all() or not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("nonempty, right-padded frame masks required")
    if not torch.isfinite(features[valid]).all() or not torch.isfinite(times[valid]).all():
        raise ValueError("nonfinite valid frame")
    if ((times[:, 1:] - times[:, :-1] <= 0) & valid[:, 1:]).any():
        raise ValueError("timestamps must increase strictly")
    if geometry:
        if poses is None or poses.shape != (b, t, 4, 4) or poses.device != features.device:
            raise ValueError("geometry mode requires aligned global_from_ego [B,T,4,4]")
        p = poses[valid].double()
        eye = torch.eye(3, device=p.device, dtype=p.dtype)
        bottom = p.new_tensor([0, 0, 0, 1]).expand(len(p), -1)
        if not torch.isfinite(p).all() or not torch.allclose(p[:, 3], bottom, atol=1e-5):
            raise ValueError("invalid rigid transform")
        r = p[:, :3, :3]
        if not torch.allclose(r.transpose(-1, -2) @ r, eye.expand_as(r), atol=1e-4) or not torch.allclose(torch.linalg.det(r), p.new_ones(len(p)), atol=1e-4):
            raise ValueError("pose rotation must be proper orthonormal")


def relative_geometry(times, poses, valid, config):
    """Pair i,j: R_i^T(p_j-p_i), relative heading and t_j-t_i.

    Subtract in float64 before casting; global coordinates/microsecond epochs
    must not be reduced to half precision before this operation.
    """
    times = torch.where(valid, times, times[:, :1]).double()
    dt = (times[:, None, :] - times[:, :, None]) / config.time_scale_s
    pairs_valid = valid[:, :, None, None] & valid[:, None, :, None]
    if not config.use_geometry:
        return dt.unsqueeze(-1).float().masked_fill(~pairs_valid, 0)
    identity = torch.eye(4, device=poses.device, dtype=torch.float64)
    poses = torch.where(valid[..., None, None], poses.double(), identity)
    r, p = poses[..., :3, :3], poses[..., :3, 3]
    delta = p[:, None, :, :] - p[:, :, None, :]
    local = torch.einsum("btji,btkj->btki", r, delta) / config.distance_scale_m
    relative_r = r.transpose(-1, -2)[:, :, None] @ r[:, None, :]
    yaw = torch.atan2(relative_r[..., 1, 0], relative_r[..., 0, 0])
    return torch.cat((local, yaw.sin()[..., None], yaw.cos()[..., None], dt[..., None]), -1).float().masked_fill(~pairs_valid, 0)


class TemporalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        dims = 6 if config.use_geometry else 1
        self.bias = nn.Sequential(nn.Linear(dims * 3, 32), nn.GELU(), nn.Linear(32, config.heads))
        self.norm1, self.norm2 = nn.LayerNorm(config.width), nn.LayerNorm(config.width)
        self.attention = nn.MultiheadAttention(config.width, config.heads, dropout=0, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(config.width, config.ff_width), nn.GELU(), nn.Linear(config.ff_width, config.width))

    def forward(self, x, relation, valid):
        encoded = torch.cat((relation, relation.sin(), relation.cos()), -1).to(x.dtype)
        bias = self.bias(encoded).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
        bias = bias.reshape(-1, x.shape[1], x.shape[1])
        z = self.norm1(x)
        x = x + self.attention(z, z, z, attn_mask=bias, need_weights=False)[0]
        return (x + self.ff(self.norm2(x))).masked_fill(~valid[..., None], 0)


class FrameTemporalAdapter(nn.Module):
    def __init__(self, config=TemporalConfig()):
        super().__init__()
        self.config = config
        dims = 6 if config.use_geometry else 1
        self.position = nn.Linear(dims * 3, config.width)
        self.blocks = nn.ModuleList([TemporalBlock(config) for _ in range(config.layers)])
        self.output = nn.Linear(config.width, config.width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, frame_times, frame_geometry=None, frame_valid_mask=None):
        valid = frame_valid_mask
        if valid is None:
            valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        validate_frames(features, frame_times, frame_geometry, valid, self.config.use_geometry)
        if features.shape[-1] != self.config.width:
            raise ValueError("feature width mismatch")
        relation = relative_geometry(frame_times, frame_geometry, valid, self.config)
        pos = relation[:, 0]
        x0 = features.masked_fill(~valid[..., None], 0)
        x = x0 + self.position(torch.cat((pos, pos.sin(), pos.cos()), -1).to(features.dtype))
        for block in self.blocks:
            x = block(x, relation, valid)
        return (x0 + self.output(x)).masked_fill(~valid[..., None], 0)

    def save(self, path):
        torch.save({"schema": "b4dl-frame-adapter-v1", "config": asdict(self.config),
                    "state_dict": self.state_dict()}, Path(path))

    @classmethod
    def load(cls, path, expected_config=None):
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if bundle["schema"] != "b4dl-frame-adapter-v1":
            raise ValueError("adapter schema mismatch")
        config = TemporalConfig(**bundle["config"])
        if expected_config is not None and config != expected_config:
            raise ValueError("adapter configuration mismatch")
        model = cls(config)
        model.load_state_dict(bundle["state_dict"], strict=True)
        return model


class TemporalProjector(nn.Module):
    """Explicit integration seam, called once before the first visual expansion.

    Do not replace VTimeLLM.mm_projector blindly: existing callers lack geometry.
    The checkpoint-specific integration supplies the three named frame tensors.
    """
    def __init__(self, projector, adapter=None):
        super().__init__()
        self.projector, self.adapter = projector, adapter

    def forward(self, features, *, frame_times=None, frame_geometry=None, frame_valid_mask=None):
        if self.adapter is not None:
            if frame_times is None:
                raise ValueError("enabled adapter requires explicit frame_times")
            features = self.adapter(features, frame_times, frame_geometry, frame_valid_mask)
        return self.projector(features)
