"""Detection/track queries plus an optional phrase-to-query alignment head.

This is a MaskPLS/Mask4D-inspired prototype, not a reproduction of their
backbones, losses or published benchmark results.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import PointMLPDebugEncoder, SparsePointEncoder


@dataclass
class TrackMemory:
    token: str
    feature: torch.Tensor
    center: torch.Tensor
    velocity: torch.Tensor
    timestamp_us: int


class MaskLayer(nn.Module):
    def __init__(self, dimension, heads):
        super().__init__()
        self.cross = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.self_attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(dimension) for _ in range(3)]
        self.ffn = nn.Sequential(nn.Linear(dimension, dimension * 4), nn.GELU(),
                                 nn.Linear(dimension * 4, dimension))

    def forward(self, queries, memory, bias):
        attended = self.cross(queries[None], memory[None], memory[None],
                              attn_mask=bias, need_weights=False)[0][0]
        queries = self.norm1(queries + attended)
        queries = self.norm2(queries + self.self_attention(
            queries[None], queries[None], queries[None], need_weights=False)[0][0])
        return self.norm3(queries + self.ffn(queries))


class Grounded4DModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        encoder = SparsePointEncoder if config.encoder == "spconv" else PointMLPDebugEncoder
        self.encoder = encoder(config)
        dimension = config.feature_dim
        self.detect_queries = nn.Parameter(torch.randn(config.num_queries, dimension) * 0.02)
        self.position_projection = nn.Linear(24, dimension)
        self.time_projection = nn.Sequential(nn.Linear(1, dimension), nn.Tanh())
        self.layers = nn.ModuleList([MaskLayer(dimension, config.attention_heads)
                                     for _ in range(config.decoder_layers)])
        self.mask_projection = nn.Linear(dimension, dimension)
        self.classifier = nn.Linear(dimension, len(config.class_names) + 1)
        self.reference_projection = nn.Sequential(nn.Linear(config.reference_dim, dimension),
                                                 nn.GELU(), nn.Linear(dimension, dimension))
        self.no_referent = nn.Parameter(torch.randn(dimension) * 0.02)

    def forward(self, frame, tracks=()):
        points = frame["points"]
        if points.ndim != 2 or points.shape[1] != 4 or not torch.isfinite(points).all():
            raise ValueError("model points must be finite [N,4] XYZ/intensity")
        features, valid = self.encoder(points)
        frequencies = points.new_tensor([1.0, 2.0, 4.0, 8.0])
        angles = points[:, :3, None] / 51.2 * frequencies * math.pi
        position = torch.cat((angles.sin(), angles.cos()), -1).flatten(1)
        features = features + self.position_projection(position)
        indices = valid.nonzero(as_tuple=True)[0]
        if len(indices) > self.config.memory_points:
            indices = indices[torch.linspace(0, len(indices) - 1, self.config.memory_points,
                                            device=points.device).long()]
        memory = features[indices]
        stamp = frame["timestamp_us"]
        active = [track for track in tracks if
                  0 <= (stamp - track.timestamp_us) / 1e6 <= self.config.track_lifetime_seconds]
        for track in tracks:
            if stamp <= track.timestamp_us:
                raise ValueError("query memory must come from earlier frames")
        queries = self.detect_queries
        position_bias = points.new_zeros((len(active) + len(queries), len(memory)))
        if active:
            deltas = points.new_tensor([(stamp - track.timestamp_us) / 1e6 for track in active])
            track_queries = torch.stack([track.feature for track in active])
            track_queries = track_queries + self.time_projection(deltas[:, None])
            queries = torch.cat((track_queries, queries), 0)
            centers = torch.stack([track.center for track in active])
            velocities = torch.stack([track.velocity for track in active])
            predicted_centers = centers + velocities * deltas[:, None]
            position_bias[:len(active)] = -((points[indices, :3][None] - predicted_centers[:, None])
                                           .square().sum(-1) / (2 * self.config.position_sigma ** 2))
        auxiliary = []
        for layer in self.layers:
            logits = self.mask_projection(queries) @ memory.T / math.sqrt(self.config.feature_dim)
            blocked = logits.detach() < 0
            blocked[blocked.all(-1)] = False  # an empty mask must retain access to geometry
            attention_bias = position_bias.masked_fill(blocked, -torch.inf)
            queries = layer(queries, memory, attention_bias)
            full_logits = self.mask_projection(queries) @ features.T / math.sqrt(self.config.feature_dim)
            full_logits = full_logits.masked_fill(~valid[None], -1e4)
            auxiliary.append(dict(mask_logits=full_logits, class_logits=self.classifier(queries)))
        output = dict(**auxiliary[-1], queries=queries, valid_points=valid,
                      track_tokens=[track.token for track in active], auxiliary=auxiliary[:-1])
        references = frame.get("reference_features")
        if references is not None:
            candidates = torch.cat((queries, self.no_referent[None]), 0)
            output["reference_logits"] = (F.normalize(self.reference_projection(references), dim=-1)
                                          @ F.normalize(candidates, dim=-1).T) / 0.1
        return output

    def update_memory(self, frame, output, assignment, previous=()):
        """Training-only GT query assignment; centers/velocity use predicted masks.

        Never expose this oracle instance-token mapping as inference tracking.
        """
        previous = {track.token: track for track in previous if
                    (frame["timestamp_us"] - track.timestamp_us) / 1e6 <= self.config.track_lifetime_seconds}
        for target_index, query_index in assignment.items():
            token = frame["instance_tokens"][target_index]
            weights = output["mask_logits"][query_index].detach().sigmoid() * output["valid_points"]
            center = (weights[:, None] * frame["points"][:, :3]).sum(0) / weights.sum().clamp_min(1e-6)
            velocity = torch.zeros_like(center)
            if token in previous:
                old = previous[token]
                delta = (frame["timestamp_us"] - old.timestamp_us) / 1e6
                velocity = (center - old.center) / delta
            previous[token] = TrackMemory(token, output["queries"][query_index], center, velocity,
                                          frame["timestamp_us"])
        return list(previous.values())
