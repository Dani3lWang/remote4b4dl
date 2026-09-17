"""Full-3D LiDAR encoder and fixed-length scene-query compressor.

The maintained LiDAR-CLIP encoder uses a 6 m-tall voxel and camera-view crop,
which is appropriate for its pooled 768-D B4DL feature but cannot recover a
point mask.  This module uses true 3-D voxels and keeps the point-to-voxel
inverse map needed to return predictions in original point order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn

from .config import ReasonSegConfig


@dataclass
class PointEncoding:
    point_features: torch.Tensor
    point_valid_mask: torch.Tensor
    batch_indices: torch.Tensor
    voxel_coordinates: torch.Tensor
    point_to_voxel: torch.Tensor


class SparseUNetPointEncoder(nn.Module):
    """Small spconv U-Net that maps raw points to 256-D point features."""

    def __init__(self, config: ReasonSegConfig, input_dim: int = 4):
        super().__init__()
        config.validate()
        self.config = config
        try:
            import spconv.pytorch as spconv
        except ImportError as exc:
            raise RuntimeError(
                "SparseUNetPointEncoder requires spconv. Install the CUDA-matched "
                "spconv package in the dedicated segmentation environment."
            ) from exc
        self._spconv = spconv
        self.input_block = spconv.SparseSequential(
            spconv.SubMConv3d(input_dim, 32, 3, padding=1, bias=False, indice_key="seg_subm0"),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(32, 32, 3, padding=1, bias=False, indice_key="seg_subm0"),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.down1 = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False, indice_key="seg_down1"),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(64, 64, 3, padding=1, bias=False, indice_key="seg_subm1"),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.down2 = spconv.SparseSequential(
            spconv.SparseConv3d(64, 128, 3, stride=2, padding=1, bias=False, indice_key="seg_down2"),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(128, 128, 3, padding=1, bias=False, indice_key="seg_subm2"),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.up2 = spconv.SparseInverseConv3d(
            128, 64, 3, bias=False, indice_key="seg_down2"
        )
        self.fuse1 = spconv.SparseSequential(
            spconv.SubMConv3d(128, 64, 3, padding=1, bias=False, indice_key="seg_up_subm1"),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.up1 = spconv.SparseInverseConv3d(
            64, 32, 3, bias=False, indice_key="seg_down1"
        )
        self.fuse0 = spconv.SparseSequential(
            spconv.SubMConv3d(64, 64, 3, padding=1, bias=False, indice_key="seg_up_subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(
                64,
                config.point_feature_dim,
                1,
                bias=True,
                indice_key="seg_output",
            ),
        )

    def forward(self, points: torch.Tensor, batch_indices: torch.Tensor) -> PointEncoding:
        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError("points must have shape [N, 4+] with XYZ and reflectance")
        if batch_indices.shape != (points.shape[0],):
            raise ValueError("batch_indices must contain one value per point")
        if points.numel() == 0:
            raise ValueError("cannot encode an empty point batch")

        voxel_features, coordinates, inverse, valid = self._voxelize(points, batch_indices)
        batch_size = int(batch_indices.max().item()) + 1
        valid_per_batch = torch.bincount(
            batch_indices[valid].long(), minlength=batch_size
        )
        if (valid_per_batch == 0).any():
            bad = torch.where(valid_per_batch == 0)[0].tolist()
            raise ValueError(f"all points are outside point_cloud_range for batches {bad}")
        sparse_shape = self._sparse_shape()
        sparse = self._spconv.SparseConvTensor(
            voxel_features,
            coordinates.int(),
            sparse_shape,
            batch_size,
        )
        skip0 = self.input_block(sparse)
        skip1 = self.down1(skip0)
        encoded = self.down2(skip1)

        decoded1 = self.up2(encoded)
        decoded1 = _concat_sparse(decoded1, skip1)
        decoded1 = self.fuse1(decoded1)
        decoded0 = self.up1(decoded1)
        decoded0 = _concat_sparse(decoded0, skip0)
        decoded0 = self.fuse0(decoded0)

        if not torch.equal(decoded0.indices, coordinates.int()):
            raise RuntimeError("spconv inverse-convolution changed base voxel order")
        point_features = torch.zeros(
            points.shape[0],
            self.config.point_feature_dim,
            dtype=decoded0.features.dtype,
            device=decoded0.features.device,
        )
        point_features[valid] = decoded0.features[inverse]
        point_to_voxel = torch.full(
            (points.shape[0],), -1, dtype=torch.long, device=points.device
        )
        point_to_voxel[valid] = inverse
        return PointEncoding(
            point_features=point_features,
            point_valid_mask=valid,
            batch_indices=batch_indices,
            voxel_coordinates=coordinates,
            point_to_voxel=point_to_voxel,
        )

    def _sparse_shape(self) -> List[int]:
        lower = self.config.point_cloud_range[:3]
        upper = self.config.point_cloud_range[3:]
        xyz = [int(round((hi - lo) / self.config.voxel_size)) for lo, hi in zip(lower, upper)]
        # spconv coordinates and spatial shape use Z, Y, X order.
        return [xyz[2], xyz[1], xyz[0]]

    def _voxelize(
        self,
        points: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lower = points.new_tensor(self.config.point_cloud_range[:3])
        upper = points.new_tensor(self.config.point_cloud_range[3:])
        valid = ((points[:, :3] >= lower) & (points[:, :3] < upper)).all(dim=1)
        if not valid.any():
            raise ValueError("all points are outside point_cloud_range")
        xyz_indices = torch.floor(
            (points[valid, :3] - lower) / self.config.voxel_size
        ).long()
        coordinates = torch.cat(
            [batch_indices[valid, None].long(), xyz_indices[:, [2, 1, 0]]], dim=1
        )
        unique, inverse = torch.unique(
            coordinates, dim=0, sorted=True, return_inverse=True
        )
        values = points[valid, :4]
        sums = torch.zeros(
            unique.shape[0], values.shape[1], dtype=values.dtype, device=values.device
        )
        sums.index_add_(0, inverse, values)
        counts = torch.bincount(inverse, minlength=unique.shape[0]).to(values.dtype)
        features = sums / counts[:, None].clamp_min(1.0)
        return features, unique, inverse, valid


class SceneQueryCompressor(nn.Module):
    """Compress variable-length point features to fixed scene tokens for Vicuna."""

    def __init__(self, config: ReasonSegConfig):
        super().__init__()
        self.config = config
        self.queries = nn.Parameter(
            torch.empty(config.num_scene_queries, config.point_feature_dim)
        )
        nn.init.normal_(self.queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            config.point_feature_dim,
            config.decoder_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.point_feature_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(config.point_feature_dim),
            nn.Linear(config.point_feature_dim, config.decoder_ffn_dim),
            nn.GELU(),
            nn.Linear(config.decoder_ffn_dim, config.point_feature_dim),
        )
        # Keep the existing Stage-1 768 -> Vicuna projector in the path.  The
        # VTimeLLM multimodal preparation code expects ``images`` in this
        # dimension and applies ``mm_projector`` itself.
        self.to_llm = nn.Linear(config.point_feature_dim, config.multimodal_input_dim)

    def forward(
        self,
        padded_point_features: torch.Tensor,
        point_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if padded_point_features.shape[:2] != point_valid_mask.shape:
            raise ValueError("point_valid_mask does not match padded point features")
        batch_size = padded_point_features.shape[0]
        queries = self.queries[None, :, :].expand(batch_size, -1, -1)
        attended, _ = self.cross_attention(
            queries,
            padded_point_features,
            padded_point_features,
            key_padding_mask=~point_valid_mask.bool(),
            need_weights=False,
        )
        features = self.norm1(queries + attended)
        features = features + self.ffn(features)
        return self.to_llm(features)


def pad_point_features(
    encoding: PointEncoding,
) -> tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Convert flat encoded points into [B, Nmax, C] while retaining indices."""

    batch_size = int(encoding.batch_indices.max().item()) + 1
    per_batch = [
        torch.where(encoding.batch_indices == batch_index)[0]
        for batch_index in range(batch_size)
    ]
    max_points = max(int(indices.numel()) for indices in per_batch)
    features = encoding.point_features.new_zeros(
        batch_size, max_points, encoding.point_features.shape[-1]
    )
    valid = torch.zeros(
        batch_size, max_points, dtype=torch.bool, device=encoding.point_features.device
    )
    for batch_index, indices in enumerate(per_batch):
        count = int(indices.numel())
        features[batch_index, :count] = encoding.point_features[indices]
        valid[batch_index, :count] = encoding.point_valid_mask[indices]
    return features, valid, per_batch


def _concat_sparse(first, second):
    if not torch.equal(first.indices, second.indices):
        raise RuntimeError("sparse U-Net skip connection indices do not align")
    return first.replace_feature(torch.cat([first.features, second.features], dim=1))
