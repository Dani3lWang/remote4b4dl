"""True 3D voxel encoder with an inverse map into original point order."""
import math

import torch
from torch import nn


class PointMLPDebugEncoder(nn.Module):
    """CPU plumbing only; not a spatial segmentation performance baseline."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(nn.Linear(4, config.feature_dim), nn.GELU(),
                                     nn.Linear(config.feature_dim, config.feature_dim))

    def forward(self, points):
        valid = in_range(points, self.config)
        return self.network(points), valid


def in_range(points, config):
    lower, upper = points.new_tensor(config.point_cloud_range).chunk(2)
    valid = ((points[:, :3] >= lower) & (points[:, :3] < upper)).all(-1)
    if not valid.any():
        raise ValueError("no points inside point_cloud_range")
    return valid


class SparsePointEncoder(nn.Module):
    """spconv U-Net; geometric conventions adapted from the audited seg encoder."""
    def __init__(self, config):
        super().__init__()
        try:
            import spconv.pytorch as spconv
        except ImportError as exc:
            raise RuntimeError("spconv encoder requires a CUDA-matched spconv installation") from exc
        self.spconv, self.config = spconv, config
        self.input_block = spconv.SparseSequential(
            spconv.SubMConv3d(4, 32, 3, padding=1, bias=False, indice_key="g4d_base"),
            nn.BatchNorm1d(32), nn.ReLU(),
            spconv.SubMConv3d(32, 32, 3, padding=1, bias=False, indice_key="g4d_base"),
            nn.BatchNorm1d(32), nn.ReLU())
        self.down = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False, indice_key="g4d_down"),
            nn.BatchNorm1d(64), nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, padding=1, bias=False, indice_key="g4d_low"),
            nn.BatchNorm1d(64), nn.ReLU())
        self.up = spconv.SparseInverseConv3d(64, 32, 3, bias=False, indice_key="g4d_down")
        self.fuse = spconv.SparseSequential(
            spconv.SubMConv3d(64, config.feature_dim, 3, padding=1, indice_key="g4d_fuse"),
            nn.BatchNorm1d(config.feature_dim), nn.ReLU())

    def forward(self, points):
        if not points.is_cuda:
            raise RuntimeError("spconv production encoder requires CUDA; use debug encoder for CPU tests")
        valid = in_range(points, self.config)
        lower = points.new_tensor(self.config.point_cloud_range[:3])
        cells = torch.floor((points[valid, :3] - lower) / self.config.voxel_size).long()
        coordinates = torch.cat((torch.zeros_like(cells[:, :1]), cells[:, [2, 1, 0]]), 1)
        unique, inverse = torch.unique(coordinates, dim=0, sorted=True, return_inverse=True)
        features = points.new_zeros((len(unique), 4))
        features.index_add_(0, inverse, points[valid])
        features /= torch.bincount(inverse, minlength=len(unique)).to(features.dtype)[:, None]
        extent = [(hi - lo) / self.config.voxel_size for lo, hi in
                  zip(self.config.point_cloud_range[:3], self.config.point_cloud_range[3:])]
        shape = [int(math.ceil(value)) for value in extent[::-1]]
        sparse = self.spconv.SparseConvTensor(features, unique.int(), shape, 1)
        skip = self.input_block(sparse)
        decoded = self.up(self.down(skip))
        if not torch.equal(decoded.indices, skip.indices):
            raise RuntimeError("sparse skip indices changed order")
        decoded = self.fuse(decoded.replace_feature(torch.cat((decoded.features, skip.features), 1)))
        if not torch.equal(decoded.indices, unique.int()):
            raise RuntimeError("sparse encoder lost original voxel ordering")
        result = decoded.features.new_zeros((len(points), self.config.feature_dim))
        result[valid] = decoded.features[inverse]
        return result, valid
