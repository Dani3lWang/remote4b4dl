"""Inverse squared-distance interpolation, as in MaskPLS (MIT, see NOTICE.md)."""
import torch
from torch import nn


class KNNInterpolation(nn.Module):
    def __init__(self, k=3, backend="keops", chunk_size=512):
        super().__init__()
        self.k, self.backend, self.chunk_size = k, backend, chunk_size

    def forward(self, voxel_coordinates, features, point_coordinates):
        if not len(voxel_coordinates):
            raise ValueError("cannot interpolate an empty sparse feature map")
        k = min(self.k, len(voxel_coordinates))
        # Coordinate neighbours are nondifferentiable; feature interpolation remains differentiable.
        with torch.no_grad():
            source = voxel_coordinates.to(device=features.device, dtype=torch.float32).contiguous()
            target = point_coordinates.to(device=features.device, dtype=torch.float32).contiguous()
            if self.backend == "keops":
                from pykeops.torch import LazyTensor
                distance = ((LazyTensor(target[:, None, :]) - LazyTensor(source[None, :, :])) ** 2).sum(-1)
                distances, indices = distance.Kmin_argKmin(k, dim=1)
            elif self.backend == "torch":
                distances, indices = [], []
                # Bounded [chunk, voxels] allocation; CPU fixtures use this exact fallback.
                for part in target.split(self.chunk_size):
                    squared = (part[:, None, :] - source[None, :, :]).square().sum(-1)
                    values, ids = squared.topk(k, largest=False, sorted=True)
                    distances.append(values)
                    indices.append(ids)
                distances, indices = torch.cat(distances), torch.cat(indices)
            else:
                raise ValueError("unsupported KNN backend")
            inverse = (distances + 1e-8).reciprocal()
            weights = inverse / inverse.sum(-1, keepdim=True)
        return (features[indices.long()] * weights.to(features.dtype).unsqueeze(-1)).sum(1)
