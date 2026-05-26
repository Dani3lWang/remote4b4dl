"""Compatibility shim for running old mmdet3d (0.x, sst/) against mmcv 2.x + mmengine.

Import this module BEFORE importing anything from mmdet3d.
It patches mmcv, mmdet, and provides pure-Python fallbacks for CUDA ops.
"""

import functools
import importlib
import inspect as _inspect
import os
import sys
from types import ModuleType
from typing import Tuple

# Workaround for Python 3.10.20 + torch 2.8.x inspect module bug:
# When mmengine imports transformers.optimization → torch.distributed.tensor
# → torch.library @register → _register_fake → inspect.getmodule() →
# getabsfile → getsourcefile, the getfile() call inside getsourcefile can
# return a non-string (function object) for certain module objects in this
# CPython version. This causes .endswith() to crash.
# Monkey-patch inspect.getabsfile and inspect.getsourcefile to be robust.
_orig_getfile = _inspect.getfile
def _patched_getfile(obj):
    result = _orig_getfile(obj)
    if not isinstance(result, str):
        raise TypeError(f'getfile({obj!r}) returned {type(result).__name__}, expected str')
    return result
_inspect.getfile = _patched_getfile

import mmcv
import mmdet
import mmengine.config
import mmengine.registry
import torch


# Meta-path finder: auto-creates stub modules for any MISSING submodule under
# mmdet3d.* / mmdet.* / mmcv.* that doesn't exist on disk.  This avoids having
# to enumerate every removed submodule individually.
_MISSING_STUB_PREFIXES = ("mmdet3d.", "mmdet.", "mmcv.", "mmseg.")

# Prevent mmcv._ext from loading if incompatible with current PyTorch ABI.
# Use a proper mock that returns dummy functions for any attribute access,
# so that real mmcv.ops submodules (e.g. active_rotated_filter.py) can
# import their CUDA kernels without triggering AssertionError.
class _MockExtModule(ModuleType):
    """A module-like object that returns dummy callables for any attribute.

    Raises AttributeError for dunder attributes (e.g. __file__) so that
    inspect.getfile/hasattr behave correctly — this module has no source file.
    """
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None
sys.modules.setdefault("mmcv._ext", _MockExtModule("mmcv._ext"))

class _AutoStubFinder:
    """Auto-creates stub modules for missing subpackages under known prefixes."""

    @classmethod
    def find_spec(cls, fullname, path, target=None):
        if fullname in sys.modules:
            return None  # already loaded
        if not any(fullname.startswith(p) for p in _MISSING_STUB_PREFIXES):
            return None  # not our domain

        # Try to find the real module first
        for finder in sys.meta_path:
            if finder is cls:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    return None  # real module exists, don't stub
            except Exception:
                continue

        # Real module not found — create a stub
        return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=False)


class _StubLoader:
    """Loader that always returns an empty stub module."""

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _AutoStubFinder)

# Ensure old SST mmdet3d is importable (sst/ directory containing mmdet3d/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SST_PATH = os.path.join(_PROJECT_ROOT, "sst")
if _SST_PATH not in sys.path:
    sys.path.insert(0, _SST_PATH)


# ---------------------------------------------------------------------------
# 1. mmcv.Config → mmengine.config.Config
# ---------------------------------------------------------------------------
mmcv.Config = mmengine.config.Config
from mmengine.config import ConfigDict
mmcv.ConfigDict = ConfigDict
mmcv.is_tuple_of = lambda x, t: isinstance(x, tuple) and all(isinstance(e, t) for e in x)
mmcv.is_list_of = lambda x, t: isinstance(x, list) and all(isinstance(e, t) for e in x)
mmcv.is_seq_of = lambda x, t, length=None: isinstance(x, (list, tuple)) and all(isinstance(e, t) for e in x) and (length is None or len(x) == length)


# ---------------------------------------------------------------------------
# 2. Pure-Python dynamic_voxelize / hard_voxelize (defined early, used by stubs below)
# ---------------------------------------------------------------------------
def _dynamic_voxelize_py(
    points: torch.Tensor,
    coors: torch.Tensor,
    voxel_size: Tuple[float, float, float],
    coors_range: Tuple[float, ...],
    num_features: int = 3,
) -> None:
    vs = points.new_tensor(voxel_size)
    cr_min = points.new_tensor(coors_range[:3])
    coors_float = (points[:, :num_features] - cr_min[None, :]) / vs[None, :]
    coors_int = torch.floor(coors_float).to(torch.int32)
    coors[:, 2] = coors_int[:, 0]  # x
    coors[:, 1] = coors_int[:, 1]  # y
    coors[:, 0] = coors_int[:, 2]  # z


def _hard_voxelize_py(
    points: torch.Tensor,
    voxels: torch.Tensor,
    coors: torch.Tensor,
    num_points_per_voxel: torch.Tensor,
    voxel_size: Tuple[float, float, float],
    coors_range: Tuple[float, ...],
    max_points: int,
    max_voxels: int,
    num_features: int = 3,
) -> int:
    vs = torch.tensor(voxel_size, device=points.device)
    cr = torch.tensor(coors_range, device=points.device)
    grid_size = torch.round((cr[3:] - cr[:3]) / vs).long()
    nx, ny, nz = grid_size[0].item(), grid_size[1].item(), grid_size[2].item()
    xyz = points[:, :num_features]
    idx = torch.floor((xyz - cr[:3][None, :]) / vs[None, :]).long()
    idx[:, 0] = idx[:, 0].clamp(0, nx - 1)
    idx[:, 1] = idx[:, 1].clamp(0, ny - 1)
    idx[:, 2] = idx[:, 2].clamp(0, nz - 1)
    flat_idx = idx[:, 0] + idx[:, 1] * nx + idx[:, 2] * nx * ny
    unique_idx, inverse = torch.unique(flat_idx, return_inverse=True)
    voxel_num = min(len(unique_idx), max_voxels)
    if voxel_num < len(unique_idx):
        unique_idx = unique_idx[:voxel_num]
        mask = inverse < voxel_num
        inverse = inverse[mask]
        _, inverse = torch.unique(inverse, return_inverse=True)
    for i in range(voxel_num):
        pt_mask = inverse == i
        pts_in_voxel = points[pt_mask][:max_points]
        n = pts_in_voxel.shape[0]
        voxels[i, :n, :] = pts_in_voxel
        num_points_per_voxel[i] = n
        u = unique_idx[i]
        coors[i, 2] = u % nx
        coors[i, 1] = (u // nx) % ny
        coors[i, 0] = u // (nx * ny)
    return voxel_num


# ---------------------------------------------------------------------------
# 3. mmcv.utils / mmdet.core stubs (must be ready before mmdet3d import)
# ---------------------------------------------------------------------------
import mmcv.utils
mmcv.utils.Registry = mmengine.registry.Registry
# Additional mmcv.utils symbols moved to mmengine
from mmengine.logging import print_log
mmcv.utils.print_log = print_log
from mmengine.registry import build_from_cfg
mmcv.utils.build_from_cfg = build_from_cfg
from mmengine.logging import MMLogger
mmcv.utils.get_logger = lambda name=None: MMLogger.get_current_instance() or MMLogger(name or "mmdet3d")
mmcv.utils.get_git_hash = lambda: ""
mmcv.utils.collect_env = lambda: ""
import mmdet.models
import mmdet.registry

# Stub module class that auto-creates missing attributes (avoids whack-a-mole)
class _StubModule(ModuleType):
    def __getattr__(self, name):
        # Only stub non-dunder names.  Dunder attrs fall through to the normal
        # ModuleType machinery (which raises AttributeError if truly missing).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        stub = type(name, (), {})
        setattr(self, name, stub)
        return stub

# Pre-stub mmcv.ops submodules so that mmdet imports don't trigger the real
# mmcv/ops/__init__.py which tries to load the incompatible _ext shared library.
for _ops_sub in ["mmcv.ops", "mmcv.ops.nms", "mmcv.ops.roi_align",
                 "mmcv.ops.iou3d", "mmcv.ops.pointnet2", "mmcv.ops.diff_iou_rotated",
                 "mmcv.ops.spconv"]:
    sys.modules.setdefault(_ops_sub, _StubModule(_ops_sub))
mmcv.ops = sys.modules["mmcv.ops"]

# mmdet.datasets stubs for removed submodules in mmdet 3.x
import mmdet.datasets
mmdet.datasets.DATASETS = mmengine.registry.Registry("datasets")
mmdet.datasets.PIPELINES = mmengine.registry.Registry("pipelines")
for _ds_mod_path in [
    "mmdet.datasets.builder",
    "mmdet.datasets.dataset_wrappers",
    "mmdet.datasets.pipelines",
]:
    if _ds_mod_path not in sys.modules:
        sys.modules[_ds_mod_path] = _StubModule(_ds_mod_path)

_mmdet_core_stubs = {
    "mmdet.core": ["bbox2result", "build_anchor_generator", "build_assigner",
                   "build_sampler", "build_bbox_coder", "multi_apply",
                   "eval_map", "images_to_levels"],
    "mmdet.core.anchor": ["ANCHOR_GENERATORS", "build_anchor_generator"],
    "mmdet.core.bbox": ["AssignResult", "BaseAssigner", "MaxIoUAssigner",
                        "BaseBBoxCoder", "bbox_overlaps", "build_bbox_coder"],
    "mmdet.core.bbox.builder": ["BBOX_CODERS", "BBOX_SAMPLERS"],
    "mmdet.core.bbox.iou_calculators": [],
    "mmdet.core.bbox.iou_calculators.builder": ["IOU_CALCULATORS"],
    "mmdet.core.bbox.samplers": [
        "BaseSampler", "CombinedSampler", "SamplingResult",
        "InstanceBalancedPosSampler", "IoUBalancedNegSampler", "OHEMSampler",
        "PseudoSampler", "RandomSampler",
    ],
    "mmdet.core.post_processing": [
        "merge_aug_bboxes", "merge_aug_masks", "merge_aug_proposals",
        "merge_aug_scores", "multiclass_nms",
    ],
}
for _mod_name, _attrs in _mmdet_core_stubs.items():
    _mod = ModuleType(_mod_name)
    for _a in _attrs:
        _found = getattr(mmdet.registry, _a, None) or getattr(mmdet.models, _a, None)
        setattr(_mod, _a, _found if _found is not None else type(_a, (), {}))
    sys.modules[_mod_name] = _mod


# ---------------------------------------------------------------------------
# 3.5. Pre-stub non-essential mmdet3d subpackages so their heavy deps
#      (numba, lyft_dataset_sdk, tensorflow, open3d, etc.) are never imported.
#      Only the SST backbone and core bbox/anchor/points/voxel modules are needed.
# ---------------------------------------------------------------------------
_MMDET3D_NONESSENTIAL_SUBPACKS = [
    # -- core subpackages with heavy deps (numba, lyft_sdk, tensorflow, open3d)
    "mmdet3d.core.evaluation",
    "mmdet3d.core.evaluation.kitti_utils",
    "mmdet3d.core.visualizer",
    "mmdet3d.core.post_processing",
    # -- datasets (all of them bring in heavy third-party deps)
    "mmdet3d.datasets",
    "mmdet3d.datasets.kitti_dataset",
    "mmdet3d.datasets.kitti_mono_dataset",
    "mmdet3d.datasets.lyft_dataset",
    "mmdet3d.datasets.nuscenes_dataset",
    "mmdet3d.datasets.nuscenes_mono_dataset",
    "mmdet3d.datasets.s3dis_dataset",
    "mmdet3d.datasets.scannet_dataset",
    "mmdet3d.datasets.sunrgbd_dataset",
    "mmdet3d.datasets.waymo_dataset",
    "mmdet3d.datasets.pipelines",
    "mmdet3d.datasets.samplers",
    "mmdet3d.datasets.builder",
    "mmdet3d.datasets.custom_3d",
    "mmdet3d.datasets.custom_3d_seg",
    # -- model components NOT used by SST encoder
    "mmdet3d.models.roi_heads",
    "mmdet3d.models.decode_heads",
    "mmdet3d.models.segmentors",
    "mmdet3d.models.backbones.second",
    "mmdet3d.models.backbones.voxelnet",
    "mmdet3d.models.backbones.centerpoint",
    "mmdet3d.models.backbones.hrnet",
    "mmdet3d.models.backbones.multi_backbone",
    "mmdet3d.models.backbones.nostem_regnet",
    "mmdet3d.models.backbones.sparse_unet",
    "mmdet3d.models.backbones.pointnet2",
    "mmdet3d.models.backbones.pillar_encoder",
    "mmdet3d.models.losses",
    # NOTE: fusion_layers is a package imported by detectors — do NOT stub
    "mmdet3d.models.utils",
    "mmdet3d.models.votenet",
    "mmdet3d.models.h3dnet",
    "mmdet3d.models.groupfree3dnet",
    "mmdet3d.models.imvotenet",
    "mmdet3d.models.parta2",
    "mmdet3d.models.ssd3dnet",
    "mmdet3d.models.voxelnet",
    "mmdet3d.models.centerpoint",
]
for _sub in _MMDET3D_NONESSENTIAL_SUBPACKS:
    sys.modules.setdefault(_sub, _StubModule(_sub))

# ---------------------------------------------------------------------------
# 4. Stub C/CUDA extension modules (old mmdet3d.ops.*)
# ---------------------------------------------------------------------------
for _ext_mod in [
    "mmdet3d.ops.ball_query.ball_query_ext",
    "mmdet3d.ops.gather_points.gather_points_ext",
    "mmdet3d.ops.furthest_point_sample.furthest_point_sample_ext",
    "mmdet3d.ops.interpolate.interpolate_ext",
    "mmdet3d.ops.paconv.assign_score_withk_ext",
    "mmdet3d.ops.iou3d.iou3d_cuda",
    "mmdet3d.ops.knn.knn_ext",
    "mmdet3d.ops.group_points.group_points_ext",
    "mmdet3d.ops.roiaware_pool3d.roiaware_pool3d_ext",
    "mmdet3d.ops.spconv.sparse_conv_ext",
    "mmdet3d.ops.voxel.voxel_layer",
]:
    _m = ModuleType(_ext_mod)
    sys.modules[_ext_mod] = _m

def _dynamic_point_to_voxel_forward_py(feats, coors, reduce_type="max"):
    """Pure-Python replacement for CUDA dynamic_point_to_voxel_forward.

    Groups point features by their voxel coordinates and reduces within each voxel.

    Args:
        feats: (N, C) float tensor, point features
        coors: (N, ndim) int32 tensor, voxel coordinates
        reduce_type: 'max', 'sum', or 'mean'
    Returns:
        voxel_feats: (M, C) reduced features
        voxel_coors: (M, ndim) unique voxel coordinates
        point2voxel_map: (N,) int64, mapping from point index to voxel index
        voxel_points_count: (M,) int32, number of points in each voxel
    """
    if coors.size(-1) == 0:
        return feats, coors, torch.zeros(0, dtype=torch.int64, device=feats.device), torch.zeros(0, dtype=torch.int32, device=feats.device)

    # Flatten coors to 1D to use unique
    coors_flat = coors[:, 0].long()
    for d in range(1, coors.size(1)):
        coors_flat = coors_flat * (coors[:, d].max() + 1) + coors[:, d].long()

    unique_coors, inverse_indices, counts = torch.unique(
        coors_flat, return_inverse=True, return_counts=True)

    n_voxels = unique_coors.size(0)
    n_points, n_channels = feats.shape
    device = feats.device

    voxel_feats = feats.new_zeros((n_voxels, n_channels))
    voxel_coors_out = torch.zeros((n_voxels, coors.size(1)), dtype=coors.dtype, device=device)

    for v in range(n_voxels):
        mask = inverse_indices == v
        pts = feats[mask]
        if reduce_type == "max":
            voxel_feats[v] = pts.max(dim=0)[0]
        elif reduce_type == "sum":
            voxel_feats[v] = pts.sum(dim=0)
        elif reduce_type == "mean":
            voxel_feats[v] = pts.mean(dim=0)

        # Recover the "first" coordinate assignment
        first_idx = torch.where(mask)[0][0]
        voxel_coors_out[v] = coors[first_idx]

    point2voxel_map = inverse_indices
    voxel_points_count = counts.int()

    return voxel_feats, voxel_coors_out, point2voxel_map, voxel_points_count


def _dynamic_point_to_voxel_backward_py(grad_feats, grad_voxel_feats, feats,
                                         voxel_feats, point2voxel_map,
                                         voxel_points_count, reduce_type):
    """Pure-Python backward for dynamic scatter (no-op for inference)."""
    pass


# Pre-populate voxel_layer stub with needed symbols
_vl = sys.modules["mmdet3d.ops.voxel.voxel_layer"]
_vl.dynamic_voxelize = _dynamic_voxelize_py
_vl.hard_voxelize = _hard_voxelize_py
_vl.dynamic_point_to_voxel_forward = _dynamic_point_to_voxel_forward_py
_vl.dynamic_point_to_voxel_backward = _dynamic_point_to_voxel_backward_py


# ---------------------------------------------------------------------------
# 5. mmcv.runner stub (force_fp32, auto_fp16, BaseModule, etc.)
# ---------------------------------------------------------------------------
class _FakeRunnerModule(ModuleType):
    @staticmethod
    def force_fp32(apply_to=None, out_fp32=False, out_fp16=False):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def auto_fp16(apply_to=None, out_fp32=False):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def load_checkpoint(model, filename, map_location=None, strict=False, logger=None, revise_keys=None):
        from mmengine.runner import load_checkpoint as _load
        return _load(model, filename, map_location=map_location, strict=strict, logger=logger, revise_keys=revise_keys)

from mmengine.model import BaseModule
_FakeRunnerModule.BaseModule = BaseModule

_runner_mod = _FakeRunnerModule("mmcv.runner")
sys.modules["mmcv.runner"] = _runner_mod
mmcv.runner = _runner_mod

# mmcv.parallel stub
_parallel_mod = ModuleType("mmcv.parallel")
_parallel_mod.collate = lambda samples, samples_per_gpu=1: samples
_parallel_mod.scatter = lambda input, target_gpus, dim=0: input
sys.modules["mmcv.parallel"] = _parallel_mod
mmcv.parallel = _parallel_mod


# ---------------------------------------------------------------------------
# 6. mmdet.models.builder stub
# ---------------------------------------------------------------------------
_BUILDER_REGISTRY_NAMES = [
    "BACKBONES", "DETECTORS", "HEADS", "LOSSES", "NECKS",
    "ROI_EXTRACTORS", "SHARED_HEADS",
]
from mmdet.registry import MODELS as _MMDET_MODELS

_builder_mod = ModuleType("mmdet.models.builder")
for _name in _BUILDER_REGISTRY_NAMES:
    setattr(_builder_mod, _name, _MMDET_MODELS)
for _fn_name in ["build_backbone", "build_neck", "build_head", "build_shared_head",
                 "build_roi_extractor", "build_loss", "build_detector"]:
    setattr(_builder_mod, _fn_name, lambda cfg, **kw: _MMDET_MODELS.build(cfg))
sys.modules["mmdet.models.builder"] = _builder_mod
mmdet.models.builder = _builder_mod
for _name in _BUILDER_REGISTRY_NAMES:
    setattr(mmdet.models, _name, _MMDET_MODELS)
# Also add build functions to mmdet.models (used by `from mmdet.models import build_backbone`)
for _fn_name in ["build_backbone", "build_neck", "build_head", "build_shared_head"]:
    setattr(mmdet.models, _fn_name, lambda cfg, **kw: _MMDET_MODELS.build(cfg))


# ---------------------------------------------------------------------------
# 7. mmcv.cnn additions for backward compat
# ---------------------------------------------------------------------------
if not hasattr(mmcv.cnn, "MODELS"):
    mmcv.cnn.MODELS = mmengine.registry.MODELS
# NORM_LAYERS must be MODELS so build_norm_layer can find naiveSyncBN types.
# CONV_LAYERS stays separate to avoid spconv conflicts with real mmcv.
mmcv.cnn.NORM_LAYERS = mmengine.registry.MODELS
if not hasattr(mmcv.cnn, "CONV_LAYERS"):
    mmcv.cnn.CONV_LAYERS = mmengine.registry.Registry("conv_layers")

from mmengine.model import (constant_init, kaiming_init, normal_init, xavier_init,
                              bias_init_with_prob, caffe2_xavier_init)
for _init_func in [constant_init, kaiming_init, normal_init, xavier_init,
                   bias_init_with_prob, caffe2_xavier_init]:
    _init_name = _init_func.__name__
    if not hasattr(mmcv.cnn, _init_name):
        setattr(mmcv.cnn, _init_name, _init_func)

# Patch mmcv.cnn.bricks.transformer / registry with removed symbols
_br_mod = ModuleType("mmcv.cnn.bricks.registry")
for _sym in ["ATTENTION", "TRANSFORMER_LAYER", "TRANSFORMER_LAYER_SEQUENCE",
             "FEEDFORWARD_NETWORK", "POSITIONAL_ENCODING"]:
    setattr(_br_mod, _sym, mmengine.registry.Registry(_sym.lower()))
sys.modules["mmcv.cnn.bricks.registry"] = _br_mod

import mmcv.cnn.bricks.transformer as _trans
for _sym in ["POSITIONAL_ENCODING", "MultiheadAttention",
             "build_positional_encoding"]:
    if not hasattr(_trans, _sym):
        setattr(_trans, _sym, mmengine.registry.Registry(_sym.lower()))

# mmcv.ops stubs
import mmcv.ops
mmcv.ops.RoIAlign = type("RoIAlign", (), {})
mmcv.ops.SigmoidFocalLoss = type("SigmoidFocalLoss", (), {})
mmcv.ops.get_compiler_version = lambda: ""
mmcv.ops.get_compiling_cuda_version = lambda: ""
mmcv.ops.nms = lambda *a, **kw: None
mmcv.ops.batched_nms = lambda *a, **kw: None
mmcv.ops.roi_align = lambda *a, **kw: None
mmcv.ops.sigmoid_focal_loss = lambda *a, **kw: None

# mmcv.parallel.DataContainer
mmcv.parallel.DataContainer = type("DataContainer", (), {})

# mmcv.image.tensor2imgs
mmcv.image = ModuleType("mmcv.image")
mmcv.image.tensor2imgs = lambda *a, **kw: None

# Fix build_norm_layer / build_conv_layer to handle cfg=None (old mmcv behavior)
import mmcv.cnn.bricks.norm as _norm_mod
_orig_bnl = _norm_mod.build_norm_layer
def _build_norm_layer_compat(norm_cfg, num_features, postfix=""):
    if norm_cfg is None:
        return [None, torch.nn.Identity()]
    return _orig_bnl(norm_cfg, num_features, postfix)
_norm_mod.build_norm_layer = _build_norm_layer_compat
mmcv.cnn.build_norm_layer = _build_norm_layer_compat  # patch at import point

# mmdet 3.x added abstract methods _forward, loss, predict to BaseDetector.
# Patch them BEFORE importing old mmdet3d models so the abstract-method check
# doesn't prevent class creation.
from mmdet.models.detectors.base import BaseDetector
for _meth_name in ["_forward", "loss", "predict"]:
    _m = getattr(BaseDetector, _meth_name, None)
    if _m is not None and hasattr(_m, "__isabstractmethod__"):
        _m.__isabstractmethod__ = False


# ---------------------------------------------------------------------------
# 8. Post-import voxel patch (called from sst.py after mmdet3d is loaded)
# ---------------------------------------------------------------------------
def _patch_voxel_ops():
    """Re-patch voxel_layer after mmdet3d is loaded."""
    import mmdet3d.ops
    import mmdet3d.ops.voxel
    import mmdet3d.ops.voxel.voxel_layer as vl
    vl.dynamic_voxelize = _dynamic_voxelize_py
    vl.hard_voxelize = _hard_voxelize_py
    vl.dynamic_point_to_voxel_forward = _dynamic_point_to_voxel_forward_py
    vl.dynamic_point_to_voxel_backward = _dynamic_point_to_voxel_backward_py
