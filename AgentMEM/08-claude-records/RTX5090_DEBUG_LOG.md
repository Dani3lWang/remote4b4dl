# RTX 5090 (sm_120) 环境调试记录

## 环境

| 项目 | 状态 |
|------|------|
| 环境名 | `wqlc` (conda) |
| Python | 3.10.20 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| GPU | RTX 5090 (sm_120) |

---

## 问题 1: 缺少 mmcv / mmdet / mmseg

**错误**: `ModuleNotFoundError: No module named 'mmcv'`

**原因**: `_mmdet3d_compat.py` 依赖 mmcv 2.x、mmdet 3.x、mmsegmentation

**解决**:
```bash
conda activate wqlc
pip install "setuptools<70"              # 先降级 setuptools (见问题 2)
pip install mmcv==2.1.0 --no-build-isolation
pip install mmdet==3.2.0 --no-build-isolation
pip install mmsegmentation --no-build-isolation
```

---

## 问题 2: setuptools 82 无 pkg_resources

**错误**: `ModuleNotFoundError: No module named 'pkg_resources'` (pip 构建 mmcv 时)

**原因**: setuptools >= 70 移除了 `pkg_resources` 模块，但 mmcv 的 setup.py 需要它

**解决**:
```bash
pip install "setuptools<70"    # 降级到 69.x，包含 pkg_resources
```

**注意**: 必须用 `--no-build-isolation` 安装 mmcv，否则 pip 会创建隔离环境并使用新版 setuptools

---

## 问题 3: mmcv._ext CUDA 扩展加载失败

**错误**: `AssertionError: active_rotated_filter_forward miss in module _ext`

**原因**: `_mmdet3d_compat.py` 创建的 `mmcv._ext` 空模块被真实 `mmcv.ops.*` 子模块加载时调用 `ext_loader.load_ext()`，断言扩展函数存在

**解决**: 修改 `_mmdet3d_compat.py`，将空 `ModuleType` 替换为 `_MockExtModule`，对非 dunder 属性返回 lambda stub:

```python
class _MockExtModule(ModuleType):
    __file__ = None
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None
```

---

## 问题 4: Python 3.10.20 inspect.getsourcefile bug

**错误**: `'function' object has no attribute 'endswith'` in `inspect.py:820`

**原因**: Python 3.10.20 配合 torch 2.8.0 时，`inspect.getsourcefile` → `getfile()` 对某些模块返回 function 而非 string。触发链: mmengine → transformers.optimization → torch.distributed.tensor → torch.library @register → `_register_fake` → `inspect.getmodule(frame)` → `getabsfile` → `getsourcefile` → crash

**解决**: 在 `_mmdet3d_compat.py` 开头 monkey-patch `inspect.getfile`:

```python
_orig_getfile = _inspect.getfile
def _patched_getfile(obj):
    result = _orig_getfile(obj)
    if not isinstance(result, str):
        raise TypeError(...)
    return result
_inspect.getfile = _patched_getfile
```

这样 `getfile` 对非字符串返回值抛出 TypeError，`getmodule` 会 catch TypeError 并返回 None。

---

## 问题 5: NumPy 2.x ABI 不兼容

**错误**: 
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
AttributeError: _ARRAY_API not found
```

**原因**: matplotlib 等包是用 NumPy 1.x 编译的，与 NumPy 2.2.6 ABI 不兼容

**解决**:
```bash
pip install "numpy<2"   # 降级到 1.26.4
```

---

## 问题 6: SST mmdet3d 旧代码导入链问题

**错误**: `ModuleNotFoundError: No module named 'lyft_dataset_sdk'` / `numba` / `open3d` / `tensorflow` / ...

**原因**: `sst/mmdet3d/` 中的旧代码有大量第三方依赖（评估、可视化、数据集处理等），这些在推理时不需要

**解决**: 在 `_mmdet3d_compat.py` 中用 `_StubModule` 预填充 `sys.modules`，stub 掉不需要的子模块:

```python
_MMDET3D_NONESSENTIAL_SUBPACKS = [
    "mmdet3d.core.evaluation",
    "mmdet3d.core.evaluation.kitti_utils",
    "mmdet3d.core.visualizer",
    "mmdet3d.core.post_processing",
    "mmdet3d.datasets",                     # 所有数据集模块
    "mmdet3d.datasets.waymo_dataset",
    # ... 更多 ...
    "mmdet3d.models.roi_heads",
    "mmdet3d.models.decode_heads",
    "mmdet3d.models.segmentors",
    # ... 更多非 SST 的 backbone ...
]

for _sub in _MMDET3D_NONESSENTIAL_SUBPACKS:
    sys.modules.setdefault(_sub, _StubModule(_sub))
```

**关键**: 以下模块**不能** stub，因为 SST 编码器需要它们:
- `mmdet3d.models.detectors` (DynamicVoxelNet)
- `mmdet3d.models.dense_heads` (Anchor3DHead)
- `mmdet3d.models.voxel_encoders` (DynamicVFE)
- `mmdet3d.models.middle_encoders` (SSTInputLayerV2)
- `mmdet3d.models.backbones.sparse_encoder` (DynamicVoxelNet 父类)
- `mmdet3d.models.necks` (配置引用)
- `mmdet3d.models.fusion_layers` (被 detectors import，且是 package)
- `mmdet3d.models.sst` (SST 核心)

---

## 问题 7: 输入格式错误

**错误1**: `torch.cat(points, dim=0)` TypeError —— 收到的是 Tensor 而非 tuple of Tensors

**原因**: `LidarEncoderSST.forward()` 期望 `list[Tensor]`（每帧一个 (N_i, C) tensor），不是 `Tensor(batch, N, C)`

**正确调用方式**:
```python
x = [torch.randn(2500, 4, device='cuda') for _ in range(2)]  # list of (N,4) tensors
y = model(x)
```

**错误2**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x11 and 10x64)`

**原因**: 模型配置 `in_channels=4`，但传入 5 通道数据。模型根据 `in_channels=4` 加上 `cluster_center(3) + voxel_center(3) = 10` 输入特征来创建 Linear(10, 64)。实际输入 5 通道 → 经过 VFE 后变成 11 特征 → 不匹配

**正确通道数**: 4 (x, y, z, intensity) —— 与配置 `in_channels=4` 一致

**错误3**: `RuntimeError: Expected all tensors to be on the same device`

**原因**: 模型在 CPU 上创建，需要显式 `.cuda()`

**正确初始化**:
```python
model = LidarEncoderSST('lidarclip/model/sst_encoder_only_config.py').cuda()
model.eval()
```

---

## 问题 8: OOM (退出码 137)

**错误**: 进程被 SIGKILL (退出码 137)

**原因**: RTX 5090 32GB 显存不足以运行完整的 DynamicVoxelNet（带 dense_heads、bbox_head 等检测头）。纯编码器前向传播仍可能 OOM

**待解决**: 需要只加载 SST backbone 部分进行特征提取，而非完整的检测器模型

---

## 环境包列表 (wqlc)

```bash
# 核心
torch==2.8.0+cu128
numpy==1.26.4
setuptools==69.5.1

# OpenMMLab
mmcv==2.1.0
mmdet==3.2.0
mmengine==0.10.6
mmsegmentation==1.2.2

# 调试/工具 (SST 依赖)
ipdb==0.13.13
numba==0.65.1
```

---

## 修改文件清单

### `encoders/lidarclip/lidarclip/model/_mmdet3d_compat.py`
- 加入 `inspect.getfile` monkey-patch (问题4)
- `_MockExtModule` dunder 属性处理 (问题3,5)
- `_MMDET3D_NONESSENTIAL_SUBPACKS` 预 stub 列表 (问题6)

### `mllm/vtimellm/train/train_mem.py`
- flash_attn try/except 回退

### `mllm/run_stages.sh`
- CUDA 检测加入 12.8

---

## 下次继续事项

1. **OOM 问题**: 修改 `LidarEncoderSST` 或 config，只加载 SST backbone + 必要组件，移除检测头
2. **特征提取**: 修复 OOM 后运行 `extract_pc_features.py`
3. **训练**: 在 RTX 5090 上运行 stage1/stage2 训练
