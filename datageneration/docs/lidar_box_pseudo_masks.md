# LiDAR 框内点伪掩码工具

`tools/build_lidar_box_pseudo_masks.py` 读取 `build_stage3_seg_candidates.py` 的
`candidates.jsonl`，为每个框实例生成逐点几何伪标签。只依赖 Python 3.10+、NumPy
和候选工具的标准库辅助代码，不导入或修改 B3 模型、数据 loader、QA 或训练入口。

所有输出明确标为 `label_kind=box_geometry_pseudo_mask`。
`mask_verified`、`grounding_verified`、`supervision_ready` 始终为 **false**，
输出的 `eligible_for_training` 也固定为 false。候选原来的完整内容保存在
`candidate` 内，其中的 `eligible_for_training` 仅表示原候选来自允许的 split，
不能解释为框内伪标签已可以训练。

## 核对过的实际约定

2026-10-12 核对了官方 nuScenes SDK 和仓库读取代码；官方文件的 Git blob SHA
也写入运行 summary，便于定位本次检查的源码内容：

- [LidarPointCloud.from_file / Box.corners](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py)：
  blob `fcc757b839079895865a788a077fd7f733535544`。
  原始 `.pcd.bin` 每行五个 float32，顺序为 `x,y,z,intensity,ring`；SDK reshape 为
  `(-1,5)`，保留点行顺序。SDK 去掉 ring 列，**并未删除或重排点**。
  工具显式使用小端 `<f4`，只接受原始 HDL-32 的整数 ring 0–31。
- [get_box / get_sample_data](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py)：
  blob `e64270870c402450f1fb5e93dec8e14b093976cf`。
  `sample_annotation` 的框在 global 坐标；SDK 将其经过 ego_pose 和 calibrated_sensor
  的逆变换转换到传感器坐标。原始 LiDAR XYZ 是传感器坐标。
- [官方表结构](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md)：
  blob `e69415ca97e49f50fd659aac5cafcd39bffa3cf0`。
  四元数顺序 `w,x,y,z`；框尺寸 `width,length,height`，单位米；框局部轴的半径分别是
  `length/2,width/2,height/2`，不能直接把 size 当作 XYZ 边长。
- 仓库 `datageneration/utils.py::load_lidar` 同样使用 float32、`reshape(-1,5)`。
  `nuscenes_converter.py` 保存的 `lidar2ego/ego2global` 也对应此变换链。
  编码器中转换后的训练框、多 sweep 点云和池化 `.npy` 特征不属于本工具输入。

工具使用**完整三轴旋转**，按列向量记号计算：

```text
p_global = R_ego @ (R_sensor @ p_lidar + t_sensor) + t_ego
p_box    = R_box.T @ (p_global - box_center_global)
inside   = abs(p_box) <= [length,width,height] / 2
```

框面采用包含边界规则；float64 运算加入 `64 * eps * coordinate_scale` 的机器舍入容差，
以免旋转和平移使恰好落在框面的点被误排除。没有可配置的框扩张。
只接受 `LIDAR_TOP` keyframe；非 keyframe 不使用插值框，不猜测坐标，也不回退单位变换。

## 原始点来源证明清单

候选构建器只检查了点文件存在和字节数，没有保留原始点文件哈希。
文件名、20 字节整除和合理的 XYZ 都不能证明文件未经转换或重排。
因此生成时必须提供**独立于本生成器建立的原始点来源清单** `--point-manifest`：
在官方原始数据导入时记录哈希、布局、坐标和原始行序，并保留可追溯的来源说明。
本工具不会把任意当前文件重新哈希后自动声明为“官方原始点”。
哈希证明文件与清单一致；清单来源真实性仍需由数据导入流程保证。

清单是 JSON 对象，格式如下（尖括号为实际值的占位符）：

```json
{
  "schema_version": "nuscenes-original-lidar-manifest-v1",
  "provenance": {
    "source": "<原始 nuScenes 数据版本、官方归档来源与导入记录位置>",
    "original_files": true
  },
  "metadata_sha256": {
    "scene": "<scene.json SHA-256>",
    "sample": "<sample.json SHA-256>",
    "sample_data": "<sample_data.json SHA-256>",
    "sample_annotation": "<sample_annotation.json SHA-256>",
    "instance": "<instance.json SHA-256>",
    "category": "<category.json SHA-256>",
    "sensor": "<sensor.json SHA-256>",
    "calibrated_sensor": "<calibrated_sensor.json SHA-256>",
    "ego_pose": "<ego_pose.json SHA-256>"
  },
  "points": [
    {
      "sample_data_token": "<LIDAR_TOP token>",
      "sample_token": "<sample token>",
      "filename": "samples/LIDAR_TOP/<原始文件名>.pcd.bin",
      "sha256": "<原始点文件的64位小写SHA-256>",
      "point_count": 34720,
      "layout": "float32_le_xyz_intensity_ring",
      "coordinate_frame": "lidar_sensor",
      "point_order": "original_file_rows"
    }
  ]
}
```

清单绑定九张原始表的完整 SHA-256，防止不同版本或修改后的标定/框混用。
每个候选还要重新核对 scene 帧索引、sample、LiDAR keyframe、channel、filename、
点数、annotation、instance、category 和框几何。
读取路径只由当前 root 下的原始 `sample_data.filename` 决定；候选中旧主机的
`lidar_path` 原样保留作来源，不用它重定向读取。拒绝路径越界及越界符号链接。
读取同一份 bytes 完成 SHA-256 和 reshape，避免两次打开点文件产生身份差异。

缺少清单、清单缺少帧、坐标/布局/点序声明不符、哈希/点数不符、非有限点值、
不合法 ring、缺失标定、无效四元数、不同 sample 的框、框几何不符，都拒绝该候选。
同一 sample 中其他框的几何若无效，也拒绝，避免遗漏重叠造成伪正例。

## 运行

```powershell
python datageneration/tools/build_lidar_box_pseudo_masks.py --candidates <候选目录>/candidates.jsonl --nuscenes-root <原始nuScenes目录> --nuscenes-version v1.0-trainval --point-manifest <导入审计目录>/original_lidar_manifest.json --output-dir datageneration/data/lidar-box-pseudo-masks/run-001
```

输出必须是新目录，且位于 nuScenes 根目录之外。省略 `--point-manifest` 时仍可做拒绝审计，
但不生成伪掩码。重复 candidate ID 也拒绝；多个同类实例保留，不选择回答目标，不合并。
输入为空时是明确的零产物审计。退出码 0 表示输入全部处理成功或为空，
1 表示存在被拒绝的候选且审计已完成，2 表示配置或运行失败。
只有完成运行才发布最终 JSONL 和 summary；未完成的清单保留 `.part`。

## 输出与标签含义

| 文件 | 内容 |
| --- | --- |
| `masks/<candidate_id哈希>.npz` | 每个候选一份；`labels` 为 `(实例数, 原始点数)` 的 int8；保留完整点行 |
| `pseudo_masks.jsonl` | 每个生成候选一行，含原候选、QA 来源、坐标证据、实例列表、掩码路径/哈希、逐实例计数 |
| `audit.jsonl` | 每个输入候选一行，生成/拒绝状态、候选文件哈希和一基行号 |
| `rejected.jsonl` | 拒绝候选及原因，无掩码文件 |
| `summary.json` | 输入、来源清单、代码、原始表哈希，约定、缺失表、拒绝原因与计数 |

`labels[j,i]` 对应第 j 个候选框、原始文件第 i 行点：

- `1`：只在目标框内的**几何伪正例**。
- `0`：目标框外，**只表达框几何排除，不代表真实语义背景**。
- `-1`：同时在目标框和任意其他标注框内，忽略。

重叠检查包括此 sample 的全部框，不只同类别或此次 QA 的候选框。
NPZ 同时保存 `box_membership`（原始几何布尔掩码，含重叠点）、`overlap_count`、
`point_indices=0..N-1`、`sample_annotation_tokens`、`instance_tokens`、点文件 SHA-256、
伪标签类型和三项 false 标记。无需 pickle；用 `np.load(..., allow_pickle=False)` 读取。
审计中的 `mask_row` 明确标识实例对应 NPZ 行；`instance_token` 来自原始 annotation，
并不是 panoptic ID，也没有在这里建立或验证跨帧轨迹。

审计同时记录 `num_lidar_pts` 与实际框内点数及是否相等。计数相等也不能验证分割；
框内可能含地面、背景和遮挡物，计数差异保留为诊断，不替换点、不裁剪点序、不伪造 GT。
没有框内点的候选拒绝。所有点都被重叠规则忽略时仍保留空伪正例及计数，供审计处理。

## 验证与当前数据边界

```powershell
python -m unittest discover -s datageneration/tests -p "test_*.py" -v
```

测试通过原候选构建器生成输入，再验证三轴旋转、两级平移、宽长顺序、边界、
异类/同类重叠、逐实例身份、点序哈希篡改、来源证明拒绝和 CLI 退出码。
测试点云与表均为合成数据，只证明工具行为。
本地现有 `train-final-20261012/candidates.jsonl` 为空且原始 nuScenes 缺失；
零产物审计不能算真实数据生成验收。
