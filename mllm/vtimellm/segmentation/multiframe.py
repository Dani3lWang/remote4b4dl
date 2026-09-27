"""多帧（时序）ReasonSeg 数据集：选项 A 的 A1/A2/A3 三臂共用一个装配器。

判据与约束见 `docs/learn docs/B4DL_ReasonSeg多帧A_预注册_20260927.md`。三条硬约束决定了
这里的形状：

1. 编码器输入契约不可动（`spatial_encoder.py:173-179` 对同体素取均值、`checkpoint.py:169-182`
   校验 voxel 契约且 strict 加载）⇒ 只能在**点云层面**拼接，仍是 x,y,z,reflectance 4 列，
   没有帧号/时间通道。
2. 盘上只有 2 Hz 关键帧（`samples/LIDAR_TOP` 34,149，sweeps 未解包）⇒ 邻帧相隔 0.5 s。
3. panoptic npz 只有逐帧 instance 号、无跨帧身份 ⇒ 跨帧同实例只能靠
   `sample_annotation` 的 `instance_token` 轨迹。

三臂：
  a1_compensated  邻帧点与框一起补偿到 anchor 的 ego 系
  a2_repeat       只把 anchor 帧的点/掩码复制 F 份。同体素均值不变 ⇒ 严格零新信息，
                  且不需要 nuScenes 元数据（可离线自检）
  a3_naive        同一套装配，但每帧的点与框都留在**各自本车的 ego 系**里
                  ⇒ 物体与背景一起按自车位移错开

a1 与 a3 的唯一差异是"用哪个 ego 系表达"这一个变量。

坐标约定（全模块统一，行向量）：nuScenes 的 `rotation`/`translation` 描述 源系 -> 目标系，
故 `v_dst = (v_src @ R.T + t - t_dst) @ R_dst`。框与点必须走同一个变换，否则 a3 会变成
"点错开而框不错开"的混合体，那既不是 naive 也不是 compensated。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ReasonSegConfig
from .data import ReasonSegDataset

ARMS = ("a1_compensated", "a2_repeat", "a3_naive")

# trainval 的 sample_annotation 有 85 万行，NuScenes() 一次加载要几分钟。三臂共用一份。
_NUSC_CACHE: Dict[tuple, object] = {}


def quaternion_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """nuScenes 的 [w,x,y,z] 四元数 -> 3x3（列向量约定，与 devkit 一致）。"""
    w, x, y, z = (float(value) for value in quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def ego_to_ego(points: np.ndarray, source: "Frame", target: "Frame") -> np.ndarray:
    """把 `source` ego 系里的点云换到 `target` ego 系；第 4 列（反射率）原样带过。"""
    global_xyz = points[:, :3] @ source.to_global.T + source.origin
    return np.concatenate([(global_xyz - target.origin) @ target.to_global,
                           points[:, 3:]], axis=1)


@dataclass(frozen=True)
class Frame:
    to_global: np.ndarray
    origin: np.ndarray

    def box_of(self, annotation: Dict[str, object]) -> "OrientedBox":
        """sample_annotation 的框在全局系；换算到本帧 ego 系。"""
        center_global = np.asarray(annotation["translation"], dtype=np.float64)
        size = np.asarray(annotation["size"], dtype=np.float64)  # x_len, y_width, z_height
        return OrientedBox(
            center=(center_global - self.origin) @ self.to_global,
            half_size=size / 2.0,
            rotation=self.to_global.T @ quaternion_matrix(annotation["rotation"]),
        )

    def contains(self, annotation: Dict[str, object], points: np.ndarray) -> np.ndarray:
        return self.box_of(annotation).mask_for(points)


@dataclass(frozen=True)
class OrientedBox:
    center: np.ndarray
    half_size: np.ndarray
    rotation: np.ndarray

    def mask_for(self, points: np.ndarray) -> np.ndarray:
        local = (points[:, :3] - self.center) @ self.rotation
        return (np.abs(local) <= self.half_size).all(axis=1)


class MultiFrameReasonSegDataset(Dataset):
    """包一层 `ReasonSegDataset`：anchor 那条完全按原样取，只在它前后拼帧。

    返回的 dict 键与原数据集逐字相同（外加 `frame_point_counts` / `frame_offsets`），
    所以探针的 `oracle_inputs` / `evaluate` / 训练循环不需要为多帧改任何形状。
    """

    def __init__(
        self,
        manifest_path: str,
        *,
        dataroot: str,
        config: Optional[ReasonSegConfig] = None,
        arm: str = "a1_compensated",
        num_frames: int = 3,
        require_reachable: bool = False,
        nuscenes_version: str = "v1.0-trainval",
    ):
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        if num_frames < 1 or num_frames % 2 == 0:
            raise ValueError("num_frames 必须是奇数（两侧对称取邻帧）")
        self.config = config or ReasonSegConfig()
        self.arm = arm
        self.num_frames = int(num_frames)
        self.side = (self.num_frames - 1) // 2
        self.base = ReasonSegDataset(
            manifest_path, dataroot=dataroot, config=self.config,
            require_reachable=require_reachable,
        )
        self.dataroot = self.base.dataroot
        self.records = self.base.records
        self.nuscenes_version = nuscenes_version
        self._nusc = None
        self._frames: Dict[str, List[Dict[str, object]]] = {}

    def __len__(self) -> int:
        return len(self.base)

    @property
    def nusc(self):
        if self._nusc is None:
            from nuscenes.nuscenes import NuScenes

            key = (self.nuscenes_version, str(self.dataroot))
            if key not in _NUSC_CACHE:
                _NUSC_CACHE[key] = NuScenes(version=self.nuscenes_version,
                                            dataroot=str(self.dataroot), verbose=False)
            self._nusc = _NUSC_CACHE[key]
        return self._nusc

    def __getitem__(self, index: int) -> Dict[str, object]:
        anchor = self.base[index]
        return self._repeat(anchor) if self.arm == "a2_repeat" else self._assemble(index, anchor)

    # ---- A2：密度对照（严格零新信息） ----
    def _repeat(self, anchor: Dict[str, object]) -> Dict[str, object]:
        points = anchor["points"].numpy()
        count = self.num_frames
        out = dict(anchor)
        # 逐位相同的点落进同一体素，_voxelize 的均值对它们是恒等的
        out["points"] = torch.from_numpy(np.tile(points, (count, 1)))
        out["target_masks"] = torch.from_numpy(np.tile(anchor["target_masks"].numpy(), (1, count)))
        out["target_loc_masks"] = torch.from_numpy(
            np.tile(anchor["target_loc_masks"].numpy(), (1, count)))
        out["frame_point_counts"] = [int(points.shape[0])] * count
        out["frame_offsets"] = [0] * count
        return out

    # ---- A1 / A3 ----
    def _assemble(self, index: int, anchor: Dict[str, object]) -> Dict[str, object]:
        record = self.records[index]
        frames = self._scene_frames(record.sample_token)
        reference = self._reference(frames, record.sample_token)
        # a1 把所有帧表达到 anchor 的 ego 系；a3 表达到各自的 ego 系（= 不做补偿）
        target = reference.frame if self.arm == "a1_compensated" else None

        segments: List[Dict[str, object]] = [{
            "offset": 0,
            "points": anchor["points"].numpy(),
            "panoptic": reference.panoptic,
            "frame": reference.frame,
            "anns": reference.anns,
        }]
        for entry in frames:
            if entry["offset"] == 0 or entry.get("points") is None:
                continue
            own = entry["frame"]
            destination = target if target is not None else own
            segments.append({
                "offset": entry["offset"],
                "points": ego_to_ego(entry["points"], own, destination),
                "panoptic": entry["panoptic"],
                "frame": destination,
                "anns": entry["anns"],
            })

        anchor_count = int(segments[0]["points"].shape[0])
        total = sum(int(segment["points"].shape[0]) for segment in segments)
        anchor_masks = anchor["target_masks"].numpy()
        masks = np.zeros((anchor_masks.shape[0], total), dtype=np.bool_)
        masks[:, :anchor_count] = anchor_masks

        for row, target_instance in enumerate(record.targets):
            if not anchor_masks[row].any():
                continue
            instance_token = self._anchor_instance(reference, anchor, row)
            if instance_token is None:
                continue
            cursor = anchor_count
            for segment in segments[1:]:
                count = int(segment["points"].shape[0])
                annotation = segment["anns"].get(instance_token)
                if annotation is not None:
                    inside = np.flatnonzero(
                        segment["frame"].contains(annotation, segment["points"]))
                    # 跨帧身份只由框给出；同语义类是硬条件，防止框套到别的实例身上
                    if inside.size:
                        same_class = (segment["panoptic"][inside] // 1000
                                      == target_instance.panoptic_id // 1000)
                        masks[row, cursor + inside[same_class]] = True
                cursor += count

        points = np.concatenate([segment["points"] for segment in segments], axis=0)
        out = dict(anchor)
        out["points"] = torch.from_numpy(points.astype(np.float32))
        out["target_masks"] = torch.from_numpy(masks)
        out["target_loc_masks"] = torch.from_numpy(self._loc_masks(points, masks))
        out["frame_point_counts"] = [int(segment["points"].shape[0]) for segment in segments]
        out["frame_offsets"] = [int(segment["offset"]) for segment in segments]
        return out

    def _loc_masks(self, points: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """与 `data.py:117-121` 同一公式，但中心/半径用拼接后的正例点集算。"""
        xyz = points[:, :3]
        loc = np.zeros_like(masks)
        margin = self.config.coarse_radius_margin_m
        for row in range(masks.shape[0]):
            if not masks[row].any():
                continue
            center = xyz[masks[row]].mean(axis=0)
            radius = float(np.linalg.norm(xyz[masks[row]] - center, axis=1).max())
            loc[row] = np.linalg.norm(xyz - center, axis=1) <= (radius + margin)
        return loc

    # ---- 元数据与缓存 ----
    def _reference(self, frames: List[Dict[str, object]], sample_token: str) -> Dict[str, object]:
        for entry in frames:
            if entry["offset"] == 0 and entry.get("points") is not None:
                return entry
        raise RuntimeError(f"anchor sample {sample_token} 没有可用的 LIDAR_TOP / panoptic")

    def _anchor_instance(self, reference: Dict[str, object], anchor: Dict[str, object],
                         row: int) -> Optional[str]:
        """anchor 第 row 个 panoptic 目标 -> `instance_token`。

        用该样本的 3D 框去套 anchor 自己的 GT 掩码，取覆盖正例点最多的框 —— 不依赖
        nuScenes 内部的编号顺序。冒烟门会验证框与掩码的 IoU≈1，对不上即映射失败。
        """
        mask = anchor["target_masks"].numpy()[row]
        best: Tuple[Optional[str], int] = (None, 0)
        for instance_token, annotation in reference["anns"].items():
            inside = reference["frame"].contains(annotation, reference["points"])
            overlap = int((inside & mask).sum())
            if overlap > best[1]:
                best = (instance_token, overlap)
        return best[0]

    def anchor_box_vs_mask(self, index: int) -> List[Tuple[str, int, int, int]]:
        """冒烟用：逐目标返回 (class_name, 交集, 掩码点数, 框内点数)。"""
        anchor = self.base[index]
        frames = self._scene_frames(self.records[index].sample_token)
        reference = self._reference(frames, self.records[index].sample_token)
        rows = []
        for row, target in enumerate(self.records[index].targets):
            mask = anchor["target_masks"].numpy()[row]
            instance_token = self._anchor_instance(reference, anchor, row)
            if instance_token is None:
                rows.append((target.class_name, 0, int(mask.sum()), 0))
                continue
            inside = reference["frame"].contains(reference["anns"][instance_token],
                                                 reference["points"])
            rows.append((target.class_name, int((inside & mask).sum()), int(mask.sum()),
                         int(inside.sum())))
        return rows

    def frame_summary(self, index: int) -> Dict[str, object]:
        """冒烟用：邻帧可用性、各帧点数、以及补偿前后邻帧质心的位移。"""
        record = self.records[index]
        frames = self._scene_frames(record.sample_token)
        reference = self._reference(frames, record.sample_token)
        out: Dict[str, object] = {"offsets": [0], "points": [int(reference["points"].shape[0])],
                                  "shift_m": [0.0], "anns": [len(reference["anns"])]}
        for entry in frames:
            if entry["offset"] == 0:
                continue
            out["offsets"].append(entry["offset"])
            out["anns"].append(len(entry["anns"]))
            if entry.get("points") is None:
                out["points"].append(0)
                out["shift_m"].append(None)
                continue
            # 全帧质心的位移 = 自车运动量（背景被对齐了多少），不是目标物体的位移
            compensated = ego_to_ego(entry["points"], entry["frame"], reference["frame"])
            out["points"].append(int(entry["points"].shape[0]))
            out["shift_m"].append(float(np.linalg.norm(
                entry["points"][:, :3].mean(axis=0) - compensated[:, :3].mean(axis=0))))
        return out

    def _scene_frames(self, sample_token: str) -> List[Dict[str, object]]:
        cached = self._frames.get(sample_token)
        if cached is not None:
            return cached
        chain: List[Dict[str, object]] = [{"offset": 0, "sample_token": sample_token}]
        for direction in (-1, 1):
            token = sample_token
            for offset in range(direction, direction * (self.side + 1), direction):
                sample = self.nusc.get("sample", token)
                token = sample["prev" if direction < 0 else "next"]
                if not token:
                    break
                chain.append({"offset": offset, "sample_token": token})
        for entry in chain:
            try:
                self._fill_frame(entry)
            except (KeyError, ValueError, FileNotFoundError) as exc:
                entry.update({"points": None, "panoptic": None, "anns": {},
                              "frame": None, "missing": repr(exc)})
        chain.sort(key=lambda entry: entry["offset"])
        self._frames[sample_token] = chain
        return chain

    def _fill_frame(self, entry: Dict[str, object]) -> None:
        nusc = self.nusc
        sample = nusc.get("sample", entry["sample_token"])
        sample_data_token = sample["data"]["LIDAR_TOP"]
        sample_data = nusc.get("sample_data", sample_data_token)
        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        path = self.dataroot / sample_data["filename"]
        panoptic_path = (self.dataroot / "panoptic" / self.nuscenes_version /
                         f"{sample_data_token}_panoptic.npz")
        if not path.is_file():
            raise FileNotFoundError(path)
        if not panoptic_path.is_file():
            raise FileNotFoundError(panoptic_path)
        points = np.fromfile(path, dtype=np.float32)
        if points.size % 5:
            raise RuntimeError(f"invalid nuScenes point file: {path}")
        with np.load(panoptic_path) as values:
            panoptic = np.asarray(values["data"]).reshape(-1)
        points = points.reshape(-1, 5)[:, :4]
        if panoptic.shape[0] != points.shape[0]:
            raise RuntimeError(f"point/panoptic length mismatch in {sample_data_token}")
        entry.update({
            "sample_data_token": sample_data_token,
            "points": points,
            "panoptic": panoptic,
            "frame": Frame(to_global=quaternion_matrix(ego_pose["rotation"]),
                           origin=np.asarray(ego_pose["translation"], dtype=np.float64)),
            "anns": {nusc.get("sample_annotation", token)["instance_token"]:
                     nusc.get("sample_annotation", token) for token in sample["anns"]},
        })
