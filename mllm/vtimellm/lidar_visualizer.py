"""nuScenes-backed data and Plotly helpers for the B4DL Gradio demo.

The module deliberately keeps Gradio and the B4DL model stack out of the data
layer.  It can therefore be imported and unit-tested on a CPU-only machine;
the nuScenes SDK and Plotly are imported only when their functionality is used.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


CAMERA_VIEWS: Tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

BOX_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

CATEGORY_COLORS: Mapping[str, str] = {
    "vehicle": "#ffb000",
    "pedestrian": "#ff4d6d",
    "cycle": "#00e5ff",
    "movable": "#b7ff4a",
    "static": "#b4a7d6",
    "other": "#e4e8ef",
}


@dataclass(frozen=True)
class SceneRef:
    scene_token: str
    scene_id: Optional[str]
    name: str
    split: Optional[str]
    sample_tokens: Tuple[str, ...]

    @property
    def label(self) -> str:
        identity = self.scene_id or self.name
        split = f" · {self.split}" if self.split else ""
        return f"{identity} · {self.name}{split} · {len(self.sample_tokens)} 帧"


@dataclass(frozen=True)
class BoxRender:
    token: str
    category: str
    corners: np.ndarray  # [3, 8], current LIDAR_TOP coordinates


@dataclass(frozen=True)
class TrackRender:
    token: str
    category: str
    points: np.ndarray  # [N, 3], oldest -> current, current LIDAR_TOP coordinates


@dataclass(frozen=True)
class FrameData:
    scene: SceneRef
    frame_index: int
    sample_token: str
    points: np.ndarray  # [N, 4+] in current LIDAR_TOP coordinates
    boxes: Tuple[BoxRender, ...]
    tracks: Tuple[TrackRender, ...]
    camera_paths: Mapping[str, Optional[str]]
    warnings: Tuple[str, ...] = ()


def category_group(category: str) -> str:
    """Map detailed nuScenes names to a small, stable visual palette."""
    name = (category or "").lower()
    if name.startswith("vehicle"):
        if "bicycle" in name or "motorcycle" in name:
            return "cycle"
        return "vehicle"
    if name.startswith("human.pedestrian"):
        return "pedestrian"
    if name.startswith("movable_object"):
        return "movable"
    if name.startswith("static_object") or name.startswith("flat"):
        return "static"
    return "other"


def deterministic_downsample(
    points: np.ndarray,
    max_points: int,
    seed_key: str,
) -> np.ndarray:
    """Return a repeatable, order-preserving sample without touching global RNG."""
    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError(f"points must be a 2-D array, got {points.shape}")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if len(points) <= max_points:
        return points
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(points), size=max_points, replace=False))
    return points[indices]


def quaternion_rotation_matrix(rotation: Sequence[float]) -> np.ndarray:
    """Convert a nuScenes [w, x, y, z] quaternion to a 3x3 rotation matrix."""
    q = np.array(rotation, dtype=np.float64, copy=True)
    if q.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {q.shape}")
    norm = float(np.dot(q, q))
    if norm < np.finfo(float).eps:
        return np.eye(3, dtype=np.float64)
    q *= np.sqrt(2.0 / norm)
    outer = np.outer(q, q)
    return np.array(
        [
            [1.0 - outer[2, 2] - outer[3, 3], outer[1, 2] - outer[3, 0], outer[1, 3] + outer[2, 0]],
            [outer[1, 2] + outer[3, 0], 1.0 - outer[1, 1] - outer[3, 3], outer[2, 3] - outer[1, 0]],
            [outer[1, 3] - outer[2, 0], outer[2, 3] + outer[1, 0], 1.0 - outer[1, 1] - outer[2, 2]],
        ],
        dtype=np.float64,
    )


def global_to_sensor(
    points: np.ndarray,
    ego_pose: Mapping[str, Sequence[float]],
    calibrated_sensor: Mapping[str, Sequence[float]],
) -> np.ndarray:
    """Transform row-vector points from nuScenes global to sensor coordinates."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {values.shape}")
    ego_rotation = quaternion_rotation_matrix(ego_pose["rotation"])
    sensor_rotation = quaternion_rotation_matrix(calibrated_sensor["rotation"])
    values = (values - np.asarray(ego_pose["translation"], dtype=np.float64)) @ ego_rotation
    values = (
        values - np.asarray(calibrated_sensor["translation"], dtype=np.float64)
    ) @ sensor_rotation
    return values.astype(np.float32, copy=False)


def box_line_coordinates(corners: np.ndarray) -> np.ndarray:
    """Expand [3, 8] box corners to NaN-separated line coordinates."""
    corners = np.asarray(corners)
    if corners.shape != (3, 8):
        raise ValueError(f"box corners must have shape [3, 8], got {corners.shape}")
    chunks = []
    for start, end in BOX_EDGES:
        chunks.extend((corners[:, start], corners[:, end], np.full(3, np.nan)))
    return np.asarray(chunks, dtype=np.float32)


def validate_scene_features(features: np.ndarray, frame_count: int) -> None:
    """Fail fast when a feature file cannot safely align to the viewed scene."""
    if not isinstance(features, np.ndarray):
        raise TypeError("scene features must be a numpy array")
    if features.ndim != 2 or features.shape[1] != 768:
        raise ValueError(
            f"特征应为 [N, 768]，实际为 {tuple(features.shape)}"
        )
    if features.shape[0] != frame_count:
        raise ValueError(
            f"特征帧数 {features.shape[0]} 与 nuScenes 场景帧数 {frame_count} 不一致"
        )


def step_frame(index: int, frame_count: int, delta: int, loop: bool = False) -> int:
    """Move a timeline cursor with explicit end behaviour."""
    if frame_count <= 0:
        return 0
    target = int(index) + int(delta)
    if loop:
        return target % frame_count
    return max(0, min(frame_count - 1, target))


def advance_playback(index: int, frame_count: int) -> Tuple[int, bool]:
    """Advance once and report whether playback should remain active."""
    if frame_count <= 0:
        return 0, False
    index = max(0, min(frame_count - 1, int(index)))
    if index >= frame_count - 1:
        return index, False
    return index + 1, True


class NuScenesSceneRepository:
    """Lazy, bounded-cache access to scenes and visualization-ready frames."""

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-trainval",
        scene_metadata: Optional[str] = None,
        max_points: int = 30_000,
        cache_size: int = 12,
        nusc=None,
    ) -> None:
        self.dataroot = str(Path(dataroot).expanduser().resolve())
        self.version = version
        self.max_points = int(max_points)
        self.cache_size = max(1, int(cache_size))
        if self.max_points <= 0:
            raise ValueError("max_points must be positive")

        if nusc is None:
            if not os.path.isdir(self.dataroot):
                raise FileNotFoundError(f"nuScenes 数据目录不存在：{self.dataroot}")
            try:
                from nuscenes.nuscenes import NuScenes
            except ImportError as exc:
                raise RuntimeError(
                    "缺少 nuscenes-devkit，请安装 mllm/requirements-demo.txt"
                ) from exc
            nusc = NuScenes(version=version, dataroot=self.dataroot, verbose=False)
        self.nusc = nusc

        metadata = self._load_scene_metadata(scene_metadata)
        self._metadata_by_token = {
            str(item["scene_token"]): item
            for item in metadata
            if item.get("scene_token")
        }
        self._sample_chain_cache: Dict[str, Tuple[str, ...]] = {}
        self._frame_cache: "OrderedDict[Tuple[str, int], FrameData]" = OrderedDict()
        self.scenes = self._build_scene_index()
        self._scenes_by_token = {scene.scene_token: scene for scene in self.scenes}
        if not self.scenes:
            raise RuntimeError(f"{version} 中没有可显示的场景")

    @staticmethod
    def _load_scene_metadata(path: Optional[str]) -> List[dict]:
        if not path:
            return []
        metadata_path = Path(path).expanduser()
        if not metadata_path.is_file():
            raise FileNotFoundError(f"scene_metadata 不存在：{metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            raise ValueError("scene_metadata.json 顶层必须是数组")
        return value

    def _sample_chain(self, scene_record: Mapping[str, object]) -> Tuple[str, ...]:
        scene_token = str(scene_record["token"])
        cached = self._sample_chain_cache.get(scene_token)
        if cached is not None:
            return cached
        token = str(scene_record.get("first_sample_token") or "")
        result: List[str] = []
        visited = set()
        while token:
            if token in visited:
                raise ValueError(f"场景 {scene_token} 的 sample 链存在循环：{token}")
            visited.add(token)
            result.append(token)
            sample = self.nusc.get("sample", token)
            token = str(sample.get("next") or "")
        chain = tuple(result)
        self._sample_chain_cache[scene_token] = chain
        return chain

    def _build_scene_index(self) -> Tuple[SceneRef, ...]:
        result = []
        for record in self.nusc.scene:
            token = str(record["token"])
            item = self._metadata_by_token.get(token, {})
            result.append(
                SceneRef(
                    scene_token=token,
                    scene_id=str(item["scene_id"]) if item.get("scene_id") is not None else None,
                    name=str(record.get("name") or token[:8]),
                    split=str(item["split"]) if item.get("split") is not None else None,
                    sample_tokens=self._sample_chain(record),
                )
            )
        return tuple(result)

    def get_scene(self, scene_token: str) -> SceneRef:
        try:
            return self._scenes_by_token[str(scene_token)]
        except KeyError as exc:
            raise KeyError(f"未知 scene token：{scene_token}") from exc

    def _load_points(self, lidar_path: str, sample_token: str) -> np.ndarray:
        try:
            from nuscenes.utils.data_classes import LidarPointCloud
        except ImportError as exc:
            raise RuntimeError(
                "缺少 nuscenes-devkit，请安装 mllm/requirements-demo.txt"
            ) from exc
        cloud = LidarPointCloud.from_file(lidar_path)
        points = np.asarray(cloud.points.T, dtype=np.float32)
        return deterministic_downsample(points, self.max_points, sample_token)

    def _load_boxes_and_tracks(
        self,
        lidar_token: str,
        lidar_data: Mapping[str, object],
    ) -> Tuple[Tuple[BoxRender, ...], Tuple[TrackRender, ...], List[str]]:
        warnings: List[str] = []
        boxes: List[BoxRender] = []
        tracks: List[TrackRender] = []
        try:
            _, sdk_boxes, _ = self.nusc.get_sample_data(lidar_token)
        except Exception as exc:  # annotations are optional in test/custom sets
            return (), (), [f"标注不可用：{exc}"]

        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated = self.nusc.get(
            "calibrated_sensor", lidar_data["calibrated_sensor_token"]
        )
        for box in sdk_boxes:
            token = str(getattr(box, "token", "") or "")
            category = str(getattr(box, "name", "") or "other")
            boxes.append(
                BoxRender(
                    token=token,
                    category=category,
                    corners=np.asarray(box.corners(), dtype=np.float32),
                )
            )
            if not token:
                continue
            try:
                centers = []
                annotation_token = token
                visited = set()
                while annotation_token:
                    if annotation_token in visited:
                        break
                    visited.add(annotation_token)
                    annotation = self.nusc.get("sample_annotation", annotation_token)
                    centers.append(annotation["translation"])
                    annotation_token = str(annotation.get("prev") or "")
                if centers:
                    sensor_points = global_to_sensor(
                        np.asarray(centers[::-1], dtype=np.float64),
                        ego_pose,
                        calibrated,
                    )
                    tracks.append(
                        TrackRender(
                            token=token,
                            category=category,
                            points=sensor_points,
                        )
                    )
            except Exception as exc:
                warnings.append(f"轨迹 {token[:8]} 读取失败：{exc}")
        return tuple(boxes), tuple(tracks), warnings

    def get_frame(self, scene_token: str, frame_index: int) -> FrameData:
        scene = self.get_scene(scene_token)
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= len(scene.sample_tokens):
            raise IndexError(
                f"帧索引 {frame_index} 超出范围 [0, {len(scene.sample_tokens) - 1}]"
            )
        cache_key = (scene.scene_token, frame_index)
        cached = self._frame_cache.get(cache_key)
        if cached is not None:
            self._frame_cache.move_to_end(cache_key)
            return cached

        sample_token = scene.sample_tokens[frame_index]
        sample = self.nusc.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = self.nusc.get("sample_data", lidar_token)
        lidar_path = os.path.join(self.dataroot, lidar_data["filename"])
        points = self._load_points(lidar_path, sample_token)
        boxes, tracks, warnings = self._load_boxes_and_tracks(lidar_token, lidar_data)

        camera_paths: Dict[str, Optional[str]] = {}
        for camera in CAMERA_VIEWS:
            camera_token = sample.get("data", {}).get(camera)
            if not camera_token:
                camera_paths[camera] = None
                continue
            record = self.nusc.get("sample_data", camera_token)
            path = os.path.join(self.dataroot, record["filename"])
            camera_paths[camera] = path if os.path.isfile(path) else None

        frame = FrameData(
            scene=scene,
            frame_index=frame_index,
            sample_token=sample_token,
            points=points,
            boxes=boxes,
            tracks=tracks,
            camera_paths=camera_paths,
            warnings=tuple(warnings),
        )
        self._frame_cache[cache_key] = frame
        self._frame_cache.move_to_end(cache_key)
        while len(self._frame_cache) > self.cache_size:
            self._frame_cache.popitem(last=False)
        return frame

    @staticmethod
    def camera_image(frame: FrameData, camera: str) -> Image.Image:
        path = frame.camera_paths.get(camera)
        if path:
            with Image.open(path) as image:
                return image.convert("RGB").copy()
        image = Image.new("RGB", (960, 540), "#101720")
        draw = ImageDraw.Draw(image)
        draw.rectangle((22, 22, 938, 518), outline="#293848", width=2)
        draw.text((48, 46), f"{camera} / IMAGE UNAVAILABLE", fill="#ffb000")
        draw.text((48, 84), frame.sample_token, fill="#8393a7")
        return image


def _append_grouped_boxes(fig, boxes: Iterable[BoxRender], is_3d: bool) -> None:
    import plotly.graph_objects as go

    grouped: Dict[str, List[np.ndarray]] = {}
    for box in boxes:
        group = category_group(box.category)
        grouped.setdefault(group, []).append(box_line_coordinates(box.corners))
    for group, chunks in grouped.items():
        coords = np.concatenate(chunks, axis=0)
        common = dict(
            mode="lines",
            name=group,
            line=dict(color=CATEGORY_COLORS[group], width=3),
            hoverinfo="name",
        )
        if is_3d:
            fig.add_trace(go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], **common))
        else:
            fig.add_trace(go.Scattergl(x=coords[:, 0], y=coords[:, 1], **common))


def _append_tracks(fig, tracks: Iterable[TrackRender], is_3d: bool) -> None:
    import plotly.graph_objects as go

    shown = set()
    for track in tracks:
        if len(track.points) < 2:
            continue
        group = category_group(track.category)
        common = dict(
            mode="lines+markers",
            name=f"{group} 轨迹",
            legendgroup=f"track-{group}",
            showlegend=group not in shown,
            line=dict(color=CATEGORY_COLORS[group], width=2, dash="dot"),
            marker=dict(color=CATEGORY_COLORS[group], size=3),
            opacity=0.8,
            hoverinfo="skip",
        )
        shown.add(group)
        if is_3d:
            fig.add_trace(
                go.Scatter3d(
                    x=track.points[:, 0],
                    y=track.points[:, 1],
                    z=track.points[:, 2],
                    **common,
                )
            )
        else:
            fig.add_trace(go.Scattergl(x=track.points[:, 0], y=track.points[:, 1], **common))


def make_plotly_figures(
    frame: FrameData,
    show_boxes: bool = True,
    show_tracks: bool = True,
):
    """Create coordinated 3D and BEV figures for a frame."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "缺少 Plotly，请安装 mllm/requirements-demo.txt"
        ) from exc

    points = frame.points
    color = points[:, 2]
    colorscale = [[0.0, "#123247"], [0.45, "#00d4c7"], [1.0, "#ffe66d"]]
    fig3d = go.Figure(
        go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode="markers",
            name="LiDAR",
            marker=dict(size=1.5, color=color, colorscale=colorscale, opacity=0.78),
            hoverinfo="skip",
        )
    )
    bev = go.Figure(
        go.Scattergl(
            x=points[:, 0], y=points[:, 1],
            mode="markers",
            name="LiDAR",
            marker=dict(size=2, color=color, colorscale=colorscale, opacity=0.72),
            hoverinfo="skip",
        )
    )

    if show_boxes:
        _append_grouped_boxes(fig3d, frame.boxes, is_3d=True)
        _append_grouped_boxes(bev, frame.boxes, is_3d=False)
    if show_tracks:
        _append_tracks(fig3d, frame.tracks, is_3d=True)
        _append_tracks(bev, frame.tracks, is_3d=False)

    axis = dict(
        showbackground=True,
        backgroundcolor="#0c1219",
        gridcolor="#263646",
        zerolinecolor="#547086",
        color="#9fb0c2",
        showspikes=False,
    )
    fig3d.update_layout(
        paper_bgcolor="#081018",
        plot_bgcolor="#081018",
        font=dict(color="#dce7f2", family="Bahnschrift, sans-serif"),
        margin=dict(l=0, r=0, t=12, b=0),
        legend=dict(bgcolor="rgba(8,16,24,.72)", orientation="h", y=0.98),
        scene=dict(
            xaxis={**axis, "title": "前向 X / m"},
            yaxis={**axis, "title": "左向 Y / m"},
            zaxis={**axis, "title": "高度 Z / m"},
            aspectmode="data",
            camera=dict(eye=dict(x=1.25, y=1.25, z=0.82)),
        ),
        uirevision=f"scene-{frame.scene.scene_token}",
    )
    bev.update_layout(
        paper_bgcolor="#081018",
        plot_bgcolor="#0c1219",
        font=dict(color="#dce7f2", family="Bahnschrift, sans-serif"),
        margin=dict(l=56, r=16, t=12, b=48),
        legend=dict(bgcolor="rgba(8,16,24,.72)", orientation="h", y=0.98),
        xaxis=dict(title="前向 X / m", gridcolor="#263646", zerolinecolor="#547086"),
        yaxis=dict(
            title="左向 Y / m",
            gridcolor="#263646",
            zerolinecolor="#547086",
            scaleanchor="x",
            scaleratio=1,
        ),
        uirevision=f"scene-{frame.scene.scene_token}",
    )
    return fig3d, bev


def empty_plotly_figure(message: str):
    """Return a styled placeholder instead of leaving a broken plot component."""
    import plotly.graph_objects as go

    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color="#ffb000", size=15, family="Bahnschrift, sans-serif"),
    )
    figure.update_layout(
        paper_bgcolor="#081018",
        plot_bgcolor="#0c1219",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return figure
