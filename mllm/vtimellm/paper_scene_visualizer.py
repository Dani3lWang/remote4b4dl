"""CPU orthographic LiDAR views for paper figures, without a display or GPU."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np
from PIL import Image, ImageColor, ImageDraw

try:
    from .lidar_visualizer import BOX_EDGES, FrameData
except ImportError:
    from lidar_visualizer import BOX_EDGES, FrameData


TARGET_PURPLE = "#b770df"


def points_in_box(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Select points inside an oriented annotation box, rather than its AABB."""
    points = np.asarray(points, dtype=float)
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (3, 8) or not np.isfinite(corners).all():
        raise ValueError("目标框必须包含 8 个有效三维角点")
    edges = corners[:, [1, 3, 4]] - corners[:, [0]]
    try:
        local = np.linalg.solve(edges, (points[:, :3] - corners[:, 0]).T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("目标框尺寸无效") from exc
    return np.isfinite(local).all(axis=1) & (local >= -1e-6).all(axis=1) & (local <= 1 + 1e-6).all(axis=1)


def render_lidar_scene(
    frame: FrameData,
    size: Tuple[int, int] = (640, 360),
    target_colors: Optional[Mapping[str, str]] = None,
    *,
    azimuth: float = -55.0,
    elevation: float = 28.0,
    range_m: float = 45.0,
    show_boxes: bool = False,
    show_tracks: bool = False,
) -> Image.Image:
    """Draw actual XYZ points; targets color only points inside their 3D boxes.

    This is annotation-based evidence, not a predicted segmentation mask. Raw
    and highlighted views use identical projection and extents. Selected box
    corners remain in the extent calculation even when the raw view hides them.
    """
    if not (np.isfinite([azimuth, elevation, range_m]).all() and range_m > 0):
        raise ValueError("视角和范围必须为有效数值，范围应大于 0")
    points = np.asarray(frame.points[:, :3], dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[(np.abs(points[:, :2]) <= range_m).all(axis=1)]
    # Deterministic sampling leaves enough points for evidence without loading CUDA.
    if len(points) > 30_000:
        points = points[np.linspace(0, len(points) - 1, 30_000, dtype=int)]
    a, e = np.deg2rad([azimuth, elevation])
    right = np.array([-np.sin(a), np.cos(a), 0])
    up = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    depth = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    basis = np.column_stack((right, up, depth))
    projected = points @ basis
    extent_points = [projected[:, :2]]
    boxes = []
    for box in frame.boxes:
        corners = np.asarray(box.corners, dtype=float)
        if corners.shape != (3, 8) or not np.isfinite(corners).all():
            continue
        if (target_colors or {}).get(box.token) and not (np.abs(corners[:2]) <= range_m).all():
            raise ValueError("目标超出三维显示范围，请增大范围")
        if (np.abs(corners[:2]) <= range_m).all():
            view = corners.T @ basis
            extent_points.append(view[:, :2])
            boxes.append((box, view))
    extent = np.concatenate(extent_points)
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    if not len(extent):
        draw.text((20, 20), "NO LIDAR POINTS IN VIEW", fill="#646464")
        return image
    lower, upper = extent.min(axis=0), extent.max(axis=0)
    span = np.maximum(upper - lower, 1.0)
    width, height = size
    scale = min((width - 40) / span[0], (height - 40) / span[1])
    center = (lower + upper) / 2

    def xy(view):
        return np.column_stack((width / 2 + (view[:, 0] - center[0]) * scale,
                                height / 2 - (view[:, 1] - center[1]) * scale))

    pixels = xy(projected)
    colors = np.empty((len(points), 3), dtype=np.uint8)
    if len(points):
        low, high = np.percentile(points[:, 2], [5, 95])
        shade = np.clip((points[:, 2] - low) / max(high - low, .1), 0, 1)
        colors[:] = np.column_stack((150 - 75 * shade, 159 - 75 * shade, 165 - 70 * shade))
    selected = np.zeros(len(points), dtype=bool)
    for box, _ in boxes:
        color = (target_colors or {}).get(box.token)
        if color:
            mask = points_in_box(points, box.corners)
            selected |= mask
            colors[mask] = ImageColor.getrgb(color)
    # Far-to-near drawing preserves visible scene depth. Evidence points are larger.
    for index in np.argsort(projected[:, 2], kind="stable"):
        x, y = pixels[index]
        radius = 2.2 if selected[index] else .75
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=tuple(colors[index]))
    if show_tracks:
        for track in frame.tracks:
            values = np.asarray(track.points, dtype=float)
            if len(values) > 1 and np.isfinite(values).all():
                draw.line([tuple(p) for p in xy(values @ basis)], fill="#81728d", width=2)
    for box, view in boxes:
        target = (target_colors or {}).get(box.token)
        if show_boxes or target:
            corners = xy(view)
            for start, end in BOX_EDGES:
                draw.line((tuple(corners[start]), tuple(corners[end])),
                          fill=target or "#a1a6aa", width=3 if target else 1)
    return image
