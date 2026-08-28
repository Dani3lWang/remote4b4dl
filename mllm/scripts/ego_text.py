"""
ego_text.py — B4DL Metatoken 文本渲染共享模块（论文 §4.1 / Appendix C / Figure 6）
==================================================================================
ego 运动状态 → 自然语言模板的单一实现来源，供三方共用，保证训练注入与推理注入
产生逐字符一致的 metatoken：

  - generate_ego_metadata.py  （per-scene / per-sequence 预生成条目 + 逐帧运动表）
  - inject_metatoken.py       （训练数据 metatoken 注入）
  - evaluation/test_b4dl.py   （推理时 metatoken 注入）

论文要求（Appendix C）："the metatoken descriptions of the first and last frames
referenced in the QA pair are concatenated" —— metatoken 描述的是 **QA 文本所引用
的首帧与末帧**（不一定是序列边界帧），因此需要一个能对任意 (first, last) 帧对
渲染文本的模块，而不只是预生成的序列边界条目。

逐帧运动表（ego_frame_motion.json，由 generate_ego_metadata.py --frame_motion 产出）：
    {
      "scene_id": [
        {"x": .., "y": .., "z": .., "yaw": ..,        # 该帧 ego_pose（全局系）
         "spd_next": .., "spd_prev": ..,              # 相对下一帧/上一帧的速度 m/s
         "yaw_next": .., "yaw_prev": ..,              # 相对下一帧/上一帧的航向变化（度，左+）
         "acc_next": .., "acc_prev": ..},             # 该帧加速度 m/s^2（前后向差分）
        ...   # 场景所有帧按索引排列；边界帧缺失邻帧的字段为 None
      ]
    }

本模块不依赖 nuscenes-devkit / torch，可独立 import。
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 自然语言模板（论文 Figure 6 样例风格）
# ---------------------------------------------------------------------------

def describe_position(lateral: float, longitudinal: float, is_relative: bool = True) -> str:
    """Describe the ego's position / movement direction.

    For is_relative=True (last frame relative to first frame):
        "slightly ahead and to the right"
    For is_relative=False (first frame, absolute-ish):
        "at the starting position" or movement direction
    """
    if is_relative:
        parts = []
        if abs(longitudinal) < 0.3:
            pass
        elif longitudinal > 0:
            parts.append("slightly ahead" if abs(longitudinal) < 5 else "ahead")
        else:
            parts.append("slightly behind" if abs(longitudinal) < 5 else "behind")
        if abs(lateral) < 0.3:
            pass
        elif lateral > 0:
            parts.append("and to the left" if parts else "to the left")
        else:
            parts.append("and to the right" if parts else "to the right")
        if not parts:
            return "nearly at the same position"
        return " ".join(parts)
    else:
        return "at the starting position"


def describe_terrain(z_val: float, z_ref: Optional[float] = None) -> str:
    """Describe terrain. For first frame, z_val is absolute. For last, compare to z_ref."""
    if z_ref is not None:
        dz = z_val - z_ref
        if abs(dz) < 0.5:
            return "on flat ground"
        elif dz > 0:
            return "on an uphill slope"
        else:
            return "on a downhill slope"
    return "on flat ground"


def describe_turn(yaw_change_deg: float) -> str:
    """Describe turning behaviour. yaw_change_deg is signed (positive=left)."""
    abs_yaw = abs(yaw_change_deg)
    if abs_yaw < 1.0:
        return "moving straight ahead"
    elif abs_yaw < 5.0:
        direction = "left" if yaw_change_deg > 0 else "right"
        return f"with a slight turn to the {direction} by about {abs_yaw:.0f} degrees"
    else:
        direction = "left" if yaw_change_deg > 0 else "right"
        return f"performing a {direction} turn by about {abs_yaw:.0f} degrees"


def describe_speed(speed_mps: float) -> str:
    """Describe ego speed in natural language."""
    if speed_mps < 0.5:
        return "nearly stationary"
    elif speed_mps < 2.0:
        return f"moving slowly at about {speed_mps:.1f} meters per second"
    elif speed_mps < 8.0:
        return f"moving at about {speed_mps:.0f} meters per second"
    else:
        return f"moving quickly at about {speed_mps:.0f} meters per second"


def describe_acceleration(accel_mpss: Optional[float]) -> str:
    """Describe acceleration. None → conservative 'nearly constant speed'."""
    if accel_mpss is None or abs(accel_mpss) < 0.1:
        return "with nearly constant speed"
    elif accel_mpss > 0:
        return f"with an acceleration of about {accel_mpss:.1f} meters per second squared"
    else:
        return f"with a deceleration of about {abs(accel_mpss):.1f} meters per second squared"


def frame_text(position: str, terrain: str, speed_mps: float,
               yaw_change_deg: float, accel_mpss: Optional[float]) -> str:
    """Assemble one frame description (paper Figure 6 style)."""
    return (f"The ego vehicle is {position} {terrain}, "
            f"{describe_speed(speed_mps)}, {describe_turn(yaw_change_deg)}, "
            f"{describe_acceleration(accel_mpss)}.")


# ---------------------------------------------------------------------------
# 任意帧范围渲染（论文 Appendix C："first and last frames referenced in the QA pair"）
# ---------------------------------------------------------------------------

def _disp_in_frame(x_ref: float, y_ref: float, yaw_ref: float,
                   x: float, y: float) -> Tuple[float, float]:
    """Displacement of (x, y) in the ego frame at (x_ref, y_ref, yaw_ref).

    Returns (longitudinal, lateral): forward(+)/backward(-), left(+)/right(-).
    """
    dx, dy = x - x_ref, y - y_ref
    cos_yaw = math.cos(-yaw_ref)
    sin_yaw = math.sin(-yaw_ref)
    longitudinal = cos_yaw * dx - sin_yaw * dy
    lateral = sin_yaw * dx + cos_yaw * dy
    return longitudinal, lateral


def render_meta_texts(frames: List[dict], first: int, last: int) -> Tuple[str, str]:
    """Render (first_frame_text, last_frame_text) for an arbitrary frame range.

    `frames` is one scene's per-frame motion list (see module docstring).
    `first`/`last` are indices into that list (scene-relative frame numbers as
    they appear in the QA text).

    - first frame: motion relative to the NEXT frame (paper: "relative to
      previous frames" — for the range start the informative direction is
      forward-looking), position "at the starting position".
    - last frame: motion relative to the PREVIOUS frame, position relative to
      the first frame.
    - first == last (single-frame QA reference): both descriptions use the
      frame's real neighbouring motion instead of a degenerate "stationary"
      text; the last-frame position then describes displacement from the
      previous frame.
    """
    n = len(frames)
    if n == 0:
        raise ValueError("empty frame table")
    first = max(0, min(first, n - 1))
    last = max(0, min(last, n - 1))
    if first > last:
        first, last = last, first

    f = frames[first]
    l = frames[last]

    # --- first frame: forward-looking motion ---
    f_speed = f.get("spd_next")
    f_yaw = f.get("yaw_next")
    f_acc = f.get("acc_next")
    if f_speed is None:
        f_speed = f.get("spd_prev")
    if f_yaw is None:
        f_yaw = f.get("yaw_prev")
    first_text = frame_text(
        describe_position(0, 0, is_relative=False),
        describe_terrain(f.get("z", 0.0)),
        f_speed if f_speed is not None else 0.0,
        f_yaw if f_yaw is not None else 0.0,
        f_acc,
    )

    # --- last frame: backward-looking motion, position relative to first ---
    l_speed = l.get("spd_prev")
    l_yaw = l.get("yaw_prev")
    l_acc = l.get("acc_prev")
    if l_speed is None:
        l_speed = l.get("spd_next")
    if l_yaw is None:
        l_yaw = l.get("yaw_next")

    if last == first and first > 0:
        # Single-frame reference: describe real displacement vs previous frame
        # so the text is informative (never a fabricated "stationary").
        p = frames[first - 1]
        lon, lat = _disp_in_frame(p["x"], p["y"], p["yaw"], f["x"], f["y"])
        position = describe_position(lat, lon, is_relative=True)
        terrain = describe_terrain(f.get("z", 0.0), z_ref=p.get("z"))
    else:
        lon, lat = _disp_in_frame(f["x"], f["y"], f["yaw"], l["x"], l["y"])
        position = describe_position(lat, lon, is_relative=True)
        terrain = describe_terrain(l.get("z", 0.0), z_ref=f.get("z"))
    last_text = frame_text(
        position,
        terrain,
        l_speed if l_speed is not None else 0.0,
        l_yaw if l_yaw is not None else 0.0,
        l_acc,
    )

    return first_text, last_text


def build_metatoken_line(first_text: str, last_text: str) -> str:
    """Connective-phrase metatoken body (paper Figure 6 red-highlighted words)."""
    return (f"The metadata of the first frame is '{first_text}' "
            f"and the metadata of the last frame is '{last_text}'")


def load_frame_motion(path: str) -> Dict[str, List[dict]]:
    with open(path) as fh:
        return json.load(fh)


def render_metatoken_for_range(frame_motion: Dict[str, List[dict]], scene_id: str,
                               first: int, last: int) -> Optional[str]:
    """Convenience: full '<meta>' body for a scene + frame range, or None on miss."""
    frames = frame_motion.get(scene_id)
    if not frames:
        return None
    first_text, last_text = render_meta_texts(frames, first, last)
    return build_metatoken_line(first_text, last_text)
