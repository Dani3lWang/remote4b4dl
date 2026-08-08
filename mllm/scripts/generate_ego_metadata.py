#!/usr/bin/env python3
"""
generate_ego_metadata.py — 从 nuScenes 提取 Metatoken 所需的 ego motion 文本描述
====================================================================================
论文 §4.1 / Appendix C / Figure 6：Metatoken 为序列首帧与末帧各自的 ego vehicle
运动状态文本，用连接词拼接，格式如：

  The metadata of the first frame is 'The ego vehicle is at the starting
  position on flat ground, moving at about 5 meters per second, moving
  straight ahead, with nearly constant speed.' and the metadata of the last
  frame is 'The ego vehicle is slightly ahead and to the right on flat
  ground, moving at about 3 meters per second, performing a left turn by
  about 11 degrees, with an acceleration of about 1 meters per second
  squared.'

此脚本为每个 scene 生成首帧和末帧两段独立描述，输出 ego_metadata.json：
  {scene_id: {"first_frame": "...", "last_frame": "..."}}

算法（依据论文 Figure 6）：
  1. 收集每个 scene 的所有 LIDAR_TOP sample_data token，按时间排序
  2. 取首帧和次帧的 ego_pose → 计算首帧速度、方向、加速度
  3. 取末帧和次末帧的 ego_pose → 计算末帧速度、方向、加速度
  4. 末帧位置/朝向相对于首帧计算
  5. 用自然语言模板转换为文本

用法：
    cd mllm
    python scripts/generate_ego_metadata.py \
        --nuscenes_root /path/to/nuScenes \
        --scene_metadata ../encoders/lidarclip/annotations/scene_metadata.json \
        --sequence_metadata ../encoders/lidarclip/annotations/sequence_metadata.json \
        --output ./b4dl_dataset/ego_metadata.json
"""
from __future__ import annotations  # defer annotation eval (Quaternion may be absent)

import os
import sys
import json
import argparse
import math
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

try:
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion
    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False
    print("Warning: nuscenes-devkit / pyquaternion not installed.")


def parse_args():
    p = argparse.ArgumentParser(description="Generate ego-motion metadata text for B4DL Metatoken")
    p.add_argument("--nuscenes_root", type=str, required=True,
                   help="Path to nuScenes dataset root")
    p.add_argument("--scene_metadata", type=str, required=True,
                   help="Path to scene_metadata.json")
    p.add_argument("--sequence_metadata", type=str, required=True,
                   help="Path to sequence_metadata.json")
    p.add_argument("--output", type=str, default="./b4dl_dataset/ego_metadata.json",
                   help="Output JSON path")
    p.add_argument("--version", type=str, default="v1.0-trainval",
                   help="nuScenes version")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 运动计算
# ---------------------------------------------------------------------------

def _quaternion_to_yaw(q: Quaternion) -> float:
    """Extract yaw angle (rotation around Z axis) from quaternion, in radians."""
    qw, qx, qy, qz = q.w, q.x, q.y, q.z
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _compute_frame_motion(pose_prev: dict, pose_curr: dict) -> Dict[str, float]:
    """Compute instantaneous motion at *pose_curr* relative to *pose_prev*.

    Returns dict with:
        displacement_m:   3D Euclidean displacement (meters)
        lateral_m:        Left(+)/right(-) displacement relative to prev heading
        longitudinal_m:   Forward(+)/backward(-) displacement
        speed_mps:        Instantaneous speed (m / s)
        yaw_change_deg:   Signed yaw change (degrees, positive = left turn)
        dt_sec:           Time interval (seconds)
    """
    t_prev = np.array(pose_prev["translation"])
    t_curr = np.array(pose_curr["translation"])
    disp_vec = t_curr - t_prev
    displacement = float(np.linalg.norm(disp_vec))

    # Decompose into longitudinal/lateral using prev frame's heading
    q_prev = Quaternion(pose_prev["rotation"])
    yaw_prev = _quaternion_to_yaw(q_prev)
    cos_yaw = math.cos(-yaw_prev)
    sin_yaw = math.sin(-yaw_prev)
    dx_ego = cos_yaw * disp_vec[0] - sin_yaw * disp_vec[1]
    dy_ego = sin_yaw * disp_vec[0] + cos_yaw * disp_vec[1]
    longitudinal = float(dx_ego)   # forward(+)
    lateral = float(dy_ego)         # left(+)

    # Time
    dt = abs(pose_curr["timestamp"] - pose_prev["timestamp"]) / 1e6
    speed = displacement / dt if dt > 0 else 0.0

    # Yaw change — preserve sign (positive = left turn)
    q_curr = Quaternion(pose_curr["rotation"])
    yaw_curr = _quaternion_to_yaw(q_curr)
    yaw_diff = yaw_curr - yaw_prev
    yaw_diff = math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))
    yaw_change_deg = float(math.degrees(yaw_diff))  # signed, not abs

    return {
        "displacement_m": displacement,
        "lateral_m": lateral,
        "longitudinal_m": longitudinal,
        "speed_mps": speed,
        "yaw_change_deg": yaw_change_deg,
        "dt_sec": dt,
    }


def _compute_acceleration(speed_prev: float, speed_curr: float, dt_sec: float) -> float:
    """Compute acceleration as Δspeed / Δt (m/s²)."""
    if dt_sec <= 0:
        return 0.0
    return (speed_curr - speed_prev) / dt_sec


# ---------------------------------------------------------------------------
# 自然语言模板（论文 Figure 6 样例风格）
# ---------------------------------------------------------------------------

def _describe_position(lateral: float, longitudinal: float, is_relative: bool = True) -> str:
    """Describe the ego's position / movement direction.

    For is_relative=True (last frame relative to first frame):
        "slightly ahead and to the right"
    For is_relative=False (first frame, absolute-ish):
        "at the starting position" or movement direction
    """
    if is_relative:
        parts = []
        # Forward / backward
        if abs(longitudinal) < 0.3:
            pass
        elif longitudinal > 0:
            parts.append("slightly ahead" if abs(longitudinal) < 5 else "ahead")
        else:
            parts.append("slightly behind" if abs(longitudinal) < 5 else "behind")
        # Left / right
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
        # First frame: describe as starting position
        return "at the starting position"


def _describe_terrain(z_val: float, z_ref: Optional[float] = None) -> str:
    """Describe terrain. For first frame, z_val is absolute. For last, compare to z_ref."""
    if z_ref is not None:
        dz = z_val - z_ref
        if abs(dz) < 0.5:
            return "on flat ground"
        elif dz > 0:
            return "on an uphill slope"
        else:
            return "on a downhill slope"
    else:
        # First frame — just check if near sea level or elevated
        return "on flat ground"


def _describe_turn(yaw_change_deg: float) -> str:
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


def _describe_speed(speed_mps: float) -> str:
    """Describe ego speed in natural language."""
    if speed_mps < 0.5:
        return "nearly stationary"
    elif speed_mps < 2.0:
        return f"moving slowly at about {speed_mps:.1f} meters per second"
    elif speed_mps < 8.0:
        return f"moving at about {speed_mps:.0f} meters per second"
    else:
        return f"moving quickly at about {speed_mps:.0f} meters per second"


def _describe_acceleration(accel_mpss: float) -> str:
    """Describe acceleration."""
    if abs(accel_mpss) < 0.1:
        return "with nearly constant speed"
    elif accel_mpss > 0:
        return f"with an acceleration of about {accel_mpss:.1f} meters per second squared"
    else:
        return f"with a deceleration of about {abs(accel_mpss):.1f} meters per second squared"


def frame_to_text(motion: Dict[str, float], z_val: float,
                  z_ref: Optional[float] = None,
                  is_relative: bool = True,
                  accel: Optional[float] = None) -> str:
    """Convert motion metrics into a natural-language frame description.

    Output format (paper Figure 6 style):
      "The ego vehicle is slightly ahead and to the right on flat ground,
       moving at about 5 meters per second, performing a left turn by about
       11 degrees, with an acceleration of about 1 meters per second squared."
    """
    position = _describe_position(motion["lateral_m"], motion["longitudinal_m"],
                                  is_relative=is_relative)
    terrain = _describe_terrain(z_val, z_ref)
    turn = _describe_turn(motion["yaw_change_deg"])
    speed = _describe_speed(motion["speed_mps"])
    if accel is not None:
        accel_str = _describe_acceleration(accel)
    else:
        accel_str = "with nearly constant speed"

    return f"The ego vehicle is {position} {terrain}, {speed}, {turn}, {accel_str}."


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def _collect_scene_frames(nusc: NuScenes, scene_token: str) -> List[dict]:
    """Collect all LIDAR_TOP sample_data entries for a scene, sorted by timestamp.

    Returns list of ego_pose dicts sorted by timestamp.
    """
    scene = nusc.get("scene", scene_token)
    first_sample_token = scene["first_sample_token"]

    # Walk through all samples in the scene
    poses = []
    sample_token = first_sample_token
    while sample_token:
        sample = nusc.get("sample", sample_token)
        sd_token = sample["data"].get("LIDAR_TOP")
        if sd_token:
            sd = nusc.get("sample_data", sd_token)
            pose = nusc.get("ego_pose", sd["ego_pose_token"])
            poses.append(pose)
        sample_token = sample["next"]

    # Sort by timestamp just in case
    poses.sort(key=lambda p: p["timestamp"])
    return poses


def generate_metadata(nusc: NuScenes,
                      scene_metadata: list,
                      sequence_metadata: list = None) -> Dict[str, Dict[str, str]]:
    """Generate ego-motion text for every scene.

    Returns: {scene_id: {"first_frame": "text", "last_frame": "text"}}

    For each scene, collects all ego_pose entries sorted by timestamp
    directly from the nuScenes scene data (sequence_metadata is accepted
    for backward compat but not used — we walk the sample chain instead).
    - first_frame: motion at first frame (using first→second pose)
    - last_frame: motion at last frame (using second-to-last→last pose,
      position relative to first frame)
    """
    out: Dict[str, Dict[str, str]] = {}
    missing_pose = 0

    for scene in scene_metadata:
        scene_token = scene["scene_token"]
        scene_id = scene["scene_id"]

        try:
            poses = _collect_scene_frames(nusc, scene_token)
        except Exception:
            missing_pose += 1
            continue

        if len(poses) < 2:
            missing_pose += 1
            continue

        pose_first = poses[0]
        pose_last = poses[-1]
        pose_second = poses[1] if len(poses) > 1 else pose_first
        pose_prev_last = poses[-2] if len(poses) > 1 else pose_last

        # --- First frame motion (relative to second frame) ---
        first_motion = _compute_frame_motion(pose_first, pose_second)
        z_first = pose_first["translation"][2]

        # First frame acceleration (needs at least 3 frames)
        # speed_01 is the mean speed over [t0,t1] (centered at its midpoint),
        # speed_12 over [t1,t2]; the Δt between the two samples is therefore
        # half the full span (t2 - t0) / 2.
        first_accel = None
        if len(poses) >= 3:
            pose_third = poses[2]
            speed_01 = first_motion["speed_mps"]
            motion_12 = _compute_frame_motion(pose_second, pose_third)
            speed_12 = motion_12["speed_mps"]
            dt_02 = (pose_third["timestamp"] - pose_first["timestamp"]) / 1e6 / 2.0
            first_accel = _compute_acceleration(speed_01, speed_12, dt_02)

        first_text = frame_to_text(
            first_motion, z_first, z_ref=None,
            is_relative=False, accel=first_accel)

        # --- Last frame motion (relative to second-to-last) ---
        last_motion = _compute_frame_motion(pose_prev_last, pose_last)
        z_last = pose_last["translation"][2]

        # Last frame acceleration (needs at least 3 frames from the end)
        # Same midpoint reasoning as above: Δt between the two interval-average
        # speeds is half the full span (t_last - t_prev_prev) / 2.
        last_accel = None
        if len(poses) >= 3:
            pose_prev_prev = poses[-3]
            speed_prev = _compute_frame_motion(pose_prev_prev, pose_prev_last)["speed_mps"]
            speed_curr = last_motion["speed_mps"]
            dt_end = (pose_last["timestamp"] - pose_prev_prev["timestamp"]) / 1e6 / 2.0
            last_accel = _compute_acceleration(speed_prev, speed_curr, dt_end)

        # For last frame, position is relative to first frame
        # Recompose lateral/longitudinal using first frame heading
        t_first = np.array(pose_first["translation"])
        t_last = np.array(pose_last["translation"])
        disp_vec = t_last - t_first
        q_first = Quaternion(pose_first["rotation"])
        yaw_first = _quaternion_to_yaw(q_first)
        cos_yaw = math.cos(-yaw_first)
        sin_yaw = math.sin(-yaw_first)
        dx_ego = cos_yaw * disp_vec[0] - sin_yaw * disp_vec[1]
        dy_ego = sin_yaw * disp_vec[0] + cos_yaw * disp_vec[1]

        last_motion_relative = {
            "lateral_m": float(dy_ego),
            "longitudinal_m": float(dx_ego),
            "speed_mps": last_motion["speed_mps"],
            "yaw_change_deg": last_motion["yaw_change_deg"],
        }

        last_text = frame_to_text(
            last_motion_relative, z_last, z_ref=z_first,
            is_relative=True, accel=last_accel)

        out[scene_id] = {"first_frame": first_text, "last_frame": last_text}

    if missing_pose:
        print(f"  ⚠ {missing_pose} scenes skipped (missing ego_pose data)")

    return out


def main():
    args = parse_args()

    if not HAS_NUSCENES:
        print("Error: nuscenes-devkit required. Run: pip install nuscenes-devkit pyquaternion")
        sys.exit(1)

    print(f"Loading nuScenes from {args.nuscenes_root} ...")
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=False)

    print(f"Loading metadata ...")
    with open(args.scene_metadata) as f:
        scene_meta = json.load(f)
    with open(args.sequence_metadata) as f:
        seq_meta = json.load(f)
    print(f"  Scenes: {len(scene_meta)}, Sequences: {len(seq_meta)}")

    print("Generating ego-motion metadata ...")
    ego_meta = generate_metadata(nusc, scene_meta, seq_meta)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(ego_meta, f, indent=2, ensure_ascii=False)
    print(f"Generated metadata for {len(ego_meta)}/{len(scene_meta)} scenes")
    print(f"Saved to {args.output}")

    # Print a few examples
    print("\nExample outputs:")
    for i, (sid, data) in enumerate(ego_meta.items()):
        if i >= 3:
            break
        print(f"  [{sid}]")
        print(f"    first: {data['first_frame'][:120]}...")
        print(f"    last:  {data['last_frame'][:120]}...")


if __name__ == "__main__":
    main()
