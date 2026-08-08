#!/usr/bin/env python3
"""
generate_ego_metadata.py — 从 nuScenes 提取 Metatoken 所需的 ego motion 文本描述
====================================================================================
论文 §4.1 / Appendix C / Figure 6：Metatoken 为序列首帧与末帧之间的 ego vehicle
运动信息文本，格式如：

  The ego vehicle is slightly ahead and to the right on flat ground, moving at
  about 5 meters per second, performing a left turn by about 11 degrees, with
  an acceleration of about 1 meters per second squared.

此脚本为每个 sequence 生成该文本，输出 ego_metadata.json（scene_id → 元信息文本）。

算法（依据论文 Figure 6 样例推算）：
  1. 取每个 sequence 的 first frame 和 last frame 的 ego_pose
  2. 计算两帧之间的位移、方向变化、速度、加速度
  3. 用自然语言模板转换为文本

用法：
    cd mllm
    python scripts/generate_ego_metadata.py \
        --nuscenes_root /path/to/nuScenes \
        --scene_metadata ../encoders/lidarclip/annotations/scene_metadata.json \
        --sequence_metadata ../encoders/lidarclip/annotations/sequence_metadata.json \
        --output ./b4dl_dataset/ego_metadata.json
"""

import os
import sys
import json
import argparse
import math
import numpy as np
from typing import Dict, Optional, Tuple
from collections import defaultdict
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
    # Quaternion to Euler: yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy² + qz²))
    qw, qx, qy, qz = q.w, q.x, q.y, q.z
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _compute_ego_motion(pose_first: dict, pose_last: dict) -> Dict[str, float]:
    """Compute ego motion metrics between two ego_pose records.

    Returns dict with:
        displacement_m:  3D Euclidean displacement (meters)
        lateral_m:       Left(+)/right(-) displacement (meters)
        longitudinal_m:  Forward(+)/backward(-) displacement (meters)
        speed_mps:       Average speed (m / s)
        yaw_change_deg:  Absolute yaw change (degrees, positive = left turn)
        acceleration_mpss: Average acceleration (m / s²)
        dt_sec:           Time interval (seconds)
    """
    # Translation
    t1 = np.array(pose_first["translation"])
    t2 = np.array(pose_last["translation"])
    disp_vec = t2 - t1
    displacement = float(np.linalg.norm(disp_vec))

    # Decompose into longitudinal (forward) and lateral (right) components
    # Use first frame's heading as reference
    q1 = Quaternion(pose_first["rotation"])
    yaw1 = _quaternion_to_yaw(q1)
    # Rotate displacement vector into ego frame
    cos_yaw = math.cos(-yaw1)
    sin_yaw = math.sin(-yaw1)
    dx_ego = cos_yaw * disp_vec[0] - sin_yaw * disp_vec[1]
    dy_ego = sin_yaw * disp_vec[0] + cos_yaw * disp_vec[1]
    longitudinal = float(dx_ego)   # forward(+)
    lateral = float(dy_ego)         # left(+)

    # Time
    dt = abs(pose_last["timestamp"] - pose_first["timestamp"]) / 1e6  # microseconds → seconds

    # Speed
    speed = displacement / dt if dt > 0 else 0.0

    # Yaw change (direction / turning)
    q2 = Quaternion(pose_last["rotation"])
    yaw2 = _quaternion_to_yaw(q2)
    yaw_diff = yaw2 - yaw1
    # Normalize to [-pi, pi]
    yaw_diff = math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))
    yaw_change_deg = float(abs(math.degrees(yaw_diff)))

    # Acceleration (simplified: speed change / time)
    acceleration = speed / dt if dt > 0 else 0.0

    return {
        "displacement_m": displacement,
        "lateral_m": lateral,
        "longitudinal_m": longitudinal,
        "speed_mps": speed,
        "yaw_change_deg": yaw_change_deg,
        "acceleration_mpss": acceleration,
        "dt_sec": dt,
    }


# ---------------------------------------------------------------------------
# 自然语言模板（论文 Figure 6 样例风格）
# ---------------------------------------------------------------------------

def _describe_direction(lateral: float, longitudinal: float) -> str:
    """Describe the ego's relative position / movement direction."""
    parts = []

    # Forward / backward
    if abs(longitudinal) < 0.3:
        pass  # negligible
    elif longitudinal > 0:
        parts.append("moving forward")
    else:
        parts.append("moving backward")

    # Left / right
    if abs(lateral) < 0.3:
        pass
    elif lateral > 0:
        parts.append("and to the left" if parts else "to the left")
    else:
        parts.append("and to the right" if parts else "to the right")

    if not parts:
        return "is nearly stationary"

    return "is " + " ".join(parts)


def _describe_terrain(z_first: float, z_last: float) -> str:
    """Describe vertical terrain change."""
    dz = z_last - z_first
    if abs(dz) < 0.5:
        return "on flat ground"
    elif dz > 0:
        return "on an uphill slope"
    else:
        return "on a downhill slope"


def _describe_turn(yaw_change_deg: float) -> str:
    """Describe turning behaviour."""
    if yaw_change_deg < 1.0:
        return "moving straight ahead"
    elif yaw_change_deg < 5.0:
        direction = "left" if yaw_change_deg > 0 else "right"
        return f"with a slight turn to the {direction} by about {yaw_change_deg:.0f} degrees"
    else:
        direction = "left" if yaw_change_deg > 0 else "right"
        return f"performing a turn to the {direction} by about {yaw_change_deg:.0f} degrees"


def _describe_speed(speed_mps: float) -> str:
    """Describe ego speed in natural language."""
    speed_kmh = speed_mps * 3.6
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
        return f"accelerating at about {accel_mpss:.1f} meters per second squared"
    else:
        return f"decelerating at about {abs(accel_mpss):.1f} meters per second squared"


def motion_to_text(motion: Dict[str, float], pose_first: dict, pose_last: dict) -> str:
    """Convert motion metrics dict into a natural language description.

    Output format (paper Figure 6 style):
      "The ego vehicle is slightly ahead and to the right on flat ground,
       moving at about 5 meters per second, performing a left turn by about
       11 degrees, with an acceleration of about 1 meters per second squared."
    """
    direction = _describe_direction(motion["lateral_m"], motion["longitudinal_m"])
    terrain = _describe_terrain(
        pose_first["translation"][2], pose_last["translation"][2])
    turn = _describe_turn(motion["yaw_change_deg"])
    speed = _describe_speed(motion["speed_mps"])
    accel = _describe_acceleration(motion["acceleration_mpss"])

    # Compose
    parts = [f"The ego vehicle {direction} {terrain}"]
    parts.append(f"{speed}, {turn}, {accel}")

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def generate_metadata(nusc: NuScenes,
                      scene_metadata: list,
                      sequence_metadata: list) -> Dict[str, str]:
    """Generate ego-motion text for every sequence, keyed by scene_id.

    Returns: {scene_id: metadata_text}
    """
    # Build a lookup: scene_token → list of sequences sorted by first-frame timestamp
    scene_seqs: Dict[str, list] = defaultdict(list)
    for seq in sequence_metadata:
        scene_seqs[seq["scene_token"]].append(seq)

    # For each scene, pick the sequence with the most complete frame range
    out: Dict[str, str] = {}
    missing_pose = 0

    for scene in scene_metadata:
        scene_token = scene["scene_token"]
        scene_id = scene["scene_id"]
        seqs = scene_seqs.get(scene_token, [])

        if not seqs:
            continue

        # Use the longest sequence as representative for this scene
        best_seq = max(seqs, key=lambda s: len(s.get("indices", s.get("frames", []))))

        # Get first and last sample_data entries to obtain ego_pose tokens
        frames = best_seq.get("frames", best_seq.get("indices", []))
        if len(frames) < 2:
            continue

        first_frame = frames[0]
        last_frame = frames[-1]

        # Extract frame tokens — frames may be dicts or plain sample_data tokens
        if isinstance(first_frame, dict):
            # Use LIDAR_TOP as the reference sensor for ego_pose lookup
            first_token = (first_frame.get("TOKEN_LIDAR_TOP")
                           or first_frame.get("token"))
            last_token = (last_frame.get("TOKEN_LIDAR_TOP")
                          or last_frame.get("token"))
        else:
            first_token = first_frame
            last_token = last_frame

        if not first_token or not last_token:
            missing_pose += 1
            continue

        try:
            sd_first = nusc.get("sample_data", first_token)
            sd_last = nusc.get("sample_data", last_token)
            pose_first = nusc.get("ego_pose", sd_first["ego_pose_token"])
            pose_last = nusc.get("ego_pose", sd_last["ego_pose_token"])
        except Exception:
            missing_pose += 1
            continue

        motion = _compute_ego_motion(pose_first, pose_last)
        text = motion_to_text(motion, pose_first, pose_last)
        out[scene_id] = text

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
    for i, (sid, text) in enumerate(ego_meta.items()):
        if i >= 3:
            break
        print(f"  [{sid}] {text[:120]}...")


if __name__ == "__main__":
    main()
