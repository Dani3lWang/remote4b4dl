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

此脚本为每个 sequence 生成首帧和末帧两段独立描述（论文 Appendix C：
"the metatoken descriptions of the first and last frames referenced in the
QA pair are concatenated"）。同时保留 per-scene 条目用于 fallback。

输出 ego_metadata.json：
  {
    "scene_id": {"first_frame": "...", "last_frame": "..."},           # per-scene (fallback)
    "scene_id_0_8": {"first_frame": "...", "last_frame": "..."},      # per-sequence
    "scene_id_30_38": {"first_frame": "...", "last_frame": "..."},    # per-sequence
    ...
  }

算法（依据论文 Figure 6 + Appendix C）：
  1. 从 sequence_metadata 获取每个 sequence 的 indices（如 [0,2,4,6,8]）
  2. 收集对应 scene 的所有 LIDAR_TOP ego_pose，按时间排序
  3. 对每个 sequence，取 indices[0] 和 indices[-1] 对应的 ego_pose
  4. 计算首帧速度、方向、加速度（首帧→次帧）
  5. 计算末帧速度、方向、加速度（次末帧→末帧），位置相对首帧
  6. 用自然语言模板转换为文本
  7. 同时生成 per-scene 条目（scene 首末帧）用于无帧号的问题

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

# 文本模板统一从 ego_text.py 导入（与 inject_metatoken.py / test_b4dl.py 共用同一实现）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ego_text import (  # noqa: E402
    describe_position as _describe_position,
    describe_terrain as _describe_terrain,
    describe_turn as _describe_turn,
    describe_speed as _describe_speed,
    describe_acceleration as _describe_acceleration,
    frame_text,
)

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
                   help="Output JSON path (per-scene + per-sequence text entries)")
    p.add_argument("--frame_motion", type=str, default=None,
                   help="Optional output JSON path for the per-frame motion table "
                        "(ego_frame_motion.json). Enables rendering metatokens for "
                        "ARBITRARY frame ranges referenced in QA pairs (paper "
                        "Appendix C), consumed by inject_metatoken.py --frame_motion "
                        "and test_b4dl.py --frame_motion.")
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


def _build_frame_table(poses: List[dict]) -> List[dict]:
    """Build the per-frame motion table for one scene (ego_text.py format).

    For every frame index i (over ALL scene frames, not just sequences):
        x/y/z/yaw : ego pose at frame i (global frame)
        spd_prev  : speed of segment (i-1 → i), None at i == 0
        spd_next  : speed of segment (i → i+1), None at i == n-1
        yaw_prev  : signed yaw change (i-1 → i) in degrees, None at i == 0
        yaw_next  : signed yaw change (i → i+1) in degrees, None at i == n-1
        acc_next  : forward-looking acceleration at i from segments (i→i+1, i+1→i+2), None if unavailable
        acc_prev  : backward-looking acceleration at i from segments (i-2→i-1, i-1→i), None if unavailable
    """
    n = len(poses)
    ts = [p["timestamp"] for p in poses]

    # Segment quantities: seg_speed[i] / seg_yaw[i] describe motion from frame i to i+1
    seg_speed: List[Optional[float]] = [None] * max(n - 1, 0)
    seg_yaw: List[Optional[float]] = [None] * max(n - 1, 0)
    for i in range(n - 1):
        seg_speed[i] = _compute_frame_motion(poses[i], poses[i + 1])["speed_mps"]
        q0 = Quaternion(poses[i]["rotation"])
        q1 = Quaternion(poses[i + 1]["rotation"])
        d = _quaternion_to_yaw(q1) - _quaternion_to_yaw(q0)
        d = math.atan2(math.sin(d), math.cos(d))
        seg_yaw[i] = float(math.degrees(d))

    table: List[dict] = []
    for i in range(n):
        yaw_i = _quaternion_to_yaw(Quaternion(poses[i]["rotation"]))
        rec = {
            "x": float(poses[i]["translation"][0]),
            "y": float(poses[i]["translation"][1]),
            "z": float(poses[i]["translation"][2]),
            "yaw": float(yaw_i),
            "spd_prev": seg_speed[i - 1] if i >= 1 else None,
            "spd_next": seg_speed[i] if i <= n - 2 else None,
            "yaw_prev": seg_yaw[i - 1] if i >= 1 else None,
            "yaw_next": seg_yaw[i] if i <= n - 2 else None,
            "acc_next": None,
            "acc_prev": None,
        }
        # acc_next[i] uses segments (i→i+1) and (i+1→i+2), i.e. frames i..i+2
        if i <= n - 3:
            dt = abs(ts[i + 2] - ts[i]) / 1e6 / 2.0
            rec["acc_next"] = _compute_acceleration(seg_speed[i], seg_speed[i + 1], dt)
        # acc_prev[i] uses segments (i-2→i-1) and (i-1→i), i.e. frames i-2..i
        if i >= 2:
            dt = abs(ts[i] - ts[i - 2]) / 1e6 / 2.0
            rec["acc_prev"] = _compute_acceleration(seg_speed[i - 2], seg_speed[i - 1], dt)
        table.append(rec)
    return table


# ---------------------------------------------------------------------------
# 自然语言模板已迁移至 ego_text.py（单一来源），此处仅保留旧签名包装
# ---------------------------------------------------------------------------


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
    return frame_text(
        _describe_position(motion["lateral_m"], motion["longitudinal_m"],
                           is_relative=is_relative),
        _describe_terrain(z_val, z_ref),
        motion["speed_mps"],
        motion["yaw_change_deg"],
        accel,
    )


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


def _compute_ego_for_pose_range(
    poses: List[dict],
    first_idx: int,
    last_idx: int,
) -> Dict[str, str]:
    """Compute ego-motion text for a specific frame range within a scene.

    first_idx / last_idx are indices into the *poses* list (which is the
    scene's full ego_pose array sorted by timestamp).

    Returns {"first_frame": "...", "last_frame": "..."} where:
    - first_frame: motion at poses[first_idx] (relative to poses[first_idx+1])
    - last_frame: motion at poses[last_idx] (relative to poses[last_idx-1],
      position relative to poses[first_idx])
    """
    pose_first = poses[first_idx]
    pose_last = poses[last_idx]

    # Need at least 2 frames in the range for motion computation
    if first_idx == last_idx:
        # Single frame — no motion, just position
        return {
            "first_frame": frame_to_text(
                {"lateral_m": 0, "longitudinal_m": 0, "speed_mps": 0,
                 "yaw_change_deg": 0},
                pose_first["translation"][2], z_ref=None,
                is_relative=False, accel=None),
            "last_frame": frame_to_text(
                {"lateral_m": 0, "longitudinal_m": 0, "speed_mps": 0,
                 "yaw_change_deg": 0},
                pose_first["translation"][2], z_ref=None,
                is_relative=False, accel=None),
        }

    # First frame: motion relative to the next frame in the sequence
    pose_after_first = poses[first_idx + 1] if first_idx + 1 <= last_idx else pose_first
    first_motion = _compute_frame_motion(pose_first, pose_after_first)
    z_first = pose_first["translation"][2]

    # First frame acceleration (needs at least 3 frames in range)
    first_accel = None
    if last_idx - first_idx >= 2:
        pose_third = poses[first_idx + 2]
        speed_01 = first_motion["speed_mps"]
        motion_12 = _compute_frame_motion(pose_after_first, pose_third)
        speed_12 = motion_12["speed_mps"]
        dt_02 = (pose_third["timestamp"] - pose_first["timestamp"]) / 1e6 / 2.0
        first_accel = _compute_acceleration(speed_01, speed_12, dt_02)

    first_text = frame_to_text(
        first_motion, z_first, z_ref=None,
        is_relative=False, accel=first_accel)

    # Last frame: motion relative to the previous frame in the sequence
    pose_before_last = poses[last_idx - 1] if last_idx - 1 >= first_idx else pose_last
    last_motion = _compute_frame_motion(pose_before_last, pose_last)
    z_last = pose_last["translation"][2]

    # Last frame acceleration (needs at least 3 frames from the end)
    last_accel = None
    if last_idx - first_idx >= 2:
        pose_prev_prev = poses[last_idx - 2]
        speed_prev = _compute_frame_motion(pose_prev_prev, pose_before_last)["speed_mps"]
        speed_curr = last_motion["speed_mps"]
        dt_end = (pose_last["timestamp"] - pose_prev_prev["timestamp"]) / 1e6 / 2.0
        last_accel = _compute_acceleration(speed_prev, speed_curr, dt_end)

    # Position of last frame relative to first frame
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

    return {"first_frame": first_text, "last_frame": last_text}


def generate_metadata(nusc: NuScenes,
                      scene_metadata: list,
                      sequence_metadata: list = None) -> Dict[str, Dict[str, str]]:
    """Generate ego-motion text for every scene AND every sequence.

    Returns a dict with two types of entries:
    - Per-scene (fallback): {scene_id: {"first_frame": "...", "last_frame": "..."}}
    - Per-sequence: {f"{scene_id}_{first_idx}_{last_idx}": {"first_frame": "...", "last_frame": "..."}}

    Per-sequence entries are keyed by f"{scene_id}_{indices[0]}_{indices[-1]}"
    so that inject_metatoken.py and build_query() can look up the correct
    ego data by parsing frame numbers from the question text.

    Per-scene entries use the scene's first and last frames (indices 0 and -1)
    and are used as fallback for questions that don't mention specific frame
    numbers (e.g. time_grounding questions at inference time).
    """
    out: Dict[str, Dict[str, str]] = {}
    missing_pose = 0
    missing_seq = 0
    # Build scene_token → scene_id mapping
    token_to_id: Dict[str, str] = {}
    for scene in scene_metadata:
        token_to_id[scene["scene_token"]] = scene["scene_id"]

    # Group sequences by scene_token
    sequences_by_scene: Dict[str, List[dict]] = {}
    if sequence_metadata:
        for seq in sequence_metadata:
            st = seq.get("scene_token")
            if st:
                sequences_by_scene.setdefault(st, []).append(seq)

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

        # --- Per-scene entry (fallback): first & last frame of entire scene ---
        scene_ego = _compute_ego_for_pose_range(poses, 0, len(poses) - 1)
        out[scene_id] = scene_ego

        # --- Per-sequence entries: first & last frame of each sequence ---
        seqs = sequences_by_scene.get(scene_token, [])
        if not seqs:
            missing_seq += 1
            continue

        for seq in seqs:
            indices = seq.get("indices", [])
            if not indices or len(indices) < 1:
                continue

            first_idx = indices[0]
            last_idx = indices[-1]

            # Bounds check
            if last_idx >= len(poses) or first_idx < 0:
                continue

            seq_ego = _compute_ego_for_pose_range(poses, first_idx, last_idx)
            seq_key = f"{scene_id}_{first_idx}_{last_idx}"
            out[seq_key] = seq_ego

    if missing_pose:
        print(f"  ⚠ {missing_pose} scenes skipped (missing ego_pose data)")
    if missing_seq:
        print(f"  ⚠ {missing_seq} scenes had no sequence metadata (per-scene only)")

    n_per_scene = sum(1 for k in out if "_" not in k)
    n_per_seq = len(out) - n_per_scene
    print(f"  Generated {n_per_scene} per-scene + {n_per_seq} per-sequence entries "
          f"({len(out)} total)")

    return out


def generate_frame_tables(nusc: NuScenes, scene_metadata: list) -> Dict[str, List[dict]]:
    """Build the per-frame motion table for every scene (see _build_frame_table).

    Used by inject_metatoken.py --frame_motion / test_b4dl.py --frame_motion to
    render metatokens for ARBITRARY (first, last) frame pairs referenced in QA
    text (paper Appendix C), instead of only precomputed sequence boundaries.
    """
    tables: Dict[str, List[dict]] = {}
    for scene in scene_metadata:
        try:
            poses = _collect_scene_frames(nusc, scene["scene_token"])
        except Exception:
            continue
        if len(poses) < 2:
            continue
        tables[scene["scene_id"]] = _build_frame_table(poses)
    print(f"  Frame tables: {len(tables)} scenes "
          f"({sum(len(v) for v in tables.values())} frames total)")
    return tables


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

    if args.frame_motion:
        print("Generating per-frame motion table ...")
        frame_tables = generate_frame_tables(nusc, scene_meta)
        fm_dir = os.path.dirname(args.frame_motion)
        if fm_dir:
            os.makedirs(fm_dir, exist_ok=True)
        with open(args.frame_motion, "w") as f:
            json.dump(frame_tables, f, ensure_ascii=False)
        print(f"Saved frame motion table to {args.frame_motion}")

    # Print a few examples
    print("\nExample per-scene entries:")
    n_shown = 0
    for sid, data in ego_meta.items():
        if "_" in sid:  # skip per-sequence keys
            continue
        print(f"  [{sid}]")
        print(f"    first: {data['first_frame'][:120]}...")
        print(f"    last:  {data['last_frame'][:120]}...")
        n_shown += 1
        if n_shown >= 2:
            break

    print("\nExample per-sequence entries:")
    n_shown = 0
    for sid, data in ego_meta.items():
        if "_" not in sid:  # skip per-scene keys
            continue
        print(f"  [{sid}]")
        print(f"    first: {data['first_frame'][:120]}...")
        print(f"    last:  {data['last_frame'][:120]}...")
        n_shown += 1
        if n_shown >= 4:
            break


if __name__ == "__main__":
    main()
