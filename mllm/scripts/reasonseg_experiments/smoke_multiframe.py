#!/usr/bin/env python3
"""选项 A 的冒烟中止门：跑 GPU 之前先把多帧装配的四件事证伪掉。

纯 CPU、不碰编码器。任何一条不过就不许起 25–35 h 的长跑：

1. **映射正确性** —— anchor 的 `panoptic_id -> instance_token` 是靠"3D 框套 GT 掩码"反查的，
   所以框与掩码必须几乎重合（IoU 中位数 ≥ 0.9）。对不上说明映射错了，跨帧正例全是噪声。
2. **跨帧收益** —— 邻帧到底给同一实例带来多少新正例点。若中位数≈0，A 根本没东西可测。
3. **各臂自己的门槛** —— 判据用"距本臂门槛的倍数"，所以必须先量出每臂的 K、N 中位数，
   把 `1 − K/N` 重算出来，禁止沿用单帧的 0.99963。
4. **a1 与 a3 的掩码集合必须相同** —— 点和框一起变换时框包含关系是不变量，两臂只差
   "点云的相对对齐"。若这里不相同，说明实现把某一侧漏了变换，两臂就不是单变量对照。
5. **a2 是严格零新信息** —— 复制块必须逐位相同，且 K、N 同乘 F ⇒ 门槛与单帧一致（这是
   它作失效门的前提：a2 的 AUC 应落在单帧带内）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MLLM_ROOT = Path(__file__).resolve().parents[2]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))

from vtimellm.segmentation.config import ReasonSegConfig  # noqa: E402
from vtimellm.segmentation.data import ReasonSegDataset  # noqa: E402
from vtimellm.segmentation.multiframe import MultiFrameReasonSegDataset  # noqa: E402


def in_range(points: np.ndarray, config: ReasonSegConfig) -> np.ndarray:
    lower = np.asarray(config.point_cloud_range[:3])
    upper = np.asarray(config.point_cloud_range[3:])
    return ((points[:, :3] >= lower) & (points[:, :3] < upper)).all(axis=1)


class SingleFrameView:
    """让单帧数据集也能进 metrics_for（它需要 frame_point_counts 这个键）。"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["frame_point_counts"] = [int(sample["points"].shape[0])]
        return sample


def metrics_for(dataset, limit: int, config: ReasonSegConfig) -> dict:
    """逐目标统计 K（in-range 正例点数）与 N（in-range 总点数）。"""
    ks, ns = [], []
    per_frame: list[list[int]] = []
    for index in range(min(limit, len(dataset))):
        sample = dataset[index]
        points = sample["points"].numpy()
        masks = sample["target_masks"].numpy()
        valid = in_range(points, config)
        ns.append(int(valid.sum()))
        per_frame.append([int(count) for count in sample["frame_point_counts"]])
        for row in range(masks.shape[0]):
            if not masks[row].any():
                continue
            target = masks[row] & valid
            if target.any():
                ks.append(int(target.sum()))
    ks_arr = np.asarray(ks, dtype=np.float64)
    ns_arr = np.asarray(ns, dtype=np.float64)
    if ks_arr.size == 0:
        raise RuntimeError("没有可统计的目标")
    widest = max(len(counts) for counts in per_frame)
    return {
        "records": len(per_frame),
        "objects": int(ks_arr.size),
        "k_median": float(np.median(ks_arr)),
        "k_p10_p90": [float(np.percentile(ks_arr, 10)), float(np.percentile(ks_arr, 90))],
        "n_median": float(np.median(ns_arr)),
        "threshold_1_minus_K_over_N": float(1.0 - np.median(ks_arr) / np.median(ns_arr)),
        "frames_used_median": float(np.median([len(counts) for counts in per_frame])),
        "points_per_frame_median": [
            float(np.median([counts[slot] if slot < len(counts) else 0 for counts in per_frame]))
            for slot in range(widest)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="任一实例侧清单（建议 dev，物体多）")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--records", type=int, default=120)
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = ReasonSegConfig()
    failures = []
    report: dict = {"manifest": str(args.manifest), "num_frames": args.num_frames}

    base = MultiFrameReasonSegDataset(
        args.manifest, dataroot=args.dataroot, config=config, arm="a2_repeat",
        num_frames=args.num_frames, nuscenes_version=args.version)

    # ---- 1. 映射正确性 ----
    ious = []
    for index in range(min(args.records, len(base))):
        for class_name, overlap, mask_count, box_count in base.anchor_box_vs_mask(index):
            union = mask_count + box_count - overlap
            if union:
                ious.append(overlap / union)
    ious = np.asarray(ious, dtype=np.float64)
    report["anchor_box_vs_mask_iou"] = {
        "n": int(ious.size), "median": float(np.median(ious)),
        "p10": float(np.percentile(ious, 10)), "below_0_5": int((ious < 0.5).sum())}
    if np.median(ious) < 0.9:
        failures.append(f"框与 GT 掩码的 IoU 中位数只有 {np.median(ious):.3f}（需 ≥0.9）"
                        "⇒ panoptic→instance_token 反查不可信")

    # ---- 2/3/4. 三臂装配 ----
    a1 = MultiFrameReasonSegDataset(
        args.manifest, dataroot=args.dataroot, config=config, arm="a1_compensated",
        num_frames=args.num_frames, nuscenes_version=args.version)
    a3 = MultiFrameReasonSegDataset(
        args.manifest, dataroot=args.dataroot, config=config, arm="a3_naive",
        num_frames=args.num_frames, nuscenes_version=args.version)

    extra, zero_extra, mismatch = [], 0, 0
    for index in range(min(args.records, len(a1))):
        one = a1[index]
        three = a3[index]
        if not np.array_equal(one["target_masks"].numpy(), three["target_masks"].numpy()):
            mismatch += 1
        anchor_count = int(one["frame_point_counts"][0])
        base_masks = a1.base[index]["target_masks"].numpy()
        for row in range(base_masks.shape[0]):
            if not base_masks[row].any():
                continue
            extra.append(int(one["target_masks"].numpy()[row, anchor_count:].sum()))
            if one["target_masks"].numpy()[row, anchor_count:].sum() == 0:
                zero_extra += 1
    extra_arr = np.asarray(extra, dtype=np.float64)
    report["cross_frame_positive_points"] = {
        "objects": int(extra_arr.size), "median": float(np.median(extra_arr)),
        "p10_p90": [float(np.percentile(extra_arr, 10)), float(np.percentile(extra_arr, 90))],
        "share_with_zero": float(zero_extra / max(extra_arr.size, 1)),
        "a1_a3_mask_mismatch_records": mismatch}
    if mismatch:
        failures.append(f"{mismatch} 条记录上 a1 与 a3 的掩码集合不同 —— 两臂不再是单变量对照")
    if np.median(extra_arr) == 0:
        failures.append("邻帧正例点中位数为 0 ⇒ 多帧没有带来任何新监督信号")

    report["frame_examples"] = {str(i): a1.frame_summary(i) for i in range(min(3, len(a1)))}

    for name, dataset in (("a1_compensated", a1), ("a3_naive", a3), ("a2_repeat", base)):
        report[f"threshold_{name}"] = metrics_for(dataset, args.records, config)

    plain = ReasonSegDataset(args.manifest, dataroot=args.dataroot, config=ReasonSegConfig())
    report["threshold_single_frame_reference"] = metrics_for(
        SingleFrameView(plain), args.records, config)

    # ---- 5. a2 的复制块必须逐位相同 ----
    sample = base[min(5, len(base) - 1)]
    points = sample["points"].numpy()
    masks = sample["target_masks"].numpy()
    block = points.shape[0] // args.num_frames
    report["a2_repeat_exact"] = bool(
        points.shape[0] == block * args.num_frames
        and all(np.array_equal(points[:block], points[i * block:(i + 1) * block])
                for i in range(args.num_frames))
        and np.array_equal(masks, np.tile(masks[:, :block], (1, args.num_frames))))
    if not report["a2_repeat_exact"]:
        failures.append("a2 的复制块不是逐位相同 ⇒ 它不再是零新信息的对照")

    a2 = report["threshold_a2_repeat"]
    ref = report["threshold_single_frame_reference"]
    report["a2_threshold_ratio_check"] = {
        "single": ref["threshold_1_minus_K_over_N"], "a2": a2["threshold_1_minus_K_over_N"],
        "abs_diff": abs(ref["threshold_1_minus_K_over_N"] - a2["threshold_1_minus_K_over_N"])}
    if report["a2_threshold_ratio_check"]["abs_diff"] > 1e-4:
        failures.append("a2 的门槛与单帧差超过 1e-4 ⇒ K/N 没有同倍增长，对照不成立")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    if failures:
        print("\n!!! 冒烟未通过：", file=sys.stderr)
        for item in failures:
            print("  - " + item, file=sys.stderr)
        return 1
    print("\n冒烟全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
