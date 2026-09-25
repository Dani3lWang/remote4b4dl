"""量出 voxel 特征共享给逐点 AUC 设的天花板，判断改 voxel_size 能否填上接地缺口。

背景：`SparseUNetPointEncoder` 先体素化、跑稀疏 U-Net，再用 `inverse` 把体素特征
散射回原始点（spatial_encoder.py:128-134）。因此**同一体素内的所有点拿到逐位相同
的特征向量**，掩码头对它们的打分也必然相同 —— 这些点在 AUC 里只能算并列（0.5）。

设某个物体的 GT 框内点数为 K、帧内框内点数为 N、与 GT 点共享体素的非 GT 框内点数
为 T，则并列单独造成的 AUC 上限是

    AUC_max = 1 - 0.5 * T / (K * N)

而 Phase 2.3 的先验判据要求 top-K 解码可用，即 AUC >= 1 - K/N。两式联立得到
**T <= 2 K^2** 才可能达标 —— 与 voxel_size 无关的硬条件。oracle 探针实测
(1-AUC)*N ~= 289，反推 T ~= 6936；本脚本直接数 T，看并列是否解释得掉整个缺口。

只读 manifest / lidar / panoptic，不加载任何模型，纯 CPU。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

POINT_CLOUD_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)


def tied_negatives(xyz: np.ndarray, gt: np.ndarray, valid: np.ndarray, voxel: float) -> int:
    """与任一 GT 点共享体素的非 GT 框内点数。"""
    lower = np.array(POINT_CLOUD_RANGE[:3], dtype=np.float64)
    coords = np.floor((xyz[valid] - lower) / voxel).astype(np.int64)
    gt_valid = gt[valid]
    # 按 (z,y,x) 三元组分组，与 spatial_encoder._voxelize 的 unique 口径一致
    keys = np.ascontiguousarray(coords[:, [2, 1, 0]])
    _, inverse = np.unique(
        keys.view([("", keys.dtype)] * keys.shape[1]), return_inverse=True
    )
    gt_voxel_ids = np.unique(inverse[gt_valid])
    in_gt_voxels = np.isin(inverse, gt_voxel_ids)
    return int((in_gt_voxels & ~gt_valid).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--voxel-sizes", default="0.2,0.1,0.05,0.025")
    parser.add_argument("--max-records", type=int, default=150)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    voxels = [float(v) for v in args.voxel_sizes.split(",")]
    records = [
        json.loads(line)
        for line in Path(args.manifest).read_text().splitlines()
        if line.strip()
    ][: args.max_records]

    lower = np.array(POINT_CLOUD_RANGE[:3])
    upper = np.array(POINT_CLOUD_RANGE[3:])
    rows = []
    for record in records:
        points = np.fromfile(dataroot / record["lidar_path"], dtype=np.float32).reshape(-1, 5)[:, :4]
        with np.load(dataroot / record["panoptic_path"]) as archive:
            panoptic = np.asarray(archive["data"]).reshape(-1)
        xyz = points[:, :3].astype(np.float64)
        valid = ((xyz >= lower) & (xyz < upper)).all(axis=1)
        n_valid = int(valid.sum())
        for target in record.get("targets", []):
            gt = panoptic == target["panoptic_id"]
            k = int((gt & valid).sum())
            if k == 0:
                continue  # 全越界，物理上不可分割
            row = {
                "sample_token": record["sample_token"],
                "panoptic_id": target["panoptic_id"],
                "n_valid": n_valid,
                "k": k,
            }
            for voxel in voxels:
                row[f"tied@{voxel}"] = tied_negatives(xyz, gt, valid, voxel)
            rows.append(row)

    summary = {"records": len(records), "objects": len(rows), "voxel_sizes": voxels}
    for voxel in voxels:
        tied = np.array([r[f"tied@{voxel}"] for r in rows], dtype=np.float64)
        k = np.array([r["k"] for r in rows], dtype=np.float64)
        n = np.array([r["n_valid"] for r in rows], dtype=np.float64)
        # 逐物体的 AUC 上限，再取均值（与探针的 auc_mean 同口径：先逐物体后平均）
        auc_ceiling = 1.0 - 0.5 * tied / np.maximum(k * n, 1.0)
        budget = 2.0 * k**2  # 判据要求 T <= 2K^2
        summary[f"voxel{voxel}"] = {
            "tied_median": float(np.median(tied)),
            "tied_mean": float(tied.mean()),
            "tied_p90": float(np.percentile(tied, 90)),
            "auc_ceiling_mean": float(auc_ceiling.mean()),
            "frac_within_tie_budget": float((tied <= budget).mean()),
            "required_auc": float(np.mean(1.0 - k / n)),
        }
    summary["k_median"] = float(np.median([r["k"] for r in rows]))
    summary["n_valid_median"] = float(np.median([r["n_valid"] for r in rows]))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
