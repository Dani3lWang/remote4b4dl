#!/usr/bin/env python3
"""Verify the local data/model closure required by a B3/B4-style run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


EXPECTED = {
    "stage1_train": 161_629,
    "b3_train": 148_271,
    "test": 30_145,
}


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size < 1024:
        raise RuntimeError(f"{path} 过小，可能是 Git LFS 指针")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_feature_coverage(records, feature_dir: Path, label: str) -> set[str]:
    scene_ids = {str(item["scene_id"]) for item in records}
    missing = sorted(
        scene_id
        for scene_id in scene_ids
        if not (feature_dir / f"{scene_id}.npy").is_file()
    )
    if missing:
        raise RuntimeError(
            f"{label}: {len(missing)}/{len(scene_ids)} 个特征缺失，例如 {missing[:5]}"
        )
    print(f"[OK] {label}: {len(records):,} 条，{len(scene_ids):,} 个 ID，特征缺失 0")
    return scene_ids


def check_feature_shapes(feature_dir: Path, scene_ids: set[str], stage1: bool) -> None:
    for scene_id in sorted(scene_ids):
        array = np.load(feature_dir / f"{scene_id}.npy", mmap_mode="r")
        if array.ndim != 2 or array.shape[1] != 768:
            raise RuntimeError(f"特征形状异常: {scene_id}.npy = {array.shape}")
        if stage1 and array.shape[0] != 1:
            raise RuntimeError(f"Stage 1 特征形状异常: {scene_id}.npy = {array.shape}")
    print(f"[OK] {feature_dir.name}: 已检查 {len(scene_ids):,} 个特征，维度均为 (*, 768)")


def check_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label}: {path}")
    print(f"[OK] {label}: {path} ({path.stat().st_size / 1024**2:.1f} MiB)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--train-data",
        default="stage2_full_train_seqv3_meta2_148k.json",
        help="Training JSON filename under mllm/b4dl_dataset",
    )
    parser.add_argument("--expected-train-count", type=int, default=EXPECTED["b3_train"])
    parser.add_argument("--run-label", default="B3")
    args = parser.parse_args()

    root = args.project_root.resolve()
    mllm = root / "mllm"
    dataset_dir = mllm / "b4dl_dataset"
    feature_root = root / "encoders" / "lidarclip" / "b4dl"

    stage1 = load_json(dataset_dir / "stage1_train.json")
    if len(stage1) != EXPECTED["stage1_train"]:
        raise RuntimeError(
            f"stage1_train: {len(stage1):,} 条，预期 {EXPECTED['stage1_train']:,}"
        )
    stage1_ids = check_feature_coverage(
        stage1, feature_root / "stage1_features_sample", "Stage 1"
    )
    del stage1
    check_feature_shapes(feature_root / "stage1_features_sample", stage1_ids, True)
    del stage1_ids

    train_records = load_json(dataset_dir / args.train_data)
    if len(train_records) != args.expected_train_count:
        raise RuntimeError(
            f"{args.run_label} train: {len(train_records):,} 条，"
            f"预期 {args.expected_train_count:,}"
        )
    train_ids = check_feature_coverage(
        train_records, feature_root / "stage2_features", f"{args.run_label} train"
    )
    del train_records

    test = load_json(dataset_dir / "test_qa.json")
    if len(test) != EXPECTED["test"]:
        raise RuntimeError(f"test_qa: {len(test):,} 条，预期 {EXPECTED['test']:,}")
    test_ids = check_feature_coverage(
        test, feature_root / "stage2_features", f"{args.run_label} test"
    )
    del test
    check_feature_shapes(feature_root / "stage2_features", train_ids | test_ids, False)

    base = mllm / "base_model" / "vicuna-v1-5-7b"
    index_path = base / "pytorch_model.bin.index.json"
    index = load_json(index_path)
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        raise RuntimeError(f"模型索引未列出权重分片: {index_path}")
    for shard in shards:
        check_file(base / shard, "Vicuna 权重分片")

    check_file(
        mllm
        / "checkpoints"
        / "vtimellm-vicuna-v1-5-7b-stage1"
        / "mm_projector.bin",
        "Stage 1 projector",
    )
    check_file(dataset_dir / "ego_metadata.json", "评测 ego metadata")
    check_file(dataset_dir / "ego_frame_motion.json", "评测 frame motion")

    roberta = root / "models" / "roberta-large"
    for name in (
        "config.json",
        "model.safetensors",
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        check_file(roberta / name, "RoBERTa 评测文件")

    for package in ("deepspeed", "transformers", "peft"):
        if importlib.util.find_spec(package) is None:
            raise RuntimeError(f"当前 Python 环境缺少 {package}")
        print(f"[OK] Python package: {package}")

    try:
        import torch

        print(f"[OK] torch: {torch.__version__}; CUDA available={torch.cuda.is_available()}")
        if args.require_cuda and not torch.cuda.is_available():
            raise RuntimeError(f"未检测到 CUDA GPU，不能启动 {args.run_label} 训练")
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境缺少 torch") from exc

    print(f"[PASS] {args.run_label} 训练与评测数据闭包完整")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
