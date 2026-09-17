#!/usr/bin/env python3
"""Fail-fast checks for the ReasonSeg data, environment and B3 base."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--b3-checkpoint", type=Path, required=True)
    parser.add_argument("--spatial-checkpoint", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    dataroot = args.dataroot.resolve()
    for package in ("torch", "transformers", "peft", "accelerate", "spconv"):
        if importlib.util.find_spec(package) is None:
            raise RuntimeError(f"missing Python package: {package}")
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        print(f"[OK] {package} {version}")

    import torch

    print(
        f"[INFO] torch={torch.__version__} CUDA={torch.version.cuda} "
        f"available={torch.cuda.is_available()} devices={torch.cuda.device_count()}"
    )
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ReasonSeg training")

    b3_required = (
        "config.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "non_lora_trainables.bin",
        "trainer_state.json",
    )
    for name in b3_required:
        check_file(args.b3_checkpoint / name, f"B3 {name}")
    state = json.loads((args.b3_checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    print(
        f"[INFO] B3 source epoch={state.get('epoch')} global_step={state.get('global_step')}"
    )
    if args.spatial_checkpoint:
        check_file(
            args.spatial_checkpoint / "spatial_config.json",
            "spatial checkpoint metadata",
        )
        check_file(
            args.spatial_checkpoint / "spatial_encoder.pt",
            "spatial checkpoint weights",
        )
    base = project_root / "mllm" / "base_model" / "vicuna-v1-5-7b"
    check_file(base / "pytorch_model.bin.index.json", "Vicuna index")
    check_file(
        project_root
        / "mllm"
        / "checkpoints"
        / "vtimellm-vicuna-v1-5-7b-stage1"
        / "mm_projector.bin",
        "Stage 1 projector",
    )

    metadata_dir = dataroot / "v1.0-trainval"
    for name in ("scene.json", "sample.json", "sample_data.json", "panoptic.json"):
        check_file(metadata_dir / name, f"nuScenes metadata {name}")
    if not (dataroot / "samples" / "LIDAR_TOP").is_dir():
        raise FileNotFoundError(dataroot / "samples" / "LIDAR_TOP")
    if not (dataroot / "panoptic" / "v1.0-trainval").is_dir():
        raise FileNotFoundError(dataroot / "panoptic" / "v1.0-trainval")

    for manifest in (args.train_manifest, args.val_manifest):
        validate_manifest(manifest.resolve(), dataroot)
    print("[PASS] ReasonSeg environment and data closure are ready")
    return 0


def validate_manifest(path: Path, dataroot: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    count = 0
    scenes = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for key in (
                "sample_token",
                "scene_token",
                "lidar_path",
                "panoptic_path",
                "query",
                "answer",
                "targets",
            ):
                if key not in record:
                    raise RuntimeError(f"{path}:{line_number} missing {key}")
            if len(record["targets"]) > 8:
                raise RuntimeError(f"{path}:{line_number} exceeds eight targets")
            if count < 128:
                check_file(dataroot / record["lidar_path"], "manifest point file")
                check_file(dataroot / record["panoptic_path"], "manifest panoptic file")
            scenes.add(record["scene_token"])
            count += 1
    if not count:
        raise RuntimeError(f"empty manifest: {path}")
    print(f"[OK] {path}: {count:,} records across {len(scenes):,} scenes")


def check_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label}: {path}")
    print(f"[OK] {label}: {path}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
