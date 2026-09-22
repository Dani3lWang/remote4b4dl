#!/usr/bin/env python3
"""Fail-fast checks for the ReasonSeg data, environment and B3 base."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


MLLM_ROOT = Path(__file__).resolve().parents[1]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))

from reasonseg_splits import partition_development_scenes


THING_CLASSES = (
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
)


@dataclass(frozen=True)
class ManifestSummary:
    path: Path
    count: int
    scenes: frozenset[str]
    samples: frozenset[str]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("spatial", "train", "evaluate"),
        default="train",
        help="select the exact closure required by the next command",
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--manifest", type=Path, help="evaluation manifest")
    parser.add_argument(
        "--exclude-scenes",
        type=Path,
        help="JSON scene names/tokens reserved for the B4DL test set",
    )
    parser.add_argument("--b3-checkpoint", type=Path)
    parser.add_argument("--spatial-checkpoint", type=Path)
    parser.add_argument("--seg-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-free-gib", type=float)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--spatial-label-source",
        choices=("auto", "lidarseg", "panoptic"),
        default="auto",
    )
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    dataroot = args.dataroot.resolve()
    if args.output_dir is not None:
        required_gib = args.min_free_gib
        if required_gib is None:
            required_gib = {"spatial": 2.0, "train": 5.0, "evaluate": 2.0}[args.stage]
        validate_free_space(args.output_dir, required_gib)
    packages = ["torch", "accelerate", "spconv", "numpy", "nuscenes"]
    if args.stage in ("train", "evaluate"):
        packages.extend(("transformers", "peft"))
    for package in packages:
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

    # A discoverable spconv package can still be unusable because its wheel,
    # PyTorch and CUDA runtimes do not agree.  Import it for every stage and
    # exercise one real GPU kernel whenever CUDA is required.
    import spconv.pytorch as spconv

    print(f"[OK] spconv.pytorch import: {spconv.__file__}")
    if args.require_cuda:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("configured bf16 launch requires a bf16-capable CUDA GPU")
        validate_spconv_cuda(torch, spconv)

    metadata_dir = dataroot / "v1.0-trainval"
    common_metadata = ("scene.json", "sample.json", "sample_data.json")
    for name in common_metadata:
        check_file(metadata_dir / name, f"nuScenes metadata {name}")
    if not (dataroot / "samples" / "LIDAR_TOP").is_dir():
        raise FileNotFoundError(dataroot / "samples" / "LIDAR_TOP")

    excluded_scenes = normalize_scene_tokens(metadata_dir, load_scene_ids(args.exclude_scenes))
    validate_official_test_exclusion(metadata_dir, excluded_scenes, args.exclude_scenes)

    if args.stage == "spatial":
        validate_spatial_data_closure(
            dataroot,
            metadata_dir,
            excluded_scenes=excluded_scenes,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            label_source=args.spatial_label_source,
        )
        print("[PASS] ReasonSeg spatial-pretraining closure is ready")
        return 0

    if args.b3_checkpoint is None:
        raise RuntimeError(f"--b3-checkpoint is required for stage {args.stage}")

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
    if args.stage == "train" and args.spatial_checkpoint is None:
        raise RuntimeError("--spatial-checkpoint is required for stage train")
    if args.spatial_checkpoint:
        validate_spatial_checkpoint(args.spatial_checkpoint, torch)
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

    check_file(metadata_dir / "panoptic.json", "nuScenes metadata panoptic.json")
    if not (dataroot / "panoptic" / "v1.0-trainval").is_dir():
        raise FileNotFoundError(dataroot / "panoptic" / "v1.0-trainval")

    if args.stage == "evaluate":
        if args.manifest is None or args.seg_checkpoint is None:
            raise RuntimeError("--manifest and --seg-checkpoint are required for stage evaluate")
        validate_reasonseg_checkpoint(args.seg_checkpoint, torch)
        validate_manifest(
            args.manifest.resolve(),
            dataroot,
            expected_split="val",
            excluded_scenes=excluded_scenes,
            verify_data_contents=True,
        )
        print("[PASS] ReasonSeg free-generation evaluation closure is ready")
        return 0

    if args.train_manifest is None or args.val_manifest is None:
        raise RuntimeError("--train-manifest and --val-manifest are required for stage train")
    train = validate_manifest(
        args.train_manifest.resolve(),
        dataroot,
        expected_split="train",
        excluded_scenes=excluded_scenes,
        verify_data_contents=True,
    )
    val = validate_manifest(
        args.val_manifest.resolve(),
        dataroot,
        expected_split="val",
        excluded_scenes=excluded_scenes,
        verify_data_contents=True,
    )
    scene_overlap = train.scenes & val.scenes
    if scene_overlap:
        raise RuntimeError(
            "train/val scene leakage: " + ", ".join(sorted(scene_overlap)[:5])
        )
    sample_overlap = train.samples & val.samples
    if sample_overlap:
        raise RuntimeError(
            "train/val sample leakage: " + ", ".join(sorted(sample_overlap)[:5])
        )
    if excluded_scenes:
        print(f"[OK] excluded B4DL test scenes: {len(excluded_scenes):,}")
    print("[OK] train/val scene and sample sets are disjoint")
    print("[PASS] ReasonSeg environment and data closure are ready")
    return 0


def validate_manifest(
    path: Path,
    dataroot: Path,
    *,
    expected_split: str | None = None,
    excluded_scenes: set[str] | frozenset[str] = frozenset(),
    verify_data_contents: bool = False,
) -> ManifestSummary:
    if not path.is_file():
        raise FileNotFoundError(path)
    count = 0
    scenes: set[str] = set()
    samples: set[str] = set()
    sample_queries: set[tuple[str, str]] = set()
    checked_data_paths: set[Path] = set()
    data_cache: dict[tuple[Path, Path], tuple[int, frozenset[int]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number} invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"{path}:{line_number} must contain a JSON object")
            for key in (
                "sample_token",
                "scene_token",
                "split",
                "lidar_path",
                "panoptic_path",
                "query",
                "answer",
                "targets",
            ):
                if key not in record:
                    raise RuntimeError(f"{path}:{line_number} missing {key}")
            for key in (
                "sample_token",
                "scene_token",
                "split",
                "lidar_path",
                "panoptic_path",
                "query",
                "answer",
            ):
                if not isinstance(record[key], str) or not record[key].strip():
                    raise RuntimeError(f"{path}:{line_number} invalid {key}")
            if expected_split is not None and record["split"] != expected_split:
                raise RuntimeError(
                    f"{path}:{line_number} split={record['split']!r}; "
                    f"expected {expected_split!r}"
                )
            scene_token = record["scene_token"]
            if scene_token in excluded_scenes:
                raise RuntimeError(
                    f"{path}:{line_number} contains excluded test scene {scene_token}"
                )
            targets = record["targets"]
            if not isinstance(targets, list):
                raise RuntimeError(f"{path}:{line_number} targets must be a list")
            if len(targets) > 8:
                raise RuntimeError(f"{path}:{line_number} exceeds eight targets")
            _validate_targets(path, line_number, targets)
            _validate_answer_contract(path, line_number, record["answer"], len(targets))
            if any(token in record["query"] for token in ("<LOC>", "<SEG>", "<NOOBJ>")):
                raise RuntimeError(
                    f"{path}:{line_number} query leaks segmentation supervision tokens"
                )
            point_path = resolve_data_file(dataroot, record["lidar_path"])
            panoptic_path = resolve_data_file(dataroot, record["panoptic_path"])
            for data_path, label in (
                (point_path, "manifest point file"),
                (panoptic_path, "manifest panoptic file"),
            ):
                if data_path not in checked_data_paths:
                    check_file(data_path, label, verbose=False)
                    checked_data_paths.add(data_path)
            if verify_data_contents:
                pair = (point_path, panoptic_path)
                if pair not in data_cache:
                    data_cache[pair] = inspect_reasonseg_data_pair(
                        point_path, panoptic_path
                    )
                _, panoptic_ids = data_cache[pair]
                missing_ids = {
                    target["panoptic_id"] for target in targets
                } - panoptic_ids
                if missing_ids:
                    raise RuntimeError(
                        f"{path}:{line_number} target IDs absent from panoptic data: "
                        + ", ".join(str(value) for value in sorted(missing_ids))
                    )
            sample_query = (record["sample_token"], record["query"])
            if sample_query in sample_queries:
                raise RuntimeError(
                    f"{path}:{line_number} duplicates sample/query {sample_query!r}"
                )
            sample_queries.add(sample_query)
            scenes.add(scene_token)
            samples.add(record["sample_token"])
            count += 1
    if not count:
        raise RuntimeError(f"empty manifest: {path}")
    print(
        f"[OK] {path}: {count:,} records, {len(samples):,} samples, "
        f"{len(scenes):,} scenes; {len(checked_data_paths):,} unique data files checked"
        + (
            f"; {len(data_cache):,} point/panoptic pairs content-verified"
            if verify_data_contents
            else ""
        )
    )
    return ManifestSummary(path, count, frozenset(scenes), frozenset(samples))


def inspect_reasonseg_data_pair(
    point_path: Path, panoptic_path: Path
) -> tuple[int, frozenset[int]]:
    """Read one referenced frame and prove its point/panoptic alignment."""

    import numpy as np

    point_bytes = point_path.stat().st_size
    if point_bytes % (5 * np.dtype(np.float32).itemsize):
        raise RuntimeError(f"invalid nuScenes point file size: {point_path}")
    point_count = point_bytes // (5 * np.dtype(np.float32).itemsize)
    try:
        with np.load(panoptic_path) as values:
            if "data" not in values:
                raise RuntimeError(f"panoptic archive has no 'data' array: {panoptic_path}")
            panoptic = np.asarray(values["data"]).reshape(-1)
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"cannot read panoptic archive {panoptic_path}: {exc}") from exc
    if len(panoptic) != point_count:
        raise RuntimeError(
            f"point/panoptic length mismatch: {point_path} has {point_count:,} points; "
            f"{panoptic_path} has {len(panoptic):,} labels"
        )
    return point_count, frozenset(int(value) for value in np.unique(panoptic))


def _validate_targets(path: Path, line_number: int, targets: list) -> None:
    seen_panoptic_ids: set[int] = set()
    for target_index, target in enumerate(targets):
        label = f"{path}:{line_number} target[{target_index}]"
        if not isinstance(target, dict):
            raise RuntimeError(f"{label} must be an object")
        for key in ("panoptic_id", "class_id", "class_name"):
            if key not in target:
                raise RuntimeError(f"{label} missing {key}")
        panoptic_id = target["panoptic_id"]
        class_id = target["class_id"]
        class_name = target["class_name"]
        if type(panoptic_id) is not int or panoptic_id <= 0:
            raise RuntimeError(f"{label} has invalid panoptic_id")
        if panoptic_id in seen_panoptic_ids:
            raise RuntimeError(f"{label} duplicates panoptic_id {panoptic_id}")
        seen_panoptic_ids.add(panoptic_id)
        if type(class_id) is not int or not 0 <= class_id < len(THING_CLASSES):
            raise RuntimeError(f"{label} has invalid class_id {class_id!r}")
        if class_name != THING_CLASSES[class_id]:
            raise RuntimeError(
                f"{label} class mismatch: id {class_id} is {THING_CLASSES[class_id]!r}, "
                f"got {class_name!r}"
            )


def _validate_answer_contract(
    path: Path, line_number: int, answer: str, target_count: int
) -> None:
    loc_count = answer.count("<LOC>")
    seg_count = answer.count("<SEG>")
    noobj_count = answer.count("<NOOBJ>")
    if target_count:
        if (loc_count, seg_count, noobj_count) != (target_count, target_count, 0):
            raise RuntimeError(
                f"{path}:{line_number} answer/target mismatch: targets={target_count}, "
                f"LOC={loc_count}, SEG={seg_count}, NOOBJ={noobj_count}"
            )
    elif (loc_count, seg_count, noobj_count) != (0, 0, 1):
        raise RuntimeError(
            f"{path}:{line_number} no-object answer must contain exactly one <NOOBJ>"
        )


def resolve_data_file(dataroot: Path, value: str) -> Path:
    root = dataroot.resolve()
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"manifest path escapes dataroot: {value}") from exc
    return path


def validate_spconv_cuda(torch, spconv) -> None:
    """Run a minimal sparse CUDA convolution to catch wheel/ABI mismatches."""

    device = torch.device("cuda:0")
    features = torch.randn(4, 4, device=device)
    indices = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
        dtype=torch.int32,
        device=device,
    )
    sparse = spconv.SparseConvTensor(features, indices, [4, 4, 4], 1)
    layer = spconv.SubMConv3d(4, 8, 3, padding=1, bias=False).to(device)
    output = layer(sparse)
    if output.features.shape != (4, 8) or not torch.isfinite(output.features).all():
        raise RuntimeError("spconv CUDA smoke test returned invalid output")
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    print(
        f"[OK] CUDA device: {properties.name}; capability="
        f"{properties.major}.{properties.minor}; bf16=yes"
    )
    print("[OK] spconv CUDA sparse-convolution smoke test")


def validate_free_space(output_dir: Path, minimum_gib: float) -> None:
    if minimum_gib <= 0:
        raise ValueError("--min-free-gib must be positive")
    probe = output_dir.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_bytes = shutil.disk_usage(probe).free
    free_gib = free_bytes / (1024**3)
    if free_gib < minimum_gib:
        raise RuntimeError(
            f"insufficient free space for {output_dir}: {free_gib:.2f} GiB available; "
            f"require at least {minimum_gib:.2f} GiB"
        )
    print(
        f"[OK] output filesystem free space: {free_gib:.2f} GiB "
        f"(minimum {minimum_gib:.2f} GiB)"
    )


def validate_official_test_exclusion(
    metadata_dir: Path,
    excluded_scenes: set[str],
    exclude_path: Path | None,
) -> None:
    if exclude_path is None:
        raise RuntimeError(
            "--exclude-scenes is required; the official nuScenes val split is the B4DL test set"
        )
    from nuscenes.utils.splits import create_splits_scenes

    official_test_scenes = normalize_scene_tokens(
        metadata_dir, set(create_splits_scenes()["val"])
    )
    missing_test_scenes = official_test_scenes - excluded_scenes
    if missing_test_scenes:
        raise RuntimeError(
            "excluded-scene manifest is missing official B4DL test scenes: "
            + ", ".join(sorted(missing_test_scenes)[:5])
        )
    print(f"[OK] excluded B4DL test scenes: {len(excluded_scenes):,}")


def validate_spatial_data_closure(
    dataroot: Path,
    metadata_dir: Path,
    *,
    excluded_scenes: set[str],
    validation_fraction: float,
    seed: int,
    label_source: str,
) -> None:
    """Check every official-train key frame needed by spatial pretraining."""

    import numpy as np
    from nuscenes.utils.splits import create_splits_scenes

    scenes = json.loads((metadata_dir / "scene.json").read_text(encoding="utf-8"))
    samples = json.loads((metadata_dir / "sample.json").read_text(encoding="utf-8"))
    sample_data = json.loads(
        (metadata_dir / "sample_data.json").read_text(encoding="utf-8")
    )
    if label_source == "auto":
        label_source = (
            "lidarseg" if (metadata_dir / "lidarseg.json").is_file() else "panoptic"
        )
    label_metadata = metadata_dir / f"{label_source}.json"
    check_file(label_metadata, f"nuScenes metadata {label_source}.json")
    label_records = json.loads(label_metadata.read_text(encoding="utf-8"))
    assignments = partition_development_scenes(
        scenes,
        official_train_names=create_splits_scenes()["train"],
        excluded_scenes=excluded_scenes,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    lidar_by_sample = {
        str(record["sample_token"]): record
        for record in sample_data
        if record.get("is_key_frame")
        and "/LIDAR_TOP/" in str(record.get("filename", ""))
    }
    labels_by_token = {
        str(record.get("sample_data_token") or record.get("token")): record
        for record in label_records
    }
    frame_counts = {"train": 0, "val": 0}
    checked_paths: set[Path] = set()
    for sample in samples:
        split = assignments.get(str(sample["scene_token"]))
        if split not in frame_counts:
            continue
        point_record = lidar_by_sample.get(str(sample["token"]))
        if point_record is None:
            raise RuntimeError(
                f"missing key-frame LIDAR_TOP metadata for sample {sample['token']}"
            )
        lidar_token = str(point_record["token"])
        label_record = labels_by_token.get(lidar_token)
        if label_record is None:
            raise RuntimeError(
                f"missing point/{label_source} metadata for LIDAR_TOP {lidar_token}"
            )
        point_path = resolve_data_file(dataroot, str(point_record["filename"]))
        label_path = resolve_data_file(dataroot, str(label_record["filename"]))
        check_file(point_path, "lidarseg point file", verbose=False)
        check_file(label_path, f"{label_source} label file", verbose=False)
        point_bytes = point_path.stat().st_size
        if point_bytes % 20:
            raise RuntimeError(f"invalid nuScenes point file size: {point_path}")
        point_count = point_bytes // 20
        if label_source == "lidarseg":
            label_count = label_path.stat().st_size
        else:
            try:
                with np.load(label_path) as values:
                    if "data" not in values:
                        raise RuntimeError(
                            f"panoptic archive has no 'data' array: {label_path}"
                        )
                    panoptic = np.asarray(values["data"]).reshape(-1)
            except Exception as exc:
                if isinstance(exc, RuntimeError):
                    raise
                raise RuntimeError(
                    f"cannot read panoptic archive {label_path}: {exc}"
                ) from exc
            label_count = len(panoptic)
            semantic = panoptic.astype(np.int64) // 1000
            if semantic.size and (semantic.min() < 0 or semantic.max() >= 32):
                raise RuntimeError(
                    f"panoptic semantic IDs are outside [0, 31]: {label_path}"
                )
        if label_count != point_count:
            raise RuntimeError(
                f"point/{label_source} length mismatch for {lidar_token}: "
                f"{point_count:,} points != {label_count:,} labels"
            )
        checked_paths.update((point_path, label_path))
        frame_counts[split] += 1
    if not all(frame_counts.values()):
        raise RuntimeError(f"empty internal lidarseg split: {frame_counts}")
    split_scene_counts = {
        split: sum(value == split for value in assignments.values())
        for split in ("train", "val")
    }
    print(
        f"[OK] spatial label source={label_source}; closure: "
        f"train={frame_counts['train']:,} frames/{split_scene_counts['train']:,} scenes; "
        f"val={frame_counts['val']:,} frames/{split_scene_counts['val']:,} scenes; "
        f"{len(checked_paths):,} files checked"
    )


def validate_spatial_checkpoint(path: Path, torch) -> None:
    metadata_path = path / "spatial_config.json"
    weights_path = path / "spatial_encoder.pt"
    check_file(metadata_path, "spatial checkpoint metadata")
    check_file(weights_path, "spatial checkpoint weights")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or not isinstance(metadata.get("config"), dict):
        raise RuntimeError(f"invalid spatial checkpoint metadata: {metadata_path}")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"invalid spatial checkpoint state dict: {weights_path}")
    print(f"[OK] spatial checkpoint state: {len(state):,} tensors")


def validate_reasonseg_checkpoint(path: Path, torch) -> None:
    config_path = path / "reasonseg_config.json"
    modules_path = path / "reasonseg_modules.pt"
    check_file(config_path, "ReasonSeg checkpoint metadata")
    check_file(modules_path, "ReasonSeg checkpoint modules")
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or not isinstance(metadata.get("config"), dict):
        raise RuntimeError(f"invalid ReasonSeg checkpoint metadata: {config_path}")
    modules = torch.load(modules_path, map_location="cpu", weights_only=True)
    required = {"point_encoder", "scene_compressor", "mask_head", "token_adapters"}
    missing = required - set(modules) if isinstance(modules, dict) else required
    if missing:
        raise RuntimeError(f"ReasonSeg checkpoint missing module states: {sorted(missing)}")
    check_file(path / "seg_lora" / "adapter_config.json", "ReasonSeg LoRA config")
    lora_weights = path / "seg_lora" / "adapter_model.safetensors"
    if not lora_weights.is_file():
        lora_weights = path / "seg_lora" / "adapter_model.bin"
    check_file(lora_weights, "ReasonSeg LoRA weights")
    check_file(path / "tokenizer" / "tokenizer_config.json", "ReasonSeg tokenizer")
    print("[OK] ReasonSeg checkpoint structure and module payload")


def load_scene_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(f"excluded-scene manifest: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(values, dict):
        values = values.get("scene_tokens") or values.get("scenes") or []
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"excluded-scene manifest is empty or invalid: {path}")
    scene_ids = {str(value).strip() for value in values if str(value).strip()}
    if not scene_ids:
        raise RuntimeError(f"excluded-scene manifest has no usable values: {path}")
    return scene_ids


def normalize_scene_tokens(metadata_dir: Path, scene_ids: set[str]) -> set[str]:
    """Accept scene names or tokens, then return tokens used by manifests."""

    if not scene_ids:
        return set()
    records = json.loads((metadata_dir / "scene.json").read_text(encoding="utf-8"))
    lookup: dict[str, str] = {}
    for record in records:
        token = str(record["token"])
        lookup[token] = token
        lookup[str(record["name"])] = token
    unknown = scene_ids - set(lookup)
    if unknown:
        raise RuntimeError(
            "excluded-scene manifest contains unknown values: "
            + ", ".join(sorted(unknown)[:5])
        )
    return {lookup[value] for value in scene_ids}


def check_file(path: Path, label: str, *, verbose: bool = True) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label}: {path}")
    if verbose:
        print(f"[OK] {label}: {path}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
