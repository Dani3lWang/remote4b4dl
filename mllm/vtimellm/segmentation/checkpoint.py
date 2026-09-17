"""Self-describing checkpoint format for B4DL reasoning segmentation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch

from .tokens import load_token_adapter_state_dict, token_adapter_state_dict


FORMAT_VERSION = 1
SPATIAL_FORMAT_VERSION = 1


def save_reasonseg_checkpoint(
    model,
    tokenizer,
    output_dir: str,
    *,
    b3_checkpoint: str,
    step: Optional[int] = None,
    epoch: Optional[float] = None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    spatial_checkpoint: Optional[str] = None,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(destination / "tokenizer")

    language_model = model.language_model
    if hasattr(language_model, "peft_config"):
        language_model.save_pretrained(destination / "seg_lora")
    modules = {
        "point_encoder": _cpu_state_dict(model.point_encoder.state_dict()),
        "scene_compressor": _cpu_state_dict(model.scene_compressor.state_dict()),
        "mask_head": _cpu_state_dict(model.mask_head.state_dict()),
        "token_adapters": token_adapter_state_dict(language_model),
    }
    torch.save(modules, destination / "reasonseg_modules.pt")

    metadata = {
        "format_version": FORMAT_VERSION,
        "b3_checkpoint": str(Path(b3_checkpoint)),
        "b3_fingerprint": fingerprint_b3_checkpoint(b3_checkpoint),
        "token_ids": model.token_ids.as_dict(),
        "config": model.reasonseg_config.to_dict(),
        "step": step,
        "epoch": epoch,
        "spatial_checkpoint": str(Path(spatial_checkpoint)) if spatial_checkpoint else None,
    }
    (destination / "reasonseg_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    training_state = {}
    if optimizer is not None:
        training_state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        training_state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        training_state["scaler"] = scaler.state_dict()
    if training_state:
        training_state["torch_rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            training_state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        torch.save(training_state, destination / "training_state.pt")


def load_reasonseg_checkpoint(
    model,
    tokenizer,
    checkpoint_dir: str,
    *,
    expected_b3_checkpoint: str,
) -> Dict[str, object]:
    source = Path(checkpoint_dir)
    config_path = source / "reasonseg_config.json"
    modules_path = source / "reasonseg_modules.pt"
    if not config_path.is_file() or not modules_path.is_file():
        raise FileNotFoundError(
            f"incomplete ReasonSeg checkpoint: {config_path} and {modules_path} are required"
        )
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION:
        raise RuntimeError(
            f"unsupported ReasonSeg checkpoint format: {metadata.get('format_version')}"
        )
    actual_fingerprint = fingerprint_b3_checkpoint(expected_b3_checkpoint)
    if metadata.get("b3_fingerprint") != actual_fingerprint:
        raise RuntimeError(
            "ReasonSeg checkpoint was trained from a different B3 checkpoint: "
            f"saved={metadata.get('b3_fingerprint')}, actual={actual_fingerprint}"
        )
    current_ids = model.token_ids.as_dict()
    if metadata.get("token_ids") != current_ids:
        raise RuntimeError(
            f"segmentation token ids changed: saved={metadata.get('token_ids')}, current={current_ids}"
        )
    if len(tokenizer) != max(current_ids.values()) + 1:
        raise RuntimeError(
            f"unexpected tokenizer length {len(tokenizer)} for ids {current_ids}"
        )
    modules = torch.load(modules_path, map_location="cpu", weights_only=True)
    required = {"point_encoder", "scene_compressor", "mask_head", "token_adapters"}
    missing = required - set(modules)
    if missing:
        raise RuntimeError(f"ReasonSeg checkpoint missing module states: {sorted(missing)}")
    model.point_encoder.load_state_dict(modules["point_encoder"], strict=True)
    model.scene_compressor.load_state_dict(modules["scene_compressor"], strict=True)
    model.mask_head.load_state_dict(modules["mask_head"], strict=True)
    load_token_adapter_state_dict(model.language_model, modules["token_adapters"])
    return metadata


def save_spatial_encoder_checkpoint(
    point_encoder,
    output_dir: str,
    *,
    config,
    epoch: int,
    validation_miou: float,
    num_classes: int,
) -> None:
    """Save a lidarseg-pretrained encoder without coupling it to the LLM."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        _cpu_state_dict(point_encoder.state_dict()),
        destination / "spatial_encoder.pt",
    )
    metadata = {
        "format_version": SPATIAL_FORMAT_VERSION,
        "config": config.to_dict(),
        "epoch": int(epoch),
        "validation_miou": float(validation_miou),
        "num_classes": int(num_classes),
    }
    (destination / "spatial_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_spatial_encoder_checkpoint(point_encoder, checkpoint_dir: str, *, config):
    """Restore a spatial encoder and reject incompatible voxel contracts."""

    source = Path(checkpoint_dir)
    metadata_path = source / "spatial_config.json"
    weights_path = source / "spatial_encoder.pt"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"incomplete spatial checkpoint: {metadata_path} and {weights_path} are required"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != SPATIAL_FORMAT_VERSION:
        raise RuntimeError(
            f"unsupported spatial checkpoint format: {metadata.get('format_version')}"
        )
    saved = metadata.get("config")
    if not isinstance(saved, dict):
        raise RuntimeError("spatial checkpoint has no valid config")
    current = config.to_dict()
    spatial_keys = (
        "point_feature_dim",
        "voxel_size",
        "point_cloud_range",
    )
    differences = {
        key: (saved.get(key), current.get(key))
        for key in spatial_keys
        if saved.get(key) != current.get(key)
    }
    if differences:
        raise RuntimeError(f"spatial checkpoint config mismatch: {differences}")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    point_encoder.load_state_dict(state, strict=True)
    return metadata


def fingerprint_b3_checkpoint(checkpoint_dir: str) -> str:
    """Hash small identity/config files without reading the 320 MB LoRA payload."""

    source = Path(checkpoint_dir)
    hasher = hashlib.sha256()
    identity_files = ["config.json", "adapter_config.json", "trainer_state.json"]
    for name in identity_files:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"B3 identity file missing: {path}")
        hasher.update(name.encode("utf-8"))
        hasher.update(path.read_bytes())
    adapter_path = source / "adapter_model.safetensors"
    non_lora_path = source / "non_lora_trainables.bin"
    for path in (adapter_path, non_lora_path):
        if not path.is_file():
            raise FileNotFoundError(f"B3 weight file missing: {path}")
        hasher.update(path.name.encode("utf-8"))
        hasher.update(str(path.stat().st_size).encode("ascii"))
    return hasher.hexdigest()


def _cpu_state_dict(values: Mapping[str, torch.Tensor]):
    return {name: value.detach().cpu() for name, value in values.items()}
