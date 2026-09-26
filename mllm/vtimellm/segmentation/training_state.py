"""Explicit configuration and data-position contracts for exact SEG resume."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Sampler

from .config import ReasonSegConfig


class EpochRandomSampler(Sampler):
    """Epoch order is independent of model/dropout and DataLoader RNG consumption."""

    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self):
        return len(self.dataset)


def resolve_training_config(args):
    checkpoint = args.resume_from_checkpoint or args.eval_checkpoint
    config = ReasonSegConfig()
    if checkpoint:
        metadata = json.loads((Path(checkpoint) / "reasonseg_config.json").read_text())
        config = ReasonSegConfig.from_dict(metadata["config"])
    overrides = {key: getattr(args, key, None) for key in
                 ("dropout", "bce_mode", "region_loss", "tversky_alpha", "tversky_beta")}
    if args.no_loc_prior:
        overrides["use_loc_prior"] = False
    for key, value in overrides.items():
        if value is None:
            continue
        if args.resume_from_checkpoint and getattr(config, key) != value:
            raise ValueError(f"exact resume cannot change {key}; use weights-only initialization")
        setattr(config, key, value)
    config.validate()
    return config


def resume_contract(args, num_processes, batches_per_epoch):
    keys = ("seed", "per_device_train_batch_size", "gradient_accumulation_steps",
            "learning_rate", "lora_learning_rate", "weight_decay", "warmup_ratio",
            "num_train_epochs", "mixed_precision", "model_max_length", "max_grad_norm",
            "validation_samples", "validation_threshold", "early_stopping_patience",
            "early_stopping_min_delta")
    contract = {key: getattr(args, key) for key in keys}
    for key in ("train_manifest", "validation_manifest"):
        path = getattr(args, key)
        contract[key + "_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    contract.update(num_processes=num_processes, batches_per_epoch=batches_per_epoch)
    return contract


def resume_position(state, contract):
    if state.get("resume_format_version") != 2:
        raise ValueError("legacy checkpoint has no reliable batch cursor; use --eval-checkpoint for a new run")
    if state.get("resume_contract") != contract:
        raise ValueError("resume data/schedule/topology changed; use a new weights-only run")
    cursor = int(state["next_batch_index"])
    if not 0 <= cursor <= contract["batches_per_epoch"]:
        raise ValueError("invalid resume batch cursor")
    if state["epoch_completed"]:
        if cursor != contract["batches_per_epoch"]:
            raise ValueError("completed epoch must consume all batches")
        return int(state["epoch"]) + 1, 0
    return int(state["epoch"]), cursor
