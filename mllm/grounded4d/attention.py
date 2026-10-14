"""Explicit, weight-compatible first-layer attention behavior for spatial studies."""
import hashlib
from pathlib import Path

import torch


ATTENTION_MODES = ("masked", "full")


def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def attention_behavior(mode):
    if mode not in ATTENTION_MODES:
        raise ValueError("unknown first-layer attention mode: " + str(mode))
    return dict(first_layer_attention=mode, implementation_sha256=implementation_sha256())


def checkpoint_attention_behavior(metadata):
    mode = metadata["training_config"].get("first_layer_attention", "masked")
    if mode not in ATTENTION_MODES:
        raise ValueError("unknown checkpoint attention mode")
    behavior = metadata.get("attention_behavior")
    if behavior is None:
        if mode != "masked":
            raise ValueError("non-default attention requires implementation provenance")
        return dict(first_layer_attention="masked", implementation_sha256=None, legacy_default=True)
    if behavior.get("first_layer_attention") != mode:
        raise ValueError("checkpoint attention behavior and training recipe disagree")
    if behavior.get("implementation_sha256") != implementation_sha256():
        raise ValueError("checkpoint attention implementation identity differs")
    if metadata.get("source_sha256", {}).get("attention.py") != behavior["implementation_sha256"]:
        raise ValueError("checkpoint attention provenance fields disagree")
    return dict(behavior, legacy_default=False)


def _allow_all_memory(module, inputs):
    queries, memory, bias = inputs
    # Retain finite positional bias; remove only hard exclusions in this layer.
    return queries, memory, bias.masked_fill(torch.isneginf(bias), 0.0)


def configure_first_attention(model, mode):
    if mode not in ATTENTION_MODES:
        raise ValueError("unknown first-layer attention mode: " + str(mode))
    handle = getattr(model, "_first_attention_handle", None)
    if handle is not None:
        handle.remove()
        model._first_attention_handle = None
    if mode == "full":
        # A pre-hook preserves state_dict names, tensor shapes and initialization.
        model._first_attention_handle = model.layers[0].register_forward_pre_hook(_allow_all_memory)
    return model
