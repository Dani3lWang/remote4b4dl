"""Explicit model-only migration and paired initialization for E2-Q.

This experiment does not change the production runner's resume/warm-start rules.
"""
from contextlib import contextmanager
import hashlib

import torch


ARMS = {"Q0": "learned", "Q1": "zero_learned", "Q2": "zero_fps"}
QUERY_KEYS = ("decoder.query_feat.weight", "decoder.query_embed.weight")


def tensor_digest(state, *, shared_only=False):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if shared_only and name in QUERY_KEYS:
            continue
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def configure_trainability(model, arm):
    """Shared network trains in every arm; only unused query tables are fixed."""
    if arm not in ARMS or model.config.query_initialization != ARMS[arm]:
        raise ValueError("arm and model query mode disagree")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name not in QUERY_KEYS or
                                 name == QUERY_KEYS[0] and arm == "Q0" or
                                 name == QUERY_KEYS[1] and arm != "Q2")
    return {name: p.requires_grad for name, p in model.named_parameters()}


def initialize_model(model, parent_state, arm, seed=2026):
    """Replace BOTH historical query tables; use a private CPU RNG for pairing."""
    if arm not in ARMS or model.config.query_initialization != ARMS[arm]:
        raise ValueError("arm and model query mode disagree")
    if set(parent_state) != set(model.state_dict()):
        raise ValueError("parent/model parameter keys differ")
    if any(parent_state[k].shape != v.shape or parent_state[k].dtype != v.dtype
           for k, v in model.state_dict().items()):
        raise ValueError("parent/model parameter shape or dtype differs")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fresh = {key: torch.randn(parent_state[key].shape, generator=generator,
                              dtype=parent_state[key].dtype) for key in QUERY_KEYS}
    state = dict(parent_state)
    state[QUERY_KEYS[0]] = fresh[QUERY_KEYS[0]] if arm == "Q0" else torch.zeros_like(fresh[QUERY_KEYS[0]])
    state[QUERY_KEYS[1]] = fresh[QUERY_KEYS[1]] if arm != "Q2" else torch.zeros_like(fresh[QUERY_KEYS[1]])
    model.load_state_dict(state, strict=True)
    flags = configure_trainability(model, arm)
    if tensor_digest(model.state_dict(), shared_only=True) != tensor_digest(parent_state, shared_only=True):
        raise ValueError("shared parent parameters were modified")
    return dict(arm=arm,mode=ARMS[arm],seed=seed,reset_parameters=list(QUERY_KEYS),
                shared_state_sha256=tensor_digest(parent_state,shared_only=True),
                initial_state_sha256=tensor_digest(model.state_dict()),
                query_sha256={k:tensor_digest({k:model.state_dict()[k]}) for k in QUERY_KEYS},
                trainable=flags)


@contextmanager
def preserve_model_modes(model):
    """Restore per-parameter flags and per-module modes even if evaluation fails."""
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    modes = {name: module.training for name, module in model.named_modules()}
    try:
        yield
    finally:
        for name, p in model.named_parameters():
            p.requires_grad_(flags[name])
        for name, module in model.named_modules():
            module.training = modes[name]


def selection_gates(treatment, controls):
    """The preregistered E2-Q screen; every necessary control is retained."""
    if len(controls)<2:
        raise ValueError("both frozen F and a contemporaneous control are required")
    scenes = set(treatment["strata"]["scene_token"])
    if len(scenes)!=8 or any(set(m["strata"]["scene_token"])!=scenes for m in controls):
        raise ValueError("all arms must contain the same eight development scenes")
    small = lambda m: m["strata"]["bucket"]["1-10"]
    gates = dict(
        final_tp=treatment["tp"] >= max(m["tp"] for m in controls)+10,
        precision=treatment["precision"] >= max(m["precision"] for m in controls)-.02,
        small_final=small(treatment)["tp"] >= max(small(m)["tp"] for m in controls),
        small_geometry=small(treatment)["raw_geometry_tp"] >= max(small(m)["raw_geometry_tp"] for m in controls)+20,
        ignored=treatment["ignored_fraction"] <= max(m["ignored_fraction"] for m in controls)+.01,
        ego=treatment["ego_points"] <= max(m["ego_points"] for m in controls)*1.1+100)
    # Recall/scene constraints apply to each contemporaneous trained control.
    for size, key in (("11-49","medium_recall"),("50+","large_recall")):
        gates[key] = all(treatment["strata"]["bucket"][size]["recall"] >=
                         m["strata"]["bucket"][size]["recall"]-.02 for m in controls[1:])
    gates["scenes"] = all(sum(v["recall"] >= m["strata"]["scene_token"][scene]["recall"]
                                  for scene,v in treatment["strata"]["scene_token"].items()) >= 5
                          for m in controls[1:])
    return gates
