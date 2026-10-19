"""E2-P: explicit architecture migration, without relaxing native resume rules.

This residual-head ablation is inspired by P3Former's separate position mask
mechanism. It is not an implementation of its MPE, PA-Seg or MFA modules.
"""
import torch

from .e2q import selection_gates as query_selection_gates, tensor_digest


ARMS = {"P0": "none", "P1": "feature", "P2": "position"}
RESIDUAL_PREFIX = "decoder.mask_residual_embed."


def initialize_model(model, parent_state, arm, seed=2026):
    if (arm not in ARMS or model.config.mask_residual_mode != ARMS[arm]
            or model.config.query_initialization != "learned"):
        raise ValueError("E2-P requires mature learned queries and the specified residual mode")
    state = model.state_dict()
    extra = sorted(set(state) - set(parent_state))
    expected = sorted(RESIDUAL_PREFIX + f"layers.{i}.{suffix}"
                      for i in range(3) for suffix in ("weight", "bias")) if arm != "P0" else []
    if set(parent_state) - set(state) or extra != expected:
        raise ValueError("unexpected architecture migration keys")
    for name, value in parent_state.items():
        if value.shape != state[name].shape or value.dtype != state[name].dtype:
            raise ValueError("parent tensor shape/dtype mismatch: " + name)
    # The new head has a private CPU RNG, independent of backbone/model creation.
    # Its paired hidden layers are random; final weight AND bias are zero.
    if extra:
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed also reseeds CUDA, which a CPU-only fork does
            # not restore. Touch only the CPU default generator here.
            torch.random.default_generator.manual_seed(seed)
            head = model.decoder.mask_residual_embed
            for layer in head.layers:
                layer.reset_parameters()
            torch.nn.init.zeros_(head.layers[-1].weight)
            torch.nn.init.zeros_(head.layers[-1].bias)
        state = model.state_dict()
    state.update(parent_state)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(True)
    shared = {k: v for k, v in model.state_dict().items() if k in parent_state}
    if tensor_digest(shared) != tensor_digest(parent_state):
        raise ValueError("inherited parent tensors changed")
    return dict(arm=arm,mode=ARMS[arm],seed=seed,reset_parameters=[],added_parameters=extra,
                inherited_state_sha256=tensor_digest(shared),
                initial_state_sha256=tensor_digest(model.state_dict()),
                added_state_sha256=tensor_digest({k:model.state_dict()[k] for k in extra}),
                added_parameter_count=sum(state[k].numel() for k in extra),
                trainable={n:p.requires_grad for n,p in model.named_parameters()})


def small_diagnostics(metrics):
    """Medians over EVERY fixed 1-10 point target, including misses."""
    return metrics["geometry_diagnostics"]["1-10"]


def selection_gates(treatment, controls):
    gates = query_selection_gates(treatment, controls)
    small = lambda m:m["strata"]["bucket"]["1-10"]
    gates["small_final"] = small(treatment)["tp"] >= max(small(m)["tp"] for m in controls) + 10
    diag = small_diagnostics(treatment)
    gates["small_purity"] = diag["median_best_query_purity"] >= max(
        small_diagnostics(m)["median_best_query_purity"] for m in controls) + .01
    gates["small_coverage"] = diag["median_best_query_coverage"] >= max(
        small_diagnostics(m)["median_best_query_coverage"] for m in controls) - .05
    gates["small_iou"] = diag["median_best_iou"] >= max(
        small_diagnostics(m)["median_best_iou"] for m in controls)
    return gates
