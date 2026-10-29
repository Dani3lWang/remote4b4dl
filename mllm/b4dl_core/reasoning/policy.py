"""Multimodal completion log probabilities and shared-base LoRA policies."""
from contextlib import contextmanager
import copy
from pathlib import Path
import random

import numpy as np
import torch

from ..integration import FRAME_KEYS, check_context, language_model, padding_side, prompt_tokens
from ..inputs import collate_frames
from ..io import atomic_output
from .protocol import encode_trace


def sequence_batch(model, tokenizer, items, completions=None):
    """Build assistant-only labels without re-tokenizing generated completions."""
    from vtimellm.constants import IGNORE_INDEX
    if not items or completions is not None and len(items) != len(completions):
        raise ValueError("one completion per nonempty input batch required")
    rows, labels = [], []
    for i, item in enumerate(items):
        prefix = prompt_tokens(tokenizer, item["prompt"])
        suffix = [] if completions is None else [int(t) for t in completions[i]]
        if completions is not None and not suffix:
            raise ValueError("empty completion")
        rows.append(prefix + suffix)
        labels.append([IGNORE_INDEX]*len(prefix) + suffix)
    device = language_model(model).get_input_embeddings().weight.device
    size = max(map(len, rows))
    ids = torch.full((len(rows), size), tokenizer.eos_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    truth = torch.full_like(ids, IGNORE_INDEX)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row, device=device)
        mask[i, :len(row)] = 1
        truth[i, :len(row)] = torch.tensor(labels[i], device=device)
    frames = {k: v.to(device) for k, v in collate_frames(items).items()}
    check_context(model, ids, mask, frames["frame_valid_mask"])
    batch = {"input_ids": ids, "attention_mask": mask, "images": frames.pop("features"), **frames}
    if completions is not None:
        batch["labels"] = truth
    return batch


def teacher_completions(tokenizer, targets, max_new_tokens):
    rows = [tokenizer.encode(encode_trace(t["trace"]), add_special_tokens=False) + [tokenizer.eos_token_id]
            for t in targets]
    if any(len(ids) > max_new_tokens for ids in rows):
        raise ValueError("teacher trace exceeds generation budget")
    return rows


def expanded_log_probs(model, batch):
    """Return each completion's causal log-probs after multimodal expansion."""
    from vtimellm.constants import IGNORE_INDEX
    base = language_model(model)
    with padding_side(base, "right"):
        _, positions, attention, _, embeds, labels = base.prepare_inputs_labels_for_multimodal(
            batch["input_ids"], None, batch["attention_mask"], None, batch["labels"],
            batch["images"], **{key: batch[key] for key in FRAME_KEYS})
    expected = batch["labels"].ne(IGNORE_INDEX).sum(-1)
    if labels is None or not torch.equal(labels.ne(IGNORE_INDEX).sum(-1), expected):
        raise ValueError("multimodal expansion changed completion token coverage")
    if labels[:, 0].ne(IGNORE_INDEX).any():
        raise ValueError("a completion token has no causal predecessor")
    logits = model(inputs_embeds=embeds, attention_mask=attention, position_ids=positions,
                   use_cache=False).logits[:, :-1].float()
    shifted = labels[:, 1:]
    mask = shifted.ne(IGNORE_INDEX)
    safe = shifted.masked_fill(~mask, 0)
    selected = torch.log_softmax(logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    result = [selected[i][mask[i]] for i in range(len(expected))]
    if any(len(row) != int(n) for row, n in zip(result, expected)):
        raise ValueError("completion log-probability alignment failed")
    return result


def decode_completion(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def step_token_ends(tokenizer, ids, char_ends):
    """Map exact decoded character ends to one-based original token positions."""
    text = decode_completion(tokenizer, ids)
    result, previous = [], 0
    for end in char_ends:
        if not previous < end <= len(text):
            raise ValueError("invalid character step boundary")
        lo, hi = 1, len(ids)
        while lo < hi:
            mid = (lo + hi)//2
            prefix = decode_completion(tokenizer, ids[:mid])
            if not text.startswith(prefix):
                raise ValueError("tokenizer prefix decoding is not stable")
            if len(prefix) >= end:
                hi = mid
            else:
                lo = mid+1
        prefix = decode_completion(tokenizer, ids[:lo])
        if not text.startswith(prefix) or len(prefix) < end:
            raise ValueError("step boundary cannot be mapped to generated tokens")
        result.append(lo)
        previous = end
    return result


def add_reference_adapter(model):
    """Copy the active SFT LoRA; disabling LoRA is not the SFT reference."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    if "reference" in model.peft_config:
        raise ValueError("reference adapter already exists")
    state = get_peft_model_state_dict(model, adapter_name="default")
    model.add_adapter("reference", copy.deepcopy(model.peft_config["default"]))
    result = set_peft_model_state_dict(model, state, adapter_name="reference")
    if getattr(result, "unexpected_keys", []):
        raise ValueError("unexpected reference adapter keys")
    activate_adapter(model, "default")


def activate_adapter(model, name):
    if name not in ("default", "reference") or name not in model.peft_config:
        raise ValueError("unknown policy adapter")
    model.set_adapter(name)
    for param_name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "default" and ".lora_" in param_name and ".default." in param_name)


@contextmanager
def reference_policy(model):
    activate_adapter(model, "reference")
    try:
        yield
    finally:
        activate_adapter(model, "default")


def extension_state(model):
    state = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
             if ".frame_adapter." in name or ".mm_projector." in name or ".lora_" in name}
    if not state or any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("missing or nonfinite extension state")
    return state


def restore_extension(model, state):
    current = {name: p for name, p in model.named_parameters()
               if ".frame_adapter." in name or ".mm_projector." in name or ".lora_" in name}
    if set(current) != set(state):
        raise ValueError("extension parameter set mismatch")
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != current[name].shape or not torch.isfinite(value).all():
                raise ValueError("invalid extension tensor: " + name)
            current[name].copy_(value)


def rng_state():
    kind, values, pos, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": (kind, values.tolist(), pos, has_gauss, cached),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    kind, values, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((kind, np.asarray(values, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA topology differs from checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, identity, stage, recipe, step, optimizer=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError("checkpoint already exists")
    payload = {"schema": "b4dl-reasoning-checkpoint-v1", "identity": identity, "stage": stage,
               "recipe": recipe, "step": step, "parameters": extension_state(model),
               "optimizer": optimizer.state_dict() if optimizer is not None else None, "rng": rng_state()}
    with atomic_output(path, "wb") as stream:
        torch.save(payload, stream)


def load_checkpoint(path, model, expected_identity=None, expected_stage=None, optimizer=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema") != "b4dl-reasoning-checkpoint-v1"
            or expected_identity is not None and payload["identity"] != expected_identity
            or expected_stage is not None and payload["stage"] != expected_stage):
        raise ValueError("reasoning checkpoint identity or stage mismatch")
    restore_extension(model, payload["parameters"])
    if optimizer is not None:
        if payload["optimizer"] is None:
            raise ValueError("checkpoint has no optimizer")
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])
    return payload
