"""B3 visual expansion, assistant-only batches, and explicit extension state."""
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import torch

from .inputs import collate_frames
from .temporal import FrameTemporalAdapter, TemporalConfig, validate_frames

FRAME_KEYS = ("frame_times", "frame_geometry", "frame_valid_mask")


def language_model(model):
    return model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model


def attach(model, arm="A3", temporal=None, lora_rank=0):
    from peft import PeftModel
    from peft.tuners.tuners_utils import BaseTunerLayer
    if arm not in ("A0", "A1", "A2", "A3") or lora_rank < 0 or arm == "A0" and lora_rank:
        raise ValueError("invalid ablation arm/LoRA rank")
    if isinstance(model, PeftModel) or any(isinstance(m, BaseTunerLayer) for m in model.modules()):
        raise ValueError("attach only to a restored and merged original B3")
    if hasattr(model.get_model(), "frame_adapter"):
        raise ValueError("attach only to a restored and merged original B3")
    # PEFT can leave configuration metadata after merge_and_unload.
    if hasattr(model, "peft_config"):
        delattr(model, "peft_config")
    config = temporal or TemporalConfig(use_geometry=arm == "A3")
    if arm in ("A2", "A3") and config.use_geometry != (arm == "A3"):
        raise ValueError("temporal configuration disagrees with arm")
    model.requires_grad_(False)
    if lora_rank:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=2*lora_rank,
            lora_dropout=0., bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]))
    base = language_model(model)
    core = base.get_model()
    if arm in ("A2", "A3"):
        core.frame_adapter = FrameTemporalAdapter(config).to(device=core.mm_projector.weight.device, dtype=torch.float32)
    if arm != "A0":
        core.mm_projector.float().requires_grad_(True)
    # Optimizer master parameters stay FP32 even when the frozen B3 uses FP16.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    base.config.use_cache = False
    options = {"arm": arm, "temporal": asdict(config) if arm in ("A2", "A3") else None, "lora_rank": lora_rank}
    return model, options


def project_frames(model, images, times, poses, valid):
    if not isinstance(images, torch.Tensor) or times is None:
        raise ValueError("core B3 requires dense [B,T,768] features and explicit frame_times")
    if valid is None:
        valid = torch.ones(images.shape[:2], dtype=torch.bool, device=images.device)
    core = model.get_model()
    adapter = getattr(core, "frame_adapter", None)
    validate_frames(images, times, poses, valid, geometry=adapter is not None and adapter.config.use_geometry)
    with torch.autocast(device_type=images.device.type, enabled=False):
        values = images
        if adapter is not None:
            values = adapter(images.to(next(adapter.parameters()).dtype), times, poses, valid)
        dtype = model.get_input_embeddings().weight.dtype
        projector = core.mm_projector
        values = torch.nn.functional.linear(values[valid].to(dtype), projector.weight.to(dtype),
            projector.bias.to(dtype) if projector.bias is not None else None)
    values = values.to(model.get_input_embeddings().weight.dtype)
    return list(torch.split(values, valid.sum(-1).tolist(), dim=0))


@contextmanager
def padding_side(model, side):
    config = language_model(model).config
    previous = getattr(config, "tokenizer_padding_side", None)
    config.tokenizer_padding_side = side
    try:
        yield
    finally:
        if previous is None:
            delattr(config, "tokenizer_padding_side")
        else:
            config.tokenizer_padding_side = previous


def check_context(model, ids, attention, valid, extra=0):
    from vtimellm.constants import IMAGE_TOKEN_INDEX
    if ids.ndim != 2 or attention.shape != ids.shape or valid.shape[0] != ids.shape[0]:
        raise ValueError("incompatible text/frame batch axes")
    if not ((ids == IMAGE_TOKEN_INDEX) & attention.bool()).sum(-1).eq(1).all():
        raise ValueError("exactly one visual slot per core request required")
    lengths = attention.sum(-1) - 1 + valid.sum(-1)
    limits = [getattr(language_model(model).config, name, None)
              for name in ("max_position_embeddings", "tokenizer_model_max_length")]
    if any(limit is not None and int(lengths.max()) + extra > limit for limit in limits):
        raise ValueError("expanded multimodal context exceeds limit; refusing truncation")
    return lengths


def expand_generation(model, inputs, kwargs):
    kwargs = dict(kwargs)
    if inputs is None:
        inputs = kwargs.pop("input_ids", None)
    elif "input_ids" in kwargs:
        raise ValueError("provide inputs or input_ids, not both")
    if inputs is None or kwargs.get("inputs_embeds") is not None or kwargs.get("past_key_values") is not None:
        raise ValueError("core generation requires fresh token IDs")
    images = kwargs.pop("images")
    frames = {k: kwargs.pop(k, None) for k in FRAME_KEYS}
    valid = frames["frame_valid_mask"]
    if valid is None:
        valid = torch.ones(images.shape[:2], device=images.device, dtype=torch.bool)
        frames["frame_valid_mask"] = valid
    attention = kwargs.pop("attention_mask", torch.ones_like(inputs))
    budget = kwargs.get("max_new_tokens")
    if type(budget) is not int or budget < 1:
        raise ValueError("explicit positive max_new_tokens required for core generation")
    lengths = check_context(model, inputs, attention, valid, budget)
    if kwargs.pop("position_ids", None) is not None or kwargs.pop("cache_position", None) is not None:
        raise ValueError("core generation computes positions after visual expansion")
    with padding_side(model, "left"):
        _, _, expanded_attention, _, embeds, _ = model.prepare_inputs_labels_for_multimodal(
            inputs, None, attention, None, None, images, **frames)
    if not torch.equal(expanded_attention.sum(-1), lengths):
        raise ValueError("visual expansion lost valid tokens")
    kwargs.update(inputs_embeds=embeds, attention_mask=expanded_attention)
    return None, kwargs


def prompt_tokens(tokenizer, question):
    from vtimellm.conversation import conv_templates
    from vtimellm.constants import IMAGE_TOKEN_INDEX
    if not isinstance(question, str) or not question.strip() or "<video>" in question:
        raise ValueError("request prompt must not contain a reserved visual marker")
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], "<video>\n" + question)
    conv.append_message(conv.roles[1], None)
    before, after = conv.get_prompt().split("<video>")
    return tokenizer.encode(before, add_special_tokens=True) + [IMAGE_TOKEN_INDEX] + tokenizer.encode(after, add_special_tokens=False)


def batch_inputs(model, tokenizer, items, targets=None):
    from vtimellm.constants import IGNORE_INDEX
    if targets is not None and len(targets) != len(items):
        raise ValueError("one target per training input required")
    rows, labels = [], []
    for i, item in enumerate(items):
        prefix = prompt_tokens(tokenizer, item["prompt"])
        answer = []
        if targets is not None:
            target = targets[i]
            content = json.dumps({"answer": target["answer"], "evidence": target["evidence"]}, ensure_ascii=False, allow_nan=False)
            answer = tokenizer.encode(content, add_special_tokens=False) + [tokenizer.eos_token_id]
        rows.append(prefix + answer)
        labels.append([IGNORE_INDEX]*len(prefix) + answer)
    device = language_model(model).get_input_embeddings().weight.device
    size = max(map(len, rows))
    ids = torch.full((len(rows), size), tokenizer.eos_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    truth = torch.full_like(ids, IGNORE_INDEX)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row, device=device)
        attention[i, :len(row)] = 1
        truth[i, :len(row)] = torch.tensor(labels[i], device=device)
    tensors = {k: v.to(device) for k, v in collate_frames(items).items()}
    check_context(model, ids, attention, tensors["frame_valid_mask"])
    batch = {"input_ids": ids, "attention_mask": attention, "images": tensors.pop("features"), **tensors}
    if targets is not None:
        batch["labels"] = truth
    return batch


def trainable_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, state):
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(state):
        raise ValueError("checkpoint trainable parameter set mismatch")
    for name, value in state.items():
        if parameters[name].shape != value.shape or not torch.isfinite(value).all():
            raise ValueError("invalid checkpoint parameter: " + name)
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)


def save_checkpoint(path, model, options, identity, training, optimizer=None, scaler=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError("checkpoint already exists")
    payload = {"schema": "b4dl-core-b3-extension-v1", "options": options, "identity": identity,
        "parameters": trainable_state(model), "training": training,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError("partial checkpoint exists")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, options, identity, optimizer=None, scaler=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema"] != "b4dl-core-b3-extension-v1" or payload["options"] != options or payload["identity"] != identity:
        raise ValueError("checkpoint schema, configuration or baseline/data identity mismatch")
    restore_trainable(model, payload["parameters"])
    if optimizer is not None:
        if payload["optimizer"] is None:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and payload["scaler"] is not None:
            scaler.load_state_dict(payload["scaler"])
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"]:
            if len(payload["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("resume CUDA device topology mismatch")
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["training"]
