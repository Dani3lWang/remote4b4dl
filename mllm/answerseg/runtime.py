"""Restore B3 first, attach new adapters, and save/reload an identifiable extension."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import torch

from . import SCHEMA
from .data import digest, write_json
from .language import register_tokens, token_rows, train_new_token_rows
from .model import AnswerSegModel


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def artifact_identity(path):
    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and (item.suffix in (".json", ".bin", ".safetensors", ".model", ".txt"))
        and not item.name.startswith(("optimizer", "scheduler", "training_args"))
    )
    if not files:
        raise ValueError("empty model artifact: " + str(path))
    return digest(
        {item.relative_to(path).as_posix(): file_sha256(item) for item in files}
    )


def source_identity(runtime, *, include_spatial=True):
    result = {
        key: artifact_identity(runtime[key])
        for key in ("model_base", "pretrain_mm_mlp_adapter", "stage2", "stage3")
        if runtime.get(key)
    }
    if "model_base" not in result or "stage2" not in result:
        raise ValueError("runtime must identify a restored B3 base and stage2 adapter")
    if include_spatial:
        result["spatial_checkpoint"] = artifact_identity(runtime["spatial_checkpoint"])
    return result


def implementation_identity():
    root = Path(__file__).resolve().parent
    files = list(root.glob("*.py"))
    files.extend((root.parent / "vtimellm" / "model").glob("*.py"))
    files.append(root.parent / "vtimellm" / "conversation.py")
    return digest(
        {
            str(path.relative_to(root.parent)).replace("\\", "/"): file_sha256(path)
            for path in sorted(files)
        }
    )


def configure_lora(language, tokens, *, rank=16, alpha=32):
    from peft import LoraConfig, get_peft_model

    if type(rank) is not int or rank < 1 or type(alpha) is not int or alpha < 1:
        raise ValueError("LoRA rank/alpha must be positive integers")
    language.requires_grad_(False)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    language = get_peft_model(language, config)
    train_new_token_rows(language, tokens)
    return language


def load_b3(runtime, device):
    from vtimellm.model.builder import load_pretrained_model

    device = runtime.get("language_device", device)
    if not runtime.get("pretrain_mm_mlp_adapter"):
        raise ValueError("restore B3 with its explicit stage1/mm_projector artifact")
    if "chatglm" in runtime["model_base"].lower():
        raise ValueError("the first AnswerSeg bridge supports the B3 Vicuna/Llama path")
    args = SimpleNamespace(
        model_base=runtime["model_base"],
        pretrain_mm_mlp_adapter=runtime.get("pretrain_mm_mlp_adapter"),
    )
    tokenizer, model, _ = load_pretrained_model(
        args, runtime.get("stage2"), runtime.get("stage3")
    )
    model = model.to(
        device=device,
        dtype=torch.float32 if torch.device(device).type == "cpu" else torch.float16,
    )
    model.config.use_cache = False
    return tokenizer, model


def build_model(
    runtime,
    device,
    *,
    stage="binding",
    tune_spatial=False,
    minimum_iou=0.5,
    rank=16,
    alpha=32,
    marker_loss_weight=1.0,
):
    from maskpls_b4dl.config import MaskPLSConfig
    from maskpls_b4dl.model import MaskPLS
    from maskpls_b4dl.runner import load_native

    tokenizer, language = load_b3(runtime, device)
    tokens = register_tokens(tokenizer, language)
    if stage == "joint":
        language = configure_lora(language, tokens, rank=rank, alpha=alpha)
    checkpoint = load_native(runtime["spatial_checkpoint"])
    spatial = MaskPLS(MaskPLSConfig.from_dict(checkpoint["metadata"]["config"])).to(
        device
    )
    spatial.load_state_dict(checkpoint["model"], strict=True)
    model = AnswerSegModel(
        language,
        tokenizer,
        tokens,
        spatial,
        stage=stage,
        tune_spatial=tune_spatial,
        minimum_iou=minimum_iou,
        marker_loss_weight=marker_loss_weight,
    )
    model.head.to(device)
    options = {
        "stage": stage,
        "tune_spatial": tune_spatial,
        "minimum_iou": minimum_iou,
        "rank": rank,
        "alpha": alpha,
        "marker_loss_weight": marker_loss_weight,
    }
    return model, options


def optimizer_for(model, lr):
    if not 0 < lr < 1:
        raise ValueError("learning rate must lie in (0,1)")
    embeddings = {
        id(model.language.get_input_embeddings().weight),
        id(model.language.get_output_embeddings().weight),
    }
    ordinary, rows = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (rows if id(parameter) in embeddings else ordinary).append(parameter)
    groups = [{"params": ordinary, "weight_decay": 0.01}]
    if rows:
        groups.append({"params": rows, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=lr)


def extension_state(model):
    excluded = {
        id(model.language.get_input_embeddings().weight),
        id(model.language.get_output_embeddings().weight),
    }
    state = {
        key: value.detach().cpu()
        for key, value in model.named_parameters()
        if value.requires_grad
        and id(value) not in excluded
        and not key.startswith("spatial.")
    }
    rows = {
        key: token_rows(layer, model.tokens.original_vocab).detach().cpu()
        for key, layer in (
            ("input", model.language.get_input_embeddings()),
            ("output", model.language.get_output_embeddings()),
        )
    }
    # Running statistics are part of a tuned spatial module's state as well.
    spatial = (
        {key: value.detach().cpu() for key, value in model.spatial.state_dict().items()}
        if model.tune_spatial
        else None
    )
    return {"parameters": state, "token_rows": rows, "spatial": spatial}


def restore_extension(model, state):
    expected = extension_state(model)
    if set(expected["parameters"]) != set(state["parameters"]):
        raise ValueError(
            "trainable extension parameter names differ from the saved stage"
        )
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for key, value in state["parameters"].items():
            if value.shape != parameters[key].shape:
                raise ValueError("extension parameter shape mismatch: " + key)
            parameters[key].copy_(value)
        for key, layer in (
            ("input", model.language.get_input_embeddings()),
            ("output", model.language.get_output_embeddings()),
        ):
            target = token_rows(layer, model.tokens.original_vocab)
            value = state["token_rows"][key]
            if target.shape != value.shape:
                raise ValueError("saved new-token embedding shape mismatch")
            target.copy_(value)
    if (state["spatial"] is not None) != model.tune_spatial:
        raise ValueError("saved spatial tuning stage mismatch")
    if model.tune_spatial:
        model.spatial.load_state_dict(state["spatial"], strict=True)


def save_bundle(directory, model, options, identity, training, optimizer=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.bridge.tokenizer.save_pretrained(directory / "tokenizer")
    payload = {
        "schema_version": SCHEMA,
        "extension": extension_state(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "training": training,
    }
    path = directory / "extension.pt"
    torch.save(payload, path)
    metadata = {
        "schema_version": SCHEMA,
        "options": options,
        "source_identity": identity,
        "implementation_sha256": implementation_identity(),
        "tokens": vars(model.tokens),
        "extension_sha256": file_sha256(path),
        "tokenizer_sha256": artifact_identity(directory / "tokenizer"),
        "training": training,
        "language_grounding_accepted": False,
        "model_performance_accepted": False,
        "tracking_accepted": False,
        "runtime_versions": {
            "torch": str(torch.__version__),
            "transformers": version("transformers"),
            "peft": version("peft"),
        },
        "language_dtype": str(model.language.get_input_embeddings().weight.dtype),
    }
    write_json(directory / "bundle.json", metadata)
    return metadata


def load_bundle(directory, runtime, device):
    directory = Path(directory)
    metadata = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA:
        raise ValueError("not an AnswerSeg bundle")
    if metadata["implementation_sha256"] != implementation_identity():
        raise ValueError("AnswerSeg source differs from the bundle implementation")
    if source_identity(runtime) != metadata["source_identity"]:
        raise ValueError("B3/spatial base identity differs from the saved extension")
    if (
        file_sha256(directory / "extension.pt") != metadata["extension_sha256"]
        or artifact_identity(directory / "tokenizer") != metadata["tokenizer_sha256"]
    ):
        raise ValueError("extension/tokenizer integrity mismatch")
    model, _options = build_model(runtime, device, **metadata["options"])
    if vars(model.tokens) != metadata["tokens"]:
        raise ValueError("B3 vocabulary/token IDs changed")
    payload = torch.load(
        directory / "extension.pt", map_location="cpu", weights_only=True
    )
    if (
        payload["schema_version"] != SCHEMA
        or payload["training"] != metadata["training"]
    ):
        raise ValueError("bundle payload/metadata mismatch")
    restore_extension(model, payload["extension"])
    return model, metadata
