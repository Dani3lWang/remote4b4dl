"""Build the segmentation model on top of a fully restored B3 checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

import torch

from vtimellm.model.builder import load_pretrained_model

from .checkpoint import load_reasonseg_checkpoint, load_spatial_encoder_checkpoint
from .config import ReasonSegConfig
from .model import ReasonSegModel
from .tokens import install_trainable_token_adapters, register_segmentation_tokens


SEGMENTATION_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_reasonseg_model(
    args,
    *,
    b3_checkpoint: str,
    reasonseg_config: Optional[ReasonSegConfig] = None,
    segmentation_checkpoint: Optional[str] = None,
    spatial_checkpoint: Optional[str] = None,
    trainable: bool = False,
    point_encoder=None,
):
    """Load Vicuna -> B4DL rows -> Stage 1 -> B3 -> segmentation rows.

    This order is an invariant.  ``non_lora_trainables.bin`` in the real B3
    checkpoint stores a full 32,002-row input embedding matrix, so adding the
    three segmentation rows before B3 restoration causes a shape mismatch.
    """

    if segmentation_checkpoint and spatial_checkpoint:
        raise ValueError(
            "spatial_checkpoint is only for a new ReasonSeg run; a ReasonSeg "
            "checkpoint already contains the complete spatial encoder"
        )

    tokenizer, language_model, context_length = load_pretrained_model(
        args, stage2=b3_checkpoint
    )
    base_vocab_size = len(tokenizer)
    if base_vocab_size < 32002:
        raise RuntimeError(
            f"B3 restoration is incomplete: expected at least 32002 tokens, got {base_vocab_size}"
        )

    token_ids = register_segmentation_tokens(tokenizer, language_model)
    install_trainable_token_adapters(language_model, token_ids)
    checkpoint_config = None
    if segmentation_checkpoint:
        metadata_path = Path(segmentation_checkpoint) / "reasonseg_config.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint_config = ReasonSegConfig.from_dict(metadata["config"])
    if reasonseg_config is not None and checkpoint_config is not None:
        requested = reasonseg_config.architecture_dict()
        stored = checkpoint_config.architecture_dict()
        if requested != stored:
            differing = sorted(
                key for key in requested if requested.get(key) != stored.get(key)
            )
            raise RuntimeError(
                "requested ReasonSeg architecture differs from checkpoint "
                f"(loss-only fields are ignored); differing keys: {differing}"
            )
    config = reasonseg_config or checkpoint_config or ReasonSegConfig(
        hidden_size=int(language_model.config.hidden_size)
    )
    config.validate()

    if segmentation_checkpoint:
        lora_dir = Path(segmentation_checkpoint) / "seg_lora"
        if lora_dir.is_dir():
            from peft import PeftModel

            language_model = PeftModel.from_pretrained(
                language_model, str(lora_dir), is_trainable=trainable
            )
    elif trainable:
        from peft import LoraConfig, get_peft_model

        for parameter in language_model.parameters():
            parameter.requires_grad_(False)
        language_model = get_peft_model(
            language_model,
            LoraConfig(
                r=64,
                lora_alpha=128,
                lora_dropout=0.05,
                bias="none",
                target_modules=SEGMENTATION_LORA_TARGETS,
                task_type="CAUSAL_LM",
            ),
        )
    if trainable:
        # PEFT freezes every non-LoRA parameter while wrapping the model, so
        # restore the two independently checkpointed token parameter blocks
        # afterwards.
        language_model.get_input_embeddings().delta.requires_grad_(True)
        language_model.get_output_embeddings().rows.requires_grad_(True)
        language_model.config.use_cache = False
        if hasattr(language_model, "gradient_checkpointing_enable"):
            language_model.gradient_checkpointing_enable()
        if hasattr(language_model, "enable_input_require_grads"):
            language_model.enable_input_require_grads()

    model = ReasonSegModel(
        language_model=language_model,
        token_ids=token_ids,
        config=config,
        point_encoder=point_encoder,
    )
    if spatial_checkpoint:
        load_spatial_encoder_checkpoint(
            model.point_encoder,
            spatial_checkpoint,
            config=config,
        )
    if segmentation_checkpoint:
        load_reasonseg_checkpoint(
            model,
            tokenizer,
            segmentation_checkpoint,
            expected_b3_checkpoint=b3_checkpoint,
        )
    if not trainable:
        model.requires_grad_(False)
        model.eval()
    return tokenizer, model, context_length
