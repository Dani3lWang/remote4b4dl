"""Token registration and trainable row adapters for reasoning segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


B4DL_SPECIAL_TOKENS = ("<4DLiDAR>", "<meta>")
SEGMENTATION_SPECIAL_TOKENS = ("<LOC>", "<SEG>", "<NOOBJ>")


@dataclass(frozen=True)
class SegmentationTokenIds:
    loc: int
    seg: int
    noobj: int

    def as_dict(self) -> Dict[str, int]:
        return {"<LOC>": self.loc, "<SEG>": self.seg, "<NOOBJ>": self.noobj}

    @property
    def ordered(self) -> tuple[int, int, int]:
        return self.loc, self.seg, self.noobj


def _token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise RuntimeError(f"special token was not registered as one token: {token}")
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if encoded != [token_id]:
        raise RuntimeError(f"special token splits into multiple ids: {token} -> {encoded}")
    return int(token_id)


def register_segmentation_tokens(tokenizer, model) -> SegmentationTokenIds:
    """Append segmentation tokens *after* the 32,002-row B3 state is loaded.

    The caller must first load and merge the B3 checkpoint.  Loading B3's full
    ``[32002, hidden]`` embedding matrix into a model already resized to 32005
    is a hard shape error even with ``strict=False``.
    """

    missing_b4dl = [token for token in B4DL_SPECIAL_TOKENS if _safe_unknown(tokenizer, token)]
    if missing_b4dl:
        raise RuntimeError(
            "B3 tokens must be registered before segmentation tokens; missing "
            + ", ".join(missing_b4dl)
        )

    old_size = len(tokenizer)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(B4DL_SPECIAL_TOKENS + SEGMENTATION_SPECIAL_TOKENS)}
    )
    if added:
        model.resize_token_embeddings(len(tokenizer))
        _mean_initialize_new_rows(model, old_size)

    ids = SegmentationTokenIds(
        loc=_token_id(tokenizer, "<LOC>"),
        seg=_token_id(tokenizer, "<SEG>"),
        noobj=_token_id(tokenizer, "<NOOBJ>"),
    )
    if len(set(ids.ordered)) != 3:
        raise RuntimeError(f"segmentation token ids are not unique: {ids.as_dict()}")
    if min(ids.ordered) < old_size and added:
        raise RuntimeError(
            f"new segmentation tokens did not append after the B3 vocabulary: {ids.as_dict()}"
        )
    model.config.vocab_size = len(tokenizer)
    return ids


def _safe_unknown(tokenizer, token: str) -> bool:
    token_id = tokenizer.convert_tokens_to_ids(token)
    return token_id is None or token_id == tokenizer.unk_token_id


def _mean_initialize_new_rows(model, old_size: int) -> None:
    input_weight = model.get_input_embeddings().weight.data
    output_weight = model.get_output_embeddings().weight.data
    if old_size <= 0 or old_size >= input_weight.shape[0]:
        return
    with torch.no_grad():
        input_weight[old_size:] = input_weight[:old_size].float().mean(dim=0).to(input_weight.dtype)
        output_weight[old_size:] = output_weight[:old_size].float().mean(dim=0).to(output_weight.dtype)


class SegTokenInputAdapter(nn.Module):
    """Add independent trainable deltas at selected token rows.

    The underlying full embedding stays frozen.  This avoids a dense gradient
    hook over the entire vocabulary and remains safe under ZeRO partitioning.
    """

    def __init__(self, base: nn.Embedding, token_ids: Sequence[int]):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.register_buffer("token_ids", torch.tensor(token_ids, dtype=torch.long))
        self.delta = nn.Parameter(
            torch.zeros(len(token_ids), base.embedding_dim, dtype=base.weight.dtype)
        )

    @property
    def weight(self):
        return self.base.weight

    @property
    def num_embeddings(self) -> int:
        return self.base.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.base.embedding_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        values = self.base(input_ids)
        for row, token_id in enumerate(self.token_ids.tolist()):
            values = values + (input_ids == token_id).unsqueeze(-1).to(values.dtype) * self.delta[row]
        return values


class SegTokenOutputAdapter(nn.Module):
    """Replace the frozen LM-head rows for new tokens with trainable rows."""

    def __init__(self, base: nn.Linear, token_ids: Sequence[int]):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.register_buffer("token_ids", torch.tensor(token_ids, dtype=torch.long))
        initial = self.base.weight.detach()[list(token_ids)].clone()
        self.rows = nn.Parameter(initial)

    @property
    def weight(self):
        return self.base.weight

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.base(hidden_states)
        replacement = F.linear(hidden_states, self.rows)
        # index_copy is differentiable for the source and does not mutate the
        # frozen base-head output in place.
        return logits.index_copy(-1, self.token_ids, replacement)


def install_trainable_token_adapters(model, token_ids: SegmentationTokenIds) -> None:
    """Install independent input/output parameters for segmentation tokens."""

    if isinstance(model.get_input_embeddings(), SegTokenInputAdapter):
        return
    input_adapter = SegTokenInputAdapter(model.get_input_embeddings(), token_ids.ordered)
    output_adapter = SegTokenOutputAdapter(model.get_output_embeddings(), token_ids.ordered)
    model.set_input_embeddings(input_adapter)
    model.set_output_embeddings(output_adapter)


def token_adapter_state_dict(model) -> Mapping[str, torch.Tensor]:
    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    if not isinstance(input_module, SegTokenInputAdapter):
        raise TypeError("segmentation input-token adapter is not installed")
    if not isinstance(output_module, SegTokenOutputAdapter):
        raise TypeError("segmentation output-token adapter is not installed")
    return {
        "input_delta": input_module.delta.detach().cpu(),
        "output_rows": output_module.rows.detach().cpu(),
        "token_ids": input_module.token_ids.detach().cpu(),
    }


def load_token_adapter_state_dict(model, state: Mapping[str, torch.Tensor]) -> None:
    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    if not isinstance(input_module, SegTokenInputAdapter):
        raise TypeError("segmentation input-token adapter is not installed")
    if not isinstance(output_module, SegTokenOutputAdapter):
        raise TypeError("segmentation output-token adapter is not installed")
    saved_ids = tuple(int(value) for value in state["token_ids"].tolist())
    current_ids = tuple(int(value) for value in input_module.token_ids.tolist())
    if saved_ids != current_ids:
        raise RuntimeError(f"segmentation token ids changed: saved={saved_ids}, current={current_ids}")
    if tuple(state["input_delta"].shape) != tuple(input_module.delta.shape):
        raise RuntimeError("input token adapter shape mismatch")
    if tuple(state["output_rows"].shape) != tuple(output_module.rows.shape):
        raise RuntimeError("output token adapter shape mismatch")
    with torch.no_grad():
        input_module.delta.copy_(state["input_delta"].to(input_module.delta))
        output_module.rows.copy_(state["output_rows"].to(output_module.rows))
