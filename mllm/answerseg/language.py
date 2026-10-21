"""Assistant-only supervision and hidden states aligned by the real visual expansion."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100
MARKERS = ("<SEG>", "<NOOBJ>")


@dataclass(frozen=True)
class SegTokens:
    seg: int
    noobj: int
    original_vocab: int


def register_tokens(tokenizer, model):
    """Call AFTER restoring/merging B3. Preserve its existing special tokens."""
    original = len(tokenizer)
    if model.get_input_embeddings().weight.shape[0] != original:
        raise ValueError(
            "restore the matching B3 tokenizer and embeddings before adding SEG tokens"
        )
    if any(token in tokenizer.get_vocab() for token in MARKERS):
        raise ValueError(
            "expected an unextended B3; restore AnswerSeg tokens from its bundle instead"
        )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(MARKERS)},
        replace_additional_special_tokens=False,
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    with torch.no_grad():
        for embedding in (model.get_input_embeddings(), model.get_output_embeddings()):
            embedding.weight[original:] = embedding.weight[:original].mean(0)
    ids = [tokenizer.convert_tokens_to_ids(token) for token in MARKERS]
    if ids != [original, original + 1]:
        raise ValueError("SEG/NOOBJ must be distinct, newly appended single tokens")
    for marker, token_id in zip(MARKERS, ids):
        if tokenizer.encode(marker, add_special_tokens=False) != [token_id]:
            raise ValueError("segmentation marker is not a single token")
    return SegTokens(*ids, original)


class TokenEmbedding(nn.Module):
    """Keep original weights frozen; optimize only two FP32 embedding rows."""

    def __init__(self, base, start):
        super().__init__()
        self.base, self.start = base, start
        base.requires_grad_(False)
        self.new_rows = nn.Parameter(base.weight[start:].detach().float().clone())

    @property
    def weight(self):
        return self.base.weight

    def forward(self, ids):
        original = self.base(ids)
        selected = ids.ge(self.start)
        replacement = F.embedding((ids - self.start).clamp_min(0), self.new_rows).to(
            original.dtype
        )
        return torch.where(selected[..., None], replacement, original)


class TokenHead(nn.Module):
    """FP32 new-token output rows avoid FP16 Adam underflow and full-vocab states."""

    def __init__(self, base, start):
        super().__init__()
        self.base, self.start = base, start
        self.in_features, self.out_features = base.in_features, base.out_features
        base.requires_grad_(False)
        self.new_rows = nn.Parameter(base.weight[start:].detach().float().clone())

    @property
    def weight(self):
        return self.base.weight

    def forward(self, states):
        original = self.base(states)[..., : self.start]
        new = F.linear(states.float(), self.new_rows).to(original.dtype)
        return torch.cat((original, new), -1)


def token_rows(layer, start):
    return layer.new_rows if hasattr(layer, "new_rows") else layer.weight[start:]


def train_new_token_rows(model, tokens):
    if getattr(model.config, "pretraining_tp", 1) != 1:
        raise ValueError("AnswerSeg token output head requires pretraining_tp=1")
    model.set_input_embeddings(
        TokenEmbedding(model.get_input_embeddings(), tokens.original_vocab)
    )
    model.set_output_embeddings(
        TokenHead(model.get_output_embeddings(), tokens.original_vocab)
    )


def mark_answer(answer, targets):
    """Each reviewed span identifies exactly one instance or an explicit absent target."""
    if (
        not answer.strip()
        or any(marker in answer for marker in MARKERS)
        or "<video>" in answer
    ):
        raise ValueError("expected an unmarked, nonempty source answer")
    if not isinstance(targets, list) or any(
        not isinstance(item, dict) for item in targets
    ):
        raise ValueError("reviewed targets must be an array of span records")
    for item in targets:
        span = item.get("span")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(v) is not int for v in span)
        ):
            raise ValueError("invalid reviewed answer span")
    ordered = sorted(targets, key=lambda item: item["span"])
    if not ordered or len(ordered) > 8:
        raise ValueError("one to eight reviewed answer spans are required")
    pieces, cursor = [], 0
    for target in ordered:
        span = target.get("span")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(value) is not int for value in span)
            or not cursor <= span[0] < span[1] <= len(answer)
        ):
            raise ValueError("invalid or overlapping reviewed spans")
        if not answer[span[0] : span[1]].strip():
            raise ValueError("empty target phrase")
        status, token = target.get("status"), target.get("instance_token")
        if status == "matched" and isinstance(token, str) and token:
            marker = MARKERS[0]
        elif status == "no_target" and token is None:
            marker = MARKERS[1]
        else:
            raise ValueError("pending/ambiguous targets cannot become NOOBJ")
        pieces.extend((answer[cursor : span[1]], " " + marker))
        cursor = span[1]
    pieces.append(answer[cursor:])
    return "".join(pieces), ordered


def make_prompt(question, *, segmentation=False):
    # Import lazily: data checks do not require a language-model runtime.
    from vtimellm.conversation import conv_templates

    if not isinstance(question, str) or not question.strip():
        raise ValueError("nonempty question required")
    if any(marker in question for marker in MARKERS) or question.count("<video>") > 1:
        raise ValueError("question contains reserved markers or multiple visual slots")
    query = question if "<video>" in question else "<video>\n" + question
    if segmentation:
        query += (
            "\nAnswer briefly. Immediately after each referred object phrase, emit <SEG>."
            " For an explicitly absent target emit <NOOBJ>."
        )
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def prompt_ids(tokenizer, prompt):
    if prompt.count("<video>") != 1:
        raise ValueError("exactly one <video> slot is required")
    before, after = prompt.split("<video>")
    left = tokenizer.encode(before, add_special_tokens=True)
    right = tokenizer.encode(after, add_special_tokens=False)
    return torch.tensor([left + [IMAGE_TOKEN_INDEX] + right], dtype=torch.long)


def training_sequence(tokenizer, prompt, marked_answer):
    prefix = prompt_ids(tokenizer, prompt)
    continuation = tokenizer.encode(" " + marked_answer, add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("the B3 tokenizer needs an EOS token")
    suffix = torch.tensor([continuation + [tokenizer.eos_token_id]], dtype=torch.long)
    ids = torch.cat((prefix, suffix), 1)
    labels = ids.clone()
    labels[:, : prefix.shape[1]] = IGNORE_INDEX
    return ids, labels


class LanguageBridge:
    def __init__(self, model, tokenizer, tokens):
        self.model, self.tokenizer, self.tokens = model, tokenizer, tokens

    def forward(self, ids, labels, features, *, marker_loss_weight=1.0):
        """Use expanded labels as positions, never original text token offsets."""
        if not math.isfinite(marker_loss_weight) or marker_loss_weight < 1:
            raise ValueError("marker_loss_weight must be finite and at least one")
        if ids.shape[0] != 1 or ids.shape != labels.shape:
            raise ValueError(
                "AnswerSeg currently supports one unpadded sample per forward"
            )
        if features.ndim != 2 or features.shape[1] != 768 or not len(features):
            raise ValueError("B3 features must be nonempty [frames,768]")
        if not torch.isfinite(features).all():
            raise ValueError("nonfinite scene features")
        device = self.model.get_input_embeddings().weight.device
        dtype = self.model.get_model().mm_projector.weight.dtype
        ids, labels = ids.to(device), labels.to(device)
        if int(ids.eq(IMAGE_TOKEN_INDEX).sum()) != 1:
            raise ValueError("exactly one visual slot is required")
        expected = ids.shape[1] - 1 + len(features)
        limits = [
            getattr(self.model.config, name, None)
            for name in ("tokenizer_model_max_length", "max_position_embeddings")
        ]
        if any(limit is not None and expected > limit for limit in limits):
            raise ValueError(
                "expanded prompt/answer exceeds context; refusing silent target truncation"
            )
        prepared = self.model.prepare_inputs_labels_for_multimodal(
            ids,
            torch.arange(ids.shape[1], device=device)[None],
            torch.ones_like(ids, dtype=torch.bool),
            None,
            labels,
            features.to(device=device, dtype=dtype)[None],
        )
        _, positions, attention, _, embeds, expanded_labels = prepared
        if (
            embeds is None
            or embeds.shape[1] != expected
            or expanded_labels.shape[1] != expected
        ):
            raise ValueError(
                "visual expansion changed/truncated the expected token sequence"
            )
        output = self.model(
            inputs_embeds=embeds,
            position_ids=positions,
            attention_mask=attention,
            labels=expanded_labels,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        marker = expanded_labels.eq(self.tokens.seg) | expanded_labels.eq(
            self.tokens.noobj
        )
        states = output.hidden_states[-1][marker].float()
        kinds = expanded_labels[marker]
        loss = output.loss
        if marker_loss_weight != 1:
            # Causal shift: supervise logits BEFORE each assistant marker. Never
            # score a marker against the hidden state that already contains it.
            shifted = expanded_labels[:, 1:]
            weights = shifted.ne(IGNORE_INDEX).to(output.logits.dtype)
            is_marker = shifted.eq(self.tokens.seg) | shifted.eq(self.tokens.noobj)
            weights = weights * (1 + is_marker * (marker_loss_weight - 1))
            per_token = F.cross_entropy(
                output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                shifted.reshape(-1),
                reduction="none",
                ignore_index=IGNORE_INDEX,
            ).reshape_as(shifted)
            loss = (per_token * weights).sum() / weights.sum().clamp_min(1)
        return loss, states, kinds

    @torch.no_grad()
    def generate(self, prompt, features, max_new_tokens=256):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        ids = prompt_ids(self.tokenizer, prompt).to(
            self.model.get_input_embeddings().weight.device
        )
        if (
            features.ndim != 2
            or features.shape[1] != 768
            or not len(features)
            or not torch.isfinite(features).all()
        ):
            raise ValueError("B3 features must be finite nonempty [frames,768]")
        expanded = ids.shape[1] - 1 + len(features)
        limits = [
            getattr(self.model.config, key, None)
            for key in ("tokenizer_model_max_length", "max_position_embeddings")
        ]
        if any(
            limit is not None and expanded + max_new_tokens > limit for limit in limits
        ):
            raise ValueError("generation budget exceeds expanded B3 context")
        dtype = self.model.get_model().mm_projector.weight.dtype
        output = self.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            images=features.to(device=ids.device, dtype=dtype)[None],
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        if output.shape[1] < ids.shape[1] or not torch.equal(
            output[:, : ids.shape[1]], ids
        ):
            raise ValueError(
                "unsupported generation output: B3 prompt prefix was not retained"
            )
        labels = output.clone()
        labels[:, : ids.shape[1]] = IGNORE_INDEX
        answer = self.tokenizer.decode(
            output[0, ids.shape[1] :], skip_special_tokens=False
        )
        if self.tokenizer.eos_token:
            answer = answer.removesuffix(self.tokenizer.eos_token)
        return {
            "answer": answer.strip(),
            "output_ids": output,
            "labels": labels,
            "finished": bool(output[0, -1] == self.tokenizer.eos_token_id),
        }
