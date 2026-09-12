#!/usr/bin/env python3
"""Small CPU smoke test for the optional frame-position path.

Run in the training environment (any working directory):

    python mllm/scripts/verify_frame_position.py

The test deliberately uses a tiny dummy multimodal model, so it does not
load Vicuna, a checkpoint, or LiDAR features.
"""

from types import SimpleNamespace

import os
import sys

import torch
import torch.nn as nn

# vtimellm is not pip-installed in this repo; every entry point adds mllm/ itself.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vtimellm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from vtimellm.model.vtimellm_arch import (
    VTimeLLMMetaForCausalLM,
    VTimeLLMMetaModel,
)


class DummyVision(nn.Module, VTimeLLMMetaModel):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=4,
            model_type="llama",
        )
        self.token_embeddings = nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.token_embeddings


class DummyCausal(VTimeLLMMetaForCausalLM):
    def __init__(self, vision):
        self.model = vision
        self.config = vision.config

    def get_model(self):
        return self.model


def main():
    vision = DummyVision()
    vision.initialize_vision_modules(
        SimpleNamespace(
            pretrain_mm_mlp_adapter=None,
            use_frame_position_embedding=True,
            frame_position_max=8,
        )
    )
    model = DummyCausal(vision)

    input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    images = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32
    )

    _, _, _, _, baseline_embeds, _ = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, attention_mask, None, labels, images,
        frame_indices=torch.tensor([[2, 4]], dtype=torch.long),
    )
    projected = vision.mm_projector(images)[0]
    assert torch.allclose(baseline_embeds[0, 1:3], projected)

    with torch.no_grad():
        vision.frame_position_embedding.weight[2].fill_(1.0)
        vision.frame_position_embedding.weight[4].fill_(2.0)
    _, _, _, _, positioned_embeds, _ = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, attention_mask, None, labels, images,
        frame_indices=torch.tensor([[2, 4]], dtype=torch.long),
    )
    expected = projected + torch.tensor([[1.0] * 4, [2.0] * 4])
    assert torch.allclose(positioned_embeds[0, 1:3], expected)

    try:
        model.prepare_inputs_labels_for_multimodal(
            input_ids, None, attention_mask, None, labels, images,
            frame_indices=torch.tensor([[2]], dtype=torch.long),
        )
    except ValueError as exc:
        assert "feature frames" in str(exc)
    else:
        raise AssertionError("frame-count mismatch did not fail fast")

    print("[PASS] frame-position embedding CPU smoke test")


if __name__ == "__main__":
    main()
