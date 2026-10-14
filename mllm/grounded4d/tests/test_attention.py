"""Attention scope, weight compatibility and checkpoint provenance boundaries."""
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from grounded4d.attention import attention_behavior, checkpoint_attention_behavior, configure_first_attention


class CaptureLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.seen = []

    def forward(self, queries, memory, bias):
        self.seen.append(bias.detach().clone())
        return queries * self.scale


class AttentionTests(unittest.TestCase):
    def model(self):
        model = nn.Module()
        model.layers = nn.ModuleList([CaptureLayer() for _ in range(3)])
        return model

    def run_layers(self, model):
        queries, memory = torch.ones(2, 4), torch.ones(3, 4)
        bias = torch.tensor([[0., -float("inf"), -.3], [-float("inf"), 0., -.7]])
        for layer in model.layers:
            queries = layer(queries, memory, bias)
        return bias

    def test_default_leaves_all_layer_biases_unchanged(self):
        model = configure_first_attention(self.model(), "masked")
        bias = self.run_layers(model)
        for layer in model.layers:
            torch.testing.assert_close(layer.seen[-1], bias)

    def test_full_changes_only_first_layer_and_preserves_parameters(self):
        model = self.model()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        configure_first_attention(model, "full")
        bias = self.run_layers(model)
        torch.testing.assert_close(model.layers[0].seen[-1], bias.masked_fill(torch.isneginf(bias), 0))
        for layer in model.layers[1:]:
            torch.testing.assert_close(layer.seen[-1], bias)
        self.assertEqual(list(before), list(model.state_dict()))
        for key, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[key])

    def test_reconfiguration_is_idempotent_and_can_restore_default(self):
        model = self.model()
        configure_first_attention(model, "full")
        configure_first_attention(model, "full")
        self.assertEqual(len(model.layers[0]._forward_pre_hooks), 1)
        configure_first_attention(model, "masked")
        bias = self.run_layers(model)
        torch.testing.assert_close(model.layers[0].seen[-1], bias)
        self.assertEqual(len(model.layers[0]._forward_pre_hooks), 0)

    def test_legacy_and_explicit_checkpoint_contracts(self):
        self.assertEqual(checkpoint_attention_behavior(dict(training_config={}))['first_layer_attention'], 'masked')
        behavior = attention_behavior('full')
        metadata = dict(training_config=dict(first_layer_attention='full'), attention_behavior=behavior,
                        source_sha256={'attention.py': behavior['implementation_sha256']})
        self.assertEqual(checkpoint_attention_behavior(metadata)['first_layer_attention'], 'full')
        with self.assertRaises(ValueError):
            checkpoint_attention_behavior(dict(training_config=dict(first_layer_attention='full')))
        metadata['attention_behavior']['implementation_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            checkpoint_attention_behavior(metadata)


if __name__ == "__main__":
    unittest.main()
