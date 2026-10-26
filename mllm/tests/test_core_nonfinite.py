"""Numerical failure regressions; only CPU fixtures, never real training."""
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from b4dl_core import b3
from b4dl_core.data import build_dataset
from b4dl_core.inputs import GeometryDataset
from b4dl_core.integration import attach, load_checkpoint
from b4dl_core.io import read_json, write_json
from test_core_b3_integration import TinyTokenizer, small_config, tensors, tiny
from test_core_geometry import raw_fixture


class NonfiniteTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def scalar_setup(self):
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = b3.optimizer_for(model, 1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        return model, optimizer, scaler

    def test_streamed_diagnostic_decoder_matches_native_sdpa_layer(self):
        import importlib.util
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        path = Path(__file__).resolve().parents[1]/"scripts/diagnose_core_nonfinite.py"
        spec = importlib.util.spec_from_file_location("diagnostic", path)
        diagnostic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diagnostic)
        model = tiny().half()
        layer = model.model.layers[0]
        rotary = LlamaRotaryEmbedding(config=model.config)
        hidden = torch.randn(1, 11, 32).half()
        def linear(name, value):
            return layer.get_submodule(name)(value)
        def norm(name, value):
            return layer.get_submodule(name)(value)
        positions = torch.arange(hidden.shape[1]).unsqueeze(0)
        with torch.no_grad():
            expected = layer(hidden, position_ids=positions, use_cache=False,
                             position_embeddings=rotary(hidden, positions))[0]
            actual = diagnostic.decoder_forward(model.config, hidden, rotary, linear, norm)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_later_micro_loss_failure_clears_partial_gradients_without_update(self):
        model, optimizer, scaler = self.scalar_setup()
        before = model.weight.detach().clone()
        def loss(unused, batch):
            return model.weight.sum() if batch["ok"].item() else model.weight.sum()*float("nan")
        with patch.object(b3, "teacher_loss", loss):
            with self.assertRaises(b3.NonfiniteTrainingError) as caught:
                b3.train_step(model, [{"ok": torch.tensor(True)}, {"ok": torch.tensor(False)}], optimizer, scaler)
        self.assertEqual(caught.exception.details["micro"], 1)
        self.assertEqual(len(caught.exception.details["finite_micro_losses"]), 1)
        self.assertEqual(caught.exception.details["loss"]["nonfinite_count"], 1)
        torch.testing.assert_close(before, model.weight, rtol=0, atol=0)
        self.assertIsNone(model.weight.grad)
        self.assertFalse(optimizer.state)

    def test_nonfinite_gradient_is_named_and_optimizer_is_untouched(self):
        model, optimizer, scaler = self.scalar_setup()
        before = model.weight.detach().clone()
        model.weight.register_hook(lambda grad: grad*float("inf"))
        with patch.object(b3, "teacher_loss", lambda *unused: model.weight.sum()):
            with self.assertRaises(b3.NonfiniteTrainingError) as caught:
                b3.train_step(model, [{}], optimizer, scaler)
        self.assertEqual(caught.exception.details["stage"], "unscaled_gradient")
        self.assertIn("weight", caught.exception.details["nonfinite_gradients"])
        torch.testing.assert_close(before, model.weight, rtol=0, atol=0)
        self.assertIsNone(model.weight.grad)
        self.assertFalse(optimizer.state)

    def test_finite_gradients_with_overflowed_norm_cannot_update(self):
        model, optimizer, scaler = self.scalar_setup()
        before = model.weight.detach().clone()
        with patch.object(b3, "teacher_loss", lambda *unused: model.weight.sum()*1e30):
            with self.assertRaises(b3.NonfiniteTrainingError) as caught:
                b3.train_step(model, [{}], optimizer, scaler)
        self.assertEqual(caught.exception.details["stage"], "gradient_norm")
        torch.testing.assert_close(before, model.weight, rtol=0, atol=0)
        self.assertIsNone(model.weight.grad)
        self.assertFalse(optimizer.state)

    def test_loss_diagnostic_distinguishes_finite_logits_and_empty_supervision(self):
        model = tiny()
        batch = tensors()
        batch["labels"] = torch.full_like(batch["input_ids"], -100)
        with self.assertRaises(b3.NonfiniteTrainingError) as caught:
            b3.teacher_loss(model, batch)
        self.assertEqual(caught.exception.details["stage"], "forward_loss")
        self.assertEqual(caught.exception.details["logits"]["nonfinite_count"], 0)
        self.assertEqual(caught.exception.details["loss"]["nonfinite_count"], 1)

    def test_failure_receipt_identifies_attempt_and_requests_without_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, metadata, features, splits = raw_fixture(root)
            build_dataset(tables, metadata, features, splits, root/"dataset")
            dataset = GeometryDataset(root/"dataset", "train")
            write_json(root/"baseline.json", {"fixture": True})
            output = root/"output"; output.mkdir()
            args = SimpleNamespace(device="cpu", arm="A3", lora_rank=0, lr=1e-4, seed=42,
                batch_size=1, grad_accum=2, steps=1, save_every=1, resume=None,
                engineering=True, output=str(output), baseline=str(root/"baseline.json"))
            with patch.object(b3, "load_original", lambda *unused: (TinyTokenizer(), tiny())), \
                 patch.object(b3, "teacher_loss", lambda *unused: torch.tensor(float("nan"))):
                with self.assertRaises(b3.NonfiniteTrainingError):
                    b3.train(args, {"frame_adapter": asdict(small_config())}, {}, dataset)
            failure = read_json(output/"failure.json")
            self.assertEqual(failure["attempted_step"], 1)
            self.assertEqual(failure["completed_updates"], 0)
            self.assertFalse(failure["optimizer_update_applied"])
            self.assertEqual(failure["diagnostics"]["micro"], 0)
            for micro in range(2):
                index = b3.batch_indices(len(dataset.requests), 0, micro, 1, 2, 42)[0]
                self.assertEqual(failure["requests_by_micro"][micro][0]["request_id"], dataset.requests[index]["request_id"])
            self.assertFalse((output/"completion.json").exists())
            self.assertFalse((output/"latest.json").exists())
            self.assertFalse(list(output.glob("step-*.pt")))
            snapshot = output/failure["diagnostic_checkpoint"]
            self.assertTrue(snapshot.is_file())
            payload = torch.load(snapshot, map_location="cpu", weights_only=True)
            self.assertTrue(payload["training"]["diagnostic_only"])
            self.assertEqual(payload["training"]["step"], 0)
            model, options = attach(tiny(), temporal=small_config())
            with self.assertRaisesRegex(ValueError, "diagnostic checkpoint"):
                load_checkpoint(snapshot, model, options, payload["identity"])


if __name__ == "__main__":
    unittest.main()
