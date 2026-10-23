from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from vtimellm.model.vtimellm_llama import VTimeLLMConfig, VTimeLLMLlamaForCausalLM
from b4dl_core.integration import (attach, batch_inputs, check_context, language_model,
                                  load_checkpoint, save_checkpoint, trainable_state)
from b4dl_core.temporal import TemporalConfig
from b4dl_core import b3
from b4dl_core.io import write_json, read_json


class TinyTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return ([1] if add_special_tokens else []) + [3 + ord(c) % 57 for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


def tiny():
    torch.manual_seed(21)
    config = VTimeLLMConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=1024, bos_token_id=1, eos_token_id=2, pad_token_id=2)
    config._attn_implementation = "sdpa"
    model = VTimeLLMLlamaForCausalLM(config)
    model.get_model().mm_projector = torch.nn.Linear(768, 32)
    return model.eval()


def tensors():
    return dict(input_ids=torch.tensor([[1, 4, -200, 7, 8], [1, -200, 9, 2, 2]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
        images=torch.randn(2, 5, 768), frame_times=torch.tensor([[0., .5, 1., 1.5, 2.], [0., .5, 1., 0., 0.]], dtype=torch.float64),
        frame_geometry=torch.eye(4, dtype=torch.float64).repeat(2, 5, 1, 1),
        frame_valid_mask=torch.tensor([[1]*5, [1, 1, 1, 0, 0]], dtype=torch.bool))


def small_config():
    return TemporalConfig(layers=1, ff_width=32)


class B3IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_legacy_list_projection_and_zero_adapter_logits(self):
        model = tiny()
        batch = tensors()
        old = {k: v for k, v in batch.items() if not k.startswith("frame_")}
        old["images"] = [batch["images"][i, mask] for i, mask in enumerate(batch["frame_valid_mask"])]
        with torch.no_grad(): before = model(**old).logits
        model, _ = attach(model, temporal=small_config())
        with torch.no_grad(): after = model(**batch).logits
        torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_generation_variable_lengths_batch_equals_single_and_adapter_called_once(self):
        model, _ = attach(tiny(), temporal=small_config())
        model.eval()
        batch = tensors()
        torch.nn.init.normal_(model.get_model().frame_adapter.output.weight, std=.001)
        calls = []
        hook = model.get_model().frame_adapter.register_forward_hook(lambda *args: calls.append(1))
        generated = model.generate(**batch, max_new_tokens=4, do_sample=False, use_cache=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(generated.shape[0], 2)
        hook.remove()
        for i in range(2):
            single = {k: v[i:i+1] for k, v in batch.items()}
            one = model.generate(**single, max_new_tokens=4, do_sample=False, use_cache=True)
            n = one.shape[1]
            torch.testing.assert_close(generated[i, :n], one[0], atol=0, rtol=0)
        uncached = model.generate(**batch, max_new_tokens=4, do_sample=False, use_cache=False)
        torch.testing.assert_close(generated, uncached, atol=0, rtol=0)

    def test_visual_and_prompt_labels_masked_and_finite_gradient(self):
        model, _ = attach(tiny(), temporal=small_config(), lora_rank=4)
        items = [{"prompt": "Q", "features": np.ones((3, 768), np.float32), "frame_times": np.arange(3)*.5,
                  "frame_geometry": np.tile(np.eye(4), (3, 1, 1))}]
        batch = batch_inputs(model, TinyTokenizer(), items, [{"answer": "Yes", "evidence": {"count": 1}}])
        self.assertTrue(batch["labels"][batch["input_ids"] == -200].eq(-100).all())
        raw = language_model(model)
        expanded = raw.prepare_inputs_labels_for_multimodal(batch["input_ids"], None, batch["attention_mask"], None,
            batch["labels"], batch["images"], **{k: batch[k] for k in ("frame_times", "frame_geometry", "frame_valid_mask")})
        self.assertEqual(int(expanded[-1].ne(-100).sum()), int(batch["labels"].ne(-100).sum()))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        for _ in range(2):
            optimizer.zero_grad()
            loss = model(**batch).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        self.assertGreater(float(raw.get_model().frame_adapter.blocks[0].attention.in_proj_weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad))
        model.eval()
        inference = batch_inputs(model, TinyTokenizer(), items)
        self.assertNotIn("labels", inference)
        result = model.generate(**inference, max_new_tokens=2, do_sample=False)
        self.assertEqual(result.shape[0], 1)

    def test_checkpoint_restores_optimizer_and_same_next_step(self):
        model, options = attach(tiny(), temporal=small_config(), lora_rank=4)
        batch = tensors()
        batch["labels"] = torch.tensor([[-100, -100, -100, 7, 8], [-100, -100, 9, -100, -100]])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        def step(m, opt):
            opt.zero_grad(); loss = m(**batch).loss; loss.backward(); opt.step()
        step(model, optimizer)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"checkpoint.pt"
            save_checkpoint(path, model, options, {"baseline": "fixture", "dataset": "fixture"}, {"step": 1}, optimizer)
            step(model, optimizer)
            expected = trainable_state(model)
            restored, restored_options = attach(tiny(), temporal=small_config(), lora_rank=4)
            opt2 = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.001)
            status = load_checkpoint(path, restored, restored_options, {"baseline": "fixture", "dataset": "fixture"}, opt2)
            self.assertEqual(status, {"step": 1})
            step(restored, opt2)
            for key, value in trainable_state(restored).items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                load_checkpoint(path, restored, restored_options, {"baseline": "WRONG"})

    def test_fail_closed_missing_geometry_context_multiple_visual_slots(self):
        model, _ = attach(tiny(), temporal=small_config())
        batch = tensors()
        for key in ("frame_times", "frame_geometry"):
            bad = dict(batch); bad.pop(key)
            with self.assertRaises(ValueError): model(**bad)
        model.config.tokenizer_model_max_length = 5
        with self.assertRaises(ValueError): model.generate(**batch, max_new_tokens=4)
        model.config.tokenizer_model_max_length = 1024
        batch["input_ids"][0, 1] = -200
        with self.assertRaises(ValueError): model.generate(**batch, max_new_tokens=4)

    def test_a0_a1_a2_paths(self):
        batch = tensors()
        for arm in ("A0", "A1", "A2"):
            m, _ = attach(tiny(), arm=arm, temporal=TemporalConfig(layers=1, ff_width=32, use_geometry=False))
            self.assertEqual(hasattr(m.get_model(), "frame_adapter"), arm == "A2")
            self.assertTrue(torch.isfinite(m(**batch).logits).all())
            if arm == "A0": self.assertFalse(any(p.requires_grad for p in m.parameters()))

    def test_merged_b3_accepts_extension_but_live_lora_is_rejected(self):
        from peft import LoraConfig, get_peft_model
        for rank in (0, 4):
            model = get_peft_model(tiny(), LoraConfig(r=2, lora_alpha=4,
                target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
            with self.assertRaises(ValueError):
                attach(model, temporal=small_config())
            merged = model.merge_and_unload()
            self.assertTrue(hasattr(merged, "peft_config"))
            extended, _ = attach(merged, temporal=small_config(), lora_rank=rank)
            self.assertTrue(torch.isfinite(extended(**tensors()).logits).all())

    def test_training_entry_resume_and_inference_prediction_contract(self):
        from test_core_geometry import raw_fixture
        from b4dl_core.data import build_dataset
        from b4dl_core.inputs import GeometryDataset
        from b4dl_core.evaluation import evaluate
        from b4dl_core.io import read_jsonl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, metadata, features, splits = raw_fixture(root)
            build_dataset(tables, metadata, features, splits, root/"dataset")
            dataset = GeometryDataset(root/"dataset", "train")
            write_json(root/"baseline.json", {"fixture": True})
            preparation = {"frame_adapter": asdict(small_config())}
            args = SimpleNamespace(device="cpu", arm="A3", lora_rank=4, lr=.001, seed=42,
                batch_size=1, grad_accum=2, steps=1, save_every=1, resume=None, engineering=True,
                output=str(root/"first"), baseline=str(root/"baseline.json"), condition="identity", split="train", max_new_tokens=2)
            Path(args.output).mkdir()
            def loader(*unused): return TinyTokenizer(), tiny()
            with patch.object(b3, "load_original", loader):
                first = b3.train(args, preparation, {}, dataset)
                self.assertEqual(first["optimizer_steps"], 1)
                args.resume = str(root/"first/step-1.pt")
                args.steps = 2
                args.output = str(root/"resume")
                Path(args.output).mkdir()
                resumed = b3.train(args, preparation, {}, dataset)
                self.assertEqual(resumed["resumed_from_step"], 1)
                args.checkpoint = str(root/"resume/step-2.pt")
                args.output = str(root/"inference")
                Path(args.output).mkdir()
                b3.infer(args, preparation, {"baseline_id": "tiny-test-fixture"}, dataset)
            predictions = read_jsonl(Path(args.output)/"predictions.jsonl")
            self.assertEqual(len(predictions), len(dataset.requests))
            report = evaluate(dataset, predictions, read_json(Path(args.output)/"run.json"), repetitions=10)
            self.assertEqual(report["summary"]["n"], len(dataset.requests))
            receipts = read_jsonl(Path(args.output)/"input-receipts.jsonl")
            self.assertTrue(all(len(r["feature_array_sha256"]) == 64 for r in receipts))

    def test_baseline_files_fail_closed_before_loading(self):
        import hashlib
        import json
        from b4dl_core.io import sha256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root/"config.json", {"fixture": True})
            hashes = {"config.json": sha256(root/"config.json")}
            artifact = hashlib.sha256(json.dumps(hashes, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            lock = {"components": {"model_base": {"path": str(root), "files_sha256": hashes, "artifact_sha256": artifact}}}
            b3.verify_baseline(lock)
            write_json(root/"config.json", {"fixture": "tampered"})
            with self.assertRaises(ValueError): b3.verify_baseline(lock)


if __name__ == "__main__":
    unittest.main()
