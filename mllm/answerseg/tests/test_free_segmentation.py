"""Inline SEG learning and bounded free-generation checks."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from answerseg.language import make_prompt, mark_answer, training_sequence
from answerseg.runner import prediction_metrics, score_prediction, train
from test_answerseg import example_fixture, model_fixture
from torch.nn import functional as F


class FreeSegmentationTests(unittest.TestCase):
    def test_same_segmentation_instruction_for_training_and_prediction_only(self):
        ordinary = make_prompt("Which car?")
        segmentation = make_prompt("Which car?", segmentation=True)
        self.assertNotIn("<SEG>", ordinary)
        self.assertIn("<SEG>", segmentation)
        self.assertEqual(segmentation.count("<video>"), 1)
        with self.assertRaisesRegex(ValueError, "reserved"):
            make_prompt("Which car? <SEG>", segmentation=True)

    def test_marker_loss_uses_causal_preceding_logits_after_visual_expansion(self):
        model = model_fixture("joint")
        example = example_fixture()
        marked, _ = mark_answer(example["answer"], example["targets"])
        ids, labels = training_sequence(
            model.bridge.tokenizer, example["prompt"], marked
        )
        captured = {}

        def capture(module, args, kwargs, result):
            captured["result"] = result
            captured["labels"] = kwargs["labels"]

        hook = model.language.register_forward_hook(capture, with_kwargs=True)
        try:
            loss, _, _ = model.bridge.forward(
                ids, labels, example["features"], marker_loss_weight=8
            )
        finally:
            hook.remove()
        logits, target = captured["result"].logits[:, :-1], captured["labels"][:, 1:]
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape_as(target)
        weights = target.ne(-100).float()
        weights[target.eq(model.tokens.seg)] = 8
        torch.testing.assert_close(loss, (per_token * weights).sum() / weights.sum())
        loss.backward()
        self.assertGreater(
            float(model.language.get_output_embeddings().new_rows.grad.abs().sum()), 0
        )
        self.assertIsNone(model.language.get_output_embeddings().base.weight.grad)

    def test_emission_and_recall_retain_missing_token_failures(self):
        totals = Counter()
        prediction = {
            "status": "missing_seg_token",
            "masks": [],
            "objects": [],
            "marker_counts": {"seg": 0, "noobj": 0},
        }
        score_prediction(prediction, example_fixture(), totals)
        metrics = prediction_metrics(totals, Counter(missing_seg_token=1))
        self.assertEqual(metrics["seg_output_rate"], 0)
        self.assertEqual(metrics["recall_at_0_5"], 0)
        self.assertEqual(metrics["false_negatives"], 1)

    def test_step_budget_saves_partial_epoch_and_runs_unmocked_generation(self):
        model = model_fixture("joint")
        examples = [dict(example_fixture(), request_id=f"q{i}") for i in range(3)]
        options = {
            "stage": "joint",
            "tune_spatial": False,
            "minimum_iou": 0.5,
            "rank": 2,
            "alpha": 4,
        }
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                epochs=10,
                max_steps=1,
                free_dev_samples=1,
                max_new_tokens=2,
                marker_loss_weight=8,
                seed=2026,
                device="cpu",
                stage="joint",
                tune_spatial=False,
                minimum_iou=0.5,
                lora_rank=2,
                lora_alpha=4,
                init_bundle=None,
                lr=1e-3,
                output_dir=Path(directory),
            )
            with (
                patch(
                    "answerseg.runner.preflight", return_value=(examples, examples, {})
                ),
                patch("answerseg.runner.source_identity", return_value={}),
                patch("answerseg.runner.build_model", return_value=(model, options)),
                patch.object(
                    model.language, "generate", wraps=model.language.generate
                ) as generate,
            ):
                result = train(
                    args, {"spatial_checkpoint": "same"}, {"spatial_checkpoint": "same"}
                )
            self.assertEqual(result["global_step"], 1)
            self.assertEqual(result["completed_epoch"], 0)
            self.assertTrue(result["budget_exhausted"])
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(
                result["free_generation_development"]["positive_targets"], 1
            )
            self.assertTrue((Path(directory) / "step-1/bundle.json").is_file())


if __name__ == "__main__":
    unittest.main()
