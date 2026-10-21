"""Geometry provenance, causal token learning and bounded free-generation checks."""

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from answerseg.bootstrap import geometry_records, verify_geometry_record
from answerseg.data import AnswerSegDataset, digest
from answerseg.language import make_prompt, mark_answer, training_sequence
from answerseg.runner import prediction_metrics, score_prediction, train
from test_answerseg import example_fixture, model_fixture, request_fixture
from torch.nn import functional as F


def geometry_fixture():
    frame = {
        "points": torch.tensor(
            [[1.0, 0, 0, 0], [1.0, 1, 0, 0], [5.0, 0, 0, 0], [5.0, 1, 0, 0]]
        ),
        "labels": torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]]),
        "class_ids": torch.tensor([0, 0]),
        "instance_tokens": ["near", "far"],
        "sample_data_token": "frame1",
    }
    clip = {
        "scene_token": "scene1",
        "split": "train",
        "label_kind": "panoptic_gt",
        "frames": [
            {
                "sample_data_token": "frame1",
                "mask_verified": True,
                "point_file": {"path": "a", "sha256": "a" * 64},
                "mask_file": {"path": "b", "sha256": "b" * 64},
                "source_panoptic_file": {"path": "c", "sha256": "c" * 64},
            }
        ],
    }
    config = SimpleNamespace(
        point_cloud_range=(-10, -10, -3, 10, 10, 3), class_names=("car",)
    )
    return frame, clip, config


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

    def test_geometry_requires_multiple_instances_and_clear_distance_margin(self):
        frame, clip, config = geometry_fixture()
        rows = geometry_records(
            frame, clip, request_fixture()["features"], config, min_points=2
        )
        self.assertEqual(
            [r["targets"][0]["instance_token"] for r in rows], ["near", "far"]
        )
        self.assertTrue(all(r["grounding_verified"] is False for r in rows))
        self.assertTrue(all(r["review_status"] == "not_human_reviewed" for r in rows))
        frame["labels"] = frame["labels"][:1]
        frame["class_ids"] = frame["class_ids"][:1]
        self.assertEqual(geometry_records(frame, clip, {}, config, min_points=2), [])

    def test_geometry_ties_and_unverified_masks_never_enter_training(self):
        frame, clip, config = geometry_fixture()
        frame["points"][2:, :2] = frame["points"][:2, :2]
        self.assertEqual(geometry_records(frame, clip, {}, config, min_points=2), [])
        clip["frames"][0]["mask_verified"] = False
        with self.assertRaisesRegex(ValueError, "verified panoptic"):
            geometry_records(frame, clip, {}, config, min_points=2)

    def test_geometry_target_and_hash_cannot_be_rewritten_without_recomputation(self):
        frame, clip, config = geometry_fixture()
        row = geometry_records(
            frame, clip, request_fixture()["features"], config, min_points=2
        )[0]
        verify_geometry_record(row, frame, clip, config)
        for field in ("targets", "answer", "grounding_verified"):
            tampered = copy.deepcopy(row)
            tampered[field] = {
                "targets": [
                    {"span": [4, 15], "status": "matched", "instance_token": "far"}
                ],
                "answer": "The farthest car.",
                "grounding_verified": True,
            }[field]
            tampered["source_fingerprint"] = digest(
                {k: v for k, v in tampered.items() if k != "source_fingerprint"}
            )
            with self.assertRaisesRegex(ValueError, "panoptic rule"):
                verify_geometry_record(tampered, frame, clip, config)

    def test_dataset_reports_geometry_separately_and_rechecks_labels(self):
        frame, clip, config = geometry_fixture()
        row = geometry_records(
            frame, clip, request_fixture()["features"], config, min_points=2
        )[0]

        class Spatial:
            def __init__(self):
                self.records = [clip]

            def __getitem__(self, index):
                return {"frames": [frame]}

        spatial = Spatial()
        spatial.config = config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = AnswerSegDataset(path, spatial, directory)
            with patch(
                "answerseg.data.load_features", return_value=torch.zeros(1, 768)
            ):
                report = dataset.preflight()
                self.assertEqual(
                    report["supervision_sources"], {"geometry_bootstrap": 1}
                )
                self.assertIn("<SEG>", dataset[0]["prompt"])
                frame["points"][:2, 0] = 8
                with self.assertRaisesRegex(ValueError, "panoptic rule"):
                    dataset[0]

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
        self.assertEqual(metrics["seg_emission_rate"], 0)
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
