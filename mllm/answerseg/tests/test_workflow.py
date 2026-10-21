"""Data provenance, multiple referents and training-entry integration checks."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from answerseg import SCHEMA
from answerseg.data import digest, prepare_records, verify_feature_frames
from answerseg.language import make_prompt, mark_answer, training_sequence
from answerseg.runner import train
from test_answerseg import example_fixture, model_fixture, request_fixture


class WorkflowTests(unittest.TestCase):
    def test_multiple_objects_and_noobj_keep_order_after_visual_expansion(self):
        model = model_fixture()
        answer = "truck and car; no target"
        targets = [
            {"span": [15, len(answer)], "status": "no_target", "instance_token": None},
            {"span": [0, 5], "status": "matched", "instance_token": "a"},
            {"span": [10, 13], "status": "matched", "instance_token": "b"},
        ]
        marked, ordered = mark_answer(answer, targets)
        ids, labels = training_sequence(
            model.bridge.tokenizer, make_prompt("truck"), marked
        )
        _, states, kinds = model.bridge.forward(ids, labels, torch.randn(4, 768))
        self.assertEqual(
            [target["instance_token"] for target in ordered], ["a", "b", None]
        )
        self.assertEqual(
            kinds.tolist(), [model.tokens.seg, model.tokens.seg, model.tokens.noobj]
        )
        self.assertEqual(states.shape, (3, 32))

    def test_feature_context_cannot_leak_development_scene_or_camera(self):
        request = request_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "v1.0-mini"
            root.mkdir()
            tables = {
                "sample_data": [
                    {
                        "token": "frame1",
                        "sample_token": "s1",
                        "calibrated_sensor_token": "c1",
                    }
                ],
                "sample": [{"token": "s1", "scene_token": "scene1"}],
                "calibrated_sensor": [{"token": "c1", "sensor_token": "lidar"}],
                "sensor": [{"token": "lidar", "channel": "LIDAR_TOP"}],
            }
            for name, rows in tables.items():
                (root / (name + ".json")).write_text(json.dumps(rows), encoding="utf-8")
            self.assertTrue(
                verify_feature_frames([request], directory, "v1.0-mini")[
                    "same_scene_lidar_context"
                ]
            )
            (root / "sample.json").write_text(
                json.dumps([{"token": "s1", "scene_token": "dev-scene"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "another scene"):
                verify_feature_frames([request], directory, "v1.0-mini")
            (root / "sample.json").write_text(
                json.dumps(tables["sample"]), encoding="utf-8"
            )
            (root / "sensor.json").write_text(
                json.dumps([{"token": "lidar", "channel": "CAM_FRONT"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "LIDAR_TOP"):
                verify_feature_frames([request], directory, "v1.0-mini")

    def test_legacy_bindings_group_spans_and_preserve_original_answer(self):
        answer = "truck and car"
        sources = []
        for index, (span, token) in enumerate((([0, 5], "t1"), ([10, 13], "t2"))):
            candidate_id = f"candidate{index}"
            review = {
                "candidate_id": candidate_id,
                "candidate_fingerprint": str(index) * 64,
                "decision": "accept",
                "reviewer": "a",
                "reviewed_at": "today",
                "evidence": "checked",
                "target_span": span,
                "instance_token": token,
            }
            sources.append(
                {
                    "schema_version": "b4dl-answer-binding-v1",
                    "candidate_id": candidate_id,
                    "candidate_fingerprint": review["candidate_fingerprint"],
                    "grounding_verified": True,
                    "review_status": "accepted",
                    "review": review,
                    "target_span": span,
                    "selected_instance_token": token,
                    "target_mask_row": index,
                    "objects": [{"instance_token": token, "mask_row": index}],
                    "answer": answer,
                    "target_phrase": answer[span[0] : span[1]],
                    "answer_source": "dataset_ground_truth",
                    "question": "Which objects?",
                    "scene_token": "scene1",
                    "sample_data_token": "frame1",
                    "split": "train",
                    "point_file": {"path": "a", "sha256": "a" * 64},
                    "mask_file": {"path": "b", "sha256": "b" * 64},
                }
            )
        index = [
            {
                "scene_token": "scene1",
                "sample_data_token": "frame1",
                "features": request_fixture()["features"],
            }
        ]
        records = prepare_records(sources, index)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["answer"], answer)
        self.assertEqual(len(records[0]["targets"]), 2)

    def test_generated_supervision_cannot_switch_feature_files(self):
        source = dict(
            request_fixture(),
            schema_version=SCHEMA + "-generated",
            answer="truck",
            answer_model_id="b3-hash",
            finished=True,
        )
        source["record_fingerprint"] = digest(source)
        review = {
            "request_id": "q1",
            "record_fingerprint": source["record_fingerprint"],
            "decision": "accept",
            "reviewer": "a",
            "reviewed_at": "today",
            "evidence": "checked",
            "targets": [{"span": [0, 5], "status": "matched", "instance_token": "t1"}],
        }
        features = copy.deepcopy(source["features"])
        features["sha256"] = "b" * 64
        index = [
            {
                "scene_token": "scene1",
                "sample_data_token": "frame1",
                "features": features,
            }
        ]
        with self.assertRaisesRegex(ValueError, "features differ"):
            prepare_records([source], index, [review])

    def test_training_entry_runs_optimizer_development_and_bundle_publication(self):
        example = dict(example_fixture(), request_id="q1")
        model = model_fixture("joint")
        options = {
            "stage": "joint",
            "tune_spatial": False,
            "minimum_iou": 0.5,
            "rank": 2,
            "alpha": 4,
        }
        report = {
            "train": {"manifest_sha256": "a" * 64},
            "internal_dev": {"manifest_sha256": "b" * 64},
        }
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                epochs=1,
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
                    "answerseg.runner.preflight",
                    return_value=([example], [example], report),
                ),
                patch("answerseg.runner.source_identity", return_value={}),
                patch("answerseg.runner.build_model", return_value=(model, options)),
            ):
                result = train(
                    args,
                    {"spatial_checkpoint": "same.pt"},
                    {"spatial_checkpoint": "same.pt"},
                )
            self.assertEqual(result["global_step"], 1)
            self.assertFalse(result["model_performance_accepted"])
            self.assertTrue((Path(directory) / "epoch-1/bundle.json").is_file())
            log = json.loads(
                (Path(directory) / "training.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(log["covered_targets"], 1)
            self.assertTrue(log["optimizer_updated"])
