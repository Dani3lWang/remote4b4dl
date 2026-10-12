"""Evidence-boundary tests using raw nuScenes JSON schemas, no SDK/model imports."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/build_stage3_seg_candidates.py"
SPEC = importlib.util.spec_from_file_location("seg_candidates", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class CandidateEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.table_dir = self.root / "v1.0-trainval"
        self.table_dir.mkdir()
        self.scene = {"scene_id": "003833660", "scene_token": "scene", "num_frames": 3, "split": "train",
                      "paths": {"PATH_LIDAR_TOP": {f"PATH_{i:03}": f"samples/LIDAR_TOP/{i}.bin" for i in range(3)}}}
        self.sequence = {"scene_token": "scene", "sequence_id": "sequence", "split": "train", "indices": [0, 2],
                         "frames": [{"frame_id": f"random-id-{i}", "sample_token": f"sample-{i}",
                                     "TOKEN_LIDAR_TOP": f"lidar-{i}", "PATH_LIDAR_TOP": f"samples/LIDAR_TOP/{i}.bin"} for i in (0, 2)]}
        categories = [{"token": "cat-car", "name": "vehicle.car"}, {"token": "cat-truck", "name": "vehicle.truck"}]
        tables = {
            "scene": [{"token": "scene", "name": "scene-0001", "nbr_samples": 3,
                       "first_sample_token": "sample-0", "last_sample_token": "sample-2"}],
            "sample": [{"token": f"sample-{i}", "scene_token": "scene", "prev": f"sample-{i-1}" if i else "",
                        "next": f"sample-{i+1}" if i < 2 else ""} for i in range(3)],
            "sample_data": [{"token": f"lidar-{i}", "sample_token": f"sample-{i}", "is_key_frame": True,
                             "calibrated_sensor_token": "calib", "filename": f"samples/LIDAR_TOP/{i}.bin"} for i in range(3)],
            "sample_annotation": [{"token": f"ann-{i}", "sample_token": f"sample-{i}", "instance_token": "instance-car",
                                   "num_lidar_pts": 2, "translation": [1, 2, 0], "size": [2, 4, 1.5],
                                   "rotation": [1, 0, 0, 0]} for i in range(3)] +
                                  [{"token": "zero-truck", "sample_token": "sample-1", "instance_token": "instance-truck", "num_lidar_pts": 0}],
            "instance": [{"token": "instance-car", "category_token": "cat-car"}, {"token": "instance-truck", "category_token": "cat-truck"}],
            "category": categories,
            "sensor": [{"token": "sensor", "channel": "LIDAR_TOP", "modality": "lidar"}],
            "calibrated_sensor": [{"token": "calib", "sensor_token": "sensor"}],
        }
        for name, rows in tables.items():
            self.write(self.table_dir / f"{name}.json", rows)
        for i in range(3):
            path = self.root / f"samples/LIDAR_TOP/{i}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(struct.pack("10f", *range(10)))
        self.metadata = builder.Metadata([self.scene], [self.sequence])
        self.tables = builder.NuScenesTables(self.root, "v1.0-trainval")

    def write(self, path, rows):
        path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def record(self, question="Which object appears in frame 002?", answer="CAR.", **metadata):
        return {"row": {"scene_id": self.scene["scene_id"], "scene_token": "scene", "split": "train", **metadata},
                "question": question, "answer": answer, "ground_truth": answer,
                "source": {"path": "fixture.json", "sha256": "fixture", "row_index": 0, "answer_field": "answer"}, "reasons": []}

    def build(self, record=None, **options):
        return builder.build_record(record or self.record(), self.metadata, self.tables,
                                    options.pop("expected_split", "train"), {"scene"}, **options)

    def assert_reason(self, record, reason):
        audit, candidates, rejected = self.build(record)
        self.assertEqual(candidates, [])
        self.assertIn(reason, audit["reasons"])
        self.assertTrue(rejected)

    def test_class_and_frame_link_preserves_mask_boundary(self):
        audit, candidates, _ = self.build()
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate["frame"]["sample_token"], "sample-2")
        self.assertEqual(candidate["annotation_candidates"][0]["sample_annotation_token"], "ann-2")
        self.assertEqual(candidate["annotation_candidates"][0]["annotation_kind"], "3d_box")
        self.assertTrue(candidate["eligible_for_training"])
        for field in ("mask_verified", "grounding_verified", "supervision_ready"):
            self.assertFalse(candidate[field])
        self.assertEqual(candidate["mask_status"], "not_generated")

    def test_sparse_sequence_uses_indices_not_random_frame_id(self):
        audit, candidates, _ = self.build(self.record(question="Which object is present from frame 0 to frame 2?", sequence_id="sequence"))
        self.assertEqual(audit["frame_indices"], [0, 2])
        self.assertEqual([c["frame"]["sample_token"] for c in candidates], ["sample-0", "sample-2"])

    def test_question_is_not_used_to_invent_answer_class(self):
        self.assert_reason(self.record(question="Is a car in frame 0?", answer="Yes."), "no_supported_answer_category")

    def test_missing_frame_is_not_replaced_with_scene_or_zero(self):
        self.assert_reason(self.record(question="Describe the sequence.", answer="A car is present."), "missing_exact_frame_reference")

    def test_answer_frame_reference_is_explicitly_sourced(self):
        audit, candidates, _ = self.build(self.record(question="Where is the car?", answer="The car is ahead from frame 0 to frame 2."))
        self.assertEqual(audit["frame_source"], "answer_frame_reference")
        self.assertEqual(len(candidates), 3)

    def test_out_of_range_and_reversed_intervals_rejected(self):
        for question in ("What appears in frame 3?", "What appears from frame 2 to frame 0?", "What appears in frame -1?"):
            with self.subTest(question=question):
                self.assert_reason(self.record(question=question), "invalid_frame_reference")

    def test_negated_and_hypothetical_classes_are_not_positive(self):
        for answer in ("No cars are present.", "There isn't a car.", "A car might appear.", "If a car appears, stop.",
                       "A car is not visible.", "A car isn't visible.", "The car has not appeared."):
            with self.subTest(answer=answer):
                self.assert_reason(self.record(answer=answer), "only_negated_or_uncertain_mentions")

    def test_boundaries_plurals_and_narrow_aliases(self):
        mentions = builder.category_mentions("SCAR busyness, cargo and construction work. SUVs, trafficcones, construction vehicles and buses.")
        self.assertEqual({m["category"] for m in mentions}, {"car", "traffic_cone", "construction_vehicle", "bus"})

    def test_scene_id_and_token_conflict_rejected(self):
        other = {**self.scene, "scene_token": "other", "scene_id": "other-id"}
        self.metadata = builder.Metadata([self.scene, other], [self.sequence])
        self.assert_reason(self.record(scene_token="other"), "scene_id_token_mismatch")

    def test_metadata_vs_raw_frame_token_mismatch_rejected(self):
        self.metadata.frames["scene"][2]["sample_token"] = "wrong"
        self.assert_reason(self.record(), "metadata_nuscenes_frame_mismatch")

    def test_conflicting_overlapping_sequences_fail_closed(self):
        other = copy.deepcopy(self.sequence)
        other["sequence_id"] = "other-sequence"
        other["frames"][0]["sample_token"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Conflicting metadata"):
            builder.Metadata([self.scene], [self.sequence, other])

    def test_test_split_cannot_be_training_candidate(self):
        self.scene["split"] = "val"
        record = self.record(split="test")
        self.assert_reason(record, "excluded_split")
        audit, candidates, _ = self.build(record, expected_split="test")
        self.assertEqual(audit["split"], "test")
        self.assertFalse(candidates[0]["eligible_for_training"])

    def test_training_whitelist_required(self):
        audit, candidates, _ = builder.build_record(self.record(), self.metadata, self.tables, "train", set())
        self.assertEqual(candidates, [])
        self.assertIn("scene_not_in_training_whitelist", audit["reasons"])

    def test_annotation_class_absent_and_zero_point_class_rejected(self):
        self.assert_reason(self.record(answer="BUS."), "class_annotation_not_found")
        self.assert_reason(self.record(question="Which object appears in frame 1?", answer="TRUCK."), "class_has_no_lidar_points")

    def test_multiple_instances_remain_unresolved_candidates(self):
        self.tables.by_sample["sample-2"].append({**self.tables.by_sample["sample-2"][0], "token": "ann-second", "num_lidar_pts": 1})
        _, candidates, _ = self.build()
        self.assertTrue(candidates[0]["multiple_instance_candidates"])
        self.assertEqual(len(candidates[0]["annotation_candidates"]), 2)
        self.assertFalse(candidates[0]["grounding_verified"])

    def test_malformed_box_geometry_is_not_a_mask_source(self):
        self.tables.by_sample["sample-2"][0]["rotation"] = [0, 0, 0, 0]
        self.assert_reason(self.record(), "invalid_annotation_box_geometry")

    def test_qa_wheelchair_and_debris_are_not_pedestrians_or_cars(self):
        mentions = builder.category_mentions("Wheelchairs and debris.")
        self.assertEqual([m["category"] for m in mentions], ["wheelchair", "debris"])
        self.assertEqual(mentions[0]["nuscenes_categories"], ["human.pedestrian.wheelchair"])

    def test_missing_and_unsafe_lidar_paths_rejected(self):
        sd = self.tables.lidar["sample-2"]
        sd["filename"] = "../escape.bin"
        self.metadata.frames["scene"][2]["lidar_filename"] = "../escape.bin"
        self.assert_reason(self.record(), "unsafe_data_path")
        for filename in ("C:\\outside.bin", "/outside.bin", "\\\\server\\share\\file.bin"):
            with self.assertRaises(ValueError):
                builder.safe_file(self.root, filename)

    def test_missing_lidar_file_rejected(self):
        (self.root / "samples/LIDAR_TOP/2.bin").unlink()
        self.assert_reason(self.record(), "missing_lidar_file")

    def test_point_label_path_does_not_verify_contents_or_mask(self):
        path = self.root / "lidarseg.bin"
        path.write_bytes(bytes([9, 9]))
        self.tables.tables["lidarseg"] = {"lidar-2": {"token": "lidar-2", "filename": "lidarseg.bin"}}
        _, candidates, _ = self.build(require_point_labels=True)
        label = candidates[0]["frame"]["point_label_sources"][0]
        self.assertTrue(label["file_exists"])
        self.assertFalse(label["contents_verified"])
        self.assertFalse(candidates[0]["mask_verified"])

    def test_point_labels_can_be_required_separately(self):
        self.assert_reason_when_point_labels_required()

    def assert_reason_when_point_labels_required(self):
        audit, candidates, _ = self.build(require_point_labels=True)
        self.assertEqual(candidates, [])
        self.assertIn("missing_point_label_file", audit["reasons"])

    def test_missing_tables_produce_audit_rejection(self):
        self.tables = builder.NuScenesTables(self.root / "absent", "v1.0-trainval")
        self.assert_reason(self.record(), "nuscenes_tables_unavailable")

    def test_legacy_predictions_need_unique_question_and_gt_match(self):
        qa = {**self.record()["row"], "question": "Which object appears in frame 002?", "answer": "CAR."}
        original = self.write(self.root / "existence.json", [qa, qa])
        predictions = self.write(self.root / "predictions.json", {"existence": {"questions": [qa["question"]],
                                  "ground_truths": ["CAR."], "predictions": ["A truck."]}})
        record = list(builder.input_rows(predictions, [original], "prediction"))[0]
        self.assertEqual(record["qa_match_count"], 2)
        self.assertIn("ambiguous_original_qa", record["reasons"])
        self.assertNotIn("linked_qa_source", record)
        self.write(original, [qa])
        record = list(builder.input_rows(predictions, [original], "prediction"))[0]
        self.assertEqual(record["answer"], "A truck.")
        self.assertEqual(record["ground_truth"], "CAR.")
        self.assertEqual(record["linked_qa_source"]["row_index"], 0)
        self.assertEqual(record["source"]["answer_field"], "predictions")
        self.write(original, [{**qa, "answer": "BUS."}])
        record = list(builder.input_rows(predictions, [original], "prediction"))[0]
        self.assertIn("original_qa_not_found", record["reasons"])

    def test_prediction_array_length_mismatch_fails(self):
        path = self.write(self.root / "bad.json", {"existence": {"questions": ["Q"], "ground_truths": [], "predictions": ["CAR"]}})
        with self.assertRaisesRegex(ValueError, "inconsistent lengths"):
            list(builder.input_rows(path, [], "prediction"))

    def test_conversations_source_and_multiturn_rejection(self):
        conversations = [{"from": "human", "value": "<video>\nQ <meta> cars"}, {"from": "gpt", "value": "CAR"}]
        path = self.write(self.root / "conversation.json", [{"conversations": conversations}])
        record = list(builder.input_rows(path, [], "answer"))[0]
        self.assertEqual(record["question"], "Q")
        self.assertEqual(record["source"]["answer_field"], "conversations[1].value")
        conversations += [{"from": "human", "value": "Q2"}, {"from": "gpt", "value": "BUS"}]
        self.write(path, [{"conversations": conversations}])
        self.assertIn("invalid_qa_record", list(builder.input_rows(path, [], "answer"))[0]["reasons"])

    def test_cli_run_writes_complete_traceable_jsonl(self):
        record = self.record()
        input_path = self.write(self.root / "qa.json", [{**record["row"], "question": record["question"], "answer": record["answer"]}])
        scene_path = self.write(self.root / "scenes.json", [self.scene])
        sequence_path = self.write(self.root / "sequences.json", [self.sequence])
        whitelist = self.write(self.root / "train.json", ["scene"])
        args = argparse.Namespace(input=input_path, original_qa=[], answer_source="answer", scene_metadata=scene_path,
                                  sequence_metadata=sequence_path, nuscenes_root=self.root, nuscenes_version="v1.0-trainval",
                                  train_scene_tokens=whitelist, expected_split="train", require_point_labels=False,
                                  output_dir=self.root / "output")
        summary = builder.run(args)
        self.assertEqual(summary["counts"]["candidate_rows"], 1)
        candidate = json.loads((args.output_dir / "candidates.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(candidate["source"]["sha256"], builder.file_info(input_path)["sha256"])
        self.assertEqual((args.output_dir / "rejected.jsonl").read_text(), "")
        self.assertEqual(len(list(args.output_dir.glob("*.part"))), 0)
        with self.assertRaisesRegex(ValueError, "already exist"):
            builder.run(args)


if __name__ == "__main__":
    unittest.main()
