"""Geometric/identity refusal tests; fixtures are synthetic, never nuScenes GT."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools/build_lidar_box_pseudo_masks.py"
SPEC = importlib.util.spec_from_file_location("box_pseudo_masks", SCRIPT)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)
cb = tool.candidate_builder


class BoxPseudoMaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.root = self.work / "nuscenes"
        self.table_dir = self.root / "v1.0-mini"
        self.table_dir.mkdir(parents=True)
        self.filename = "samples/LIDAR_TOP/scan.pcd.bin"
        self.point_path = self.root / self.filename
        self.point_path.parent.mkdir(parents=True)
        xyz = [[2, 1, 0], [3.5, 1, 0], [4, 1, 0], [4.1, 1, 0],
               [2, 2, 1], [2, 2.01, 0], [-.01, 1, 0], [0, 1, 0]]
        self.points = np.column_stack([xyz, np.arange(8) + 10, np.arange(8)]).astype("<f4")
        self.point_path.write_bytes(self.points.tobytes())
        h = math.sqrt(.5)
        # R_sensor = Rz(90deg); R_ego = Rx(90deg). Their product has
        # quaternion [.5,.5,-.5,.5]. A yaw-only transform cannot pass this fixture.
        tables = {
            "scene": [{"token": "scene", "name": "scene-0001", "nbr_samples": 1,
                       "first_sample_token": "sample", "last_sample_token": "sample"}],
            "sample": [{"token": "sample", "scene_token": "scene", "prev": "", "next": "", "timestamp": 100}],
            "sample_data": [{"token": "lidar", "sample_token": "sample", "is_key_frame": True,
                             "calibrated_sensor_token": "calib", "ego_pose_token": "pose", "timestamp": 100,
                             "filename": self.filename, "fileformat": "bin"}],
            "sensor": [{"token": "sensor", "channel": "LIDAR_TOP", "modality": "lidar"}],
            "calibrated_sensor": [{"token": "calib", "sensor_token": "sensor", "translation": [1, 2, 3],
                                   "rotation": [h, 0, 0, h]}],
            "ego_pose": [{"token": "pose", "translation": [10, 20, 30], "rotation": [h, h, 0, 0], "timestamp": 100}],
            "sample_annotation": [
                {"token": "ann-car", "sample_token": "sample", "instance_token": "car", "num_lidar_pts": 5,
                 "translation": [10, 17, 34], "size": [2, 4, 2], "rotation": [.5, .5, -.5, .5],
                 "visibility_token": "4", "attribute_tokens": ["moving"]},
                {"token": "ann-truck", "sample_token": "sample", "instance_token": "truck", "num_lidar_pts": 3,
                 "translation": [10, 17, 37], "size": [2, 4, 2], "rotation": [.5, .5, -.5, .5]},
            ],
            "instance": [{"token": "car", "category_token": "cat-car"},
                         {"token": "truck", "category_token": "cat-truck"}],
            "category": [{"token": "cat-car", "name": "vehicle.car"},
                         {"token": "cat-truck", "name": "vehicle.truck"}],
        }
        for name, rows in tables.items():
            self.write(self.table_dir / (name + ".json"), rows)
        self.scene = {"scene_id": "0001", "scene_token": "scene", "num_frames": 1, "split": "train",
                      "paths": {"PATH_LIDAR_TOP": {"PATH_000": self.filename}}}
        self.candidate_path = self.work / "candidates.jsonl"
        self.candidate = self.make_candidate()
        self.write_candidates([self.candidate])
        self.manifest_path = self.work / "points.json"
        self.manifest = self.make_manifest()
        self.write(self.manifest_path, self.manifest)

    def write(self, path, data):
        path.write_text(json.dumps(data), encoding="utf-8")

    def update_table(self, name, mutate):
        path = self.table_dir / (name + ".json")
        data = cb.load_json(path)
        mutate(data)
        self.write(path, data)

    def make_candidate(self):
        record = {"row": {"scene_id": "0001", "scene_token": "scene", "split": "train"},
                  "question": "Which object appears in frame 0?", "answer": "The car is moving.",
                  "ground_truth": "The car is moving.",
                  "source": {"path": "synthetic-qa.json", "sha256": "synthetic", "row_index": 0}, "reasons": []}
        _, candidates, rejected = cb.build_record(record, cb.Metadata([self.scene], []),
                                                  cb.NuScenesTables(self.root, "v1.0-mini"), "train", {"scene"})
        self.assertEqual(rejected, [])
        return candidates[0]

    def make_manifest(self):
        return {"schema_version": tool.POINT_SCHEMA,
                "provenance": {"source": "synthetic test fixture, not real nuScenes", "original_files": True},
                "metadata_sha256": {name: cb.file_info(self.table_dir / (name + ".json"))["sha256"] for name in tool.REQUIRED},
                "points": [{"sample_data_token": "lidar", "sample_token": "sample", "filename": self.filename,
                            "sha256": cb.file_info(self.point_path)["sha256"], "point_count": len(self.points),
                            "layout": tool.LAYOUT, "coordinate_frame": "lidar_sensor", "point_order": "original_file_rows"}]}

    def refresh_manifest(self):
        self.manifest = self.make_manifest()
        self.write(self.manifest_path, self.manifest)

    def write_candidates(self, rows):
        self.candidate_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def args(self, **changes):
        args = dict(candidates=self.candidate_path, nuscenes_root=self.root, nuscenes_version="v1.0-mini",
                    point_manifest=self.manifest_path, output_dir=self.work / "output")
        args.update(changes)
        return argparse.Namespace(**args)

    def reject(self, reason):
        summary = tool.run(self.args())
        self.assertEqual(summary["counts"]["generated_candidates"], 0)
        self.assertIn(reason, summary["rejection_reason_candidates"])
        self.assertEqual(list((self.work / "output/masks").iterdir()), [])
        rows = (self.work / "output/rejected.jsonl").read_text().splitlines()
        for raw in rows:
            row = json.loads(raw)
            for key in tool.FLAGS:
                self.assertIs(row[key], False)
            self.assertEqual(row["mask_status"], "not_generated")

    def test_full_transform_order_width_length_boundary_overlap_and_provenance(self):
        original = self.point_path.read_bytes()
        summary = tool.run(self.args())
        self.assertEqual(summary["counts"]["generated_instance_pseudo_masks"], 1)
        row = json.loads((self.work / "output/pseudo_masks.jsonl").read_text())
        self.assertEqual(row["candidate"], self.candidate)
        self.assertEqual(row["instances"][0]["instance_token"], "car")
        self.assertEqual(row["instances"][0]["inside_box_point_count"], 5)
        self.assertEqual(row["instances"][0]["ignored_overlap_point_count"], 2)
        self.assertTrue(row["instances"][0]["count_matches_annotation"])
        self.assertEqual(row["frame_evidence"]["overlap_annotation_tokens"], ["ann-car", "ann-truck"])
        path = Path(row["mask_file"]["path"])
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["labels"], [[1, -1, -1, 0, 1, 0, 0, 1]])
            np.testing.assert_array_equal(data["box_membership"], [[1, 1, 1, 0, 1, 0, 0, 1]])
            np.testing.assert_array_equal(data["point_indices"], np.arange(8))
            self.assertEqual(data["point_file_sha256"].item(), hashlib.sha256(original).hexdigest())
            for key in tool.FLAGS:
                self.assertFalse(data[key].item())
        self.assertEqual(self.point_path.read_bytes(), original)
        for obj in (summary, row, row["instances"][0]):
            for key in tool.FLAGS:
                self.assertIs(obj[key], False)
        self.assertIs(row["eligible_for_training"], False)

    def test_inverse_box_transform_has_independent_known_answer(self):
        # Global x aligns with local y after Rz(90): length extends along global y.
        box = {"translation": [10, 20, 0], "size": [2, 6, 2], "rotation": [math.sqrt(.5), 0, 0, math.sqrt(.5)]}
        points = np.array([[10, 22.9, 0], [10.9, 20, 0], [11.1, 20, 0], [10, 23.1, 0]])
        np.testing.assert_array_equal(tool.points_in_box(points, box), [True, True, False, False])

    def test_same_class_candidates_remain_separate_instances(self):
        self.update_table("instance", lambda rows: rows[1].update(category_token="cat-car"))
        self.candidate = self.make_candidate()
        self.write_candidates([self.candidate])
        self.refresh_manifest()
        summary = tool.run(self.args())
        self.assertEqual(summary["counts"]["generated_instance_pseudo_masks"], 2)
        row = json.loads((self.work / "output/pseudo_masks.jsonl").read_text())
        self.assertTrue(row["multiple_instance_candidates"])
        with np.load(row["mask_file"]["path"], allow_pickle=False) as data:
            self.assertEqual(data["labels"].shape, (2, 8))
            self.assertEqual(data["instance_tokens"].tolist(), ["car", "truck"])
            np.testing.assert_array_equal(data["labels"][1], [0, -1, -1, 1, 0, 0, 0, 0])

    def test_reordered_points_with_same_size_are_rejected_by_hash(self):
        self.point_path.write_bytes(self.points[::-1].tobytes())
        self.reject("point_sha256_mismatch")

    def test_missing_manifest_refuses_even_valid_looking_points(self):
        summary = tool.run(self.args(point_manifest=None))
        self.assertEqual(summary["rejection_reason_candidates"], {"point_provenance_manifest_missing": 1})
        self.assertEqual(summary["counts"]["generated_candidates"], 0)

    def test_missing_manifest_entry_does_not_guess_from_filename(self):
        self.manifest["points"] = []
        self.write(self.manifest_path, self.manifest)
        self.reject("point_manifest_entry_missing")

    def test_duplicate_manifest_entries_refuse(self):
        self.manifest["points"].append(copy.deepcopy(self.manifest["points"][0]))
        self.write(self.manifest_path, self.manifest)
        self.reject("Missing/duplicate sample_data_token in point manifest: lidar")

    def test_unknown_original_provenance_refuses(self):
        self.manifest["provenance"]["original_files"] = False
        self.write(self.manifest_path, self.manifest)
        self.reject("original_point_provenance_missing")

    def test_manifest_layout_coordinate_and_order_are_required(self):
        cases = [("layout", "float32_xyz", "unproven_point_layout"),
                 ("coordinate_frame", "global", "unproven_point_coordinate_frame"),
                 ("point_order", "voxelized", "unproven_point_order")]
        for index, (field, value, reason) in enumerate(cases):
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest)
                manifest["points"][0][field] = value
                self.write(self.manifest_path, manifest)
                summary = tool.run(self.args(output_dir=self.work / f"out-{index}"))
                self.assertEqual(summary["rejection_reason_candidates"], {reason: 1})

    def test_manifest_must_bind_current_metadata_snapshot(self):
        self.update_table("calibrated_sensor", lambda rows: rows[0].update(translation=[0, 0, 0]))
        self.reject("point_manifest_metadata_mismatch")

    def test_missing_pose_refuses_without_identity_fallback(self):
        (self.table_dir / "ego_pose.json").unlink()
        self.reject("nuscenes_tables_unavailable")

    def test_missing_transform_vector_refuses(self):
        self.update_table("calibrated_sensor", lambda rows: rows[0].pop("rotation"))
        self.refresh_manifest()
        self.reject("invalid_quaternion")

    def test_non_unit_quaternion_refuses(self):
        self.update_table("ego_pose", lambda rows: rows[0].update(rotation=[2, 0, 0, 0]))
        self.refresh_manifest()
        self.reject("invalid_quaternion_norm")

    def test_non_keyframe_refuses_without_box_interpolation(self):
        self.update_table("sample_data", lambda rows: rows[0].update(is_key_frame=False))
        self.refresh_manifest()
        self.reject("not_corresponding_lidar_keyframe")

    def test_non_lidar_sensor_refuses(self):
        self.update_table("sensor", lambda rows: rows[0].update(channel="CAM_FRONT", modality="camera"))
        self.refresh_manifest()
        self.reject("not_lidar_top_keyframe")

    def test_modified_box_and_instance_are_rejected(self):
        for index, (field, value, reason) in enumerate([
                ("box_global", {"translation": [0, 0, 0]}, "candidate_box_geometry_mismatch"),
                ("instance_token", "truck", "candidate_instance_category_mismatch")]):
            candidate = copy.deepcopy(self.candidate)
            candidate["annotation_candidates"][0][field] = value
            self.write_candidates([candidate])
            summary = tool.run(self.args(output_dir=self.work / f"out-{index}"))
            self.assertIn(reason, summary["rejection_reason_candidates"])

    def test_frame_sample_and_filename_mismatches_refuse(self):
        for index, (field, value, reason) in enumerate([
                ("frame_index", 1, "scene_frame_sample_mismatch"),
                ("sample_token", "unknown", "unknown_sample_or_lidar_token"),
                ("lidar_filename", "samples/LIDAR_TOP/other.bin", "candidate_lidar_filename_mismatch")]):
            candidate = copy.deepcopy(self.candidate)
            candidate["frame"][field] = value
            self.write_candidates([candidate])
            summary = tool.run(self.args(output_dir=self.work / f"out-{index}"))
            self.assertIn(reason, summary["rejection_reason_candidates"])

    def test_nonfinite_points_refuse_without_filtering_rows(self):
        self.points[0, 0] = np.nan
        self.point_path.write_bytes(self.points.tobytes())
        self.refresh_manifest()
        self.reject("nonfinite_point_values")

    def test_bad_ring_refuses_instead_of_guessing_layout(self):
        self.points[0, 4] = .5
        self.point_path.write_bytes(self.points.tobytes())
        self.refresh_manifest()
        self.reject("invalid_hdl32_ring_layout")

    def test_point_count_and_partial_row_refuse(self):
        self.candidate["frame"]["lidar_point_count"] = 99
        self.write_candidates([self.candidate])
        self.reject("point_count_mismatch")
        self.point_path.write_bytes(self.point_path.read_bytes()[:-1])
        self.refresh_manifest()
        summary = tool.run(self.args(output_dir=self.work / "truncated"))
        self.assertIn("invalid_lidar_file_size", summary["rejection_reason_candidates"])

    def test_invalid_other_category_box_refuses_ambiguous_overlap(self):
        self.update_table("sample_annotation", lambda rows: rows[1].update(size=[0, 4, 2]))
        self.refresh_manifest()
        self.reject("invalid_box_size")

    def test_empty_box_refuses(self):
        self.points[:, :3] += 100
        self.point_path.write_bytes(self.points.tobytes())
        self.refresh_manifest()
        self.reject("empty_box_pseudo_mask")

    def test_duplicate_id_is_audited_and_does_not_overwrite(self):
        self.write_candidates([self.candidate, self.candidate])
        summary = tool.run(self.args())
        self.assertEqual(summary["counts"]["generated_candidates"], 1)
        self.assertEqual(summary["rejection_reason_candidates"], {"duplicate_candidate_id": 1})
        self.assertEqual(len(list((self.work / "output/masks").glob("*.npz"))), 1)

    def test_malformed_json_and_nonfinite_constants_are_audited(self):
        self.candidate_path.write_text('{broken}\n{"x": NaN}\n[]\n', encoding="utf-8")
        summary = tool.run(self.args())
        self.assertEqual(summary["counts"]["rejected_candidates"], 3)
        self.assertIn("nonfinite_json_constant", summary["rejection_reason_candidates"])
        self.assertIn("candidate_record_not_object", summary["rejection_reason_candidates"])

    def test_empty_input_is_explicit_zero_output_audit(self):
        self.write_candidates([])
        summary = tool.run(self.args())
        self.assertEqual(summary["counts"]["input_candidates"], 0)
        self.assertEqual(summary["counts"]["generated_candidates"], 0)
        self.assertEqual(summary["mask_status"], "not_generated")

    def test_output_failure_never_publishes_completed_audit(self):
        with patch.object(tool.np, "savez_compressed", side_effect=OSError("simulated disk write failure")):
            with self.assertRaises(OSError):
                tool.run(self.args())
        self.assertFalse((self.work / "output/summary.json").exists())
        self.assertFalse((self.work / "output/pseudo_masks.jsonl").exists())
        self.assertTrue((self.work / "output/audit.jsonl.part").exists())

    def test_unsafe_point_path_refuses(self):
        filename = "../scan.bin"
        self.update_table("sample_data", lambda rows: rows[0].update(filename=filename))
        self.candidate["frame"]["lidar_filename"] = filename
        self.write_candidates([self.candidate])
        self.refresh_manifest()
        self.reject("unsafe_data_path")

    def test_output_inside_dataset_or_existing_directory_refuses(self):
        for output in (self.root / "outputs", self.work):
            with self.subTest(output=output), self.assertRaises(tool.Refusal):
                tool.run(self.args(output_dir=output))
        self.assertFalse((self.root / "outputs").exists())

    def test_cli_refusal_has_nonzero_exit_and_completed_audit(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--candidates", str(self.candidate_path),
                                 "--nuscenes-root", str(self.root), "--nuscenes-version", "v1.0-mini",
                                 "--output-dir", str(self.work / "output")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertTrue((self.work / "output/summary.json").is_file())
        self.assertTrue((self.work / "output/rejected.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
