import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MLLM_ROOT = Path(__file__).resolve().parents[1]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))

from scripts.preflight_reasonseg import validate_manifest


SPLIT_HELPERS = runpy.run_path(
    str(MLLM_ROOT / "reasonseg_splits.py")
)
partition_development_scenes = SPLIT_HELPERS["partition_development_scenes"]


class ReasonSegPreflightTests(unittest.TestCase):
    def test_internal_validation_uses_only_official_train_scenes(self):
        scenes = [
            {"name": f"scene-{index}", "token": f"token-{index}"}
            for index in range(12)
        ]
        assignments = partition_development_scenes(
            scenes,
            official_train_names={f"scene-{index}" for index in range(10)},
            excluded_scenes={"scene-10", "scene-11"},
            validation_fraction=0.2,
            seed=7,
        )
        self.assertEqual(len(assignments), 10)
        self.assertEqual(sum(value == "val" for value in assignments.values()), 2)
        self.assertNotIn("token-10", assignments)
        self.assertNotIn("token-11", assignments)

    def test_manifest_checks_every_unique_data_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panoptic = root / "panoptic.npz"
            panoptic.write_bytes(b"npz")
            records = []
            for index in range(129):
                lidar = root / f"frame-{index}.bin"
                if index < 128:
                    lidar.write_bytes(b"points")
                records.append(
                    self._record(
                        sample_token=f"sample-{index}",
                        lidar_path=lidar.name,
                        panoptic_path=panoptic.name,
                    )
                )
            manifest = root / "train.jsonl"
            self._write_manifest(manifest, records)
            with self.assertRaises(FileNotFoundError):
                validate_manifest(manifest, root, expected_split="train")

    def test_manifest_rejects_excluded_test_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frame.bin").write_bytes(b"points")
            (root / "panoptic.npz").write_bytes(b"npz")
            manifest = root / "train.jsonl"
            self._write_manifest(manifest, [self._record()])
            with self.assertRaisesRegex(RuntimeError, "excluded test scene"):
                validate_manifest(
                    manifest,
                    root,
                    expected_split="train",
                    excluded_scenes={"scene-train"},
                )

    def test_manifest_rejects_answer_target_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frame.bin").write_bytes(b"points")
            (root / "panoptic.npz").write_bytes(b"npz")
            record = self._record()
            record["answer"] = "I found the car <SEG>."
            manifest = root / "train.jsonl"
            self._write_manifest(manifest, [record])
            with self.assertRaisesRegex(RuntimeError, "answer/target mismatch"):
                validate_manifest(manifest, root, expected_split="train")

    def test_manifest_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "panoptic.npz").write_bytes(b"npz")
            record = self._record(lidar_path="../outside.bin")
            manifest = root / "train.jsonl"
            self._write_manifest(manifest, [record])
            with self.assertRaisesRegex(RuntimeError, "escapes dataroot"):
                validate_manifest(manifest, root, expected_split="train")

    def test_content_verification_rejects_absent_target_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.zeros((2, 5), dtype=np.float32).tofile(root / "frame.bin")
            np.savez_compressed(
                root / "panoptic.npz",
                data=np.array([2001, 2001], dtype=np.uint16),
            )
            manifest = root / "train.jsonl"
            self._write_manifest(manifest, [self._record()])
            with self.assertRaisesRegex(RuntimeError, "target IDs absent"):
                validate_manifest(
                    manifest,
                    root,
                    expected_split="train",
                    verify_data_contents=True,
                )

    @staticmethod
    def _record(**overrides):
        record = {
            "sample_token": "sample-train",
            "scene_token": "scene-train",
            "split": "train",
            "lidar_path": "frame.bin",
            "panoptic_path": "panoptic.npz",
            "query": "Segment the car.",
            "answer": "I found the car <LOC> <SEG>.",
            "targets": [
                {"panoptic_id": 1001, "class_id": 3, "class_name": "car"}
            ],
        }
        record.update(overrides)
        return record

    @staticmethod
    def _write_manifest(path: Path, records):
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
