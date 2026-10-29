"""End-to-end data compilation against independently reconstructed raw truth."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from b4dl_core.data import build_dataset
from b4dl_core.io import sha256, write_json, write_jsonl
from b4dl_core.inputs import control_plan
from b4dl_core.reasoning.data import ReasoningDataset, audit_data, build_data
from b4dl_core.sst_source import SUPPORT_METHOD


def raw_fixture(root):
    tables, features = root/"tables", root/"features"
    tables.mkdir(); features.mkdir()
    rows = {key: [] for key in ("scene", "sample", "sample_data", "ego_pose", "calibrated_sensor",
                                "sensor", "sample_annotation", "instance", "category")}
    rows["sensor"] = [{"token": "sensor", "channel": "LIDAR_TOP"}]
    rows["calibrated_sensor"] = [{"token": "cal", "sensor_token": "sensor", "rotation": [1, 0, 0, 0], "translation": [0, 0, 1]}]
    rows["category"] = [{"token": "car", "name": "vehicle.car"}]
    metadata, support = [], []
    for s in range(3):
        sid = "s"+str(s)
        rows["scene"].append({"token": sid, "name": "scene-"+str(s), "nbr_samples": 10,
                              "first_sample_token": sid+"-0", "last_sample_token": sid+"-9"})
        rows["instance"].append({"token": sid+"-obj", "category_token": "car"})
        paths = {}
        for i in range(10):
            token, stamp = f"{sid}-{i}", 1500000000000000+i*500000
            rows["sample"].append({"token": token, "scene_token": sid,
                "prev": f"{sid}-{i-1}" if i else "", "next": f"{sid}-{i+1}" if i < 9 else ""})
            rows["sample_data"].append({"token": token+"-sd", "sample_token": token, "filename": token+".bin",
                "timestamp": stamp, "is_key_frame": True, "ego_pose_token": token+"-pose", "calibrated_sensor_token": "cal"})
            rows["ego_pose"].append({"token": token+"-pose", "timestamp": stamp, "rotation": [1, 0, 0, 0], "translation": [i*.5, 0, 0]})
            rows["sample_annotation"].append({"token": token+"-ann", "sample_token": token,
                "instance_token": sid+"-obj", "translation": [13 if i == 9 else 10, 0, 0], "num_lidar_pts": 5})
            support.append({"scene_id": sid, "frame_index": i, "sample_token": token,
                "lidar_sample_data_token": token+"-sd", "annotation_counts": {token+"-ann": 2 if i in (0, 1, 8, 9) else 0}})
            paths[f"PATH_{i:03d}"] = token+".bin"
        np.save(features/(sid+".npy"), np.arange(10*768, dtype=np.float32).reshape(10, 768)/1000+s)
        metadata.append({"scene_id": sid, "scene_token": sid, "num_frames": 10, "paths": {"PATH_LIDAR_TOP": paths}})
    for key, value in rows.items(): write_json(tables/(key+".json"), value)
    write_json(root/"metadata.json", metadata)
    write_json(root/"splits.json", {"version": "fixture", "source": "test", "train": ["scene-0", "scene-1"], "validation": ["scene-2"]})
    support_path = root/"support.jsonl"
    write_jsonl(support_path, support)
    write_json(support_path.with_suffix(".receipt.json"), {"method": SUPPORT_METHOD, "records_sha256": sha256(support_path),
        "metadata_sha256": sha256(root/"metadata.json"), "source_tables_sha256": {name: sha256(tables/(name+".json")) for name in
            ("sample", "sample_data", "ego_pose", "calibrated_sensor", "sample_annotation")}})
    return tables, features, support_path


class ReasoningDataTests(unittest.TestCase):
    def test_compile_audit_and_scope_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tables, features, support = raw_fixture(root)
            build_dataset(tables, root/"metadata.json", features, root/"splits.json", root/"geometry", input_support=support)
            manifest = build_data(root/"geometry", tables, support, root/"reasoning")
            dataset = ReasoningDataset(root/"reasoning", root/"geometry")
            self.assertEqual(audit_data(dataset, tables, support)["questions"], manifest["selected_questions"])
            self.assertTrue({"existence", "motion"} <= {r["task"] for r in dataset.requests})
            motion = next(r for r in dataset.requests if r["task"] == "motion" and r["frames"] == [0, 8])
            teacher = dataset.targets[motion["request_id"]]["trace"]
            self.assertEqual(teacher["steps"][0]["value"]["d1"], 6.)
            self.assertNotIn("trace", dataset.model_input(motion))
            controls = {r["request_id"]: r for r in control_plan(dataset.geometry, "zero")}
            self.assertTrue(np.any(dataset.model_input(motion)["features"] != 0))
            self.assertTrue(np.all(dataset.model_input(motion, controls[motion["request_id"]])["features"] == 0))
            support.write_text(support.read_text()+"\n")
            with self.assertRaises(ValueError): audit_data(dataset, tables, support)

    def test_model_request_and_supervision_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tables, features, support = raw_fixture(root)
            build_dataset(tables, root/"metadata.json", features, root/"splits.json", root/"geometry", input_support=support)
            build_data(root/"geometry", tables, support, root/"reasoning")
            path = root/"reasoning/requests.jsonl"
            path.write_text(path.read_text()+"\n")
            with self.assertRaises(ValueError): ReasoningDataset(root/"reasoning", root/"geometry")


if __name__ == "__main__":
    unittest.main()
