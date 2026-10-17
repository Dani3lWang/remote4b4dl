"""Real I/O fixtures for point provenance, split isolation and review gating."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_answer_binding_dataset import apply_review, rows, run, sha, verify_mask
from grounded4d.taxonomy import FINE_NAMES


class BindingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = argparse.Namespace(point_root=self.root / "points", label_root=self.root / "labels",
            panoptic_root=self.root / "panoptic", point_cloud_range=[-51.2,-51.2,-5,51.2,51.2,3])
        for path in (self.args.point_root, self.args.label_root, self.args.panoptic_root):
            path.mkdir()
        points = np.array([[1,0,0,.5,0],[2,0,0,.2,1],[0,0,0,1,2]], dtype="<f4")
        point = self.args.point_root / "points.bin"
        point.write_bytes(points.tobytes())
        np.savez(self.args.panoptic_root / "panoptic.npz", data=np.array([17001,17002,31000]))
        self.labels = np.array([[1,0,-1],[0,1,-1]], dtype=np.int8)
        self.save_mask(np.arange(3))
        self.frame = dict(point_file=dict(path="points.bin",sha256=sha(point)),
            mask_file=dict(path="mask.npz",sha256=sha(self.args.label_root / "mask.npz")),
            source_panoptic_file=dict(path="panoptic.npz",sha256=sha(self.args.panoptic_root / "panoptic.npz")),
            class_ids=[3,3],lidar_to_anchor=np.eye(4).tolist(),
            instance_mapping_evidence=[dict(panoptic_id=17001,instance_token="one"),dict(panoptic_id=17002,instance_token="two")])

    def save_mask(self, indices):
        np.savez(self.args.label_root / "mask.npz", labels=self.labels, instance_tokens=np.array(["one","two"]),
            point_indices=indices, label_kind=np.asarray("panoptic_gt"), mask_verified=np.asarray(True),
            point_file_sha256=np.asarray(sha(self.args.point_root / "points.bin")))

    def candidate(self, split="train"):
        return dict(candidate_id="id", candidate_fingerprint="fingerprint", split=split, answer="the car ahead",
                    mention_span=[4,7], objects=[dict(instance_token="one",mask_row=0),dict(instance_token="two",mask_row=1)])

    def review(self):
        return dict(candidate_id="id",candidate_fingerprint="fingerprint",decision="accept",reviewer="fixture",
                    reviewed_at="2026-10-17",evidence="fixture evidence",target_span=[0,13],instance_token="two")

    def test_original_point_labels(self):
        _, labels, tokens, _, _ = verify_mask(self.frame, {}, self.args)
        np.testing.assert_array_equal(labels,self.labels)
        self.assertEqual(tokens,["one","two"])

    def test_reordered_point_axis_rejected(self):
        self.save_mask(np.array([1,0,2]))
        self.frame["mask_file"]["sha256"] = sha(self.args.label_root / "mask.npz")
        with self.assertRaisesRegex(ValueError,"point order"):
            verify_mask(self.frame, {}, self.args)

    def test_wrong_mask_rejected_even_with_new_hash(self):
        self.labels[0,1]=1
        self.save_mask(np.arange(3))
        self.frame["mask_file"]["sha256"] = sha(self.args.label_root / "mask.npz")
        with self.assertRaisesRegex(ValueError,"disagrees"):
            verify_mask(self.frame, {}, self.args)

    def test_review_requires_evidence_and_exact_candidate(self):
        for changes in ({"evidence":""},{"instance_token":"other_frame"},{"candidate_fingerprint":"old"},{"target_span":[8,13]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                apply_review(self.candidate(),dict(self.review(),**changes))

    def test_dev_review_never_enters_training(self):
        accepted=apply_review(self.candidate("internal_dev"),self.review())
        self.assertFalse(accepted["eligible_for_binding_training"])
        self.assertTrue(accepted["eligible_for_binding_evaluation"])
        self.assertEqual(accepted["target_mask_row"],1)

    def test_source_hash_tampering_rejected(self):
        with (self.args.point_root / "points.bin").open("ab") as stream:
            stream.write(b"bad")
        with self.assertRaisesRegex(ValueError,"hash mismatch"):
            verify_mask(self.frame, {}, self.args)

    def test_streaming_large_unicode_record_and_invalid_json(self):
        file=self.root/"array.json"
        data=[{"text":"中文"*70000},{"tail":1}]
        file.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
        self.assertEqual(list(rows(file)),data)
        for bad in ('[{"x":1},]','[{"x":1}', '[{"x":1}] extra'):
            file.write_text(bad,encoding="utf-8")
            with self.assertRaises(ValueError):
                list(rows(file))

    def test_scene_overlap_rejected_before_data_read(self):
        manifests=[]
        for split in ("train","internal_dev"):
            path=self.root/(split+".jsonl")
            row=dict(schema_version="b4dl-grounded4d-v1",split=split,dataset_version="v1.0-trainval",
                     label_kind="panoptic_gt",annotation_scope="all_configured_instances",scene_token="scene",
                     frames=[dict(sample_data_token=split)])
            path.write_text(json.dumps(row)+"\n")
            manifests.append(path)
        white=self.root/"white.json"; white.write_text('["scene"]')
        args=copy.copy(self.args)
        args.output_dir=self.root/"out"; args.metadata_root=self.root/"metadata"
        args.train_manifest,args.dev_manifest=manifests
        args.qa=[];args.scene_metadata=self.root/"scenes.json";args.sequence_metadata=self.root/"sequences.json"
        args.train_scene_tokens=white
        with self.assertRaisesRegex(ValueError,"scene leakage"):
            run(args)


if __name__ == "__main__":
    unittest.main()
