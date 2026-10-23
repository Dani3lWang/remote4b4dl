import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from b4dl_core.data import build_dataset, load_aligned_scenes
from b4dl_core.audit import audit_supervision
from b4dl_core.inputs import GeometryDataset
from b4dl_core.io import digest, read_json, sha256, write_json, write_jsonl
from b4dl_core.sst_source import SCOPE, SUPPORT_METHOD, box_support, retained_points, seal_provenance
from b4dl_core.trainval import opposite_pair_capacity
from b4dl_core.evaluation import opposite_pairs
from test_core_geometry import binary
from b4dl_core.b3 import verify_sst_binding
from types import SimpleNamespace
from test_core_geometry import raw_fixture


def support_fixture(tables, metadata, root):
    records = []
    for sample in read_json(tables/'sample.json'):
        ident, index = sample['token'].split('-')
        records.append({'scene_id': ident, 'frame_index': int(index), 'sample_token': sample['token'],
            'lidar_sample_data_token': sample['token']+'-sd',
            'annotation_counts': {sample['token']+'-ann': 2 if int(index) % 2 == 0 else 0}})
    path = root/'support.jsonl'
    write_jsonl(path, records)
    receipt = {'method': SUPPORT_METHOD, 'records_sha256': sha256(path), 'metadata_sha256': sha256(metadata),
        'source_tables_sha256': {name: sha256(tables/(name+'.json')) for name in
            ('sample', 'sample_data', 'ego_pose', 'calibrated_sensor', 'sample_annotation')}}
    write_json(path.with_suffix('.receipt.json'), receipt)
    return path


class SSTSourceTests(unittest.TestCase):
    def test_pair_capacity_matches_explicit_disjoint_matching(self):
        rng = np.random.default_rng(19)
        for _ in range(80):
            requests, targets = [], {}
            for index in range(20):
                request, target = binary('scene-'+str(rng.integers(5)), 'Yes' if rng.integers(2) else 'No')
                request['request_id'] = str(index)
                target['request_id'] = str(index)
                requests.append(request); targets[str(index)] = target
            self.assertEqual(opposite_pair_capacity(requests, targets).get('test', 0), len(opposite_pairs(requests, targets)))

    @unittest.skipUnless(importlib.util.find_spec('pyquaternion'), 'remote wqlc provides pyquaternion')
    def test_rotated_box_counts_and_empty_input(self):
        row = {'token': 'box', 'translation': [0, 0, 0], 'size': [2, 6, 2],
               'rotation': [2**-.5, 0, 0, 2**-.5]}
        points = np.array([[0, 2, 0], [2, 0, 0], [0, 4, 0]], dtype=float)
        self.assertEqual(box_support(points, [row]), {'box': 1})
        self.assertEqual(box_support(np.empty((0, 3)), [row]), {'box': 0})

    @unittest.skipUnless(importlib.util.find_spec('nuscenes'), 'remote wqlc provides nuScenes')
    def test_crop_and_voxel_range_apply_before_visibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = np.array([[0, 0, 10, 255, 0], [6, 0, 10, 255, 0], [0, 0, 45, 255, 0],
                            [0, 3, 10, 255, 0], [0, 0, -1, 255, 0]], dtype=np.float32)
            raw.tofile(root/'lidar.bin')
            identity = {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}
            context = {'sample': {'s': {'data': {'LIDAR_TOP': 'l', 'CAM_BACK_RIGHT': 'c'}}},
                'sample_data': {'l': {'token': 'l', 'filename': 'lidar.bin', 'calibrated_sensor_token': 'lc', 'ego_pose_token': 'p'},
                    'c': {'token': 'c', 'width': 200, 'height': 100, 'calibrated_sensor_token': 'cc', 'ego_pose_token': 'p'}},
                'ego_pose': {'p': identity}, 'calibrated_sensor': {'lc': identity,
                    'cc': {**identity, 'camera_intrinsic': [[100, 0, 100], [0, 100, 50], [0, 0, 1]]}}}
            cropped, points, receipt = retained_points(context, 's', root)
            self.assertEqual(len(cropped), 3)
            np.testing.assert_array_equal(points, [[0, 0, 10]])
            self.assertEqual(receipt['retained_points'], 1)
            self.assertTrue(np.all(cropped[:, 3] == 1))

    def test_scope_filters_nearest_and_preserves_original_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, metadata, features, splits = raw_fixture(root)
            support = support_fixture(tables, metadata, root)
            scenes = load_aligned_scenes(tables, metadata, features, read_json(splits), input_support=support)
            self.assertEqual([len(f['objects']) for f in scenes[0]['frames']], [1, 0]*5)
            self.assertEqual(scenes[0]['frames'][0]['objects'][0]['num_lidar_pts'], 2)
            manifest = build_dataset(tables, metadata, features, splits, root/'dataset', input_support=support)
            dataset = GeometryDataset(root/'dataset')
            self.assertTrue(all(r['question'].startswith(SCOPE) for r in dataset.requests))
            self.assertTrue(any(t['answer'] == 'No' for t in dataset.targets.values()))
            np.testing.assert_array_equal(dataset.scene('s0')['features'], np.load(features/'s0.npy'))
            self.assertEqual(audit_supervision(dataset, tables)['questions'], len(dataset.requests))
            self.assertFalse(manifest['encoder_checkpoint_verified'])

    def test_support_tampering_and_wrong_frame_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, metadata, features, splits = raw_fixture(root)
            support = support_fixture(tables, metadata, root)
            support.write_text(support.read_text()+'\n')
            with self.assertRaises(ValueError):
                load_aligned_scenes(tables, metadata, features, read_json(splits), input_support=support)

    def test_false_encoder_verification_does_not_pass_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, metadata, features, splits = raw_fixture(root)
            provenance = {'schema': 'b4dl-sst-provenance-v1', 'verified': False}
            provenance['provenance_id'] = digest(provenance)
            write_json(root/'provenance.json', provenance)
            with self.assertRaises(ValueError):
                build_dataset(tables, metadata, features, splits, root/'dataset', encoder_provenance=root/'provenance.json')
            self.assertFalse((root/'dataset').exists())

    def test_formal_training_rejects_unbound_or_changed_sst(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'provenance.json'
            write_json(path, {'fixture': True})
            binding = {'path': str(path), 'sha256': sha256(path), 'provenance_id': 'fixture'}
            dataset = SimpleNamespace(manifest={'encoder_provenance': binding, 'input_scope': {'fixture': True}})
            verify_sst_binding({'sst_provenance': binding}, dataset)
            with self.assertRaises(ValueError): verify_sst_binding({}, dataset)
            with self.assertRaises(ValueError):
                verify_sst_binding({'sst_provenance': {**binding, 'provenance_id': 'wrong'}}, dataset)
            path.write_text('{}')
            with self.assertRaises(ValueError): verify_sst_binding({'sst_provenance': binding}, dataset)


if __name__ == '__main__':
    unittest.main()
