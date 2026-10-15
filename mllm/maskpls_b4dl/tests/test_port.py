"""Contract tests. Run in the isolated wqlc environment before the real CUDA smoke."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from maskpls_b4dl.config import MaskPLSConfig
from maskpls_b4dl.criterion import MaskPLSCriterion, sample_supervised_points
from maskpls_b4dl.data import build_targets, semantic_from_panoptic
from maskpls_b4dl.interpolate import KNNInterpolation
from maskpls_b4dl.model import MaskPLS, extract_instances, import_official_state
from maskpls_b4dl.objects import resolve_answer_objects, save_prediction
from maskpls_b4dl.protocol import score_frame, verify_source_split
from maskpls_b4dl import runner


class ToyBackbone(nn.Module):
    """CPU fixture only; never available through a production config or CLI."""
    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(4, size) for size in (256, 128, 96, 96)])
        self.sem = nn.Linear(96, 20)

    def forward(self, points):
        features = [layer(points)[None] for layer in self.projections]
        coordinates = [points[None, :, :3] for _ in features]
        padding = [torch.zeros((1, len(points)), dtype=torch.bool) for _ in features]
        return features, coordinates, padding, self.sem(features[-1])


class WeightFixture(nn.Module):
    def __init__(self, queries):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.decoder = nn.Module()
        self.decoder.query_feat = nn.Embedding(queries, 8)
        self.decoder.query_embed = nn.Embedding(queries, 8)
        self.decoder.class_embed = nn.Linear(8, 18)


def prediction_fixture(classes, masks):
    logits = torch.full((len(classes), 18), -12.0)
    for index, category in enumerate(classes):
        logits[index, category] = 12
    return dict(class_logits=logits, mask_logits=torch.tensor(masks, dtype=torch.float32) * 24 - 12,
                valid_points=torch.ones(len(masks[0]), dtype=torch.bool),
                queries=torch.arange(len(classes) * 256, dtype=torch.float32).reshape(len(classes), 256))


class InterpolationAndModelTests(unittest.TestCase):
    def test_inverse_squared_distance_and_gradient_with_fewer_than_three_voxels(self):
        coordinates = torch.tensor([[0., 0., 0.], [2., 0., 0.]])
        features = torch.tensor([[0.], [10.]], requires_grad=True)
        points = torch.tensor([[0.5, 0., 0.]])
        result = KNNInterpolation(3, "torch", 1)(coordinates, features, points)
        torch.testing.assert_close(result, torch.tensor([[1.]]))
        result.sum().backward()
        torch.testing.assert_close(features.grad, torch.tensor([[0.9], [0.1]]))

    def test_raw_point_permutation_and_outside_roi_are_preserved(self):
        torch.manual_seed(3)
        config = MaskPLSConfig(num_queries=3, checkpoint_decoder=False, knn_backend="torch")
        model = MaskPLS(config, backbone=ToyBackbone()).eval()
        points = torch.tensor([[2., 1., 0., 1.], [100., 0., 0., 2.],
                               [-2., 1., 0., 3.], [3., -1., 0., 4.]])
        permutation = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            first, second = model(points), model(points[permutation])
        torch.testing.assert_close(first["mask_logits"][:, permutation], second["mask_logits"], rtol=1e-4, atol=1e-4)
        self.assertEqual(first["valid_points"].tolist(), [True, False, True, True])
        self.assertTrue((first["mask_logits"][:, 1] == -1e4).all())
        self.assertEqual(len(first["auxiliary"]), 9)
        self.assertEqual(first["queries"].shape, (3, 256))

    def test_activation_checkpoint_preserves_outputs_and_gradients(self):
        torch.manual_seed(5)
        config = MaskPLSConfig(num_queries=2, checkpoint_decoder=False)
        plain = MaskPLS(config, backbone=ToyBackbone()).train()
        checked = copy.deepcopy(plain)
        checked.decoder.checkpoint_decoder = True
        points = torch.randn(6, 4)
        a, b = plain(points), checked(points)
        for model_output in (a, b):
            (model_output["mask_logits"].square().mean()
             + model_output["class_logits"].square().mean()).backward()
        torch.testing.assert_close(a["mask_logits"], b["mask_logits"])
        for left, right in zip(plain.parameters(), checked.parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-5)

    def test_intensity_normalization_uses_full_frame_without_mutation(self):
        config = MaskPLSConfig(num_queries=2, intensity_mode="frame_minmax", checkpoint_decoder=False)
        backbone = ToyBackbone()
        observed = []
        backbone.register_forward_pre_hook(lambda module, args: observed.append(args[0].clone()))
        model = MaskPLS(config, backbone=backbone).eval()
        points = torch.tensor([[0., 0., 0., 10.], [1., 0., 0., 20.], [100., 0., 0., 30.]])
        original = points.clone()
        with torch.no_grad():
            model(points)
        torch.testing.assert_close(points, original)
        torch.testing.assert_close(observed[0][:, 3], torch.tensor([0., 0.5]))


class SupervisionTests(unittest.TestCase):
    def test_fine_category_mapping_and_ignore(self):
        encoded = np.array([9001, 17001, 2001, 24000, 30000, 31000, 1001])
        self.assertEqual(semantic_from_panoptic(encoded).tolist(), [1, 4, 7, 11, 16, 0, 0])

    def test_one_point_target_is_retained_with_stuff(self):
        frame = dict(labels=torch.tensor([[1., 0., -1.]]), class_ids=torch.tensor([0]),
                     semantic_labels=torch.tensor([1, 11, 0]))
        labels, classes, valid = build_targets(frame, torch.ones(3, dtype=torch.bool))
        self.assertEqual(classes.tolist(), [1, 11])
        self.assertEqual((labels == 1).sum(1).tolist(), [1, 1])
        self.assertEqual(valid.tolist(), [True, True, False])

    def test_sampling_cannot_lose_tiny_foreground_or_include_ignore(self):
        labels = torch.zeros(2, 100)
        labels[0, 98] = 1
        labels[1, 99] = 1
        valid = torch.ones(100, dtype=torch.bool)
        valid[0] = False
        for _ in range(5):
            indices = sample_supervised_points(labels, valid, 4, 500)
            self.assertIn(98, indices.tolist())
            self.assertIn(99, indices.tolist())
            self.assertNotIn(0, indices.tolist())
            self.assertEqual(len(indices), 4)

    def test_ignore_and_out_of_range_have_no_mask_or_semantic_gradient(self):
        frame = dict(labels=torch.tensor([[1., 0., 0., 0., -1.]]), class_ids=torch.tensor([0]),
                     semantic_labels=torch.tensor([1, 11, 11, 11, 0]))
        output = dict(mask_logits=torch.randn(3, 5, requires_grad=True),
                      class_logits=torch.randn(3, 18, requires_grad=True),
                      semantic_logits=torch.randn(5, 20, requires_grad=True),
                      valid_points=torch.tensor([True, True, False, True, True]), auxiliary=[])
        loss, _, _ = MaskPLSCriterion(MaskPLSConfig())(output, frame)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue((output["mask_logits"].grad[:, [2, 4]] == 0).all())
        self.assertTrue((output["semantic_logits"].grad[[2, 4]] == 0).all())

    def test_insufficient_query_capacity_is_not_silent_target_truncation(self):
        criterion = MaskPLSCriterion(MaskPLSConfig())
        with self.assertRaisesRegex(ValueError, "capacity"):
            criterion.match(dict(mask_logits=torch.zeros(1, 2)), torch.eye(2),
                            torch.tensor([1, 4]), torch.arange(2))


class CheckpointTests(unittest.TestCase):
    def test_query_expansion_is_explicit_and_preserves_pretrained_rows(self):
        source, destination = WeightFixture(2), WeightFixture(3)
        state = source.state_dict()
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            import_official_state(destination, state)
        new_row = destination.decoder.query_feat.weight[2].detach().clone()
        report = import_official_state(destination, state, allow_query_expansion=True)
        torch.testing.assert_close(destination.decoder.query_feat.weight[:2], source.decoder.query_feat.weight)
        torch.testing.assert_close(destination.decoder.query_feat.weight[2], new_row)
        self.assertEqual(report["query_expansion"]["decoder.query_feat.weight"]["new_rows"], 1)

    def test_wrong_class_mapping_shape_and_missing_keys_fail(self):
        model = WeightFixture(2)
        state = dict(model.state_dict())
        state["decoder.class_embed.weight"] = torch.zeros(21, 8)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            import_official_state(model, state)
        del state["decoder.class_embed.weight"]
        with self.assertRaisesRegex(ValueError, "missing"):
            import_official_state(model, state)


class PredictionTests(unittest.TestCase):
    def test_nan_is_rejected_instead_of_becoming_an_empty_prediction(self):
        output = prediction_fixture([1], [[1, 0]])
        output["class_logits"][0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            extract_instances(output, MaskPLSConfig())

    def test_tiny_instance_is_kept_and_stuff_is_not_exported(self):
        output = prediction_fixture([1, 11, 17], [[1, 0, 0], [0, 1, 1], [0, 0, 0]])
        instances = extract_instances(output, MaskPLSConfig())
        self.assertEqual(instances["query_indices"].tolist(), [0])
        self.assertEqual(instances["masks"].sum(1).tolist(), [1])
        self.assertEqual(instances["class_ids"].tolist(), [0])

    def test_wrong_class_is_not_a_formal_hit_even_with_perfect_geometry(self):
        output = prediction_fixture([4], [[1, 0, 0]])
        frame = dict(labels=torch.tensor([[1., 0., 0.]]), class_ids=torch.tensor([0]),
                     semantic_labels=torch.tensor([1, 11, 11]))
        instances = extract_instances(output, MaskPLSConfig())
        counts, _, diagnostic = score_frame(frame, output, instances, 0.5)
        self.assertEqual((counts["tp"], counts["fp"], counts["fn"]), (0, 1, 1))
        self.assertEqual(diagnostic["raw_query_geometry_hits"], 1)
        self.assertEqual(diagnostic["point_buckets"]["1-10"], dict(gt=1, tp=0))

    def test_empty_predictions_and_query_no_object_are_not_answer_no_object(self):
        output = prediction_fixture([17], [[1, 0]])
        instances = extract_instances(output, MaskPLSConfig())
        self.assertEqual(instances["masks"].shape, (0, 2))
        self.assertEqual(instances["query_features"].shape, (0, 256))

    def test_export_does_not_read_ground_truth_and_rejects_foreign_frame_ids(self):
        output = prediction_fixture([1], [[1, 0]])
        instances = extract_instances(output, MaskPLSConfig())
        frame = dict(points=torch.tensor([[0., 0., 0., 1.], [1., 0., 0., 1.]]),
                     sample_data_token="abc123", sample_token="sample123", point_sha256="0" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            save_prediction(temporary, frame, output, instances, {"weights_sha256": "1" * 64})
            record = json.loads((Path(temporary) / "abc123.json").read_text())
            with np.load(Path(temporary) / "abc123.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["point_indices"].tolist(), [0, 1])
                self.assertEqual(archive["query_features"].shape, (1, 256))
            self.assertFalse(record["grounding_verified"])
            self.assertIsNone(record["objects"][0]["track_id"])
            self.assertEqual(resolve_answer_objects(record["objects"], ["abc123:q0"])[0]["query_index"], 0)
            with self.assertRaises(ValueError):
                resolve_answer_objects(record["objects"], ["other:q0"])
            with self.assertRaises(FileExistsError):
                save_prediction(temporary, frame, output, instances, {})


class OutputSafetyTests(unittest.TestCase):
    def test_existing_experiment_is_not_modified_when_new_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "preflight.json").write_text('{"old":true}')
            args = argparse.Namespace(command="preflight", config=None, output_dir=directory)
            with patch.object(runner, "parse_args", return_value=args):
                with self.assertRaises(FileExistsError):
                    runner.main()
            self.assertEqual((directory / "preflight.json").read_text(), '{"old":true}')
            self.assertFalse((directory / "failure.json").exists())

    def test_rejected_resume_preserves_previous_run_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            old_status = '{"status":"completed"}'
            (directory / "status.json").write_text(old_status)
            (directory / "completion.json").write_text(old_status)
            checkpoint = {"metadata": {"config": MaskPLSConfig().to_dict()}}
            args = argparse.Namespace(command="train", split="train", official_val=False,
                                      config=None, resume=directory / "last.pt", output_dir=directory)
            with (patch.object(runner, "parse_args", return_value=args),
                  patch.object(runner, "load_native", return_value=checkpoint),
                  patch.object(runner, "make_dataset", side_effect=ValueError("changed dataset"))):
                with self.assertRaisesRegex(ValueError, "changed dataset"):
                    runner.main()
            self.assertEqual((directory / "status.json").read_text(), old_status)
            self.assertEqual((directory / "completion.json").read_text(), old_status)
            self.assertFalse((directory / "failure.json").exists())


class SourceIdentityTests(unittest.TestCase):
    def fixture(self, directory, scene_name="scene-train", panoptic_filename="panoptic/a.npz"):
        version = "v1.0-trainval"
        root = Path(directory)
        (root / version).mkdir()
        tables = {
            "scene": [{"token": "scene1", "name": scene_name}],
            "sample": [{"token": "sample1", "scene_token": "scene1"}],
            "sample_data": [{"token": "lidar1", "sample_token": "sample1", "is_key_frame": True,
                             "timestamp": 100, "filename": "samples/LIDAR_TOP/a.bin"}],
            "panoptic": [{"sample_data_token": "lidar1", "filename": "panoptic/a.npz"}],
        }
        for name, value in tables.items():
            (root / version / (name + ".json")).write_text(json.dumps(value))
        frame = dict(sample_data_token="lidar1", sample_token="sample1", timestamp_us=100,
                     original_sample_data_filename="samples/LIDAR_TOP/a.bin",
                     point_file={"path": "samples/LIDAR_TOP/a.bin"},
                     source_panoptic_file={"path": panoptic_filename})
        dataset = SimpleNamespace(records=[dict(dataset_version=version, split="train", scene_token="scene1",
                                                frames=[frame])], scenes={"scene1"}, panoptic_root=root)
        utils = ModuleType("nuscenes.utils")
        utils.splits = SimpleNamespace(train=["scene-train"], val=["scene-val"])
        nuscenes = ModuleType("nuscenes")
        nuscenes.utils = utils
        return dataset, {"nuscenes": nuscenes, "nuscenes.utils": utils}

    def test_official_val_cannot_be_claimed_as_training(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, modules = self.fixture(directory, scene_name="scene-val")
            with patch.dict("sys.modules", modules):
                with self.assertRaisesRegex(ValueError, "official split"):
                    verify_source_split(dataset, directory)

    def test_panoptic_from_another_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, modules = self.fixture(directory, panoptic_filename="panoptic/wrong.npz")
            with patch.dict("sys.modules", modules):
                with self.assertRaisesRegex(ValueError, "different LiDAR frame"):
                    verify_source_split(dataset, directory)


if __name__ == "__main__":
    unittest.main()
