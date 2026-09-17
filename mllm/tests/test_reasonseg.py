import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


MLLM_ROOT = Path(__file__).resolve().parents[1]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))

from vtimellm.constants import IMAGE_TOKEN_INDEX
from vtimellm.lidar_visualizer import FrameData, SceneRef, project_full_mask_to_display
from vtimellm.segmentation.config import ReasonSegConfig
from vtimellm.segmentation.checkpoint import (
    load_spatial_encoder_checkpoint,
    save_spatial_encoder_checkpoint,
)
from vtimellm.segmentation.heads import HierarchicalMaskDecoder
from vtimellm.segmentation.metrics import (
    SegmentationMetricAccumulator,
    optimal_iou_matching,
)
from vtimellm.segmentation.model import extract_object_hidden_states
from vtimellm.segmentation.semantic_pretrain import semantic_metrics
from vtimellm.segmentation.tokens import (
    SegTokenInputAdapter,
    SegTokenOutputAdapter,
    SegmentationTokenIds,
    install_trainable_token_adapters,
    register_segmentation_tokens,
)
from scripts.build_reasonseg_nuscenes import describe_same_class_instances


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self):
        self.size = 32002
        self.ids = {"<4DLiDAR>": 32000, "<meta>": 32001}

    def __len__(self):
        return self.size

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, self.unk_token_id)

    def encode(self, token, add_special_tokens=False):
        return [self.convert_tokens_to_ids(token)]

    def add_special_tokens(self, values):
        added = 0
        for token in values["additional_special_tokens"]:
            if token not in self.ids:
                self.ids[token] = self.size
                self.size += 1
                added += 1
        return added


class FakeLanguageModel(nn.Module):
    def __init__(self, vocab_size=32002, hidden_size=16):
        super().__init__()
        self.input = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = type("Config", (), {"vocab_size": vocab_size})()

    def get_input_embeddings(self):
        return self.input

    def get_output_embeddings(self):
        return self.output

    def set_input_embeddings(self, module):
        self.input = module

    def set_output_embeddings(self, module):
        self.output = module

    def resize_token_embeddings(self, size):
        old_input = self.input
        old_output = self.output
        self.input = nn.Embedding(size, old_input.embedding_dim)
        self.output = nn.Linear(old_output.in_features, size, bias=False)
        with torch.no_grad():
            self.input.weight[: old_input.num_embeddings].copy_(old_input.weight)
            self.output.weight[: old_output.out_features].copy_(old_output.weight)
        self.config.vocab_size = size


class TokenTests(unittest.TestCase):
    def test_b3_is_resized_only_after_three_segmentation_tokens_are_added(self):
        tokenizer = FakeTokenizer()
        model = FakeLanguageModel()
        ids = register_segmentation_tokens(tokenizer, model)
        self.assertEqual(ids, SegmentationTokenIds(32002, 32003, 32004))
        self.assertEqual(len(tokenizer), 32005)
        self.assertEqual(model.input.num_embeddings, 32005)

    def test_input_and_output_rows_are_independently_trainable(self):
        tokenizer = FakeTokenizer()
        model = FakeLanguageModel()
        ids = register_segmentation_tokens(tokenizer, model)
        install_trainable_token_adapters(model, ids)
        self.assertIsInstance(model.get_input_embeddings(), SegTokenInputAdapter)
        self.assertIsInstance(model.get_output_embeddings(), SegTokenOutputAdapter)
        values = model.get_input_embeddings()(torch.tensor([[ids.loc, 4, ids.seg]]))
        logits = model.get_output_embeddings()(values)
        logits[..., list(ids.ordered)].sum().backward()
        self.assertIsNotNone(model.get_input_embeddings().delta.grad)
        self.assertIsNotNone(model.get_output_embeddings().rows.grad)
        self.assertIsNone(model.get_input_embeddings().base.weight.grad)
        self.assertIsNone(model.get_output_embeddings().base.weight.grad)

    def test_expanded_positions_account_for_scene_tokens(self):
        ids = SegmentationTokenIds(100, 101, 102)
        input_ids = torch.tensor(
            [[10, IMAGE_TOKEN_INDEX, 20, ids.loc, 30, ids.seg, 2]]
        )
        attention = torch.ones_like(input_ids)
        hidden = torch.arange(9, dtype=torch.float32).view(1, 9, 1)
        loc, seg, valid, status = extract_object_hidden_states(
            final_hidden_states=hidden,
            input_ids=input_ids,
            attention_mask=attention,
            image_token_length=3,
            token_ids=ids,
            max_objects=8,
        )
        self.assertEqual(status, ["ok"])
        self.assertTrue(valid[0, 0])
        self.assertEqual(float(loc[0, 0, 0]), 5.0)
        self.assertEqual(float(seg[0, 0, 0]), 7.0)

    def test_token_mismatch_is_explicit_failure(self):
        ids = SegmentationTokenIds(100, 101, 102)
        input_ids = torch.tensor([[10, ids.loc, ids.seg, ids.seg]])
        hidden = torch.zeros(1, 4, 2)
        _, _, valid, status = extract_object_hidden_states(
            final_hidden_states=hidden,
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            image_token_length=3,
            token_ids=ids,
            max_objects=8,
        )
        self.assertIn("token_pair_mismatch", status[0])
        self.assertFalse(valid.any())


class DecoderTests(unittest.TestCase):
    def test_hierarchical_decoder_shapes_losses_and_gradients(self):
        config = ReasonSegConfig(
            hidden_size=12,
            point_feature_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            decoder_ffn_dim=16,
            max_objects=3,
            class_names=["car", "pedestrian", "truck"],
        )
        decoder = HierarchicalMaskDecoder(config)
        point_features = torch.randn(2, 11, 8, requires_grad=True)
        valid_points = torch.ones(2, 11, dtype=torch.bool)
        valid_points[1, -2:] = False
        loc_hidden = torch.randn(2, 2, 12)
        seg_hidden = torch.randn(2, 2, 12)
        object_valid = torch.tensor([[True, True], [True, False]])
        target_masks = torch.zeros(2, 2, 11, dtype=torch.bool)
        target_masks[0, 0, :3] = True
        target_masks[0, 1, 4:7] = True
        target_masks[1, 0, 2:6] = True
        target_loc = target_masks.clone()
        target_classes = torch.tensor([[0, 1], [2, 0]])
        output = decoder(
            point_features,
            valid_points,
            loc_hidden,
            seg_hidden,
            object_valid,
            target_masks,
            target_loc,
            target_classes,
        )
        self.assertEqual(output.mask_logits.shape, (2, 2, 11))
        self.assertEqual(output.loc_logits.shape, (2, 2, 11))
        self.assertEqual(output.class_logits.shape, (2, 2, 3))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(point_features.grad)


class MetricAndViewerTests(unittest.TestCase):
    def test_exact_matching_prefers_maximum_total_iou(self):
        matrix = np.array([[0.9, 0.8], [0.85, 0.1]])
        self.assertEqual(set(optimal_iou_matching(matrix)), {(0, 1), (1, 0)})

    def test_metric_counts_extra_and_missing_instances(self):
        accumulator = SegmentationMetricAccumulator(num_classes=2)
        targets = np.array([[1, 1, 0, 0]], dtype=bool)
        predictions = np.array(
            [[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool
        )
        accumulator.update(predictions, np.array([0, 1]), targets, np.array([0]))
        values = accumulator.compute()
        self.assertEqual(values["instance_precision@0.5"], 0.5)
        self.assertEqual(values["instance_recall@0.5"], 1.0)
        self.assertEqual(values["count_accuracy"], 0.0)

    def test_full_mask_projects_to_displayed_point_indices(self):
        frame = FrameData(
            scene=SceneRef("scene", None, "scene", None, ("sample",)),
            frame_index=0,
            sample_token="sample",
            points=np.zeros((3, 4), dtype=np.float32),
            boxes=(),
            tracks=(),
            camera_paths={},
            point_indices=np.array([1, 4, 7]),
        )
        full = np.zeros(8, dtype=bool)
        full[[1, 7]] = True
        np.testing.assert_array_equal(
            project_full_mask_to_display(frame, full), [True, False, True]
        )

    def test_same_class_descriptions_are_unique(self):
        instances = [
            {"class_name": "car", "center": np.array([distance, 1.0, 0.0])}
            for distance in (1.0, 2.0, 3.0, 4.0)
        ]
        descriptions = describe_same_class_instances(instances)
        self.assertEqual(len(descriptions), len(set(descriptions)))
        self.assertIn("2nd-nearest", descriptions[1])

    def test_spatial_checkpoint_validates_and_restores_weights(self):
        config = ReasonSegConfig(point_feature_dim=8, decoder_heads=2)
        source = nn.Linear(4, 8)
        restored = nn.Linear(4, 8)
        with tempfile.TemporaryDirectory() as directory:
            save_spatial_encoder_checkpoint(
                source,
                directory,
                config=config,
                epoch=2,
                validation_miou=0.4,
                num_classes=32,
            )
            metadata = load_spatial_encoder_checkpoint(
                restored, directory, config=config
            )
        self.assertEqual(metadata["epoch"], 2)
        self.assertTrue(torch.equal(source.weight, restored.weight))

    def test_semantic_metrics_exclude_absent_classes(self):
        confusion = torch.tensor([[2, 0, 0], [0, 1, 1], [0, 0, 0]])
        values = semantic_metrics(confusion)
        self.assertAlmostEqual(values["point_accuracy"], 0.75)
        self.assertAlmostEqual(values["miou"], (1.0 + 0.5 + 0.0) / 3.0)


if __name__ == "__main__":
    unittest.main()
