"""Boundary cases for reported instance-mask performance accounting."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from grounded4d.metrics import MetricTotals, frame_counts


class InstanceMetricTests(unittest.TestCase):
    def measure(self, targets, target_classes, predictions, predicted_classes):
        result, assignment = frame_counts(np.asarray(targets, dtype=bool), np.asarray(target_classes),
                                          np.asarray(predictions, dtype=bool), np.asarray(predicted_classes), 2)
        total = MetricTotals(("car", "pedestrian"))
        total.add(result)
        return total.result(), assignment

    def test_missing_prediction_counts_every_gt(self):
        result, _ = self.measure([[1, 0], [0, 1]], [0, 1], np.zeros((0, 2)), [])
        self.assertEqual((result["true_positives"], result["false_positives"], result["false_negatives"]), (0, 0, 2))
        self.assertEqual(result["mean_gt_mask_iou"], 0)

    def test_duplicate_prediction_is_false_positive(self):
        result, _ = self.measure([[1, 0]], [0], [[1, 0], [1, 0]], [0, 0])
        self.assertEqual((result["true_positives"], result["false_positives"], result["false_negatives"]), (1, 1, 0))
        self.assertEqual(result["precision_iou50"], 0.5)

    def test_wrong_category_is_not_a_successful_mask(self):
        result, _ = self.measure([[1, 0]], [0], [[1, 0]], [1])
        self.assertEqual((result["true_positives"], result["false_positives"], result["false_negatives"]), (0, 1, 1))
        self.assertEqual(result["mean_gt_mask_iou"], 0)
        self.assertEqual(result["geometry_only_diagnostic"]["mean_gt_mask_iou"], 1)

    def test_low_iou_counts_one_fp_and_one_fn(self):
        result, _ = self.measure([[1, 1, 0]], [0], [[1, 0, 1]], [0])
        self.assertEqual((result["false_positives"], result["false_negatives"]), (1, 1))
        self.assertAlmostEqual(result["mean_gt_mask_iou"], 1 / 3)

    def test_threshold_tie_maximizes_distinct_hits(self):
        result, _ = self.measure([[1, 1, 0, 0], [0, 0, 1, 1]], [0, 0],
                                 [[1, 1, 1, 1], [1, 0, 0, 0]], [0, 0])
        self.assertEqual(result["true_positives"], 2)
        self.assertEqual(result["recall_iou50"], 1)

    def test_prediction_in_empty_scene_is_false_positive(self):
        result, _ = self.measure(np.zeros((0, 2)), [], [[1, 0]], [0])
        self.assertEqual(result["false_positives"], 1)
        self.assertIsNone(result["mean_gt_mask_iou"])
        self.assertIsNone(result["per_class"]["pedestrian"]["recall_iou50"])


if __name__ == "__main__":
    unittest.main()
