"""Loss gradient accounting and query rejection cases; run on the training server."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from grounded4d.criterion import mask_losses
from grounded4d.diagnostics import query_diagnostics
from grounded4d.study import gate


class BalancedLossTests(unittest.TestCase):
    def test_group_gradient_mass_is_balanced_for_a_sparse_target(self):
        labels = torch.tensor([1.] + [0.] * 9)
        gradients = {}
        for mode in ("mean", "foreground_balanced"):
            logits = torch.zeros(10, requires_grad=True)
            bce, _ = mask_losses(logits, labels, torch.ones(10, dtype=torch.bool), mode)
            bce.backward()
            gradients[mode] = logits.grad.abs()
        self.assertAlmostEqual(float(gradients["mean"][1:].sum() / gradients["mean"][0]), 9, places=5)
        self.assertAlmostEqual(float(gradients["foreground_balanced"][1:].sum()),
                               float(gradients["foreground_balanced"][0]), places=6)

    def test_background_duplication_does_not_dilute_foreground(self):
        values = []
        for count in (1, 100):
            logits = torch.tensor([-2.] + [-3.] * count)
            labels = torch.tensor([1.] + [0.] * count)
            values.append(mask_losses(logits, labels, torch.ones_like(labels, dtype=torch.bool), "foreground_balanced")[0])
        torch.testing.assert_close(values[0], values[1])

    def test_ignore_and_out_of_range_have_zero_gradient(self):
        logits = torch.tensor([-2., 1., 80., -80.], requires_grad=True)
        labels = torch.tensor([1., 0., -1., 1.])
        valid = torch.tensor([True, True, True, False])
        bce, _ = mask_losses(logits, labels, valid, "foreground_balanced")
        bce.backward()
        torch.testing.assert_close(logits.grad[2:], torch.zeros(2))
        torch.testing.assert_close(bce, F.binary_cross_entropy_with_logits(logits[:2], labels[:2]))

    def test_mean_recipe_and_dice_remain_unchanged(self):
        logits, labels, valid = torch.tensor([-2., 1., 3.]), torch.tensor([1., 0., 0.]), torch.ones(3, dtype=torch.bool)
        mean, dice = mask_losses(logits, labels, valid)
        balanced, balanced_dice = mask_losses(logits, labels, valid, "foreground_balanced")
        torch.testing.assert_close(mean, F.binary_cross_entropy_with_logits(logits, labels))
        torch.testing.assert_close(dice, balanced_dice)
        self.assertNotEqual(float(mean), float(balanced))

    def test_single_group_is_finite_and_invalid_mode_fails(self):
        for label in (0., 1.):
            bce, _ = mask_losses(torch.zeros(3), torch.full((3,), label), torch.ones(3, dtype=torch.bool), "foreground_balanced")
            self.assertTrue(torch.isfinite(bce))
        with self.assertRaises(ValueError):
            mask_losses(torch.zeros(1), torch.zeros(1), torch.ones(1, dtype=torch.bool), "unknown")


class QueryDiagnosticTests(unittest.TestCase):
    def diagnose(self, targets, target_classes, masks, classes, retained):
        targets, masks = np.asarray(targets, bool), np.asarray(masks, bool)
        count = targets.shape[1]
        return query_diagnostics(targets, target_classes, masks, classes, np.full(len(classes), .9), retained,
                                 2, np.column_stack((np.arange(count), np.zeros((count, 3)))),
                                 np.ones(count, bool), np.ones(count, bool), count)

    def test_filtering_is_separate_from_geometry(self):
        for category, retained, reason in ((2, False, "no_object"), (1, True, "wrong_class"),
                                           (0, False, "score_filtered"), (0, True, "hit")):
            d = self.diagnose([[1, 0]], [0], [[1, 0]], [category], [retained])
            self.assertEqual(d["reasons"][reason], 1)
            self.assertEqual(d["stage_distinct_hits"]["all_query_geometry"], 1)

    def test_query_reuse_is_not_counted_as_two_hits(self):
        d = self.diagnose([[1, 1, 0, 0], [0, 0, 1, 1]], [0, 0], [[1, 1, 1, 1]], [0], [True])
        self.assertEqual(d["stage_distinct_hits"]["all_query_geometry"], 1)
        self.assertEqual(d["reasons"]["query_competition"], 1)
        self.assertEqual(d["reasons"]["hit"], 1)

    def test_missing_masks_and_zero_memory_coverage(self):
        d = self.diagnose([[1, 0]], [0], [[0, 0]], [0], [False])
        self.assertEqual(d["reasons"]["mask_below_iou"], 1)
        d = query_diagnostics(np.array([[1, 0]], bool), [0], np.array([[1, 0]], bool), [0], [.9], [True],
                              2, np.zeros((2, 4)), np.ones(2, bool), np.ones(2, bool), 1, memory_indices=[1])
        self.assertEqual(d["gt"][0]["memory_foreground_points"], 0)


class ExpansionGateTests(unittest.TestCase):
    def test_fewer_predictions_alone_cannot_expand_training(self):
        control = dict(gt_instances=255, recall_iou50=.01, mean_gt_mask_iou=.01, f1_iou50=.01)
        candidate = dict(control, f1_iou50=.03)
        self.assertFalse(gate(control, candidate)["passed"])
        candidate.update(recall_iou50=.08, mean_gt_mask_iou=.03, f1_iou50=.04)
        self.assertTrue(gate(control, candidate)["passed"])


if __name__ == "__main__":
    unittest.main()
