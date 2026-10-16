"""Gradient and compatibility contracts for the explicit small-instance loss policy."""
import unittest

import torch
from torch.nn import functional as F

from maskpls_b4dl.config import MaskPLSConfig
from maskpls_b4dl.criterion import MaskPLSCriterion, mask_binary_cross_entropy


class BalancedMaskBCETests(unittest.TestCase):
    def test_positive_and_negative_groups_have_equal_weight_for_different_sizes(self):
        logits = torch.zeros(2, 8, requires_grad=True)
        truth = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
                              [1., 1., 1., 1., 0., 0., 0., 0.]])
        mask_binary_cross_entropy(logits, truth, "instance_balanced").backward()
        positive_gradient = (logits.grad * truth).sum(1)
        negative_gradient = (logits.grad * (1 - truth)).sum(1)
        torch.testing.assert_close(positive_gradient, torch.tensor([-.125, -.125]))
        torch.testing.assert_close(negative_gradient, torch.tensor([.125, .125]))

    def test_extra_negatives_do_not_dilute_a_single_positive(self):
        gradients = []
        for points in (2, 100):
            logits = torch.zeros(1, points, requires_grad=True)
            truth = torch.zeros_like(logits); truth[0, 0] = 1
            mask_binary_cross_entropy(logits, truth, "instance_balanced").backward()
            gradients.append(logits.grad[0, 0])
        torch.testing.assert_close(gradients[0], gradients[1])

    def test_default_retains_exact_point_mean_objective(self):
        prediction = torch.tensor([[2., -1., .2]])
        truth = torch.tensor([[1., 0., 0.]])
        torch.testing.assert_close(mask_binary_cross_entropy(prediction, truth),
                                   F.binary_cross_entropy_with_logits(prediction, truth), rtol=0, atol=0)
        self.assertEqual(MaskPLSConfig.from_dict({}).mask_bce_reduction, "point_mean")
        with self.assertRaisesRegex(ValueError, "mask_bce_reduction"):
            MaskPLSConfig.from_dict({"mask_bce_reduction": "silently_drop_background"})

    def test_all_positive_target_is_finite_and_missing_positive_is_rejected(self):
        logits = torch.zeros(1, 3, requires_grad=True)
        value = mask_binary_cross_entropy(logits, torch.ones_like(logits), "instance_balanced")
        torch.testing.assert_close(value, F.softplus(torch.tensor(0.)))
        value.backward()
        self.assertTrue((logits.grad < 0).all())
        with self.assertRaisesRegex(ValueError, "positive support"):
            mask_binary_cross_entropy(logits, torch.zeros_like(logits), "instance_balanced")

    def test_background_ignore_gradients_and_roi_contract_survive_balancing(self):
        frame = dict(labels=torch.tensor([[1., 0., -1., -1.]]), class_ids=torch.tensor([0]),
                     semantic_labels=torch.tensor([1, 11, 0, 0]))
        output = dict(mask_logits=torch.full((3, 4), 2., requires_grad=True),
                      class_logits=torch.zeros(3, 18, requires_grad=True),
                      semantic_logits=torch.zeros(4, 20, requires_grad=True),
                      valid_points=torch.tensor([True, True, True, False]), auxiliary=[])
        config = MaskPLSConfig(mask_ignore_policy="background", mask_bce_reduction="instance_balanced")
        loss, _, assignment = MaskPLSCriterion(config)(output, frame)
        loss.backward()
        self.assertTrue((output["mask_logits"].grad[assignment, 2] > 0).all())
        self.assertTrue((output["mask_logits"].grad[:, 3] == 0).all())
        self.assertTrue((output["semantic_logits"].grad[2:] == 0).all())

    def test_matching_cost_does_not_change_with_loss_reduction(self):
        output = dict(mask_logits=torch.tensor([[2., -2., -1.], [-2., 2., -1.]]),
                      class_logits=torch.zeros(2, 18))
        labels = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        inputs = (output, labels, torch.tensor([1, 2]), torch.arange(3))
        first = MaskPLSCriterion(MaskPLSConfig()).match(*inputs)
        second = MaskPLSCriterion(MaskPLSConfig(mask_bce_reduction="instance_balanced")).match(*inputs)
        self.assertEqual(first, second)

    def test_sqrt_positive_caps_amplification_and_retains_negative_gradients(self):
        logits = torch.zeros(1, 10001, requires_grad=True)
        truth = torch.zeros_like(logits); truth[0, 0] = 1
        mask_binary_cross_entropy(logits, truth, "sqrt_positive").backward()
        torch.testing.assert_close(logits.grad[0, 0], torch.tensor(-.5 * 20 / 10020))
        self.assertTrue((logits.grad[0, 1:] > 0).all())

    def test_sqrt_positive_is_between_unweighted_and_full_balancing(self):
        magnitudes = []
        for mode in ("point_mean", "sqrt_positive", "instance_balanced"):
            logits = torch.zeros(1, 100, requires_grad=True)
            truth = torch.zeros_like(logits); truth[0, 0] = 1
            mask_binary_cross_entropy(logits, truth, mode).backward()
            magnitudes.append(float(logits.grad[0, 0].abs()))
        self.assertLess(magnitudes[0], magnitudes[1])
        self.assertLess(magnitudes[1], magnitudes[2])
        config = MaskPLSConfig.from_dict({"mask_bce_reduction": "sqrt_positive"})
        self.assertEqual(config.mask_bce_reduction, "sqrt_positive")


if __name__ == "__main__":
    unittest.main()
