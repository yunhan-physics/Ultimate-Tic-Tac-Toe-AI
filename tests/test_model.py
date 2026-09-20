# -*- coding: utf-8 -*-
"""策略价值网络的形状、数值和可训练性测试。"""

import copy
import unittest

import torch

from model import UltimateNet, count_parameters, masked_policy


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.net = UltimateNet(channels=16, blocks=2, value_hidden=32)

    def test_output_shape_range_and_finiteness(self):
        logits, value = self.net(torch.randn(3, 6, 9, 9))
        self.assertEqual(logits.shape, (3, 81))
        self.assertEqual(value.shape, (3, 1))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue((value >= -1).all() and (value <= 1).all())

    def test_default_parameter_count_is_expected_size(self):
        parameters = count_parameters(UltimateNet())
        self.assertGreater(parameters, 400_000)
        self.assertLess(parameters, 700_000)

    def test_group_norm_prediction_is_batch_independent(self):
        self.net.train()
        sample = torch.randn(1, 6, 9, 9)
        others = torch.randn(3, 6, 9, 9)
        single_policy, single_value = self.net(sample)
        batch_policy, batch_value = self.net(torch.cat([sample, others]))
        self.assertTrue(torch.allclose(single_policy, batch_policy[:1], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(single_value, batch_value[:1], atol=1e-5, rtol=1e-5))

    def test_shared_body_and_both_heads_receive_gradients(self):
        logits, value = self.net(torch.randn(4, 6, 9, 9))
        target = torch.tensor([0, 10, 20, 30])
        loss = torch.nn.functional.cross_entropy(logits, target) + value.square().mean()
        loss.backward()
        named = dict(self.net.named_parameters())
        for name in (
            "stem.0.weight",
            "body.0.conv1.weight",
            "policy_linear.weight",
            "value_linear2.weight",
        ):
            gradient = named[name].grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_state_dict_round_trip_preserves_outputs(self):
        self.net.eval()
        sample = torch.randn(2, 6, 9, 9)
        expected = self.net(sample)
        restored = UltimateNet(channels=16, blocks=2, value_hidden=32)
        restored.load_state_dict(copy.deepcopy(self.net.state_dict()))
        restored.eval()
        actual = restored(sample)
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))


class MaskedPolicyTests(unittest.TestCase):
    def test_illegal_actions_are_zero_and_legal_actions_sum_to_one(self):
        logits = torch.linspace(-2, 2, 162).reshape(2, 81)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[0, [0, 4, 80]] = True
        mask[1, 9:18] = True
        probabilities = masked_policy(logits, mask)
        self.assertTrue(torch.equal(probabilities[~mask], torch.zeros_like(probabilities[~mask])))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(2)))

    def test_all_illegal_row_is_rejected(self):
        with self.assertRaises(ValueError):
            masked_policy(torch.zeros(1, 81), torch.zeros(1, 81))

    def test_mask_shape_must_match(self):
        with self.assertRaises(ValueError):
            masked_policy(torch.zeros(2, 81), torch.ones(1, 81))


if __name__ == "__main__":
    unittest.main(verbosity=2)
