# -*- coding: utf-8 -*-
"""损失函数和梯度更新的快速测试。"""

import unittest

import numpy as np
import torch

from model import UltimateNet
from self_play import ReplayBuffer, TrainingExample
from train import policy_value_loss, train_updates


class TrainingTests(unittest.TestCase):
    def test_policy_value_loss_is_finite(self):
        logits = torch.zeros(2, 81, requires_grad=True)
        values = torch.zeros(2, 1, requires_grad=True)
        policies = torch.zeros(2, 81)
        policies[0, 1] = 1
        policies[1, 2] = 1
        targets = torch.tensor([[1.0], [-1.0]])
        total, policy, value = policy_value_loss(logits, values, policies, targets)
        self.assertTrue(torch.isfinite(total))
        self.assertAlmostEqual(float(policy.detach()), float(np.log(81)), places=5)
        self.assertAlmostEqual(float(value.detach()), 1.0, places=5)

    def test_train_updates_changes_parameters(self):
        rng = np.random.default_rng(12)
        replay = ReplayBuffer(16)
        examples = []
        for index in range(8):
            state = np.zeros((6, 9, 9), dtype=np.float32)
            state[5, index // 9, index % 9] = 1
            policy = np.zeros(81, dtype=np.float32)
            policy[index] = 1
            examples.append(TrainingExample(state, policy, 1.0 if index % 2 else -1.0))
        replay.add(examples)

        net = UltimateNet(channels=8, blocks=1, value_hidden=16)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        before = net.policy_linear.weight.detach().clone()
        metrics = train_updates(net, optimizer, replay, 2, 4, torch.device("cpu"), rng)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertFalse(torch.equal(before, net.policy_linear.weight.detach()))

    def test_replay_round_trip(self):
        replay = ReplayBuffer(4)
        policy = np.ones(81, dtype=np.float32) / 81
        replay.add([TrainingExample(np.zeros((6, 9, 9), dtype=np.float32), policy, 0.0)])
        restored = ReplayBuffer.from_state_dict(replay.state_dict())
        self.assertEqual(restored.capacity, replay.capacity)
        self.assertEqual(len(restored), 1)
        self.assertTrue(np.array_equal(restored.as_list()[0].state, replay.as_list()[0].state))


if __name__ == "__main__":
    unittest.main(verbosity=2)
