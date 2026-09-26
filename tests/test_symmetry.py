"""Check D4 alignment and exact ensemble equivariance."""

import unittest

import numpy as np
import torch

from game import apply_move, encode_state, new_game
from model import UltimateNet
from self_play import ReplayBuffer, TrainingExample
from symmetry import inverse_policy, inverse_spatial, transform_spatial
from train import train_updates


class SymmetryTests(unittest.TestCase):
    def test_rule_encoding_transforms_with_first_move(self):
        original, _, _ = apply_move(new_game(), 11)
        encoded = torch.from_numpy(encode_state(original))
        marker = torch.zeros(9, 9)
        marker.reshape(81)[11] = 1
        for index in range(8):
            moved_action = int(transform_spatial(marker, index).reshape(81).argmax())
            rotated, _, _ = apply_move(new_game(), moved_action)
            self.assertTrue(torch.equal(
                transform_spatial(encoded, index), torch.from_numpy(encode_state(rotated))
            ))

    def test_spatial_transform_round_trip(self):
        board = torch.arange(81).reshape(1, 1, 9, 9)
        for index in range(8):
            self.assertTrue(torch.equal(inverse_spatial(transform_spatial(board, index), index), board))

    def test_d4_ensemble_policy_and_value(self):
        torch.manual_seed(11)
        model = UltimateNet(channels=8, blocks=1, value_hidden=16)
        model.d4_ensemble = True
        model.eval()
        state = torch.randn(1, 6, 9, 9)
        with torch.inference_mode():
            reference_policy, reference_value = model(state)
            for index in range(8):
                policy, value = model(transform_spatial(state, index))
                self.assertTrue(torch.allclose(inverse_policy(policy, index), reference_policy, atol=1e-5))
                self.assertTrue(torch.allclose(value, reference_value, atol=1e-5))

    def test_symmetry_training_update_is_finite(self):
        model = UltimateNet(channels=8, blocks=1, value_hidden=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        replay = ReplayBuffer(capacity=8)
        state = np.zeros((6, 9, 9), dtype=np.float32)
        state[5] = 1.0
        policy = np.ones(81, dtype=np.float32) / 81
        replay.add([TrainingExample(state, policy, 0.0)] * 4)
        result = train_updates(
            model, optimizer, replay, steps=1, batch_size=4, device=torch.device("cpu"),
            rng=np.random.default_rng(5), symmetry_fraction=0.5,
            symmetry_policy_weight=0.2, symmetry_value_weight=0.2,
        )
        self.assertTrue(all(np.isfinite(value) for value in result.values()))


if __name__ == "__main__":
    unittest.main()
