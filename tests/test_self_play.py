# -*- coding: utf-8 -*-
"""自对弈样本、数据增强和回放缓冲测试。"""

import unittest

import numpy as np
import torch

from game import ACTION_SIZE, DRAW, EMPTY, OPEN, P1, P2, GameState, parts_to_action
from self_play import ReplayBuffer, TrainingExample, d4_augment, play_one_game, select_action


class ZeroNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return (
            torch.zeros(x.shape[0], 81, device=x.device) + self.anchor,
            torch.zeros(x.shape[0], 1, device=x.device) + self.anchor,
        )


class AugmentationTests(unittest.TestCase):
    def test_d4_keeps_policy_aligned_with_legal_plane(self):
        state = np.zeros((6, 9, 9), dtype=np.float32)
        policy = np.zeros(81, dtype=np.float32)
        row, col = 1, 2
        state[5, row, col] = 1.0
        policy[row * 9 + col] = 1.0
        augmented = d4_augment(state, policy)
        self.assertEqual(len(augmented), 8)
        locations = set()
        for transformed_state, transformed_policy in augmented:
            state_index = int(np.argmax(transformed_state[5]))
            policy_index = int(np.argmax(transformed_policy))
            self.assertEqual(state_index, policy_index)
            self.assertEqual(float(transformed_policy.sum()), 1.0)
            self.assertTrue(transformed_state.flags.c_contiguous)
            locations.add(policy_index)
        self.assertEqual(len(locations), 8)


class SamplingTests(unittest.TestCase):
    def test_zero_temperature_chooses_maximum_legal_action(self):
        policy = np.zeros(81, dtype=np.float32)
        policy[[2, 3, 4]] = [0.1, 0.7, 0.2]
        move = select_action(policy, [2, 3, 4], 0.0, np.random.default_rng(1))
        self.assertEqual(move, 3)

    def test_positive_temperature_never_selects_illegal_action(self):
        policy = np.ones(81, dtype=np.float32) / 81
        rng = np.random.default_rng(2)
        choices = {select_action(policy, [10, 20], 1.0, rng) for _ in range(100)}
        self.assertEqual(choices, {10, 20})

    def test_invalid_weights_fall_back_without_illegal_move(self):
        policy = np.zeros(81, dtype=np.float32)
        policy[10] = np.nan
        policy[20] = -1.0
        move = select_action(policy, [10, 20], 1.0, np.random.default_rng(8))
        self.assertIn(move, (10, 20))


class ReplayBufferTests(unittest.TestCase):
    def test_capacity_augmentation_and_batch_shapes(self):
        example = TrainingExample(
            np.zeros((6, 9, 9), dtype=np.float32),
            np.ones(81, dtype=np.float32) / 81,
            -1.0,
        )
        buffer = ReplayBuffer(capacity=10)
        buffer.add([example] * 12)
        self.assertEqual(len(buffer), 10)
        states, policies, values = buffer.sample(4, np.random.default_rng(3))
        self.assertEqual(states.shape, (4, 6, 9, 9))
        self.assertEqual(policies.shape, (4, 81))
        self.assertEqual(values.shape, (4, 1))
        self.assertTrue((values == -1).all())


class SelfPlayTests(unittest.TestCase):
    def test_near_terminal_losing_move_gets_negative_label(self):
        board = [EMPTY] * ACTION_SIZE
        pattern = [P1, P2, P1, P1, P1, P2, P2, EMPTY, P2]
        for cell, player in enumerate(pattern):
            if player != EMPTY:
                board[parts_to_action(8, cell)] = player
        state = GameState(
            tuple(board),
            tuple([P1] * 3 + [P2] * 4 + [DRAW, OPEN]),
            P1,
            8,
        )
        result = play_one_game(
            ZeroNet(),
            simulations=2,
            rng=np.random.default_rng(4),
            initial_state=state,
        )
        self.assertEqual(result.winner, P2)
        self.assertEqual(result.moves, 1)
        self.assertEqual(len(result.examples), 1)
        self.assertEqual(result.examples[0].value, -1.0)
        self.assertEqual(result.examples[0].policy[parts_to_action(8, 7)], 1.0)

    def test_complete_self_play_game_finishes(self):
        result = play_one_game(
            ZeroNet(),
            simulations=2,
            temperature_moves=10,
            rng=np.random.default_rng(5),
        )
        self.assertGreater(result.moves, 0)
        self.assertLessEqual(result.moves, 81)
        self.assertEqual(len(result.examples), result.moves)
        self.assertTrue(all(example.value in (-1.0, 0.0, 1.0) for example in result.examples))


if __name__ == "__main__":
    unittest.main(verbosity=2)
