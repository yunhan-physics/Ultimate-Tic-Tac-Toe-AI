# -*- coding: utf-8 -*-
"""启发式课程数据的快速测试。"""

import unittest

import numpy as np

from baseline import heuristic_policy
from curriculum import generate_teacher_game
from game import P1, GameState, OPEN, EMPTY, parts_to_action


class CurriculumTests(unittest.TestCase):
    def test_heuristic_policy_is_normalized_and_takes_capture(self):
        board = [EMPTY] * 81
        board[parts_to_action(4, 0)] = P1
        board[parts_to_action(4, 1)] = P1
        state = GameState(tuple(board), (OPEN,) * 9, P1, 4)
        policy = heuristic_policy(state)
        self.assertAlmostEqual(float(policy.sum()), 1.0)
        self.assertEqual(policy[parts_to_action(4, 2)], 1.0)

    def test_teacher_game_labels_and_shapes(self):
        examples = generate_teacher_game(np.random.default_rng(30), max_random_opening=3)
        self.assertGreater(len(examples), 0)
        for example in examples:
            self.assertEqual(example.state.shape, (6, 9, 9))
            self.assertEqual(example.policy.shape, (81,))
            self.assertAlmostEqual(float(example.policy.sum()), 1.0)
            self.assertIn(example.value, (-1.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
