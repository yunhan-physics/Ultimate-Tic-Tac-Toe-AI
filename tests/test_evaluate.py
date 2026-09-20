# -*- coding: utf-8 -*-
"""评估协议的快速测试。"""

import unittest

import numpy as np
import torch

from baseline import random_action
from evaluate import _score_confidence_interval, evaluate_against, make_mcts_policy
from game import legal_moves, new_game
from model import UltimateNet


class EvaluationTests(unittest.TestCase):
    def test_mcts_policy_returns_legal_action(self):
        torch.manual_seed(20)
        model = UltimateNet(channels=8, blocks=1, value_hidden=16)
        policy = make_mcts_policy(model, simulations=2, c_puct=1.5)
        state = new_game()
        move = policy(state, np.random.default_rng(21))
        self.assertIn(move, legal_moves(state))

    def test_evaluation_accounting(self):
        torch.manual_seed(22)
        model = UltimateNet(channels=8, blocks=1, value_hidden=16)
        report = evaluate_against(
            model,
            random_action,
            games=4,
            simulations=1,
            workers=1,
            seed=23,
        )
        self.assertEqual(report["wins"] + report["losses"] + report["draws"], 4)
        self.assertGreaterEqual(report["score_rate"], 0.0)
        self.assertLessEqual(report["score_rate"], 1.0)

    def test_confidence_interval_contains_mean(self):
        outcomes = [1.0, 1.0, 0.5, 0.0]
        lower, upper = _score_confidence_interval(outcomes)
        self.assertLessEqual(lower, np.mean(outcomes))
        self.assertGreaterEqual(upper, np.mean(outcomes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
