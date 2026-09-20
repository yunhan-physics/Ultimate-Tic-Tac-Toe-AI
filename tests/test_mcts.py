# -*- coding: utf-8 -*-
"""MCTS 的合法性、视角符号和战术选择测试。"""

import unittest

import numpy as np
import torch

from game import (
    ACTION_SIZE,
    ANY_BOARD,
    DRAW,
    EMPTY,
    OPEN,
    P1,
    P2,
    GameState,
    legal_moves,
    new_game,
    parts_to_action,
    terminal_info,
)
from mcts import _backpropagate, Node, search, terminal_value


class ZeroNet(torch.nn.Module):
    """给所有动作相同先验、价值恒为零的可控测试网络。"""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        batch = x.shape[0]
        return (
            torch.zeros(batch, 81, device=x.device) + self.anchor,
            torch.zeros(batch, 1, device=x.device) + self.anchor,
        )


def near_terminal_winning_state():
    board = [EMPTY] * ACTION_SIZE
    board[parts_to_action(8, 0)] = P1
    board[parts_to_action(8, 1)] = P1
    statuses = [P1] * 4 + [P2] * 4 + [OPEN]
    return GameState(tuple(board), tuple(statuses), P1, 8)


class PerspectiveTests(unittest.TestCase):
    def test_terminal_value_uses_node_player_not_last_mover(self):
        state = GameState((EMPTY,) * 81, (P1,) * 5 + (P2,) * 4, P2, ANY_BOARD)
        done, winner = terminal_info(state)
        self.assertTrue(done)
        self.assertEqual(terminal_value(state, winner), -1.0)

    def test_backpropagation_flips_sign_each_ply(self):
        root = Node(new_game())
        child = Node(new_game())
        leaf = Node(new_game())
        _backpropagate([root, child, leaf], 1.0)
        self.assertEqual((root.value, child.value, leaf.value), (1.0, -1.0, 1.0))


class SearchTests(unittest.TestCase):
    def test_policy_contains_only_legal_actions(self):
        state = new_game()
        policy, value = search(state, ZeroNet(), simulations=20)
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)
        self.assertTrue(np.isfinite(value))
        illegal = set(range(81)) - set(legal_moves(state))
        self.assertTrue(all(policy[action] == 0 for action in illegal))

    def test_search_finds_immediate_final_capture(self):
        state = near_terminal_winning_state()
        policy, _ = search(state, ZeroNet(), simulations=80, c_puct=1.5)
        winning_move = parts_to_action(8, 2)
        self.assertEqual(int(np.argmax(policy)), winning_move)
        self.assertGreater(policy[winning_move], 0.5)

    def test_terminal_state_skips_network(self):
        state = GameState((EMPTY,) * 81, (P1,) * 5 + (P2,) * 4, P2, ANY_BOARD)
        net = ZeroNet()
        policy, value = search(state, net, simulations=10)
        self.assertEqual(net.calls, 0)
        self.assertEqual(float(policy.sum()), 0.0)
        self.assertEqual(value, -1.0)

    def test_last_mover_can_lose_in_score_mode(self):
        # P1 被迫填完最后一个小棋盘，但最终占领数仍以 3:4 落后。
        board = [EMPTY] * ACTION_SIZE
        pattern = [P1, P2, P1, P1, P1, P2, P2, EMPTY, P2]
        for cell, player in enumerate(pattern):
            if player != EMPTY:
                board[parts_to_action(8, cell)] = player
        statuses = [P1] * 3 + [P2] * 4 + [DRAW, OPEN]
        state = GameState(tuple(board), tuple(statuses), P1, 8)
        only_move = parts_to_action(8, 7)
        self.assertEqual(legal_moves(state), [only_move])
        policy, value = search(state, ZeroNet(), simulations=4)
        self.assertEqual(policy[only_move], 1.0)
        self.assertEqual(value, -1.0)

    def test_prediction_cache_reuses_root_evaluation(self):
        net = ZeroNet()
        cache = {}
        search(new_game(), net, simulations=1, cache=cache)
        first_calls = net.calls
        search(new_game(), net, simulations=1, cache=cache)
        self.assertEqual(net.calls, first_calls)
        self.assertGreaterEqual(len(cache), 2)

    def test_noise_is_reproducible_with_seed(self):
        kwargs = dict(simulations=12, add_noise=True, dirichlet_alpha=0.3)
        first, _ = search(new_game(), ZeroNet(), rng=np.random.default_rng(9), **kwargs)
        second, _ = search(new_game(), ZeroNet(), rng=np.random.default_rng(9), **kwargs)
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main(verbosity=2)
