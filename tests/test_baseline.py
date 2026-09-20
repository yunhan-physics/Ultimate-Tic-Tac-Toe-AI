# -*- coding: utf-8 -*-
"""基线 AI 与自动比赛工具测试。"""

import unittest

import numpy as np

from baseline import heuristic_action, play_game, play_match, random_action
from game import (
    ACTION_SIZE,
    ANY_BOARD,
    EMPTY,
    OPEN,
    P1,
    P2,
    GameState,
    apply_move,
    legal_moves,
    new_game,
    parts_to_action,
)


def state_with_pieces(pieces, *, to_play=P1, next_board=ANY_BOARD):
    board = [EMPTY] * ACTION_SIZE
    for small_board, cell, player in pieces:
        board[parts_to_action(small_board, cell)] = player
    return GameState(tuple(board), (OPEN,) * 9, to_play, next_board)


class BaselinePolicyTests(unittest.TestCase):
    def test_random_action_is_legal(self):
        rng = np.random.default_rng(1)
        state = new_game()
        for _ in range(40):
            move = random_action(state, rng)
            self.assertIn(move, legal_moves(state))
            state, done, _ = apply_move(state, move)
            if done:
                break

    def test_heuristic_takes_immediate_small_board_win(self):
        state = state_with_pieces(
            [(4, 0, P1), (4, 1, P1)],
            to_play=P1,
            next_board=4,
        )
        self.assertEqual(
            heuristic_action(state, np.random.default_rng(2)),
            parts_to_action(4, 2),
        )

    def test_heuristic_blocks_immediate_capture(self):
        state = state_with_pieces(
            [(0, 0, P2), (0, 1, P2)],
            to_play=P1,
            next_board=0,
        )
        self.assertEqual(
            heuristic_action(state, np.random.default_rng(3)),
            parts_to_action(0, 2),
        )

    def test_policies_reject_terminal_state(self):
        terminal = GameState((EMPTY,) * 81, (P1,) * 5 + (P2,) * 4, P1, ANY_BOARD)
        with self.assertRaises(ValueError):
            random_action(terminal)
        with self.assertRaises(ValueError):
            heuristic_action(terminal)


class AutomatedGameTests(unittest.TestCase):
    def test_random_game_finishes_consistently(self):
        result = play_game(random_action, random_action, np.random.default_rng(4))
        self.assertLessEqual(result.moves, 81)
        self.assertEqual(result.p1_score + result.p2_score <= 9, True)
        expected = P1 if result.p1_score > result.p2_score else P2 if result.p2_score > result.p1_score else EMPTY
        self.assertEqual(result.winner, expected)

    def test_match_accounting_and_reproducibility(self):
        report1 = play_match(heuristic_action, random_action, games=20, seed=5)
        report2 = play_match(heuristic_action, random_action, games=20, seed=5)
        self.assertEqual(report1, report2)
        self.assertEqual(report1["wins"] + report1["losses"] + report1["draws"], 20)
        self.assertGreaterEqual(report1["score_rate"], 0.0)
        self.assertLessEqual(report1["score_rate"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
